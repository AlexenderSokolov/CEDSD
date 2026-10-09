"""Targeted full-data evaluation checks without model weights or a CUDA device."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import time
import unittest

from uica.common import config_digest, file_sha256, write_json, write_jsonl
from uica.full_evaluation import (
    _build_lock, aggregate_metrics, prediction_metrics, subgroup_metrics, validate_predictions,
)


def row(sample_id, label, margin, **extra):
    return {"sample_id": sample_id, "label": label, "logits": [0.0, float(margin)],
            "margin": float(margin), **extra}


class MetricsTests(unittest.TestCase):
    def test_tied_scores_and_frozen_threshold(self):
        result = prediction_metrics([row("r", 0, 1), row("f", 1, 1)], 1)
        self.assertEqual(result["eer"], .5)
        self.assertEqual(result["auc"], .5)
        self.assertEqual((result["fp"], result["fn"]), (1, 0))
        self.assertEqual((result["fpr"], result["fnr"]), (1.0, 0.0))
        expected_ll = (math.log1p(math.exp(1)) + math.log1p(math.exp(-1))) / 2
        self.assertAlmostEqual(result["ll"], expected_ll, places=14)

    def test_large_correct_margins_stable_ll(self):
        result = prediction_metrics([row("r", 0, -1e30), row("f", 1, 1e30)], 0)
        self.assertEqual(result["ll"], 0.0)
        self.assertEqual(result["eer"], 0.0)
        self.assertEqual(result["auc"], 1.0)

    def test_single_class_source_not_dropped(self):
        rows = [row("r", 0, -1), row("f", 1, 1)]
        metadata = [{"sample_id": "r", "source": "CommonVoice", "condition": "clean"},
                    {"sample_id": "f", "source": "FAD", "condition": "noise", "fad_seen": "unseen"}]
        groups = subgroup_metrics(rows, metadata, 0)
        cv = next(r for r in groups if (r["group_field"], r["group_value"]) == ("source", "CommonVoice"))
        self.assertEqual((cv["n"], cv["n_real"], cv["n_spoof"]), (1, 1, 0))
        self.assertIsNone(cv["eer"])
        self.assertIsNone(cv["auc"])
        self.assertIsNone(cv["fnr"])
        self.assertTrue(any(r["group_field"] == "fad_seen_unseen" and r["group_value"] == "unseen" for r in groups))

    def test_nonfinite_and_nonbinary_rejected(self):
        for rows, threshold in (([row("r", 0, float("inf"))], 0),
                                ([row("r", .4, 1)], 0), ([row("r", 0, 1)], float("nan"))):
            with self.assertRaises(ValueError):
                prediction_metrics(rows, threshold)

    def test_margin_is_float64_logit_subtraction(self):
        # Subtracting these FP32-representable logits in FP32 would round the
        # difference to 100000000; float64 subtraction preserves the unit.
        prediction = row("r", 0, 0, arm="acoustic", seed=17, checkpoint_id="abc",
                         component_id="c", data_role="test")
        prediction.update(logits=[100000000.0, -1.0], margin=-100000001.0)
        record = {"sample_id": "r", "label": 0, "component_id": "c", "role": "test"}
        validate_predictions([prediction], [record], "acoustic", 17, "abc")
        prediction["margin"] = -100000000.0
        with self.assertRaises(ValueError):
            validate_predictions([prediction], [record], "acoustic", 17, "abc")

    def test_identity_order_coverage_rejected(self):
        records = [{"sample_id": "a", "label": 0, "component_id": "a", "role": "test"},
                   {"sample_id": "b", "label": 1, "component_id": "b", "role": "test"}]
        rows = [row(r["sample_id"], r["label"], 1, arm="log", seed=29, checkpoint_id="abc",
                    component_id=r["component_id"], data_role="test") for r in records]
        validate_predictions(rows, records, "log", 29, "abc")
        for bad in (rows[::-1], rows[:1], [rows[0], rows[0]]):
            with self.assertRaises(ValueError):
                validate_predictions(bad, records, "log", 29, "abc")

    def test_seed_aggregation_sample_sd_and_partial_status(self):
        table = []
        for seed, margin in zip((17, 29, 43), (.5, 1.0, 1.5)):
            table.append({"scale": "full", "arm": "log", "seed": seed, "group_field": "overall",
                          "group_value": "all", "fit_status": "partial" if seed == 43 else "complete",
                          **prediction_metrics([row("r", 0, -margin), row("f", 1, margin)], 0)})
        result = aggregate_metrics(table)[0]
        self.assertEqual(result["n_seeds"], 3)
        self.assertEqual(result["n_partial_fits"], 1)
        self.assertFalse(result["three_seeds_complete"])
        self.assertEqual(result["eer_mean"], 0)
        self.assertGreater(result["ll_sd"], 0)
        with self.assertRaises(ValueError):
            aggregate_metrics(table + [table[0]])


class Budget:
    def __init__(self, root, started):
        self.root = Path(root)
        self.state = {"started_unix": started, "fits": {}}

    def save(self):
        pass


class LockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.started = time.time() - 145 * 3600
        self.budget = Budget(self.temporary.name, self.started)
        self.config = {"full": {"started_unix": self.started}}
        self.inference = {"precision": "fp32", "score": "float64_margin"}
        self.records = [
            {"sample_id": "train", "label": 1, "component_id": "train", "role": "train"},
            {"sample_id": "dev0", "label": 0, "component_id": "dev0", "role": "validation"},
            {"sample_id": "dev1", "label": 1, "component_id": "dev1", "role": "validation"},
            {"sample_id": "test0", "label": 0, "component_id": "test0", "role": "test"},
            {"sample_id": "test1", "label": 1, "component_id": "test1", "role": "test"},
        ]
        self.manifest_path = self.budget.root / "primary_manifest.jsonl"
        write_jsonl(self.manifest_path, self.records)
        self.manifest_id = file_sha256(self.manifest_path)
        write_json(self.budget.root / "prepared.json", {"manifest_identity": self.manifest_id, "base_identity": "published_weights"})
        self.folder = self.budget.root / "fits" / "full" / "acoustic_seed17"
        self.folder.mkdir(parents=True)
        (self.folder / "best.pt").write_bytes(b"immutable_checkpoint")
        self.checkpoint_id = file_sha256(self.folder / "best.pt")
        self.predictions = [row(r["sample_id"], r["label"], -1 if r["label"] == 0 else 1,
                                component_id=r["component_id"], data_role="validation", arm="acoustic", seed=17,
                                checkpoint_id=self.checkpoint_id, inference_id=config_digest(self.inference))
                            for r in self.records if r["role"] == "validation"]
        write_jsonl(self.folder / "dev_predictions.jsonl", self.predictions)
        self.result = {"arm": "acoustic", "seed": 17, "scale": "full", "status": "complete",
                       "train_n": 1, "selected_epoch": 3, "manifest_identity": self.manifest_id,
                       "checkpoint_id": self.checkpoint_id, "dev": prediction_metrics(self.predictions, 0)}
        coverage = config_digest([("train", 1, "train")])
        self.result["train_coverage_identity"] = coverage
        write_json(self.folder / "loader_coverage.json", {"sample_ids": ["train"], "coverage_identity": coverage,
                                                        "complete_frozen_train": True})
        write_json(self.folder / "result.json", self.result)

    def test_cutoff_partial_matrix_is_explicit(self):
        lock, matrix, records = _build_lock(self.config, self.budget, self.inference)
        self.assertFalse(lock["full18_complete"])
        self.assertEqual(len(lock["incomplete_full_fits"]), 17)
        self.assertEqual(len(matrix), 27)
        self.assertEqual(len(lock["models"]), 1)
        self.assertTrue(self.budget.state["training_closed"])
        self.assertEqual(len(records), 5)

    def test_no_test_lock_before_full18(self):
        self.config["full"]["started_unix"] = time.time()
        self.budget.state["started_unix"] = self.config["full"]["started_unix"]
        with self.assertRaises(RuntimeError):
            _build_lock(self.config, self.budget, self.inference)
        self.assertFalse((self.budget.root / "evaluation_lock.json").exists())

    def test_full_train_count_must_equal_manifest(self):
        self.result["train_n"] = 959
        write_json(self.folder / "result.json", self.result)
        with self.assertRaises(ValueError):
            _build_lock(self.config, self.budget, self.inference)

    def test_frozen_checkpoint_change_rejected(self):
        _build_lock(self.config, self.budget, self.inference)
        (self.folder / "best.pt").write_bytes(b"different_checkpoint")
        with self.assertRaises(ValueError):
            _build_lock(self.config, self.budget, self.inference)

    def test_frozen_threshold_change_rejected(self):
        _build_lock(self.config, self.budget, self.inference)
        self.result["dev"] = prediction_metrics(self.predictions, 1)
        write_json(self.folder / "result.json", self.result)
        with self.assertRaises(ValueError):
            _build_lock(self.config, self.budget, self.inference)

    def test_family_crosses_role_rejected(self):
        self.records[-1]["component_id"] = "train"
        write_jsonl(self.manifest_path, self.records)
        manifest_id = file_sha256(self.manifest_path)
        write_json(self.budget.root / "prepared.json", {"manifest_identity": manifest_id, "base_identity": "published_weights"})
        with self.assertRaises(ValueError):
            _build_lock(self.config, self.budget, self.inference)

    def test_usable_partial_remains_partial(self):
        self.result.update(status="partial", checkpoint_usable=True)
        write_json(self.folder / "result.json", self.result)
        lock, _, _ = _build_lock(self.config, self.budget, self.inference)
        self.assertEqual(lock["models"]["full_acoustic_seed17"]["fit_status"], "partial")
        self.assertEqual(len(lock["incomplete_full_fits"]), 18)


if __name__ == "__main__":
    unittest.main()
