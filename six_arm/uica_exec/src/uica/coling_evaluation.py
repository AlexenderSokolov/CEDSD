"""Approved COLING closeout: immutable freeze, unchanged FP32 inference, paired statistics.

Installed beside full_evaluation.py. Importing this module does not load torch,
read model predictions, open the test role, or launch model inference.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import gzip
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from .common import canonical_json, config_digest, file_sha256, read_json, read_jsonl, utc_now, write_json, write_jsonl
from .full_evaluation import (
    ARMS, METRICS, _fit_matrix, _public_matrix, _write_csv, prediction_metrics,
    subgroup_metrics, validate_predictions,
)
from .metrics import binary_metrics

SEEDS = (17, 29)
COMPARATORS = ("acoustic", "ae", "pooljoint", "ca")
ROLE_COUNTS = {"train": 34895, "validation": 10627, "test": 22289}
AUTHORIZATION = "APPROVED_PLAN.md 2026-10-07"
LOCK_NAME = "coling_evaluation_lock.json"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 1701
CI_LEVEL = 0.9875


def _clock_contract(config, budget):
    """Require the newly authorized clock, preserving the original research T0."""
    clock = budget.state.get("closeout_clock", {})
    contract = budget.state.get("closeout_contract", {})
    if clock.get("schema") != "uica.coling.closeout.clock.v1":
        raise ValueError("Authorized, frozen COLING closeout clock is required")
    started = float(config["full"]["started_unix"])
    if float(clock.get("original_research_T0", -1)) != started or float(budget.state.get("started_unix", -2)) != started:
        raise ValueError("Original research T0 cannot change")
    if clock.get("reset_on_resume") is not False:
        raise ValueError("Closeout clock must prohibit resume reset")
    if (contract.get("authorization") != AUTHORIZATION or contract.get("seeds") != list(SEEDS)
            or contract.get("arms") != list(ARMS) or contract.get("original_full_expected") != 18):
        raise ValueError("Explicit approved two-seed/six-arm contract is required")
    for field in ("contract_sha256", "clock_sha256"):
        value = contract.get(field, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing authorization identity: {field}")
    for field, identity in (("plan_path", "contract_sha256"), ("clock_path", "clock_sha256")):
        if contract.get(field) and file_sha256(contract[field]) != contract[identity]:
            raise ValueError(f"Authorized file changed: {field}")
    raw = clock["deadlines_unix"]
    expected = {"storage_restore_by": 6, "train_stop": 12, "primary_evaluation_by": 24,
                "new_experiments_stop": 36, "results_lock": 48, "delivery": 72}
    origin = float(clock["closeout_started_unix"])
    if not math.isfinite(origin) or origin < started:
        raise ValueError("Invalid closeout start")
    if any(float(raw[k]) != origin + hours * 3600 for k, hours in expected.items()):
        raise ValueError("Closeout deadlines differ from the approved immutable schedule")
    original_delivery = min(float(clock["original_delivery_deadline"]), started + 168 * 3600)
    deadlines = {k: min(float(v), original_delivery) for k, v in raw.items()}
    deadlines["train_stop"] = min(deadlines["train_stop"], started + 144 * 3600)
    effective = budget.state.get("effective_stage_deadlines")
    if effective is not None:
        if not isinstance(effective, dict) or any(k not in effective for k in deadlines):
            raise ValueError("Incomplete effective stage deadlines")
        deadlines = {k: min(v, float(effective[k])) for k, v in deadlines.items()}
    if any(not math.isfinite(v) for v in deadlines.values()):
        raise ValueError("Nonfinite effective deadline")
    return clock, contract, deadlines


def guard_inference(config, budget, *, starting=False):
    """Every batch stops by 48h; each newly started model must begin before 36h."""
    _, _, deadlines = _clock_contract(config, budget)
    now = time.time()
    if now >= deadlines["results_lock"]:
        raise RuntimeError("COLING results lock: no further model evaluation")
    if starting and now >= deadlines["new_experiments_stop"]:
        raise RuntimeError("COLING 36h cutoff: no new inference or diagnostic")
    return deadlines


def finalize_partial(config, budget, *, arm="log", seed=29):
    """Fresh dev-only reload when training hits its deadline before core finalization.

    This produces checkpoint_usable from actual unchanged FP32 inference. It
    does not reopen training, close the queue, create a test lock or use test.
    The caller must first verify the exact prior native run is terminal.
    """
    from .full_runtime import Dataset, INFERENCE, infer, load_delta, run_identity, score_rows
    import torch
    if (arm, seed) != ("log", 29):
        raise ValueError("Closeout partial finalizer is limited to approved Log29")
    _clock_contract(config, budget)
    guard_inference(config, budget, starting=True)
    if budget.state.get("active_session"):
        raise RuntimeError("Verify/reconcile the exact prior native run before finalization")
    folder = budget.root / "fits/full" / f"{arm}_seed{seed}"
    result_path = folder / "result.json"
    previous = read_json(result_path)
    if previous.get("status") == "complete":
        raise ValueError("Completed Log29 must not be relabeled partial")
    if (previous.get("arm"), previous.get("seed"), previous.get("scale", "full")) != (arm, seed, "full"):
        raise ValueError("Prior Log29 result identity mismatch")
    if previous.get("selection_pair_complete") is not True:
        raise ValueError("A durable selected checkpoint/prediction pair is required")
    before_path = folder / ("coling_before_partial_finalize_" + config_digest(previous) + ".json")
    if not before_path.exists():
        write_json(before_path, previous)
    dev = Dataset(budget.root / "primary_manifest.jsonl", budget.root / "cache", ["validation"])
    if len(dev.records) != ROLE_COUNTS["validation"]:
        raise ValueError("Dev-only reload needs complete frozen validation role")
    checkpoint_id = file_sha256(folder / "best.pt")
    original = read_jsonl(folder / "best_dev_predictions.jsonl")
    model = None
    with budget.gpu("coling_partial_log29_fresh_dev_fp32_reload"):
        try:
            model, saved = load_delta(folder / "best.pt", config, budget)
            if (saved["arm"], saved["seed"], saved.get("scale", "full")) != (arm, seed, "full"):
                raise ValueError("Selected best checkpoint model identity mismatch")
            train_records=sorted((r for r in read_jsonl(budget.root/'primary_manifest.jsonl')
                if r['role']=='train'),key=lambda r:r['sample_id'])
            coverage_id=config_digest([(r['sample_id'],r['label'],r['component_id']) for r in train_records])
            coverage=read_json(folder/'loader_coverage.json')
            manifest_id=file_sha256(budget.root/'primary_manifest.jsonl')
            if (len(train_records)!=ROLE_COUNTS['train'] or
                    coverage.get('sample_ids')!=[r['sample_id'] for r in train_records] or
                    coverage.get('coverage_identity')!=coverage_id or
                    saved.get('train_coverage_identity')!=coverage_id or
                    saved.get('train_manifest_identity')!=manifest_id or
                    saved.get('train_n')!=ROLE_COUNTS['train']):
                raise ValueError('Partial finalizer requires the exact saved full-train identity')
            restored = infer(model, dev, arm, seed, checkpoint_id, InferenceBudget(config, budget))
            guard_inference(config, budget)
            validate_predictions(restored, dev.records, arm, seed, checkpoint_id, config_digest(INFERENCE))
            if [(r["sample_id"], r["label"], r["component_id"]) for r in original] != [(r["sample_id"], r["label"], r["component_id"]) for r in restored]:
                raise ValueError("Selected/reloaded development IDs differ")
            selected_logits = np.asarray([r["logits"] for r in original], dtype=np.float64)
            restored_logits = np.asarray([r["logits"] for r in restored], dtype=np.float64)
            if selected_logits.shape != restored_logits.shape or not np.isfinite(selected_logits).all():
                raise ValueError("Selected development logits are malformed")
            difference = float(np.max(np.abs(selected_logits - restored_logits)))
            if difference > 2e-5:
                raise ValueError("Fresh FP32 checkpoint logits differ from selection")
            dev_metrics = score_rows(restored, saved["threshold"])
            _check_metrics(dev_metrics, prediction_metrics(restored, saved["threshold"]))
            result = {**previous, "status": "partial", "arm": arm, "seed": seed, "scale": "full",
                      "manifest_identity":manifest_id,"train_manifest_identity":manifest_id,
                      "train_coverage_identity":coverage_id,"train_n":ROLE_COUNTS['train'],
                      "dev_n":ROLE_COUNTS['validation'],"source_identity":saved['source_identity'],
                      "checkpoint_id": checkpoint_id, "selected_epoch": saved["epoch"], "dev": dev_metrics,
                      "checkpoint_usable": True, "reload_max_logit_abs_diff": difference,
                      "finalized_at": utc_now(), "finalize_run_id": run_identity(),
                      "finalize_reason": "user_requested_early_stop_dev_only_fresh_fp32_reload",
                      "previous_result_identity": config_digest(previous)}
            # No result usable=True is persisted until every fresh inference and
            # selected-logit check above actually passed.
            write_jsonl(folder / "dev_predictions.jsonl", restored)
            write_json(result_path, result)
            budget.state["fits"][f"full_{arm}_seed{seed}"] = result
            budget.save()
            print("COLING_PARTIAL_FINALIZE " + __import__("json").dumps({"arm": arm, "seed": seed, "status": result["status"], "checkpoint_id": checkpoint_id, "reload_max_logit_abs_diff": difference}), flush=True)
            return result
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()


class InferenceBudget:
    """Batch-level deadline hook for unchanged full_runtime.infer."""
    def __init__(self, config, budget):
        self.config, self.budget = config, budget

    def guard(self, *args, **kwargs):
        guard_inference(self.config, self.budget)
        return self.budget.guard(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.budget, name)


class FourRecordView:
    """Persistence unit only: the same original four records and cache items."""
    def __init__(self, dataset, start):
        self.dataset, self.start = dataset, start
        self.records = dataset.records[start:start + 4]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.dataset[self.start + index]


class SavedPartialInference(RuntimeError):
    def __init__(self, original, partial):
        super().__init__(str(original))
        self.original, self.partial = original, partial


def infer_with_partial(model, dataset, arm, seed, checkpoint, config, budget, partial_path, *, infer_fn=None):
    """Run unchanged infer on each original batch, saving before post-forward guard.

    Original batch4/order/pad80000/FP32 and model are unchanged. ``infer_fn`` is
    an injection point for CPU-only controls, never a production replacement.
    A partial file remains diagnostic evidence and is never a scoring input.
    """
    if infer_fn is None:
        from .full_runtime import infer
        infer_fn = infer
    path = Path(partial_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep any previous partial evidence; never overwrite or delete it.
    attempt = 0
    while path.exists():
        attempt += 1
        path = Path(partial_path).with_name(Path(partial_path).name + f".attempt{attempt}")
    rows = []
    proxy = InferenceBudget(config, budget)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for start in range(0, len(dataset), 4):
                deadlines = guard_inference(config, budget)
                if time.time() >= deadlines["results_lock"] - 60:
                    raise RuntimeError("Insufficient 60-second results-lock margin for another evaluation batch")
                view = FourRecordView(dataset, start)
                batch_rows = infer_fn(model, view, arm, seed, checkpoint, proxy)
                expected = [(r["sample_id"], r["label"], r["component_id"]) for r in view.records]
                if [(r["sample_id"], r["label"], r["component_id"]) for r in batch_rows] != expected:
                    raise ValueError("Original four-record batch identity/order changed")
                for row in batch_rows:
                    stream.write(canonical_json(row) + "\n")
                stream.flush()
                rows.extend(batch_rows)
                # Save first even when this exact final batch crossed 48h.
                guard_inference(config, budget)
            if [r["sample_id"] for r in rows] != [r["sample_id"] for r in dataset.records]:
                raise ValueError("All original full-test records must finish before completion")
            guard_inference(config, budget)
    except BaseException as error:
        partial = {"partial_path": str(path), "partial_n": len(rows),
                   "partial_sha256": file_sha256(path), "required_n": len(dataset),
                   "partial_status": "incomplete", "original_error_type": type(error).__name__}
        raise SavedPartialInference(error, partial) from error
    return rows, {"partial_path": str(path), "partial_n": len(rows), "partial_sha256": file_sha256(path),
                  "required_n": len(dataset), "partial_status": "full_forward_saved_pending_complete_validation"}


def _diagnostic_binding(root, path, manifest_id, cache_id):
    root, path = Path(root).resolve(), Path(path)
    if not path.is_absolute():
        path = root / path
    spec = read_json(path)
    if (spec.get("schema") != "uica.coling.diagnostics.v1" or
            spec.get("primary_manifest_sha256") != manifest_id or
            spec.get("cache_index_sha256") != cache_id):
        raise ValueError("Frozen diagnostics do not bind the current manifest/cache")
    if spec.get("max_receivers") != 2048 or spec.get("mapping_seeds") != [101, 102, 103]:
        raise ValueError("Diagnostic sample/mapping contract changed")
    files = spec.get("files", {})
    if set(files) != {"metadata", "subset", "donors_101", "donors_102", "donors_103"}:
        raise ValueError("Diagnostic inventory is incomplete")
    bound = {}
    for name, item in files.items():
        relative = Path(item["path"])
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root):
            raise ValueError("Diagnostic referenced files must remain under output root")
        if file_sha256(target) != item["sha256"]:
            raise ValueError(f"Diagnostic referenced file changed: {name}")
        bound[name] = {"path": str(relative), "sha256": item["sha256"]}
    return {"path": str(path), "sha256": file_sha256(path), "files": bound}


def _check_metrics(recorded, actual):
    for metric in ("n", "n_real", "n_spoof", "fp", "fn", "eer", "auc", "ll"):
        a, b = recorded.get(metric), actual[metric]
        if a is None or b is None:
            if a != b:
                raise ValueError(f"Development metric mismatch: {metric}")
        elif not math.isclose(float(a), float(b), abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError(f"Development metric mismatch: {metric}")


def _reload_binding(folder, result, dev, arm, seed, checkpoint, inference_id):
    """Validate persisted fresh-FP32 reload evidence; never manufacture usable=True."""
    if result.get("checkpoint_usable") is not True or result.get("selection_pair_complete") is not True:
        raise ValueError("Checkpoint has no successful selected/fresh-reload pair")
    witnessed = result.get("reload_max_logit_abs_diff")
    if witnessed is None or not math.isfinite(float(witnessed)) or not 0 <= float(witnessed) <= 2e-5:
        raise ValueError("Checkpoint has no valid fresh-FP32 reload witness")
    path = folder / "dev_predictions.jsonl"
    selected_path = folder / "best_dev_predictions.jsonl"
    rows, selected = read_jsonl(path), read_jsonl(selected_path)
    validate_predictions(rows, dev, arm, seed, checkpoint, inference_id)
    # Selection rows may carry the pre-save selection identity; check their
    # samples/inference/model explicitly without inventing checkpoint equality.
    if [(r["sample_id"], r["label"], r["component_id"]) for r in selected] != [
            (r["sample_id"], r["label"], r["component_id"]) for r in rows]:
        raise ValueError("Selected/reloaded development sample identity mismatch")
    for row in selected:
        if (row.get("arm"), row.get("seed"), row.get("inference_id")) != (arm, seed, inference_id):
            raise ValueError("Selected development model/FP32 identity mismatch")
    a, b = np.asarray([r["logits"] for r in selected]), np.asarray([r["logits"] for r in rows])
    if a.shape != (len(dev), 2) or not np.isfinite(a).all():
        raise ValueError("Malformed selected development logits")
    difference = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    if difference > 2e-5 or not math.isclose(difference, float(witnessed), abs_tol=1e-12, rel_tol=1e-10):
        raise ValueError("Saved reload witness differs from actual paired logits")
    threshold = float(result["dev"]["threshold"])
    actual = prediction_metrics(rows, threshold)
    _check_metrics(result["dev"], actual)
    return {"dev_prediction_identity": file_sha256(path), "selected_dev_prediction_identity": file_sha256(selected_path),
            "reload_max_logit_abs_diff": difference, "threshold": threshold, "dev": actual}


def choose_candidate(models):
    """Development EER -> LL -> Linear, only with two complete seeds per arm."""
    def complete_pair(arm):
        return all(f"full_{arm}_seed{s}" in models and models[f"full_{arm}_seed{s}"]["fit_status"] == "complete" for s in SEEDS)
    if not complete_pair("linear"):
        raise ValueError("Primary candidate requires two complete Linear fits")
    means = {}
    for arm in ("linear", "log"):
        if complete_pair(arm):
            means[arm] = {m: float(np.mean([models[f"full_{arm}_seed{s}"]["dev"][m] for s in SEEDS])) for m in ("eer", "ll")}
    candidate = min(means, key=lambda arm: (means[arm]["eer"], means[arm]["ll"], arm != "linear"))
    return {"arm": candidate, "rule": "two_complete_seed_mean_dev_EER_then_LL_then_Linear",
            "eligible_dev_means": means, "log_pair_complete": complete_pair("log"),
            "partial_log_fallback": not complete_pair("log")}


def freeze(config, budget, diagnostic_manifest_path=None, inference=None):
    """Read dev only; bind all checkpoints, thresholds, grouping and diagnostics."""
    if inference is None:
        from .full_runtime import INFERENCE
        inference = INFERENCE
    clock, contract, deadlines = _clock_contract(config, budget)
    if (budget.state.get("training_closed") is not True or
            budget.state.get("termination_reason") != "user_requested_early_stop"):
        raise RuntimeError("Authorized two-seed training queue must be terminal before test freeze")
    root = budget.root
    manifest_path, cache_path = root / "primary_manifest.jsonl", root / "cache/cache_index.json"
    manifest_id, cache_id = file_sha256(manifest_path), file_sha256(cache_path)
    prepared = read_json(root / "prepared.json")
    if prepared.get("manifest_identity", prepared.get("manifest")) != manifest_id or prepared.get("cache_index_identity") != cache_id:
        raise ValueError("Frozen prepared manifest/cache identity mismatch")
    records = read_jsonl(manifest_path)
    if dict(Counter(r["role"] for r in records)) != ROLE_COUNTS:
        raise ValueError("Frozen full split counts differ from 34895/10627/22289")
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate frozen sample IDs")
    family_roles = defaultdict(set)
    for row in records:
        if not row.get("component_id") or row["label"] not in (0, 1):
            raise ValueError("Missing recording-family component or binary label")
        family_roles[row["component_id"]].add(row["role"])
    if any(len(v) != 1 for v in family_roles.values()):
        raise ValueError("Recording dependency family crosses frozen roles")
    diagnostics = _diagnostic_binding(root, diagnostic_manifest_path or contract.get("diagnostic_manifest_path", "coling_diagnostic_manifest.json"), manifest_id, cache_id)
    matrix = _fit_matrix(budget)
    dev = sorted((r for r in records if r["role"] == "validation"), key=lambda r: r["sample_id"])
    train = sorted((r for r in records if r["role"] == "train"), key=lambda r: r["sample_id"])
    coverage_id = config_digest([(r["sample_id"], r["label"], r["component_id"]) for r in train])
    models = {}
    for item in matrix:
        if item["scale"] != "full" or item["seed"] not in SEEDS or item["status"] not in {"complete", "partial"}:
            continue
        if item["status"] == "partial" and (item["arm"], item["seed"]) != ("log", 29):
            raise ValueError("Only Log29 may be admitted as an explicitly labeled partial fit")
        result, folder = item["result"], item["folder"]
        if item["status"] == "partial" and result.get("checkpoint_usable") is not True:
            continue
        if (result.get("arm"), result.get("seed"), result.get("scale")) != (item["arm"], item["seed"], "full"):
            raise ValueError("Fit result identity differs from scientific queue")
        if result.get("manifest_identity") != manifest_id or result.get("train_n") != ROLE_COUNTS["train"]:
            raise ValueError("Fit used a different manifest/full training pool")
        coverage = read_json(folder / "loader_coverage.json")
        if (coverage.get("sample_ids") != [r["sample_id"] for r in train] or coverage.get("coverage_identity") != coverage_id
                or result.get("train_coverage_identity") != coverage_id or coverage.get("complete_frozen_train") is not True):
            raise ValueError("Exact frozen full-train loader coverage is required")
        checkpoint = file_sha256(folder / "best.pt")
        if result.get("checkpoint_id") != checkpoint:
            raise ValueError("Selected checkpoint SHA changed")
        reload = _reload_binding(folder, result, dev, item["arm"], item["seed"], checkpoint, config_digest(inference))
        models[item["key"]] = {"scale": "full", "arm": item["arm"], "seed": item["seed"],
            "checkpoint_id": checkpoint, "fit_status": item["status"], "selected_epoch": result["selected_epoch"],
            "train_n": result["train_n"], "train_coverage_identity": coverage_id,
            "loader_coverage_sha256": file_sha256(folder / "loader_coverage.json"), **reload}
    for arm in (*COMPARATORS, "linear"):
        if not all(models.get(f"full_{arm}_seed{s}", {}).get("fit_status") == "complete" for s in SEEDS):
            raise ValueError(f"Primary complete comparison pair is missing: {arm}")
    if models.get("full_log_seed17", {}).get("fit_status") != "complete":
        raise ValueError("Retained complete Log17 fit is missing")
    lock = {"schema": "uica.coling.evaluation.v1", "locked_at": utc_now(),
        "manifest_identity": manifest_id, "cache_index_identity": cache_id, "base_identity": prepared["base_identity"],
        "inference": inference, "inference_id": config_digest(inference), "role_counts": ROLE_COUNTS,
        "models": models, "candidate": choose_candidate(models), "diagnostics": diagnostics,
        "grouping_identity": config_digest([{k: r.get(k) for k in ("sample_id", "label", "component_id", "family_id", "source", "condition", "fad_seen", "seen", "file")} for r in records]),
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "ci_level": CI_LEVEL,
                      "resampling_unit": "admitted_recording_dependency_component", "seed_aggregation": "equal_mean_of_separate_seed_metrics"},
        "closeout_clock": clock, "closeout_contract": contract, "effective_stage_deadlines": deadlines,
        "full18_complete": all(r["status"] == "complete" for r in matrix if r["scale"] == "full"),
        "closeout12_complete": len(models) == 12 and all(m["fit_status"] == "complete" for m in models.values()),
        "three_seeds_complete": False, "original_full_expected": 18,
        "incomplete_original_full_fits": [r["key"] for r in matrix if r["scale"] == "full" and r["status"] != "complete"],
        "original_fit_matrix": _public_matrix(matrix),
        "scope_excluded": [r["key"] for r in matrix if r["scale"] == "small" or r["seed"] == 43]}
    path = root / LOCK_NAME
    if path.exists():
        existing = read_json(path)
        if {k: v for k, v in existing.items() if k != "locked_at"} != {k: v for k, v in lock.items() if k != "locked_at"}:
            raise ValueError("Immutable COLING evaluation lock changed; test cannot reopen")
        return existing
    write_json(path, lock)
    write_json(root / "tables/coling_fit_matrix.json", lock["original_fit_matrix"])
    _write_csv(root / "tables/coling_fit_matrix.csv", lock["original_fit_matrix"])
    budget.state["coling_evaluation_locked_at"] = lock["locked_at"]
    budget.save()
    print("COLING_FREEZE " + __import__("json").dumps({"candidate": lock["candidate"], "models": len(models), "closeout12_complete": lock["closeout12_complete"], "full18_complete": lock["full18_complete"]}), flush=True)
    return lock


def load_lock(config, budget):
    """Verify bound files without reselecting any model or development threshold."""
    clock, contract, deadlines = _clock_contract(config, budget)
    lock = read_json(budget.root / LOCK_NAME)
    if lock["closeout_clock"] != clock or lock["closeout_contract"] != contract or lock["effective_stage_deadlines"] != deadlines:
        raise ValueError("Frozen clock/authorization cannot change after test freeze")
    for name, identity in (("primary_manifest.jsonl", "manifest_identity"), ("cache/cache_index.json", "cache_index_identity")):
        if file_sha256(budget.root / name) != lock[identity]:
            raise ValueError(f"Frozen file changed: {name}")
    binding = _diagnostic_binding(budget.root, lock["diagnostics"]["path"], lock["manifest_identity"], lock["cache_index_identity"])
    if binding != lock["diagnostics"]:
        raise ValueError("Diagnostics changed after test freeze")
    for key, info in lock["models"].items():
        folder = budget.root / "fits/full" / f"{info['arm']}_seed{info['seed']}"
        for filename, field in (("best.pt", "checkpoint_id"), ("dev_predictions.jsonl", "dev_prediction_identity"),
                                ("best_dev_predictions.jsonl", "selected_dev_prediction_identity"), ("loader_coverage.json", "loader_coverage_sha256")):
            if file_sha256(folder / filename) != info[field]:
                raise ValueError(f"Frozen model file changed: {key}/{filename}")
    return lock


def aggregate_metrics(table):
    """Actual complete seeds only in main means; partial fits remain separate."""
    groups = defaultdict(list)
    for row in table:
        status = "complete" if row["fit_status"] == "complete" else "partial_descriptive"
        groups[(row["arm"], row["group_field"], row["group_value"], status)].append(row)
    summaries = []
    for (arm, field, value, status), rows in sorted(groups.items()):
        seeds = sorted(r["seed"] for r in rows)
        if len(set(seeds)) != len(seeds) or set(seeds) - set(SEEDS):
            raise ValueError("Duplicate or out-of-contract seed in summary")
        if len({tuple(r[k] for k in ("n", "n_real", "n_spoof")) for r in rows}) != 1:
            raise ValueError("Seeds do not cover identical subgroup samples")
        item = {"arm": arm, "group_field": field, "group_value": value, "fit_class": status,
                "seeds": seeds, "n_seeds": len(seeds), "two_complete_seeds": status == "complete" and seeds == list(SEEDS),
                "three_seeds_complete": False, "aggregation": "separate_seed_metrics_equal_mean_sample_sd",
                **{k: rows[0][k] for k in ("n", "n_real", "n_spoof")}}
        for metric in (*METRICS, "tnr", "tpr"):
            vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
            item[metric + "_mean"] = float(np.mean(vals)) if vals else None
            item[metric + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            item[metric + "_n_defined"] = len(vals)
        summaries.append(item)
    return summaries


def evaluate(config, budget):
    """Evaluate every retained model on the same exact 22289 test records."""
    from .full_runtime import Dataset, INFERENCE, load_delta, run_identity
    import torch
    lock = load_lock(config, budget)
    if lock["inference"] != INFERENCE:
        raise ValueError("FP32 inference contract differs from freeze")
    manifest = read_jsonl(budget.root / "primary_manifest.jsonl")
    records = sorted((r for r in manifest if r["role"] == "test"), key=lambda r: r["sample_id"])
    if len(records) != ROLE_COUNTS["test"]:
        raise ValueError("Full test coverage is required")
    test = Dataset(budget.root / "primary_manifest.jsonl", budget.root / "cache", ["test"])
    if test.records != records:
        raise ValueError("Runtime test manifest differs from evaluation lock")
    state_path = budget.root / "coling_evaluation_status.json"
    state = {"status": "running", "run_id": run_identity(), "lock_sha256": file_sha256(budget.root / LOCK_NAME),
             "required_models": list(lock["models"]), "evaluated": [], "test_n": len(records), "started_at": utc_now()}
    table = []
    write_json(state_path, state)
    current_partial = None
    current_key = None
    try:
        with budget.gpu("coling_uniform_full_test_fp32"):
            for key, info in lock["models"].items():
                current_key, current_partial = key, None
                folder = budget.root / "fits/full" / f"{info['arm']}_seed{info['seed']}"
                path = folder / "coling_test_predictions.jsonl"
                if path.exists():
                    rows = read_jsonl(path)
                else:
                    guard_inference(config, budget, starting=True)
                    model = None
                    try:
                        model, saved = load_delta(folder / "best.pt", config, budget)
                        if (saved["arm"], saved["seed"], float(saved["threshold"])) != (info["arm"], info["seed"], info["threshold"]):
                            raise ValueError("Reloaded checkpoint differs from frozen model/threshold")
                        rows, current_partial = infer_with_partial(model, test, info["arm"], info["seed"],
                            info["checkpoint_id"], config, budget, folder / "coling_test_predictions.partial.jsonl")
                        # Check the 48h lock again before admitting a completed forward pass.
                        guard_inference(config, budget)
                        validate_predictions(rows, records, info["arm"], info["seed"], info["checkpoint_id"], lock["inference_id"])
                        budget.guard(required_bytes=len(rows) * 2400, force=True)
                        write_jsonl(path, rows)
                    finally:
                        if model is not None:
                            del model
                        gc.collect()
                        torch.cuda.empty_cache()
                validate_predictions(rows, records, info["arm"], info["seed"], info["checkpoint_id"], lock["inference_id"])
                groups = subgroup_metrics(rows, records, info["threshold"])
                model_table = [{"scale": "full", "arm": info["arm"], "seed": info["seed"], "fit_status": info["fit_status"],
                                "checkpoint_id": info["checkpoint_id"], **group} for group in groups]
                table.extend(model_table)
                write_json(folder / "coling_test_metrics.json", model_table)
                state["evaluated"].append({"key": key, "prediction_sha256": file_sha256(path)})
                write_json(state_path, state)
                write_json(budget.root / "tables/coling_test_metrics.json", {"per_seed": table, "aggregate": aggregate_metrics(table)})
                _write_csv(budget.root / "tables/coling_test_per_seed.csv", table)
                _write_csv(budget.root / "tables/coling_test_seed_summary.csv", aggregate_metrics(table))
                overall = next(g for g in groups if g["group_field"] == "overall")
                print("COLING_TEST " + __import__("json").dumps({"fit": key, **overall}, allow_nan=False), flush=True)
    except BaseException as error:
        if isinstance(error, SavedPartialInference):
            current_partial = error.partial
        if current_partial is not None:
            state["interrupted_model"] = {"key": current_key, **current_partial}
        state.update(status="incomplete", ended_at=utc_now(), reason=str(error), error_type=type(error).__name__)
        state["missing_models"] = sorted(set(lock["models"]) - {r["key"] for r in state["evaluated"]})
        write_json(state_path, state)
        budget.state["coling_evaluation"] = state
        budget.save()
        raise
    state.update(status="complete", ended_at=utc_now(), closeout12_complete=lock["closeout12_complete"],
                 full18_complete=lock["full18_complete"], all_retained_models_full_test=True)
    write_json(state_path, state)
    budget.state["coling_evaluation"] = state
    budget.save()
    return state


def _array_metrics(labels, margins, threshold):
    value = binary_metrics(labels, margins, threshold)
    value["ll"] = float(np.logaddexp(0.0, np.where(labels == 1, -margins, margins)).mean())
    return {k: value[k] for k in ("eer", "auc", "ll", "fpr", "fnr")}


def shared_bootstrap(records, predictions, models, candidate, *, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, ci_level=CI_LEVEL, trace_callback=None):
    """One family draw per replicate for ALL models; metric then equal seed mean.

    ``predictions`` and ``models`` share full_arm_seedN keys. This CPU helper
    accepts synthetic controls but production statistics fixes 2000/1701/98.75%.
    """
    if not records or set(predictions) != set(models) or not 0 < ci_level < 1 or n_bootstrap < 1:
        raise ValueError("Complete aligned prediction inventory and valid bootstrap specification required")
    keys = list(models)
    labels = np.asarray([r["label"] for r in records])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Binary labels required")
    groups = [r.get("component_id") for r in records]
    if any(g is None or g == "" for g in groups):
        raise ValueError("Every test record requires an admitted recording-family component")
    identities = sorted(set(groups))
    group_positions = defaultdict(list)
    for index, family in enumerate(groups):
        group_positions[family].append(index)
    members = [np.asarray(group_positions[f], dtype=np.int64) for f in identities]
    scores = {}
    for key, info in models.items():
        rows = predictions[key]
        if [(r["sample_id"], r["label"], r["component_id"]) for r in rows] != [(r["sample_id"], r["label"], r["component_id"]) for r in records]:
            raise ValueError("Bootstrap models must have exact identical sample order/labels/families")
        logits = np.asarray([r["logits"] for r in rows], dtype=np.float64)
        if logits.shape != (len(records), 2) or not np.isfinite(logits).all():
            raise ValueError("Bootstrap logits malformed/nonfinite")
        scores[key] = logits[:, 1] - logits[:, 0]
        if not math.isfinite(float(info["threshold"])):
            raise ValueError("Bootstrap requires a finite frozen dev threshold")
    primary_keys = [f"full_{arm}_seed{s}" for arm in (candidate, *COMPARATORS) for s in SEEDS]
    if any(k not in models or models[k]["fit_status"] != "complete" for k in primary_keys):
        raise ValueError("Bootstrap primary comparisons require every complete matched seed")
    observed = {k: _array_metrics(labels, scores[k], models[k]["threshold"]) for k in keys}
    differences = {arm: {"eer": [], "ll": []} for arm in COMPARATORS}
    draws = []
    # All models are retained in each shared resample, including descriptive Log
    # if its second seed is partial. Partial Log never enters the primary pair.
    model_replicates = {k: {"eer": [], "ll": []} for k in keys}
    rng = np.random.default_rng(seed)
    attempts = 0
    invalid = 0
    while len(draws) < n_bootstrap and attempts < n_bootstrap * 10:
        attempts += 1
        selected_families = rng.integers(len(identities), size=len(identities))
        indices = np.concatenate([members[i] for i in selected_families])
        if len(np.unique(labels[indices])) != 2:
            invalid += 1
            continue
        sampled = {k: _array_metrics(labels[indices], scores[k][indices], models[k]["threshold"]) for k in keys}
        draws.append({"accepted_replicate": len(draws), "attempt": attempts, "sample_n": len(indices),
                      "family_draw_sha256": config_digest(selected_families.tolist())})
        if trace_callback is not None:
            trace_callback({"accepted_replicate":len(draws)-1,
                            "family_indices":selected_families.tolist()})
        for key in keys:
            for metric in ("eer", "ll"):
                model_replicates[key][metric].append(sampled[key][metric])
        for arm in COMPARATORS:
            for metric in ("eer", "ll"):
                deltas = [sampled[f"full_{candidate}_seed{s}"][metric] - sampled[f"full_{arm}_seed{s}"][metric] for s in SEEDS]
                differences[arm][metric].append(float(np.mean(deltas)))
        if len(draws) % 100 == 0:
            print(f"COLING_BOOTSTRAP {len(draws)}/{n_bootstrap} shared_family_replicates", flush=True)
    contrasts = []
    tail = (1.0 - ci_level) / 2.0
    for arm in COMPARATORS:
        item = {"candidate": candidate, "comparator": arm, "seeds": list(SEEDS), "n_seeds": len(SEEDS),
                "delta_direction": "candidate_minus_comparator", "ci_level": ci_level,
                "valid_replicates": len(draws), "requested_replicates": n_bootstrap,
                "per_seed": {}}
        for training_seed in SEEDS:
            a, b = observed[f"full_{candidate}_seed{training_seed}"], observed[f"full_{arm}_seed{training_seed}"]
            item["per_seed"][str(training_seed)] = {"eer_delta": None if a["eer"] is None else a["eer"] - b["eer"], "ll_delta": a["ll"] - b["ll"]}
        for metric in ("eer", "ll"):
            values = [item["per_seed"][str(s)][metric + "_delta"] for s in SEEDS]
            item[metric + "_delta"] = float(np.mean(values)) if all(v is not None for v in values) else None
            item[metric + "_ci"] = np.quantile(differences[arm][metric], [tail, 1 - tail]).tolist() if differences[arm][metric] else None
        item["eer_delta_pp"] = item["eer_delta"] * 100 if item["eer_delta"] is not None else None
        item["eer_ci_pp"] = [v * 100 for v in item["eer_ci"]] if item["eer_ci"] else None
        item["both_seed_eer_improve"] = all(item["per_seed"][str(s)]["eer_delta"] is not None and item["per_seed"][str(s)]["eer_delta"] < 0 for s in SEEDS)
        item["both_seed_ll_improve"] = all(item["per_seed"][str(s)]["ll_delta"] < 0 for s in SEEDS)
        item["eer_gate_pass"] = (len(draws) == n_bootstrap and item["both_seed_eer_improve"] and
                                item["eer_delta"] is not None and item["eer_delta"] < 0 and
                                item["eer_ci"] is not None and item["eer_ci"][1] < 0)
        contrasts.append(item)
    overall_eer_gate = all(r["eer_gate_pass"] for r in contrasts)
    ll_consistent = all(r["ll_delta"] < 0 and r["both_seed_ll_improve"] for r in contrasts)
    return {"schema": "uica.coling.shared_bootstrap.v1", "candidate": candidate,
            "n_bootstrap": n_bootstrap, "bootstrap_seed": seed, "ci_level": ci_level,
            "family_count": len(identities), "sample_n": len(records), "attempts": attempts,
            "single_class_skipped": invalid, "valid_replicates": len(draws),
            "family_inventory_sha256": config_digest(identities), "draws": draws,
            "seed_aggregation": "separate_seed_metrics_equal_mean_no_logit_pooling",
            "resampling_unit": "admitted_recording_dependency_component", "all_models": keys,
            "observed_model_metrics": observed, "all_model_bootstrap_seed_metrics": model_replicates,
            "contrast_bootstrap_deltas": differences, "contrasts": contrasts,
            "overall_eer_advantage_gate": overall_eer_gate, "ll_direction_consistent": ll_consistent,
            "eer_ranking_claim_only": overall_eer_gate and not ll_consistent,
            "scope": "conditional_on_existing_training_seeds_and_identified_families_unknown_speaker_reference_dependencies_remain"}


def statistics(config, budget):
    """Read saved complete predictions only; does not instantiate models or CUDA."""
    lock = load_lock(config, budget)
    records = sorted((r for r in read_jsonl(budget.root / "primary_manifest.jsonl") if r["role"] == "test"), key=lambda r: r["sample_id"])
    inventory, predictions, per_seed = {}, {}, []
    for key, info in lock["models"].items():
        path = budget.root / "fits/full" / f"{info['arm']}_seed{info['seed']}" / "coling_test_predictions.jsonl"
        if not path.exists():
            raise RuntimeError(f"Complete retained-model coverage required before primary statistics: {key}")
        rows = read_jsonl(path)
        validate_predictions(rows, records, info["arm"], info["seed"], info["checkpoint_id"], lock["inference_id"])
        predictions[key] = rows
        inventory[key] = {"path": str(path), "sha256": file_sha256(path), "n": len(rows)}
        per_seed.extend({"scale": "full", "arm": info["arm"], "seed": info["seed"], "fit_status": info["fit_status"],
                         "checkpoint_id": info["checkpoint_id"], **g} for g in subgroup_metrics(rows, records, info["threshold"]))
    trace_path=budget.root/"tables"/("coling_bootstrap_family_draws_"+uuid.uuid4().hex+".jsonl.gz")
    trace_path.parent.mkdir(parents=True,exist_ok=True)
    budget.guard(required_bytes=len(records)*BOOTSTRAP_N*8,force=True)
    with gzip.open(trace_path,'xt',encoding='utf-8') as trace:
        def save_draw(draw):
            trace.write(json.dumps(draw,separators=(',',':'))+'\n')
        result = shared_bootstrap(records, predictions, lock["models"], lock["candidate"]["arm"],trace_callback=save_draw)
    result['family_draw_trace']={'path':str(trace_path),'sha256':file_sha256(trace_path),
        'family_inventory_sha256':result['family_inventory_sha256'],'n_replicates':result['valid_replicates']}
    result.update(lock_sha256=file_sha256(budget.root / LOCK_NAME), prediction_inventory=inventory, computed_at=utc_now())
    table_root = budget.root / "tables"
    write_json(table_root / "coling_primary_statistics.json", result)
    _write_csv(table_root / "coling_primary_comparisons.csv", result["contrasts"])
    # Independently regenerate all subgroup tables from the saved inventory;
    # this remains a saved-data numerical correction after the model cutoff.
    write_json(table_root / "coling_test_metrics.json", {"per_seed": per_seed, "aggregate": aggregate_metrics(per_seed)})
    _write_csv(table_root / "coling_test_per_seed.csv", per_seed)
    _write_csv(table_root / "coling_test_seed_summary.csv", aggregate_metrics(per_seed))
    print("COLING_PRIMARY_STATISTICS " + __import__("json").dumps({"candidate": result["candidate"], "contrasts": result["contrasts"],
          "overall_eer_advantage_gate": result["overall_eer_advantage_gate"], "ll_direction_consistent": result["ll_direction_consistent"]}, allow_nan=False), flush=True)
    return result


def summarize(config, budget):
    """Emit evidence facts, explicit coverage gates and original-matrix omissions."""
    lock = load_lock(config, budget)
    stats_path = budget.root / "tables/coling_primary_statistics.json"
    metrics_path = budget.root / "tables/coling_test_metrics.json"
    stats = read_json(stats_path) if stats_path.exists() else None
    metrics = read_json(metrics_path) if metrics_path.exists() else {"per_seed": [], "aggregate": []}
    validated = []
    records = sorted((r for r in read_jsonl(budget.root / "primary_manifest.jsonl") if r["role"] == "test"), key=lambda r: r["sample_id"])
    for key, info in lock["models"].items():
        path = budget.root / "fits/full" / f"{info['arm']}_seed{info['seed']}" / "coling_test_predictions.jsonl"
        if path.exists():
            validate_predictions(read_jsonl(path), records, info["arm"], info["seed"], info["checkpoint_id"], lock["inference_id"])
            validated.append(key)
    complete = set(validated) == set(lock["models"])
    if stats is not None:
        if stats.get("lock_sha256") != file_sha256(budget.root / LOCK_NAME):
            raise ValueError("Statistics used a different freeze")
        for key, entry in stats["prediction_inventory"].items():
            if key not in lock["models"] or file_sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("Saved predictions changed after statistics")
        if set(stats["prediction_inventory"]) != set(lock["models"]):
            raise ValueError("Statistics lacks a retained model")
    facts = {"schema": "uica.coling.delivery_facts.v1", "candidate": lock["candidate"], "actual_seeds": list(SEEDS),
             "three_seeds_complete": False, "full18_complete": lock["full18_complete"],
             "closeout12_complete": lock["closeout12_complete"], "original_full_expected": 18,
             "original_incomplete_fits": lock["incomplete_original_full_fits"], "scope_excluded": lock["scope_excluded"],
             "retained_models": lock["models"], "validated_full_test_models": validated,
             "missing_models": sorted(set(lock["models"]) - set(validated)), "full_test_coverage_complete": complete,
             "test_n_per_model": ROLE_COUNTS["test"], "role_counts": lock["role_counts"],
             "primary_statistics_complete": stats is not None and stats["valid_replicates"] == BOOTSTRAP_N and complete,
             "overall_eer_advantage_gate": bool(stats and complete and stats["overall_eer_advantage_gate"]),
             "ll_direction_consistent": None if stats is None else stats["ll_direction_consistent"],
             "seed_summary": metrics["aggregate"], "closeout_clock": lock["closeout_clock"],
             "effective_stage_deadlines": lock["effective_stage_deadlines"], "created_at": utc_now()}
    write_json(budget.root / "tables/coling_delivery_facts.json", facts)
    print("COLING_DELIVERY_FACTS " + __import__("json").dumps({k: facts[k] for k in ("full_test_coverage_complete", "primary_statistics_complete", "overall_eer_advantage_gate", "closeout12_complete", "full18_complete", "missing_models")}), flush=True)
    return facts
