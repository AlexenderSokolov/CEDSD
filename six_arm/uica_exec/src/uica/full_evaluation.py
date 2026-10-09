"""Uniform, locked FP32 test evaluation for the approved full-data contract.

This module never trains or selects a model using test scores.  A checkpoint and
its development threshold must be frozen before any final prediction is read.
The numerical helpers can be imported without CUDA or model dependencies.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .common import config_digest, file_sha256, read_json, read_jsonl, utc_now, write_json, write_jsonl
from .metrics import binary_metrics

ARMS = ("acoustic", "ae", "pooljoint", "ca", "linear", "log")
SEEDS = (17, 29, 43)
SMALL_ARMS = ("acoustic", "ae", "log")
ARM_NAMES = {"acoustic": "Acoustic", "ae": "AE", "pooljoint": "PoolJoint",
             "ca": "CA", "linear": "Linear", "log": "Log"}
METRICS = ("eer", "auc", "ll", "fpr", "fnr", "fp", "fn", "accuracy")
TERMINAL = {"complete", "partial", "failed", "budget_unstarted", "not_started"}


def prediction_metrics(rows, threshold):
    """Float64 margins, tied-score ROC, stable LL and a dev-frozen threshold."""
    if not rows:
        raise ValueError("Cannot score an empty subgroup")
    labels = np.asarray([r["label"] for r in rows])
    logits = np.asarray([r["logits"] for r in rows], dtype=np.float64)
    if logits.shape != (len(rows), 2) or not np.isfinite(logits).all():
        raise ValueError("Each prediction must contain two finite FP32 logits")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Prediction labels must be binary")
    margin = logits[:, 1] - logits[:, 0]
    result = binary_metrics(labels, margin, float(threshold))
    # This signed softplus form is also stable when a correctly classified
    # positive margin is too large for exp or subtractive LL formulations.
    result["ll"] = float(np.logaddexp(0.0, np.where(labels == 1, -margin, margin)).mean())
    result["fp"] = int(np.sum((labels == 0) & (margin >= threshold)))
    result["fn"] = int(np.sum((labels == 1) & (margin < threshold)))
    result["metric_status"] = "defined" if result["n_real"] and result["n_spoof"] else "single_class_eer_auc_NA"
    return result


def validate_predictions(rows, records, arm, seed, checkpoint, inference_id=None):
    """Reject missing, duplicate, reordered or mislabeled cached predictions."""
    expected = [(r["sample_id"], int(r["label"]), r["component_id"]) for r in records]
    observed = [(r["sample_id"], r["label"], r["component_id"]) for r in rows]
    if expected != observed or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Predictions do not cover the frozen role exactly in sample-ID order")
    for row, record in zip(rows, records):
        if (row.get("arm"), row.get("seed"), row.get("checkpoint_id")) != (arm, seed, checkpoint):
            raise ValueError("Prediction model identity differs from evaluation lock")
        if inference_id is not None and row.get("inference_id") != inference_id:
            raise ValueError("Prediction FP32 inference identity differs from evaluation lock")
        if row.get("data_role", row.get("role")) != record["role"]:
            raise ValueError("Prediction role differs from frozen manifest")
        logits = np.asarray(row["logits"], dtype=np.float64)
        if logits.shape != (2,) or not np.isfinite(logits).all():
            raise ValueError("Nonfinite or malformed prediction logits")
        if not math.isfinite(float(row["margin"])) or float(logits[1] - logits[0]) != row["margin"]:
            raise ValueError("Saved margin differs from float64 logit subtraction")


def _fad_seen(record):
    value = record.get("fad_seen", record.get("seen"))
    if isinstance(value, bool):
        return "seen" if value else "unseen"
    if value is not None and str(value).lower() in {"seen", "unseen"}:
        return str(value).lower()
    # Only the source's explicit path protocol can supply a fallback.  No
    # unknown speaker is silently promoted to a seen/unseen identity claim.
    parts = str(record.get("file", "")).replace("\\", "/").lower().split("/")
    if "unseen" in parts or "test_unseen" in parts:
        return "unseen"
    if "seen" in parts or "test_seen" in parts:
        return "seen"
    return "unknown"


def subgroup_metrics(rows, records, threshold):
    """Use manifest metadata, including unknowns; never drop single-class groups."""
    by_id = {r["sample_id"]: r for r in records}
    groups = {("overall", "all"): list(rows)}
    for row in rows:
        record = by_id[row["sample_id"]]
        values = [("source", record.get("source") or "unknown"),
                  ("clean_noise", record.get("condition") or "unknown"),
                  ("generator_id", record.get("generator_id") or "unknown")]
        if str(record.get("source", "")).lower() == "fad":
            values.append(("fad_seen_unseen", _fad_seen(record)))
        for field, value in values:
            groups.setdefault((field, str(value)), []).append(row)
    return [{"group_field": field, "group_value": value,
             "small_sample": len(subset) < 30, **prediction_metrics(subset, threshold)}
            for (field, value), subset in sorted(groups.items())]


def aggregate_metrics(table):
    """Equal-weight means across seeds; never pool logits across different models."""
    groups = defaultdict(list)
    for row in table:
        groups[(row["scale"], row["arm"], row["group_field"], row["group_value"])].append(row)
    result = []
    for (scale, arm, field, value), rows in sorted(groups.items()):
        seeds = sorted(r["seed"] for r in rows)
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate seed in aggregate table")
        count_sets = {tuple(r[k] for k in ("n", "n_real", "n_spoof")) for r in rows}
        if len(count_sets) != 1:
            raise ValueError("Models were not evaluated on identical subgroup samples")
        summary = {"scale": scale, "arm": arm, "group_field": field, "group_value": value,
                   "seeds": "/".join(map(str, seeds)), "n_seeds": len(rows),
                   "three_seeds_complete": seeds == list(SEEDS) and all(r["fit_status"] == "complete" for r in rows),
                   "n_partial_fits": sum(r["fit_status"] != "complete" for r in rows),
                   "n": rows[0]["n"], "n_real": rows[0]["n_real"], "n_spoof": rows[0]["n_spoof"],
                   "aggregation": "separate_seed_metrics_equal_mean_sample_sd"}
        for metric in METRICS:
            values = [float(r[metric]) for r in rows if r[metric] is not None]
            summary[metric + "_mean"] = float(np.mean(values)) if values else None
            summary[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary[metric + "_n_defined"] = len(values)
        result.append(summary)
    return result


def _write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, ensure_ascii=False, sort_keys=True)
                                   if isinstance(value, (dict, list, tuple)) else value)
                             for key, value in row.items() if key in fields})
    os.replace(temporary, path)


def _fit_matrix(budget):
    matrix = []
    for scale, arms in (("full", ARMS), ("small", SMALL_ARMS)):
        for seed in SEEDS:
            for arm in arms:
                key = f"{scale}_{arm}_seed{seed}"
                folder = budget.root / "fits" / scale / f"{arm}_seed{seed}"
                state = budget.state.get("fits", {}).get(key, {})
                result_path = folder / "result.json"
                result = read_json(result_path) if result_path.exists() else dict(state)
                status = result.get("status", "not_started")
                matrix.append({"key": key, "scale": scale, "arm": arm, "seed": seed,
                               "status": status, "selected_epoch": result.get("selected_epoch"),
                               "train_n": result.get("train_n"),
                               "checkpoint_id": result.get("checkpoint_id"),
                               "elapsed_seconds": result.get("elapsed_seconds"),
                               "run_id": result.get("run_id"), "result_path": str(result_path),
                               "reason": result.get("reason", result.get("error", "")),
                               "checkpoint_usable": result.get("checkpoint_usable", status == "complete"),
                               "result": result, "folder": folder})
    return matrix


def _training_cutoff(config, budget):
    started = float(config["full"]["started_unix"])
    stored = budget.state.get("started_unix", budget.state.get("t0_unix", started))
    if float(stored) != started:
        raise ValueError("State T0 differs from configured original T0")
    return started + 144 * 3600


def _build_lock(config, budget, inference):
    manifest_path = budget.root / "primary_manifest.jsonl"
    manifest_id = file_sha256(manifest_path)
    prepared = read_json(budget.root / "prepared.json")
    if prepared.get("manifest_identity", prepared.get("manifest")) != manifest_id:
        raise ValueError("Prepared cache does not belong to the frozen full manifest")
    if prepared.get("cache_index_identity") is not None:
        if prepared["cache_index_identity"] != file_sha256(budget.root / "cache" / "cache_index.json"):
            raise ValueError("Frozen cache index changed after preparation")
    records = read_jsonl(manifest_path)
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate sample IDs in frozen primary manifest")
    if set(r["role"] for r in records) - {"train", "validation", "test"}:
        raise ValueError("Full evaluation requires the new train/validation/test roles")
    role_counts = Counter(r["role"] for r in records)
    if any(not role_counts[role] for role in ("train", "validation", "test")):
        raise ValueError("Frozen train/validation/test must all be nonempty")
    family_roles = defaultdict(set)
    for row in records:
        if not row.get("component_id"):
            raise ValueError("Missing family component identity")
        family_roles[row["component_id"]].add(row["role"])
    if any(len(roles) > 1 for roles in family_roles.values()):
        raise ValueError("An admitted dependency family crosses frozen partitions")
    matrix = _fit_matrix(budget)
    full = [r for r in matrix if r["scale"] == "full"]
    full_complete = all(r["status"] == "complete" for r in full)
    at_cutoff = time.time() >= _training_cutoff(config, budget)
    stopped_for_block = (budget.state.get("training_closed") is True and
                         budget.state.get("termination_reason") == "true_external_block")
    if not full_complete and not at_cutoff and not stopped_for_block:
        raise RuntimeError("Complete full18 or reach the fixed training cutoff before opening final test scores")
    small_started = any(r["status"] != "not_started" or r["folder"].exists()
                        for r in matrix if r["scale"] == "small")
    if small_started and not at_cutoff and not stopped_for_block:
        if not all(r["status"] == "complete" for r in matrix if r["scale"] == "small"):
            raise RuntimeError("The started small-scale queue must stop before the final test lock")
    dev = sorted((r for r in records if r["role"] == "validation"), key=lambda r: r["sample_id"])
    models = {}
    for item in matrix:
        if item["status"] not in {"complete", "partial"} or not item["checkpoint_usable"]:
            continue
        result, folder = item["result"], item["folder"]
        if not (folder / "result.json").exists():
            raise ValueError("A usable fit needs its persisted result.json")
        if (result.get("arm"), result.get("seed"), result.get("scale")) != (item["arm"], item["seed"], item["scale"]):
            raise ValueError("Fit result identity differs from the fixed queue")
        if result.get("manifest_identity") != manifest_id:
            raise ValueError("Fit used a different full manifest")
        if item["scale"] == "full" and result.get("train_n") != role_counts["train"]:
            raise ValueError("A full fit did not use every frozen train record")
        train_manifest = records if item["scale"] == "full" else read_jsonl(budget.root / "small_manifest.jsonl")
        train_records = sorted((r for r in train_manifest if r["role"] == "train"), key=lambda r: r["sample_id"])
        coverage = read_json(folder / "loader_coverage.json")
        coverage_identity = config_digest([(r["sample_id"], r["label"], r["component_id"]) for r in train_records])
        if (coverage.get("sample_ids") != [r["sample_id"] for r in train_records] or
                coverage.get("coverage_identity") != coverage_identity or
                result.get("train_coverage_identity") != coverage_identity or result.get("train_n") != len(train_records)):
            raise ValueError("Loader evidence does not cover the exact frozen training records")
        if item["scale"] == "full" and coverage.get("complete_frozen_train") is not True:
            raise ValueError("Full loader evidence does not certify the complete frozen train pool")
        checkpoint = file_sha256(folder / "best.pt")
        if result.get("checkpoint_id") != checkpoint:
            raise ValueError("Fit checkpoint changed after selection")
        threshold = float(result["dev"]["threshold"])
        if not math.isfinite(threshold):
            raise ValueError("Nonfinite frozen development threshold")
        predictions = read_jsonl(folder / "dev_predictions.jsonl")
        validate_predictions(predictions, dev, item["arm"], item["seed"], checkpoint, config_digest(inference))
        actual = prediction_metrics(predictions, threshold)
        for metric in ("n", "n_real", "n_spoof", "fp", "fn", "eer", "auc", "ll"):
            recorded = result["dev"].get(metric)
            value = actual[metric]
            if recorded is None or value is None:
                if recorded != value:
                    raise ValueError(f"Persisted development {metric} differs from predictions")
            elif not math.isclose(float(recorded), float(value), abs_tol=1e-10, rel_tol=1e-10):
                raise ValueError(f"Persisted development {metric} differs from predictions")
        models[item["key"]] = {"scale": item["scale"], "arm": item["arm"], "seed": item["seed"],
                               "checkpoint_id": checkpoint, "threshold": threshold,
                               "fit_status": item["status"], "selected_epoch": item["selected_epoch"],
                               "train_n": item["train_n"],
                               "train_coverage_identity": coverage_identity,
                               "dev_prediction_identity": file_sha256(folder / "dev_predictions.jsonl")}
    lock = {"manifest_identity": manifest_id, "base_identity": prepared["base_identity"],
            "cache_index_identity": prepared.get("cache_index_identity"),
            "inference": inference, "inference_id": config_digest(inference),
            "role_counts": dict(role_counts), "full18_complete": full_complete,
            "small_started": small_started, "models": models,
            "incomplete_full_fits": [r["key"] for r in full if r["status"] != "complete"],
            "locked_at": utc_now(), "started_unix": float(config["full"]["started_unix"]),
            "training_cutoff_unix": _training_cutoff(config, budget)}
    existing_path = budget.root / "evaluation_lock.json"
    if existing_path.exists():
        existing = read_json(existing_path)
        for key in ("models", "manifest_identity", "base_identity", "cache_index_identity", "inference_id", "role_counts"):
            if existing.get(key) != lock.get(key):
                raise ValueError(f"Final evaluation lock cannot change: {key}")
        lock = existing
    else:
        write_json(existing_path, lock)
    budget.state["training_closed"] = True
    budget.state["evaluation_locked_at"] = lock["locked_at"]
    budget.save()
    return lock, matrix, records


def _public_matrix(matrix):
    return [{k: v for k, v in row.items() if k not in {"result", "folder"}} for row in matrix]


def _prediction_export(rows, metadata, model_info, manifest_id):
    by_id = {r["sample_id"]: r for r in metadata}
    exported = []
    for row in rows:
        record = by_id[row["sample_id"]]
        margin = float(np.float64(row["logits"][1]) - np.float64(row["logits"][0]))
        exported.append({**row, "scale": model_info["scale"], "fit_status": model_info["fit_status"],
                         "manifest_identity": manifest_id, "threshold": model_info["threshold"],
                         "predicted_spoof": int(margin >= model_info["threshold"]),
                         "logit_real": row["logits"][0], "logit_spoof": row["logits"][1],
                         "source": record.get("source"), "generator_id": record.get("generator_id"),
                         "audio_condition": record.get("condition"),
                         "fad_seen": _fad_seen(record) if str(record.get("source", "")).lower() == "fad" else None,
                         "family_id": record.get("family_id", record["component_id"]),
                         "historical_use": record.get("historical_use", record.get("history")),
                         "identity_provenance": record.get("identity_provenance", record.get("provenance"))})
    return exported


def evaluate(config, budget):
    """Freeze all selected models, then evaluate each on exactly the full test set."""
    from .full_runtime import Dataset, INFERENCE, infer, load_delta, run_identity, score_rows
    import torch

    budget.guard(force=True)
    lock, matrix, manifest = _build_lock(config, budget, INFERENCE)
    tables = budget.root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "fit_matrix.csv", _public_matrix(matrix))
    write_json(tables / "fit_matrix.json", _public_matrix(matrix))
    test_records = sorted((r for r in manifest if r["role"] == "test"), key=lambda r: r["sample_id"])
    test = Dataset(budget.root / "primary_manifest.jsonl", budget.root / "cache", ["test"])
    if [r["sample_id"] for r in test.records] != [r["sample_id"] for r in test_records]:
        raise ValueError("Runtime dataset changed the frozen full-test sample set")
    metrics_table = []
    evaluation_state = {"status": "running", "run_id": run_identity(), "locked_at": lock["locked_at"],
                        "manifest_identity": lock["manifest_identity"], "test_n": len(test_records),
                        "model_count": len(lock["models"]), "evaluated": []}
    write_json(budget.root / "evaluation_status.json", evaluation_state)
    if lock["models"]:
        with budget.gpu("uniform_full_test_fp32"):
            # Model order is fixed by the scientific queue, never by dev/test scores.
            for item in matrix:
                key = item["key"]
                if key not in lock["models"]:
                    continue
                info, folder = lock["models"][key], item["folder"]
                budget.guard(force=True)
                predictions_path = folder / "full_test_predictions.jsonl"
                if predictions_path.exists():
                    rows = read_jsonl(predictions_path)
                    validate_predictions(rows, test_records, item["arm"], item["seed"], info["checkpoint_id"], lock["inference_id"])
                else:
                    model, saved = load_delta(folder / "best.pt", config, budget)
                    if float(saved["threshold"]) != info["threshold"]:
                        raise ValueError("Reloaded model threshold differs from frozen development threshold")
                    rows = infer(model, test, item["arm"], item["seed"], info["checkpoint_id"], budget)
                    validate_predictions(rows, test_records, item["arm"], item["seed"], info["checkpoint_id"], lock["inference_id"])
                    # Atomic JSONL serialization can transiently need two files.
                    budget.guard(required_bytes=len(rows) * 2400, force=True)
                    write_jsonl(predictions_path, rows)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                groups = subgroup_metrics(rows, test_records, info["threshold"])
                overall = next(r for r in groups if r["group_field"] == "overall")
                runtime_metrics = score_rows(rows, info["threshold"])
                for metric in ("eer", "auc", "ll", "fp", "fn", "n"):
                    a, b = overall[metric], runtime_metrics[metric]
                    if a is None or b is None:
                        if a != b:
                            raise ValueError("Runtime and table metrics disagree")
                    elif not math.isclose(float(a), float(b), abs_tol=1e-10, rel_tol=1e-10):
                        raise ValueError("Runtime and table metrics disagree")
                model_rows = [{"scale": info["scale"], "arm": item["arm"], "seed": item["seed"],
                               "fit_status": info["fit_status"], "selected_epoch": info["selected_epoch"],
                               "train_n": info["train_n"], "checkpoint_id": info["checkpoint_id"],
                               "manifest_identity": lock["manifest_identity"], **group} for group in groups]
                metrics_table.extend(model_rows)
                write_json(folder / "full_test_metrics.json", model_rows)
                exported = _prediction_export(rows, test_records, info, lock["manifest_identity"])
                _write_csv(folder / "full_test_predictions.csv", exported,
                           ["sample_id", "data_role", "label", "component_id", "family_id", "scale", "arm", "seed",
                            "fit_status", "checkpoint_id", "manifest_identity", "inference_id", "logit_real", "logit_spoof",
                            "margin", "threshold", "predicted_spoof", "source", "generator_id", "audio_condition", "fad_seen",
                            "historical_use", "identity_provenance", "run_id"])
                _write_csv(tables / "test_per_seed.csv", metrics_table)
                aggregate = aggregate_metrics(metrics_table)
                _write_csv(tables / "test_three_seed_summary.csv", aggregate)
                write_json(tables / "test_metrics.json", {"per_seed": metrics_table, "aggregate": aggregate})
                evaluation_state["evaluated"].append(key)
                write_json(budget.root / "evaluation_status.json", evaluation_state)
                print("FULL_TEST_METRICS " + json.dumps({"fit": key, **overall}, ensure_ascii=False, allow_nan=False), flush=True)
    else:
        _write_csv(tables / "test_per_seed.csv", [], ["scale", "arm", "seed", "group_field", "group_value", "n", "eer", "auc", "ll"])
        _write_csv(tables / "test_three_seed_summary.csv", [], ["scale", "arm", "group_field", "group_value", "n_seeds", "eer_mean", "auc_mean", "ll_mean"])
        write_json(tables / "test_metrics.json", {"per_seed": [], "aggregate": []})
    evaluation_state.update(status="complete" if lock["full18_complete"] else "partial", ended_at=utc_now(),
                            full18_complete=lock["full18_complete"], incomplete_full_fits=lock["incomplete_full_fits"])
    write_json(budget.root / "evaluation_status.json", evaluation_state)
    budget.state["evaluation"] = evaluation_state
    budget.save()
    print("FULL_EVALUATION_SUMMARY " + json.dumps(evaluation_state, ensure_ascii=False, allow_nan=False), flush=True)
    return evaluation_state


def _fmt(value, percent=False):
    if value is None:
        return "NA"
    return f"{100 * value:.3f}%" if percent else f"{value:.6f}"


def _utc(timestamp):
    return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()


def _comparison_rows(per_seed):
    index = {(r["scale"], r["arm"], r["seed"], r["group_field"], r["group_value"]): r for r in per_seed}
    comparisons = []
    for row in per_seed:
        if row["arm"] == "acoustic":
            continue
        base = index.get((row["scale"], "acoustic", row["seed"], row["group_field"], row["group_value"]))
        if base is None:
            continue
        comparisons.append({"scale": row["scale"], "arm": row["arm"], "baseline": "acoustic", "seed": row["seed"],
                            "group_field": row["group_field"], "group_value": row["group_value"], "n": row["n"],
                            "both_fits_complete": row["fit_status"] == base["fit_status"] == "complete",
                            "eer_delta_pp": (row["eer"] - base["eer"]) * 100 if row["eer"] is not None and base["eer"] is not None else None,
                            "auc_delta": row["auc"] - base["auc"] if row["auc"] is not None and base["auc"] is not None else None,
                            "ll_delta": row["ll"] - base["ll"], "direction": "method_minus_acoustic"})
    scale_rows = []
    for key, full in sorted(index.items()):
        scale, arm, seed, field, value = key
        if scale != "full" or arm not in SMALL_ARMS:
            continue
        small = index.get(("small", arm, seed, field, value))
        if small is None:
            continue
        scale_rows.append({"arm": arm, "seed": seed, "group_field": field, "group_value": value,
                           "full_train_n": full["train_n"], "small_train_n": small["train_n"],
                           "both_fits_complete": full["fit_status"] == small["fit_status"] == "complete",
                           "eer_delta_pp": (full["eer"] - small["eer"]) * 100 if full["eer"] is not None and small["eer"] is not None else None,
                           "auc_delta": full["auc"] - small["auc"] if full["auc"] is not None and small["auc"] is not None else None,
                           "ll_delta": full["ll"] - small["ll"], "direction": "full_minus_small_same_new_dev_test"})
    return comparisons, scale_rows


def _allocation_reconciliation(allocation, primary, audit, allocation_id, manifest_id):
    """A line count alone does not establish complete, frozen data accounting."""
    ids = [r.get("sample_id") for r in allocation]
    primary_ids = [r.get("sample_id") for r in primary]
    admitted = [r for r in allocation if r.get("role") in {"train", "validation", "test"}]
    allocated_rows = {r.get("sample_id"): r for r in admitted}
    exact_join = (len(primary_ids) == len(set(primary_ids)) and len(admitted) == len(primary) and
                  set(primary_ids) == set(allocated_rows) and
                  all((r.get("role"), r.get("label"), r.get("component_id")) ==
                      tuple(allocated_rows[r["sample_id"]].get(k) for k in ("role", "label", "component_id"))
                      for r in primary))
    checks = {"record_count69700": len(allocation) == 69700,
              "unique_nonempty_sample_ids": len(set(ids)) == len(ids) and all(ids),
              "primary_join_exact": exact_join, "audit_frozen": audit.get("status") == "frozen",
              "audit_record_count69700": audit.get("record_count") == 69700,
              "family_cross_partition_zero": audit.get("family_graph", {}).get("cross_partition_family_violations") == 0,
              "allocation_identity_matches_audit": audit.get("allocation_sha256") == allocation_id,
              "manifest_identity_matches_audit": audit.get("manifest_identity") == manifest_id}
    return {"passed": all(checks.values()), "checks": checks}


def summarize(config, budget):
    """Write actual data/fit/resource facts and bounded Chinese conclusions."""
    tables = budget.root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    allocation_path = budget.root / "allocation_manifest.jsonl"
    allocation = read_jsonl(allocation_path) if allocation_path.exists() else []
    primary_path = budget.root / "primary_manifest.jsonl"
    primary = read_jsonl(primary_path) if primary_path.exists() else []
    audit = read_json(budget.root / "data_audit.json") if (budget.root / "data_audit.json").exists() else {}
    prepared = read_json(budget.root / "prepared.json") if (budget.root / "prepared.json").exists() else {}
    lock = read_json(budget.root / "evaluation_lock.json") if (budget.root / "evaluation_lock.json").exists() else {}
    evaluation = read_json(budget.root / "evaluation_status.json") if (budget.root / "evaluation_status.json").exists() else {}
    metrics = read_json(tables / "test_metrics.json") if (tables / "test_metrics.json").exists() else {"per_seed": [], "aggregate": []}
    matrix = _fit_matrix(budget)
    public_matrix = _public_matrix(matrix)
    _write_csv(tables / "fit_matrix.csv", public_matrix)
    write_json(tables / "fit_matrix.json", public_matrix)
    if allocation:
        fields = ["sample_id", "manifest_row", "file", "label", "original_label", "source", "generator_id", "condition",
                  "original_partition", "role", "component_id", "family_id", "parent_ids", "reference_ids", "speaker_id",
                  "identity_provenance", "identity_roles", "historical_use", "history", "exclusion_reason", "isolation_reason",
                  "duration_seconds", "usable_window_seconds", "window_num_samples", "window_sha256", "pcm_sha256", "provenance"]
        fields += sorted(set().union(*(r.keys() for r in allocation)) - set(fields))
        _write_csv(tables / "full_allocation.csv", allocation, fields)
    distribution = Counter((r.get("role", "unknown"), r.get("source") or "unknown", r.get("label")) for r in allocation)
    distribution_rows = [{"role": role, "source": source, "label": label, "n": n}
                         for (role, source, label), n in sorted(distribution.items(), key=lambda v: tuple(map(str, v[0])))]
    _write_csv(tables / "allocation_by_source_role_label.csv", distribution_rows, ["role", "source", "label", "n"])
    within, scale = _comparison_rows(metrics["per_seed"])
    _write_csv(tables / "method_minus_acoustic.csv", within,
               ["scale", "arm", "baseline", "seed", "group_field", "group_value", "n", "both_fits_complete", "eer_delta_pp", "auc_delta", "ll_delta", "direction"])
    _write_csv(tables / "full_minus_small.csv", scale,
               ["arm", "seed", "group_field", "group_value", "full_train_n", "small_train_n", "both_fits_complete", "eer_delta_pp", "auc_delta", "ll_delta", "direction"])
    full = [r for r in matrix if r["scale"] == "full"]
    small = [r for r in matrix if r["scale"] == "small"]
    n_full = sum(r["status"] == "complete" for r in full)
    n_small = sum(r["status"] == "complete" for r in small)
    small_started = any(r["status"] != "not_started" for r in small)
    small_complete = n_small == 9 and len([r for r in evaluation.get("evaluated", []) if r.startswith("small_")]) == 9
    independent_path = budget.root / "numerical_review.json"
    independent = read_json(independent_path) if independent_path.exists() else {"status": "not_completed"}
    role_counts = Counter(r["role"] for r in primary)
    started = float(config["full"]["started_unix"])
    manifest_id = file_sha256(primary_path) if primary_path.exists() else None
    allocation_id = file_sha256(allocation_path) if allocation_path.exists() else None
    reconciliation = _allocation_reconciliation(allocation, primary, audit, allocation_id, manifest_id)
    facts = {"allocation_n": len(allocation), "allocation_reconciles_69700": reconciliation["passed"],
             "allocation_reconciliation": reconciliation,
             "primary_n": len(primary), "role_counts": dict(role_counts), "full_complete_fits": n_full,
             "full_expected_fits": 18, "small_complete_fits": n_small,
             "small_status": "complete9" if small_complete else ("partial" if small_started else "not_executed"),
             "evaluation": evaluation, "independent_review": independent,
             "manifest_identity": manifest_id, "allocation_identity": allocation_id,
             "prepared": prepared, "data_audit": audit, "started_unix": started,
             "training_cutoff_unix": started + 144 * 3600, "delivery_cutoff_unix": started + 168 * 3600,
             "gpu_seconds": budget.state.get("gpu_seconds"), "disk": budget.state.get("disk"),
             "sessions": budget.state.get("sessions", []), "runs": budget.state.get("run_ids", []),
             "fit_matrix": public_matrix, "updated_at": utc_now()}
    write_json(tables / "delivery_facts.json", facts)
    lines = ["# 完整数据六臂训练与统一评价\n\n",
             f"本轮全量完成fit为 **{n_full}/18**；统一测试已评价 **{len(evaluation.get('evaluated', []))}** 个可用模型。",
             "完整矩阵完成。\n\n" if n_full == 18 else "未完成项目及原因按fit矩阵保留；partial模型不计入完成数。\n\n",
             "## 数据与协议\n\n",
             f"逐条分配表覆盖 **{len(allocation):,}** 条；69700逐条对账：{'通过' if reconciliation['passed'] else '未通过/尚未完成'}。",
             f"冻结主清单：train **{role_counts['train']:,}** / dev **{role_counts['validation']:,}** / test **{role_counts['test']:,}**。",
             "隔离及专项记录仍在全量分配表中，不能把主清单数量当作原始数据量。\n\n",
             "FAD保留原协议角色；已知原录音、生成参考、派生和精确音频重复按家族隔离。身份未知不等于整来源禁用；本轮不宣称所有说话人严格OOD。",
             "未知关系限制独立性结论，历史已暴露材料也不作为全新确认集。解码、来源、时长与隔离原因见data_audit.json和full_allocation.csv。\n\n",
             "六臂固定为Acoustic/AE/PoolJoint/CA/Linear/Log，seeds17/29/43。",
             "发布XLS-R起训并微调末四层，E2V/BERT冻结；5秒/16kHz，训练BF16、评价FP32。",
             "开发集margin EER→LL→较早epoch选best；开发阈值锁定后在全量同一test评价。",
             "margin由两个FP32 logit转float64后相减；LL用稳定softplus。所有率以0至1存储，表中EER以百分比展示。\n\n",
             "## 全量同一测试集结果\n\n",
             "| 方法 | 完成seed | EER均值±SD | AUC均值±SD | LL均值±SD |\n|---|---:|---:|---:|---:|\n"]
    aggregate = metrics["aggregate"]
    for arm in ARMS:
        row = next((r for r in aggregate if (r["scale"], r["arm"], r["group_field"]) == ("full", arm, "overall")), None)
        if row is None:
            lines.append(f"| {ARM_NAMES[arm]} | 0/3 | NA | NA | NA |\n")
        else:
            cells = [f"{_fmt(row[m + '_mean'], m == 'eer')} ± {_fmt(row[m + '_sd'], m == 'eer')}" for m in ("eer", "auc", "ll")]
            label = f"{row['n_seeds']}/3" + (f"，含{row['n_partial_fits']}个partial" if row["n_partial_fits"] else "")
            lines.append(f"| {ARM_NAMES[arm]} | {label} | {' | '.join(cells)} |\n")
    lines.append("\n均值和样本SD按各seed分别计算；不足3个完成seed不能称为完整三seed结论。SD不代表置信区间。\n\n")
    lines.append("## 科学判断及适用条件\n\n")
    for arm in ("linear", "log"):
        contrasts = [r for r in within if r["scale"] == "full" and r["arm"] == arm and r["group_field"] == "overall" and r["both_fits_complete"]]
        if len(contrasts) == 3 and all(r["eer_delta_pp"] is not None for r in contrasts):
            mean_delta = float(np.mean([r["eer_delta_pp"] for r in contrasts]))
            wins = sum(r["eer_delta_pp"] < 0 for r in contrasts)
            lines.append(f"{ARM_NAMES[arm]}相对Acoustic的平均EER差为 **{mean_delta:+.3f}个百分点**，{wins}/3个seed更低。")
            lines.append("这是同一新划分内的描述性对照，尚不代表统计显著或独立新确认。\n\n")
        else:
            lines.append(f"{ARM_NAMES[arm]}与Acoustic缺少完整三seed配对，不能给出完整方法差距结论。\n\n")
    if small_complete:
        lines.append("约959条家族规模对照已完成9fit并共用相同dev/test。full_minus_small.csv列出配对差异及实际训练量。")
        for arm in SMALL_ARMS:
            deltas = [r["eer_delta_pp"] for r in scale if r["arm"] == arm and r["group_field"] == "overall" and r["both_fits_complete"]]
            if len(deltas) == 3 and all(v is not None for v in deltas):
                lines.append(f" {ARM_NAMES[arm]}全量减小档平均EER差为{float(np.mean(deltas)):+.3f}个百分点。")
        lines.append("家族抽样来源比例和实际数量须同时查阅分配表；它不使旧划分与新划分等价。\n\n")
    else:
        lines.append(f"辅助规模对照状态：**{'部分执行' if small_started else '未执行'}**（{n_small}/9完成fit）。")
        lines.append("完整训练与旧959训练的差异同时包含数据量、来源与划分变化，不能单独归因于增加数据量；也不能仅凭历史结果断言Linear/Log差距因规模而缩小。\n\n")
    source_contrasts = [r for r in within if r["scale"] == "full" and r["arm"] in {"linear", "log"} and r["group_field"] == "source"]
    if source_contrasts:
        lines.append("跨来源判断依据method_minus_acoustic.csv的source分层；单类组EER/AUC均为NA，不能据此计作改善。")
        for arm in ("linear", "log"):
            groups = defaultdict(list)
            for row in source_contrasts:
                if row["arm"] == arm and row["both_fits_complete"]:
                    groups[row["group_value"]].append(row)
            comparable = {name: rows for name, rows in groups.items() if len(rows) == 3 and all(r["eer_delta_pp"] is not None for r in rows)}
            better = [name for name, rows in comparable.items() if np.mean([r["eer_delta_pp"] for r in rows]) < 0]
            lines.append(f" {ARM_NAMES[arm]}在{len(comparable)}个可评价且三seed齐全的来源中，{len(better)}个来源平均EER低于Acoustic。")
        lines.append("来源平均方向仍是描述性证据，不将bootstrap或新seed计作新的独立确认。\n\n")
    lines.extend(["## 完成范围、预算与复核\n\n",
                  f"T0：{_utc(started)}；训练停止：{_utc(started + 144 * 3600)}；交付停止：{_utc(started + 168 * 3600)}。恢复沿用同一T0。\n\n",
                  f"累计GPU阶段用时：{_fmt(float(budget.state.get('gpu_seconds', 0)) / 3600)}小时；",
                  "这是记录的阶段用时，不替代T0至截止的日历预算。磁盘和原生run IDs见delivery_facts.json/state.json。\n\n",
                  f"独立数值复核状态：**{independent.get('status', 'not_completed')}**。",
                  "独立复核未通过时，本报告属于待验收交付，不能宣称任务已完成。\n\n" if independent.get("status") not in {"passed", "complete", "verified"} else "复核细节及范围见numerical_review.json。\n\n",
                  "可复现入口为run_uica_full.sh；配置、源码和训练轨迹随运行快照保留。",
                  "fit_matrix.csv包含全部18个主fit及9个可选fit，逐样本预测在各fit目录的full_test_predictions.jsonl/CSV，",
                  "test_per_seed.csv与test_three_seed_summary.csv包含整体、来源、clean/noise和FAD seen/unseen。",
                  "任何未完成、失败、隔离或单类指标均按实际状态保留；不以工程成功替代科学改善。\n"])
    path = tables / "中文结论报告.md"
    path.write_text("".join(lines), encoding="utf-8")
    print("FULL_DELIVERY_SUMMARY " + json.dumps({"full_complete_fits": n_full, "full_expected_fits": 18,
          "allocation_n": len(allocation), "role_counts": dict(role_counts), "small_status": facts["small_status"],
          "independent_review_status": independent.get("status"), "report": str(path)}, ensure_ascii=False), flush=True)
    return facts
