"""Frozen score-free test diagnostics; integrated by the native closeout stages.

No ASR, training, new model selection, or inference occurs in ``prepare``.
The parent execution budget owns launch and per-batch deadlines. Only its caller
may invoke donor/components after the evaluation lock has frozen every file.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import unicodedata

SALT = "coling-closeout-20261007"
MAPPING_SEEDS = (101, 102, 103)
MAX_RECEIVERS = 2048
BUCKETS = ((1, 8), (9, 16), (17, 32), (33, 128))
ARMS = {"pooljoint", "ca", "linear", "log"}
SCHEMA = "uica.coling.diagnostics.v1"


def _hash(*parts):
    return hashlib.sha256(json.dumps([SALT, *parts], ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _freeze_bytes(path, content):
    """A changed freeze cannot overwrite a previous selection/mapping."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Previously frozen diagnostic file differs: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(content)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _rows_bytes(rows):
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                               allow_nan=False) + "\n").encode("utf-8") for row in rows)


def token_bucket(count):
    if type(count) is not int or not 0 <= count <= 128:
        raise ValueError("Effective token length must be an integer in [0,128]")
    if count == 0:
        return "empty"
    return next(f"{lo}-{hi}" for lo, hi in BUCKETS if lo <= count <= hi)


def metadata_from_dataset(dataset, guard):
    """Read actual verified cache items, without model or score access.

    Effective length equals the model's text_mask & text_present count, hence
    includes cached special tokens on nonempty text and is zero on legal empty
    content. A token_ids count is recorded separately, never used as a proxy.
    """
    result = []
    for index, record in enumerate(dataset.records):
        guard()
        item = dataset[index]
        transcript = item.get("transcript")
        if not isinstance(transcript, str) or type(item.get("text_present")) is not bool:
            raise ValueError("Frozen text cache lacks transcript/presence")
        present = item["text_present"]
        if present != any(unicodedata.category(character)[0] in {"L", "N"} for character in transcript):
            raise ValueError("Transcript content and cache text_present disagree")
        if item["text"].ndim != 2 or item["text"].shape[1] != 768:
            raise ValueError("Expected frozen BERT [tokens,768] cache")
        count = int(item["text"].shape[0]) if present else 0
        row = {key: record[key] for key in ("sample_id", "source", "label", "component_id")}
        if not row["source"] or not row["component_id"] or row["label"] not in (0, 1):
            raise ValueError("Frozen diagnostic source/label/family is missing")
        if len(item["token_ids"]) != item["text"].shape[0]:
            raise ValueError("Cache token_ids and BERT feature token counts disagree")
        row.update(transcript=transcript, text_present=present,
                   transcript_sha256=hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
                   text_feature_sha256=hashlib.sha256(item["text"].contiguous().numpy().tobytes()).hexdigest(),
                   token_ids_sha256=_hash("token-ids", list(item["token_ids"])),
                   token_count=count, token_ids_count=len(item["token_ids"]),
                   token_bucket=token_bucket(count), role=record["role"])
        result.append(row)
    return sorted(result, key=lambda row: row["sample_id"])


def _valid_pair(receiver, donor):
    return (receiver["sample_id"] != donor["sample_id"]
            and receiver["source"] == donor["source"]
            and receiver["label"] == donor["label"]
            and receiver["token_bucket"] == donor["token_bucket"]
            and receiver["token_bucket"] != "empty"
            and receiver["component_id"] != donor["component_id"]
            and receiver["transcript"] != donor["transcript"])


def _candidate_pools(metadata, seed):
    pools = defaultdict(list)
    for row in metadata:
        if row["token_bucket"] != "empty":
            pools[(row["source"], row["label"], row["token_bucket"])].append(row)
    return {key: {"rows": sorted(rows, key=lambda row: (_hash("donor-order", seed, row["sample_id"]), row["sample_id"])),
                  "families": Counter(row["component_id"] for row in rows),
                  "transcripts": Counter(row["transcript"] for row in rows),
                  "joint": Counter((row["component_id"], row["transcript"]) for row in rows)}
            for key, rows in pools.items()}


def _pick_donor(receiver, pools, seed):
    pool = pools.get((receiver["source"], receiver["label"], receiver["token_bucket"]))
    if pool is None:
        return None
    rows = pool["rows"]
    family, transcript = receiver["component_id"], receiver["transcript"]
    # Inclusion-exclusion identifies a no-donor stratum without scanning every
    # row for each receiver. Selection among legal candidates remains unchanged.
    eligible = len(rows) - pool["families"][family] - pool["transcripts"][transcript] + pool["joint"][(family, transcript)]
    if not eligible:
        return None
    offset = int(_hash("donor-offset", seed, receiver["sample_id"]), 16) % len(rows)
    for step in range(len(rows)):
        donor = rows[(offset + step) % len(rows)]
        if _valid_pair(receiver, donor):
            return donor
    return None


def freeze_selection(metadata, max_receivers=MAX_RECEIVERS):
    """Hamilton source x label quotas on eligible recipients, stable hash pick."""
    if not 0 < max_receivers <= MAX_RECEIVERS:
        raise ValueError("The predeclared receiver cap is 2048")
    metadata = [{key: value for key, value in row.items() if key != "donor_status"} for row in metadata]
    by_id = {row["sample_id"]: row for row in metadata}
    if len(by_id) != len(metadata):
        raise ValueError("Duplicate diagnostic sample ID")
    pools = _candidate_pools(metadata, MAPPING_SEEDS[0])
    strata = defaultdict(list)
    statuses = []
    for row in sorted(metadata, key=lambda row: row["sample_id"]):
        if token_bucket(row["token_count"]) != row["token_bucket"]:
            raise ValueError("Effective token bucket mismatch")
        status = "empty_text" if row["token_bucket"] == "empty" else (
            "eligible" if _pick_donor(row, pools, MAPPING_SEEDS[0]) else "no_donor")
        statuses.append({**row, "donor_status": status})
        if status == "eligible":
            strata[(row["source"], row["label"])].append(row)
    total = sum(map(len, strata.values()))
    target = min(max_receivers, total)
    quotas = {key: target * len(rows) // total for key, rows in strata.items()} if total else {}
    remaining = target - sum(quotas.values())
    order = sorted(strata, key=lambda key: (-(target * len(strata[key]) % total), _hash("quota-tie", *key), key))
    for key in order[:remaining]:
        quotas[key] += 1
    selected = sorted([row for key, rows in strata.items()
                       for row in sorted(rows, key=lambda row: (_hash("receiver", row["sample_id"]), row["sample_id"]))[:quotas[key]]],
                      key=lambda row: row["sample_id"])
    maps, reuse = {}, {}
    for seed in MAPPING_SEEDS:
        seed_pools = _candidate_pools(metadata, seed)
        mapping = []
        for receiver in selected:
            donor = _pick_donor(receiver, seed_pools, seed)
            if donor is None:
                raise ValueError("Eligible receiver lost its donor under another mapping seed")
            mapping.append({"receiver": receiver["sample_id"], "donor": donor["sample_id"], "mapping_seed": seed})
        counts = Counter(row["donor"] for row in mapping)
        maps[seed] = mapping
        reuse[str(seed)] = {"n_receivers": len(mapping), "n_unique_donors": len(counts),
                           "reused_donor_ids": sum(n > 1 for n in counts.values()),
                           "reuse_assignments": sum(max(0, n-1) for n in counts.values()),
                           "max_donor_reuse": max(counts.values(), default=0)}
    coverage = {"n_test": len(metadata), "n_eligible": total, "n_selected": target,
                "n_empty_text": sum(row["donor_status"] == "empty_text" for row in statuses),
                "n_no_donor": sum(row["donor_status"] == "no_donor" for row in statuses),
                "eligible_fraction": total / len(metadata) if metadata else None,
                "selected_fraction": target / len(metadata) if metadata else None,
                "strata": [{"source": key[0], "label": key[1], "n_eligible": len(strata[key]), "quota": quotas[key]}
                           for key in sorted(strata)], "donor_reuse": reuse}
    return statuses, selected, maps, coverage


def prepare(config, budget):
    """Native stage: score-free freeze before evaluation reads test scores."""
    from .full_runtime import datasets
    root = budget.root
    if (root / "coling_evaluation_lock.json").exists() and not (root / "coling_diagnostic_manifest.json").exists():
        raise ValueError("Cannot create a diagnostic selection after evaluation freeze")
    data = datasets(budget, config, "test")
    metadata = metadata_from_dataset(data, budget.guard)
    statuses, selected, mappings, coverage = freeze_selection(metadata)
    rows = {"metadata": ("diagnostics/frozen_metadata.jsonl", statuses),
            "subset": ("diagnostics/subset.jsonl", selected)}
    rows.update({f"donors_{seed}": (f"diagnostics/donors_{seed}.jsonl", mappings[seed]) for seed in MAPPING_SEEDS})
    files = {}
    for key, (relative, content) in rows.items():
        _freeze_bytes(root / relative, _rows_bytes(content))
        files[key] = {"path": relative, "sha256": _sha(root / relative)}
    manifest = {"schema": SCHEMA, "primary_manifest_sha256": _sha(root / "primary_manifest.jsonl"),
                "cache_index_sha256": _sha(root / "cache/cache_index.json"), "salt": SALT,
                "mapping_seeds": list(MAPPING_SEEDS), "max_receivers": MAX_RECEIVERS,
                "selection_rule": "source_x_label_eligible_hamilton_quotas_then_salted_sample_id_sha256",
                "token_length_rule": "cached_text_rows_if_text_present_else_zero_including_valid_special_tokens",
                "donor_family_rule": "different_primary_manifest_component_id",
                "files": files, "coverage": coverage,
                "scope": "label-conditioned offline functional diagnostics; descriptive; no unconditional mismatch significance"}
    _freeze_bytes(root / "coling_diagnostic_manifest.json", _json_bytes(manifest))
    print("COLING_DIAGNOSTICS_FROZEN " + json.dumps({"sha256": _sha(root / "coling_diagnostic_manifest.json"), "coverage": coverage}), flush=True)
    return manifest


def load_frozen(root):
    root = Path(root)
    manifest = _read_json(root / "coling_diagnostic_manifest.json")
    if (manifest["schema"], manifest["salt"], manifest["mapping_seeds"], manifest["max_receivers"]) != (SCHEMA, SALT, list(MAPPING_SEEDS), MAX_RECEIVERS):
        raise ValueError("Frozen diagnostic protocol mismatch")
    for name, spec in manifest["files"].items():
        path = (root / spec["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or _sha(path) != spec["sha256"]:
            raise ValueError("Frozen diagnostic file path/hash mismatch: " + name)
    for field, path in (("primary_manifest_sha256", "primary_manifest.jsonl"), ("cache_index_sha256", "cache/cache_index.json")):
        if manifest[field] != _sha(root / path):
            raise ValueError("Frozen diagnostic input identity changed")
    metadata = _read_rows(root / manifest["files"]["metadata"]["path"])
    _, selected, maps, coverage = freeze_selection(metadata)
    if _read_rows(root / manifest["files"]["subset"]["path"]) != selected or manifest["coverage"] != coverage:
        raise ValueError("Frozen subset/quota/coverage cannot be reconstructed")
    for seed in MAPPING_SEEDS:
        if _read_rows(root / manifest["files"][f"donors_{seed}"]["path"]) != maps[seed]:
            raise ValueError("Frozen donor map cannot be reconstructed")
    return manifest, metadata, selected, maps


def _verify_text_item(item, frozen):
    record = item["record"]
    if any(record[key] != frozen[key] for key in ("sample_id", "source", "label", "component_id")):
        raise ValueError("Frozen diagnostic sample/source/label/family changed")
    if (item["transcript"] != frozen["transcript"] or item["text_present"] != frozen["text_present"]
            or hashlib.sha256(item["text"].contiguous().numpy().tobytes()).hexdigest() != frozen["text_feature_sha256"]
            or _hash("token-ids", list(item["token_ids"])) != frozen["token_ids_sha256"]):
        raise ValueError("Frozen diagnostic text cache changed")


def _subset(dataset, selected):
    by_id = {row["sample_id"]: index for index, row in enumerate(dataset.records)}
    metadata = {row["sample_id"]: row for row in selected}
    class Subset:
        records = [dataset.records[by_id[row["sample_id"]]] for row in selected]
        def __len__(self):
            return len(self.records)
        def __getitem__(self, index):
            item = dataset[by_id[self.records[index]["sample_id"]]]
            _verify_text_item(item, metadata[item["sample_id"]])
            return item
    return Subset()


def component_reconstruction(out, model):
    """Runtime-only check of actual M=sum(p), C and D returned by the model.

    closeout_model calculates actual per-head/per-frame M, masked mean(V), C,
    and D. This checks that those tensors reconstruct the retained full and its
    nonlinear classifier; it never adds effects measured at separate logits.
    """
    import torch
    full, reconstructed = out["z_rel_full"], out["z_common"] + out["z_deviation"]
    if not torch.isfinite(reconstructed).all() or not torch.allclose(full, reconstructed, atol=2e-6, rtol=2e-5):
        raise ValueError("Actual relation full=C+D reconstruction failed")
    manual = model.classifier(torch.cat([out["z_acoustic"], full], dim=-1))
    if not torch.allclose(out["logits"], manual, atol=2e-6, rtol=2e-5):
        raise ValueError("Full forward/classifier reconstruction failed")
    return float((full - reconstructed).abs().max()), float((out["logits"] - manual).abs().max())


def _save_batch_then_guard(rows, budget, save_partial):
    """A completed batch remains recoverable even if it crossed results lock."""
    if save_partial is not None:
        save_partial(rows)
    budget.guard()


def _mark_complete(report, budget, persist):
    budget.guard()
    report["status"] = "complete"
    persist(report)


def _mark_incomplete(report, error, persist):
    report.update(status="incomplete", reason=str(error), error_type=type(error).__name__,
                  completed_conditions=sorted(report.get("conditions", {})))
    persist(report)


def _bind_model_specs(lock, retained_models):
    for spec in retained_models:
        locked = lock["models"].get(f"full_{spec['arm']}_seed{spec['seed']}")
        if locked is None or (spec["checkpoint_sha256"], spec["threshold"], spec.get("fit_status")) != (
                locked["checkpoint_id"], locked["threshold"], locked["fit_status"]):
            raise ValueError("Provided diagnostic model/status is outside the frozen roster")


def diagnostic_predict(model, dataset, spec, budget, condition="full", overrides=None, acoustic_cache=None,
                       save_partial=None, forward_stop_unix=None):
    """FP32/batch4 sorted frozen subset, with parent deadline checks per batch."""
    import numpy as np
    import torch
    from .common import config_digest
    from .full_runtime import INFERENCE, collate, run_identity, validate_predictions
    from .training import to_device
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model.float().eval()
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for start in range(0, len(dataset), 4):
            budget.guard()
            if forward_stop_unix is not None and time.time() >= forward_stop_unix:
                raise RuntimeError("Insufficient results-lock safety margin for another diagnostic batch")
            items = [dict(dataset[i]) for i in range(start, min(start + 4, len(dataset)))]
            for item in items:
                if overrides:
                    item.update(overrides[item["sample_id"]])
            batch = to_device(collate(items), "cuda")
            cached = None
            if acoustic_cache is not None and all(item["sample_id"] in acoustic_cache for item in items):
                cached = torch.stack([acoustic_cache[item["sample_id"]] for item in items]).cuda()
            out = model(batch, condition=condition, acoustic=cached)
            reconstruction, forward_difference = (None, None)
            logits = out["logits"].detach().float().cpu().numpy()
            for index, record in enumerate(batch["records"]):
                values = logits[index].astype(np.float64)
                row = {"sample_id": record["sample_id"], "data_role": record["role"], "label": int(record["label"]),
                       "component_id": record["component_id"], "source": record["source"], "arm": spec["arm"],
                       "seed": spec["seed"], "checkpoint_id": spec["checkpoint_sha256"],
                       "inference_id": config_digest(INFERENCE), "condition": condition, "run_id": run_identity(),
                       "logits": values.tolist(), "margin": float(values[1]-values[0]),
                       "component_reconstruction_max_abs": reconstruction, "full_forward_max_abs": forward_difference}
                for key in ("z_rel", "z_common", "z_deviation", "z_acoustic"):
                    if key in out:
                        row[key + "_norm"] = float(out[key][index].float().norm())
                rows.append(row)
                if acoustic_cache is not None:
                    acoustic_cache[record["sample_id"]] = out["z_acoustic"][index].detach().float().cpu()
            _save_batch_then_guard(rows, budget, save_partial)
            if model.variant in {"linear", "log"} and condition == "full":
                reconstruction, forward_difference = component_reconstruction(out, model)
                budget.guard()
                ordinary = model(batch, acoustic=out["z_acoustic"])
                if not torch.equal(out["logits"], ordinary["logits"]):
                    raise ValueError("Explicit full differs from ordinary full forward")
                for row in rows[-len(batch["records"]):]:
                    row.update(component_reconstruction_max_abs=reconstruction,
                               full_forward_max_abs=forward_difference)
                _save_batch_then_guard(rows, budget, save_partial)
    budget.guard()
    validate_predictions(rows, dataset, spec["arm"], spec["seed"], spec["checkpoint_sha256"])
    return rows


def _metrics(rows, threshold):
    from .full_runtime import score_rows
    return score_rows(rows, threshold) if rows else {"n": 0, "eer": None, "ll": None, "fpr": None, "fp": 0, "n_real": 0, "reason": "no_eligible_receivers"}


def _effect(changed, correct):
    return {"delta_" + key: changed[key]-correct[key] if changed.get(key) is not None and correct.get(key) is not None else None
            for key in ("eer", "ll", "fpr")}


def run_stage(config, budget, mode, retained_models=None):
    """Native donor/components stage; retained_models from verified evaluation lock.

    Parent must verify the lock's diagnostic file SHAs, frozen checkpoint roster
    and dev thresholds before this function. Models load through load_delta.
    Launch cutoff=36h and save/stop cutoff=48h are enforced by the parent hook.
    """
    import gc
    import torch
    from .common import write_json, write_jsonl
    from .full_runtime import datasets, load_delta
    if mode not in {"donor", "components"} or not (budget.root / "coling_evaluation_lock.json").exists():
        raise ValueError("Diagnostics require an existing frozen evaluation lock")
    # Reuse evaluator's read-only lock verifier and the parent's deadline hook.
    from .coling_evaluation import InferenceBudget, guard_inference, load_lock
    lock = load_lock(config, budget)
    guard_inference(config, budget, starting=True)
    execution_budget = InferenceBudget(config, budget)
    if retained_models is None:
        retained_models = [dict(spec, checkpoint_sha256=spec["checkpoint_id"],
                                checkpoint_path=str(Path("fits/full") / f"{spec['arm']}_seed{spec['seed']}" / "best.pt"))
                           for key, spec in lock["models"].items()]
    _bind_model_specs(lock, retained_models)
    manifest, metadata, selected, mappings = load_frozen(budget.root)
    data = datasets(budget, config, "test")
    subset = _subset(data, selected)
    by_id = {row["sample_id"]: index for index, row in enumerate(data.records)}
    meta = {row["sample_id"]: row for row in metadata}
    reports = []
    for spec in retained_models:
        if spec["arm"] not in ARMS or (mode == "components" and spec["arm"] not in {"linear", "log"}):
            continue
        deadlines = guard_inference(config, budget, starting=True)
        execution_budget.guard()
        path = Path(spec["checkpoint_path"])
        if not path.is_absolute():
            path = budget.root / path
        if _sha(path) != spec["checkpoint_sha256"] or not math.isfinite(float(spec["threshold"])):
            raise ValueError("Frozen diagnostic checkpoint/threshold changed")
        folder = budget.root / "diagnostics" / f"{spec['arm']}_seed{spec['seed']}"
        folder.mkdir(parents=True, exist_ok=True)
        report = {"arm": spec["arm"], "seed": spec["seed"], "mode": mode,
                  "checkpoint_sha256": spec["checkpoint_sha256"], "threshold": spec["threshold"],
                  "diagnostic_manifest_sha256": _sha(budget.root / "coling_diagnostic_manifest.json"),
                  "n": len(subset), "effects_direction": "changed_minus_correct_input",
                  "scope": "descriptive conditional on fixed saved model and label-conditioned donor maps",
                  "fit_status": spec.get("fit_status"), "status": "running"}
        write_json(folder / f"{mode}_summary.json", report)
        if not len(subset):
            report.update(status="not_run_no_eligible_receivers", correct_input=_metrics([], spec["threshold"]))
            write_json(folder / f"{mode}_summary.json", report)
            reports.append(report)
            continue
        model = saved = None
        cache = {}
        persist = lambda value: write_json(folder / f"{mode}_summary.json", value)
        try:
            model, saved = load_delta(path, config, budget)
            if (saved["arm"], saved["seed"], saved["threshold"]) != (spec["arm"], spec["seed"], spec["threshold"]):
                raise ValueError("Checkpoint and frozen roster identity mismatch")
            correct = diagnostic_predict(model, subset, spec, execution_budget, acoustic_cache=cache,
                                         save_partial=lambda rows: write_jsonl(folder / f"{mode}_correct_input.partial.jsonl", rows),
                                         forward_stop_unix=deadlines["results_lock"] - 60)
            for row in correct:
                row.update(token_count=meta[row["sample_id"]]["token_count"], token_bucket=meta[row["sample_id"]]["token_bucket"])
            write_jsonl(folder / "correct_input.jsonl", correct)
            report["correct_input"] = _metrics(correct, spec["threshold"])
            report["conditions"] = {}
            conditions = list(MAPPING_SEEDS) if mode == "donor" else ["common", "deviation", "zero"]
            for condition in conditions:
                execution_budget.guard()
                overrides, mapping = None, None
                if mode == "donor":
                    mapping = mappings[condition]
                    overrides = {}
                    for pair in mapping:
                        execution_budget.guard()
                        item = data[by_id[pair["donor"]]]
                        _verify_text_item(item, meta[pair["donor"]])
                        overrides[pair["receiver"]] = {key: item[key] for key in ("text", "token_ids", "text_present", "transcript")}
                changed = diagnostic_predict(model, subset, spec, execution_budget, "full" if mode == "donor" else condition,
                                             overrides=overrides, acoustic_cache=cache,
                                             save_partial=lambda rows: write_jsonl(folder / f"{mode}_{condition}.partial.jsonl", rows),
                                             forward_stop_unix=deadlines["results_lock"] - 60)
                pairs = {pair["receiver"]: pair["donor"] for pair in mapping} if mapping is not None else {}
                for row in changed:
                    row.update(token_count=meta[row["sample_id"]]["token_count"], token_bucket=meta[row["sample_id"]]["token_bucket"])
                    if mode == "donor":
                        row.update(donor_id=pairs[row["sample_id"]], donor_seed=condition)
                write_jsonl(folder / f"{mode}_{condition}.jsonl", changed)
                metrics = _metrics(changed, spec["threshold"])
                report["conditions"][str(condition)] = {"level": metrics, "effect": _effect(metrics, report["correct_input"])}
                persist(report)
            if mode == "donor":
                report["three_mapping_mean_effect"] = {
                    key: sum(report["conditions"][str(seed)]["effect"][key] for seed in MAPPING_SEEDS)/3
                    if all(report["conditions"][str(seed)]["effect"][key] is not None for seed in MAPPING_SEEDS) else None
                    for key in ("delta_eer", "delta_ll", "delta_fpr")}
            else:
                report["interpretation"] = "Nonlinear classifier effects cannot be added. zero is not retrained Acoustic; deviation is not retrained centered."
            _mark_complete(report, execution_budget, persist)
            reports.append(report)
            print("COLING_DIAGNOSTIC " + json.dumps(report), flush=True)
        except BaseException as error:
            _mark_incomplete(report, error, persist)
            write_json(budget.root / "diagnostics" / f"{mode}_index.json", {"manifest": manifest,
                       "models": reports + [report], "status": "incomplete", "reason": str(error)})
            print("COLING_DIAGNOSTIC_INCOMPLETE " + json.dumps(report), flush=True)
            raise
        finally:
            del model, saved, cache
            gc.collect()
            torch.cuda.empty_cache()
    comparisons = []
    if mode == "donor":
        by_model = {(report["arm"], report["seed"]): report for report in reports if report["status"] == "complete"}
        for (arm, seed), report in sorted(by_model.items()):
            if arm not in {"linear", "log"}:
                continue
            for comparator in ("pooljoint", "ca"):
                control = by_model.get((comparator, seed))
                if control:
                    comparisons.append({"arm": arm, "seed": seed, "comparator": comparator,
                                        "fit_status": report["fit_status"],
                                        "delta_effect_relative_to_control": {key: report["three_mapping_mean_effect"][key]-control["three_mapping_mean_effect"][key]
                                             if report["three_mapping_mean_effect"][key] is not None and control["three_mapping_mean_effect"][key] is not None else None
                                             for key in ("delta_eer", "delta_ll", "delta_fpr")}})
    write_json(budget.root / "diagnostics" / f"{mode}_index.json", {"manifest": manifest, "models": reports,
              "relative_changes_by_seed": comparisons, "unconditional_mismatch_significance": "not_claimed"})
    return reports


def donor(config, budget, retained_models=None):
    return run_stage(config, budget, "donor", retained_models)


def components(config, budget, retained_models=None):
    return run_stage(config, budget, "components", retained_models)


def saved_prediction_analysis(config, budget):
    """Saved-score-only length/empty/case analysis; performs no new inference.

    Case rule is frozen in this source: first three salted sample-ID hashes per
    confusion category, including legal empty text. No manual success picking.
    """
    from .coling_evaluation import load_lock
    from .common import read_jsonl, write_json
    from .full_runtime import datasets
    from .full_evaluation import validate_predictions
    lock = load_lock(config, budget)
    manifest, metadata, _, _ = load_frozen(budget.root)
    data = datasets(budget, config, "test")
    by_id = {row["sample_id"]: row for row in metadata}
    reports = []
    for key, spec in lock["models"].items():
        folder = budget.root / "fits/full" / f"{spec['arm']}_seed{spec['seed']}"
        path = folder / "coling_test_predictions.jsonl"
        if not path.exists():
            reports.append({"arm": spec["arm"], "seed": spec["seed"], "status": "missing_saved_predictions"})
            continue
        rows = read_jsonl(path)
        validate_predictions(rows, data.records, spec["arm"], spec["seed"], spec["checkpoint_id"], lock["inference_id"])
        groups, case_groups = defaultdict(list), defaultdict(list)
        for row in rows:
            meta = by_id[row["sample_id"]]
            groups[meta["token_bucket"]].append(row)
            predicted = row["margin"] >= spec["threshold"]
            category = ("real_false_positive" if predicted else "real_true_negative") if row["label"] == 0 else ("spoof_detected" if predicted else "spoof_miss")
            case_groups[category].append({**row, "transcript": meta["transcript"], "token_count": meta["token_count"],
                                          "token_bucket": meta["token_bucket"], "case_category": category})
        reports.append({"arm": spec["arm"], "seed": spec["seed"], "fit_status": spec["fit_status"],
                        "status": "saved_predictions_analyzed", "prediction_sha256": _sha(path),
                        "length_buckets": {bucket: _metrics(group, spec["threshold"]) for bucket, group in sorted(groups.items())},
                        "cases": {category: sorted(group, key=lambda row: (_hash("case", row["sample_id"]), row["sample_id"]))[:3]
                                  for category, group in sorted(case_groups.items())}})
    report = {"diagnostic_manifest_sha256": _sha(budget.root / "coling_diagnostic_manifest.json"),
              "scope": "existing frozen predictions only; no new model evaluation", "models": reports}
    write_json(budget.root / "diagnostics" / "saved_length_empty_cases.json", report)
    return report
