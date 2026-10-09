"""Finite training runs, complete predictions, and reproducible checkpoints."""
from __future__ import annotations

import contextlib
import copy
import csv
import gc
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .common import config_digest, file_sha256, read_json, read_jsonl, scientific_config, source_digest, utc_now, write_json, write_jsonl
from .data import FeatureDataset, collate_features
from .metrics import binary_metrics, choose_threshold, weighted_ce_sum
from .model import Detector


def assert_complete_predictions(rows, expected_ids):
    observed = [row["sample_id"] for row in rows]
    if len(observed) != len(set(observed)) or len(expected_ids) != len(set(expected_ids)) or set(observed) != set(expected_ids):
        raise ValueError("Prediction sample coverage is incomplete or duplicated")


def checkpoint_rank(eer, ce, epoch):
    if eer is None or not math.isfinite(eer) or not math.isfinite(ce):
        raise ValueError("Checkpoint selection requires finite validation EER and CE")
    return float(eer), float(ce), int(epoch)


def slice_batch(batch, start, end):
    return {key: value[start:end] if isinstance(value, (torch.Tensor, list, tuple)) else value for key, value in batch.items()}


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def amp_context(device, precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") and precision == "bf16" else contextlib.nullcontext()


def predict(model, dataset, device="cuda", batch_size=4, precision="bf16", include_vectors=False):
    model.eval()
    rows, total_ce, offset = [], 0.0, 0
    with torch.inference_mode():
        while offset < len(dataset):
            end = min(offset + batch_size, len(dataset))
            try:
                batch = to_device(collate_features([dataset[i] for i in range(offset, end)]), device)
                with amp_context(device, precision):
                    output = model(batch)
                logits = output["logits"].float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite prediction")
                scores = logits.softmax(dim=-1)[:, 1].cpu().tolist()
                batch_ce = float(functional.cross_entropy(logits, batch["labels"], reduction="sum"))
                batch_rows = []
                for i, record in enumerate(batch["records"]):
                    positive_count = int(output["positive_count"][i])
                    row = {"sample_id": record["sample_id"], "label": int(record["label"]), "score": scores[i],
                           "speaker_id": record.get("speaker_id"), "role": record["role"], "source": record.get("source"),
                           "base_corpus": record.get("base_corpus"), "generator_id": record.get("generator_id"),
                           "condition": record.get("condition"), "strength": float(output["strength"][i]),
                           "relation_norm": float(output["z_rel"][i].float().norm()),
                           "clipping_all": float(output["clipping_all"][i]),
                           "clipping_positive": float(output["clipping_positive"][i]) if positive_count else None,
                           "positive_count": positive_count, "clipped_count": int(output["clipped_count"][i]),
                           "valid_position_count": int(output["valid_position_count"][i]),
                           "text_length": int(batch["text_mask"][i].sum()),
                           "text_present": bool(batch["text_present"][i])}
                    if include_vectors:
                        row["z_rel"] = output["z_rel"][i].float().cpu().tolist()
                    batch_rows.append(row)
                # Commit only a complete batch, so an OOM retry cannot duplicate
                # earlier rows or add its CE numerator twice.
                rows.extend(batch_rows)
                total_ce += batch_ce
                offset = end
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                batch = output = None
                gc.collect()
                torch.cuda.empty_cache()
    assert_complete_predictions(rows, [row["sample_id"] for row in dataset.records])
    if not rows:
        raise ValueError("An empty evaluation is not a successful result")
    return rows, total_ce / len(rows)


def write_predictions(path, rows):
    write_jsonl(path, rows)
    csv_path = Path(path).with_suffix(".csv")
    fields = [key for key in rows[0] if key != "z_rel"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inferred_config(config, dataset):
    result = copy.deepcopy(config)
    sample = dataset[0]
    result["model"]["affective_dim"] = int(sample["affective"].shape[-1])
    result["model"]["text_dim"] = int(sample["text"].shape[-1])
    return result


def load_run_model(run_dir, device="cuda"):
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result.get("status") == "complete" and result.get("checkpoint_sha256") != file_sha256(run_dir / "best.pt"):
            raise ValueError("Checkpoint hash differs from the verified completed result")
    # Only this project's locally generated checkpoint is trusted for resume.
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("contract_digest") != config_digest(scientific_config(checkpoint["config"])):
        raise ValueError("Checkpoint config does not match its recorded contract")
    model = Detector(checkpoint["config"], checkpoint["variant"], checkpoint["seed"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _gradient_report(model):
    entries = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            entries[name] = {"has_gradient": parameter.grad is not None,
                             "finite": parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                             "norm": float(parameter.grad.float().norm()) if parameter.grad is not None else None}
    if any(not item["finite"] for item in entries.values()):
        raise ValueError("A trainable parameter has a missing or non-finite gradient")
    return entries


def _reconcile_epoch_history(run_dir, committed_epoch):
    """Retain abandoned attempt evidence without counting an epoch twice."""
    path = run_dir / "history.jsonl"
    if not path.exists():
        return
    history = read_jsonl(path)
    committed = [row for row in history if row["epoch"] <= committed_epoch]
    uncommitted = [row for row in history if row["epoch"] > committed_epoch]
    if uncommitted:
        archive = run_dir / "uncommitted_history.jsonl"
        previous = read_jsonl(archive) if archive.exists() else []
        write_jsonl(archive, previous + uncommitted)
        write_jsonl(path, committed)


def train_run(config, manifest_path, cache_dir, run_dir, variant, seed, resume=False, device="cuda", deadline=None, model_factory=Detector):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "result.json").exists():
        result = read_json(run_dir / "result.json")
        if result.get("status") == "complete":
            raise RuntimeError("Run is already complete; reuse its recorded result, do not retrain")
    if (run_dir / "last.pt").exists() and not resume:
        raise RuntimeError("A checkpoint already exists; explicit --resume is required")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("This registered experiment requires the authorized CUDA GPU")
    train_data = FeatureDataset(manifest_path, cache_dir, ["train"])
    validation = FeatureDataset(manifest_path, cache_dir, ["validation"])
    if not len(train_data) or not len(validation):
        raise ValueError("Training and validation must both be nonempty")
    from .preprocess import _processing_spec
    processing_fingerprint = train_data.processing_fingerprint
    if not processing_fingerprint or validation.processing_fingerprint != processing_fingerprint:
        raise ValueError("Cache processing fingerprint is missing or inconsistent")
    if processing_fingerprint != config_digest(_processing_spec(config)):
        raise ValueError("Cache processing fingerprint differs from the configured encoders/preprocessing")
    config = inferred_config(config, train_data)
    contract_digest = config_digest(scientific_config(config))
    manifest_digest = file_sha256(manifest_path)
    code_digest = source_digest()
    settings = config["training"]
    seed_everything(seed)
    model = model_factory(config, variant=variant, seed=seed).to(device)
    groups = model.trainable_parameter_groups(settings["backbone_lr"], settings["head_lr"])
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Optimizer parameter coverage mismatch")
    counts = np.bincount([row["label"] for row in train_data.records], minlength=2)
    if min(counts) == 0:
        raise ValueError("Both classes must be present in train")
    weights = torch.tensor(len(train_data) / (2 * counts), dtype=torch.float32, device=device)
    effective = int(settings["effective_batch"])
    microbatch = int(settings["microbatch"])
    steps_per_epoch = math.ceil(len(train_data) / effective)
    max_steps = steps_per_epoch * settings["max_epochs"]
    warmup = max(1, round(max_steps * settings["warmup_fraction"]))

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    start_epoch, global_step, bad_epochs = 1, 0, 0
    best_rank, best_eer = (float("inf"),) * 3, float("inf")
    elapsed_before = 0.0
    oom_adjustments = 0
    if resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved["contract_digest"] != contract_digest or saved["manifest_digest"] != manifest_digest or saved["variant"] != variant or saved["seed"] != seed:
            raise ValueError("Resume identity differs from checkpoint")
        if saved.get("processing_fingerprint") != processing_fingerprint:
            raise ValueError("Resume cache processing fingerprint differs from checkpoint")
        if saved.get("source_digest") != code_digest:
            raise ValueError("Resume source code differs from checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        start_epoch, global_step, bad_epochs = saved["epoch"] + 1, saved["global_step"], saved["bad_epochs"]
        best_rank, best_eer = tuple(saved["best_rank"]), saved["best_eer"]
        microbatch, elapsed_before = saved["microbatch"], saved["elapsed_seconds"]
        oom_adjustments = int(saved.get("oom_adjustments", 0))
        _reconcile_epoch_history(run_dir, saved["epoch"])
    write_json(run_dir / "config.json", config)
    previous_identity = read_json(run_dir / "identity.json") if resume and (run_dir / "identity.json").exists() else {}
    write_json(run_dir / "identity.json", {"seed": seed, "variant": variant, "contract_digest": contract_digest,
        "manifest_digest": manifest_digest, "processing_fingerprint": processing_fingerprint, "source_digest": code_digest,
        "class_counts": counts.tolist(), "class_weights": weights.cpu().tolist(),
        "initialization": "module-specific shared seeds; same acoustic/projection and Linear/Log initial values",
        "max_epochs": settings["max_epochs"], "started_at": previous_identity.get("started_at", utc_now()),
        "resumed_at": utc_now() if resume else None})
    began = time.monotonic()
    stop_reason, completed_epoch = "max_epochs", start_epoch - 1
    # A process can exit after last.pt is durable but before writing result.json.
    # Recover the stopping decision as well as weights/RNG; do not add an epoch.
    reached_early_stop = completed_epoch >= settings["min_epochs"] and bad_epochs >= settings["patience"]
    if reached_early_stop:
        stop_reason = "early_stopping"
    epochs = () if reached_early_stop else range(start_epoch, settings["max_epochs"] + 1)
    for epoch in epochs:
        if deadline is not None and time.time() >= deadline:
            stop_reason = "budget_exhausted"
            break
        model.train()
        order = np.random.default_rng(seed * 100000 + epoch).permutation(len(train_data))
        total_loss, weight_total, seen = 0.0, 0.0, 0
        epoch_started = time.monotonic()
        for batch_start in range(0, len(order), effective):
            if deadline is not None and time.time() >= deadline:
                stop_reason = "budget_exhausted"
                break
            items = [train_data[int(i)] for i in order[batch_start:batch_start + effective]]
            batch = collate_features(items)
            denominator = weights[batch["labels"].to(device)].sum()
            before = rng_state()
            while True:
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                try:
                    for start in range(0, len(items), microbatch):
                        micro = to_device(slice_batch(batch, start, start + microbatch), device)
                        with amp_context(device, settings["precision"]):
                            output = model(micro)
                            numerator = weighted_ce_sum(output["logits"].float(), micro["labels"], weights)
                        if not torch.isfinite(numerator):
                            raise ValueError("Non-finite training loss")
                        (numerator / denominator).backward()
                        batch_loss += float(numerator.detach())
                    if global_step < 3:
                        write_json(run_dir / f"gradients_step{global_step}.json", _gradient_report(model))
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if microbatch == 1:
                        raise
                    microbatch = max(1, microbatch // 2)
                    oom_adjustments += 1
                    micro = output = numerator = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    restore_rng(before)
                    write_json(run_dir / "microbatch_adjustment.json", {"epoch": epoch, "step": global_step,
                        "microbatch": microbatch, "effective_batch": effective, "reason": "cuda_oom",
                        "oom_adjustments": oom_adjustments})
            # A failed optimizer.step may already have mutated weights or Adam
            # state. Never retry it in-place; recover from the last epoch instead.
            optimizer.step()
            scheduler.step()
            global_step += 1
            seen += len(items)
            total_loss += batch_loss
            weight_total += float(denominator)
            if global_step % 10 == 0:
                write_json(run_dir / "progress.json", {"status": "training", "epoch": epoch, "seen": seen,
                    "global_step": global_step, "microbatch": microbatch, "updated_at": utc_now()})
        if stop_reason == "budget_exhausted":
            break  # Last completed epoch is the recoverable checkpoint; partial epoch is not promoted.
        rows, validation_ce = predict(model, validation, device, microbatch, settings["precision"])
        labels, scores = [r["label"] for r in rows], [r["score"] for r in rows]
        threshold = choose_threshold(labels, scores)
        metrics = binary_metrics(labels, scores, threshold)
        rank = checkpoint_rank(metrics["eer"], validation_ce, epoch)
        improved = rank < best_rank
        if metrics["eer"] < best_eer:
            bad_epochs = 0
            best_eer = metrics["eer"]
        else:
            bad_epochs += 1
        if improved:
            best_rank = rank
        epoch_record = {"epoch": epoch, "global_step": global_step, "train_ce_weighted": total_loss / weight_total,
            "validation_ce": validation_ce, "validation": metrics, "microbatch": microbatch,
            "elapsed_seconds": elapsed_before + time.monotonic() - began,
            "epoch_seconds": time.monotonic() - epoch_started, "bad_epochs": bad_epochs, "updated_at": utc_now()}
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            import json
            stream.write(json.dumps(epoch_record, allow_nan=False) + "\n")
        common = {"model": model.state_dict(), "config": config, "variant": variant, "seed": seed,
            "epoch": epoch, "threshold": threshold, "validation": metrics, "validation_ce": validation_ce,
            "contract_digest": contract_digest, "manifest_digest": manifest_digest}
        common["processing_fingerprint"] = processing_fingerprint
        common["source_digest"] = code_digest
        if improved:
            atomic_torch_save(common, run_dir / "best.pt")
            write_predictions(run_dir / "validation_predictions.jsonl", rows)
        atomic_torch_save({**common, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": rng_state(), "global_step": global_step, "bad_epochs": bad_epochs, "best_rank": best_rank,
            "best_eer": best_eer, "microbatch": microbatch, "oom_adjustments": oom_adjustments,
            "elapsed_seconds": epoch_record["elapsed_seconds"]}, run_dir / "last.pt")
        completed_epoch = epoch
        write_json(run_dir / "progress.json", {"status": "epoch_complete", **epoch_record})
        print(f"epoch={epoch} val_eer={metrics['eer']:.6f} val_ce={validation_ce:.6f} seconds={epoch_record['epoch_seconds']:.1f}", flush=True)
        if epoch >= settings["min_epochs"] and bad_epochs >= settings["patience"]:
            stop_reason = "early_stopping"
            break
    complete = stop_reason in {"max_epochs", "early_stopping"} and completed_epoch >= settings["min_epochs"]
    result = {"status": "complete" if complete else "partial", "variant": variant, "seed": seed,
        "stop_reason": stop_reason, "completed_epoch": completed_epoch, "global_step": global_step,
        "elapsed_seconds": elapsed_before + time.monotonic() - began, "best_rank": list(best_rank) if math.isfinite(best_rank[0]) else None,
        "contract_digest": contract_digest, "manifest_digest": manifest_digest, "updated_at": utc_now()}
    result.update(processing_fingerprint=processing_fingerprint, source_digest=code_digest, actual_microbatch=microbatch,
                  effective_batch=effective, oom_adjustments=oom_adjustments,
                  numerical_limit="Microbatch changes preserve the weighted objective, but may change dropout and floating-point trajectories.")
    if complete:
        saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        restored, _ = predict(model, validation, device, microbatch, settings["precision"])
        original = read_jsonl(run_dir / "validation_predictions.jsonl")
        if [r["sample_id"] for r in restored] != [r["sample_id"] for r in original] or not np.allclose([r["score"] for r in restored], [r["score"] for r in original], atol=2e-5, rtol=2e-4):
            raise ValueError("Restored checkpoint predictions do not match selected validation predictions")
        result.update(validation=saved["validation"], checkpoint_sha256=file_sha256(run_dir / "best.pt"), restore_verified=True)
    write_json(run_dir / "result.json", result)
    if not complete:
        raise RuntimeError("Training budget ended before the planned stopping criterion")
    return result
