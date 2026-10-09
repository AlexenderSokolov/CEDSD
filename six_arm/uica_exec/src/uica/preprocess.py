"""Sequential frozen encoders on the exact unpadded five-second audio window.

This module never dispatches experiments. The authorized executor calls it.
Failed ASR is a stage failure, not an empty transcript or a fabricated feature.
"""
from __future__ import annotations

import gc
import importlib.metadata
import os
from pathlib import Path
import re
import unicodedata

from .common import config_digest, file_sha256, read_json, read_jsonl, write_json
from .data import load_window, resolve_audio_path, window_digest


def has_text_content(text: str) -> bool:
    """Check actual alphanumeric content before tokenizer special tokens."""
    return any(unicodedata.category(character)[0] in {"L", "N"} for character in str(text))


def clean_asr_text(text: str) -> str:
    """Remove SenseVoice task/language/emotion markers, not ordinary words."""
    return re.sub(r"<\|[^<>]*\|>", "", text).strip()


def _device(config):
    import torch
    requested = config.get("execution", {}).get("device")
    if requested:
        return requested
    return f"cuda:{config.get('execution', {}).get('gpu', 0)}" if torch.cuda.is_available() else "cpu"


def _release_models():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _model_spec(config, name):
    value = config["models"][name]
    if isinstance(value, dict):
        return value["id"], value.get("revision")
    return value, config["models"].get("revisions", {}).get(name)


def _frozen_auto_model(config, name, device):
    from funasr import AutoModel
    model_id, revision = _model_spec(config, name)
    args = {"model": model_id, "hub": "hf", "device": device, "disable_update": True,
            "disable_progress_bar": True, "disable_log": True}
    if revision:
        args["model_revision"] = revision
    model = AutoModel(**args)
    if hasattr(model, "model"):
        model.model.eval()
        for parameter in model.model.parameters():
            parameter.requires_grad_(False)
    return model


class TextFeatureEncoder:
    """Frozen BERT; encode returns text tensor, IDs and the pre-tokenization guard."""
    def __init__(self, config, device):
        from transformers import AutoModel, AutoTokenizer
        model_id, revision = _model_spec(config, "bert")
        kwargs = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.model = AutoModel.from_pretrained(model_id, **kwargs).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.max_length = int(config.get("text", {}).get("max_length", 128))
        if self.max_length < 2:
            raise ValueError("BERT text.max_length must retain CLS and SEP")
        if config.get("text", {}).get("include_special_tokens", True) is not True:
            raise ValueError("This protocol requires preserving CLS and SEP")
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision

    def encode(self, text: str) -> dict:
        import torch
        if not isinstance(text, str):
            raise TypeError("TextFeatureEncoder requires a string transcript")
        present = has_text_content(text)
        tokens = self.tokenizer(text, add_special_tokens=True, truncation=True,
                                max_length=self.max_length, padding=False, return_tensors="pt")
        token_ids = tokens["input_ids"][0].tolist()
        with torch.inference_mode():
            features = self.model(**{key: value.to(self.device) for key, value in tokens.items()}).last_hidden_state[0].float().cpu()
        if features.ndim != 2 or features.shape[0] != len(token_ids) or features.shape[1] != 768 or not torch.isfinite(features).all():
            raise ValueError("BERT returned invalid or non-finite token features")
        return {"text": features, "token_ids": token_ids, "text_present": present, "transcript": text}


def _processing_spec(config):
    models = {name: {"id": _model_spec(config, name)[0], "revision": _model_spec(config, name)[1]}
              for name in ("sensevoice", "emotion2vec", "bert")}
    packages = {}
    for name in ("funasr", "transformers", "torch", "soundfile", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"version": "same-window-frozen-v1", "models": models, "packages": packages,
            "audio": {"sample_rate": int(config.get("audio", {}).get("sample_rate", 16000)),
                      "max_seconds": float(config.get("audio", {}).get("max_seconds", 5)),
                      "window": "first-unpadded-cut-before-resample"},
            "text": {"max_length": int(config.get("text", {}).get("max_length", 128)),
                     "include_special_tokens": True, "normalization": "strip-sensevoice-markers-v1",
                     "empty_guard": "unicode-letter-or-number-before-tokenizer"},
            "asr": {"language": "zh", "use_itn": True, "vad": False},
            "affective": {"granularity": "frame", "extract_embedding": True}}


def _identity(row, fingerprint, processing):
    return {"processing_fingerprint": fingerprint, "window_sha256": row["window_sha256"],
            "content_sha256": row["content_sha256"], "models": processing["models"]}


def _load_checked_window(row, data_root, audio):
    path = resolve_audio_path(data_root, row["file"])
    if file_sha256(path) != row["content_sha256"]:
        raise ValueError(f"Audio changed since audit: {row['sample_id']}")
    values = load_window(path, sample_rate=audio["sample_rate"], max_seconds=audio["max_seconds"])
    if window_digest(values, audio["sample_rate"]) != row["window_sha256"]:
        raise ValueError(f"Audio window changed since audit: {row['sample_id']}")
    return values


def _cached_torch(path, identity, complete=False):
    import torch
    if not path.exists():
        return None
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("cache_identity") != identity:
        return None
    if complete and not value.get("cache_complete", False):
        return None
    return value


def preprocess_dataset(manifest_path, data_root, cache_dir, config, roles: list | None = None) -> dict:
    """Materialize resumable per-sample features, releasing each model in order.

No encoder or manifest label is passed to another encoder. Only waveform arrays
reach ASR/Emotion2vec, and only the returned transcript reaches BERT.
"""
    import torch
    cache_dir, data_root = Path(cache_dir), Path(data_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    processing = _processing_spec(config)
    fingerprint = config_digest(processing)
    records = [row for row in read_jsonl(manifest_path) if row["role"] != "quarantine" and (roles is None or row["role"] in roles)]
    if not records:
        raise ValueError("No admitted records match the requested preprocessing roles")
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Preprocessing requires unique sample IDs")
    identities = {row["sample_id"]: _identity(row, fingerprint, processing) for row in records}
    report_path = cache_dir / "preprocess.json"
    report = {"status": "running", "processing": processing, "processing_fingerprint": fingerprint,
              "manifest_sha256": file_sha256(manifest_path), "roles": roles,
              "requested_count": len(records), "cached_count": 0, "completed_count": 0,
              "empty_transcript_count": 0, "errors": []}
    write_json(report_path, report)
    pending = []
    for row in records:
        try:
            cached = _cached_torch(cache_dir / f"{row['sample_id']}.pt", identities[row["sample_id"]], complete=True)
            if cached is None:
                pending.append(row)
            else:
                # Reuse still checks source integrity; paths are not a content ID.
                _load_checked_window(row, data_root, processing["audio"])
                report["cached_count"] += 1
                report["empty_transcript_count"] += not cached["text_present"]
        except Exception as error:
            report.update(status="failed", stage="source_integrity")
            report["errors"].append({"stage": "source_integrity", "sample_id": row["sample_id"], "message": str(error)})
            write_json(report_path, report)
            raise RuntimeError(f"source_integrity failed: {error}") from error
    if not pending:
        report.update(status="completed", completed_count=len(records))
        write_json(report_path, report)
        return report
    device = _device(config)
    asr_dir, affect_dir = cache_dir / "asr", cache_dir / "affective"
    asr_dir.mkdir(exist_ok=True)
    affect_dir.mkdir(exist_ok=True)

    def failure(stage, row, error):
        report["errors"].append({"stage": stage, "sample_id": row["sample_id"],
                                 "error_type": type(error).__name__, "message": str(error)})

    def finish_stage(stage):
        report["stage"] = stage
        if report["errors"]:
            report["status"] = "failed"
            write_json(report_path, report)
            raise RuntimeError(f"{stage} failed for {len(report['errors'])} sample(s); see {report_path}")
        write_json(report_path, report)

    # First pass: ASR only. An empty valid result is distinct from any failure.
    model = None
    try:
        for row in pending:
            path = asr_dir / f"{row['sample_id']}.json"
            try:
                if path.exists() and read_json(path).get("cache_identity") == identities[row["sample_id"]]:
                    continue
                if model is None:
                    model = _frozen_auto_model(config, "sensevoice", device)
                values = _load_checked_window(row, data_root, processing["audio"])
                with torch.inference_mode():
                    result = model.generate(input=values, cache={}, language="zh", use_itn=True,
                                            batch_size_s=60, merge_vad=False, fs=processing["audio"]["sample_rate"])
                if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0].get("text"), str):
                    raise ValueError("ASR returned no unambiguous transcript result")
                write_json(path, {"sample_id": row["sample_id"], "raw_text": result[0]["text"],
                                  "transcript": clean_asr_text(result[0]["text"]), "asr_status": "completed",
                                  "cache_identity": identities[row["sample_id"]]})
            except Exception as error:
                failure("asr", row, error)
                if model is None:
                    break  # Never retry a failed model download/init per corpus row.
    finally:
        del model
        _release_models()
    finish_stage("asr")

    # Second pass: frame-level Emotion2vec on exactly the same unpadded window.
    model = None
    try:
        for row in pending:
            path = affect_dir / f"{row['sample_id']}.pt"
            try:
                if _cached_torch(path, identities[row["sample_id"]]) is not None:
                    continue
                if model is None:
                    model = _frozen_auto_model(config, "emotion2vec", device)
                values = _load_checked_window(row, data_root, processing["audio"])
                with torch.inference_mode():
                    result = model.generate(input=values, granularity="frame", extract_embedding=True,
                                            fs=processing["audio"]["sample_rate"])
                if not isinstance(result, list) or len(result) != 1 or "feats" not in result[0]:
                    raise ValueError("Emotion2vec returned no frame embeddings")
                features = torch.as_tensor(result[0]["feats"]).detach().float().cpu()
                if features.ndim != 2 or not len(features) or not torch.isfinite(features).all():
                    raise ValueError("Invalid or non-finite Emotion2vec frame embeddings")
                _atomic_torch_save(path, {"sample_id": row["sample_id"], "affective": features,
                                          "cache_identity": identities[row["sample_id"]]})
            except Exception as error:
                failure("emotion2vec", row, error)
                if model is None:
                    break
    finally:
        del model
        _release_models()
    finish_stage("emotion2vec")

    # Third pass: BERT. XLSR is intentionally absent; its output is trainable.
    encoder = None
    try:
        for row in pending:
            try:
                if encoder is None:
                    encoder = TextFeatureEncoder(config, device)
                    report["bert_resolved_revision"] = encoder.resolved_revision
                asr = read_json(asr_dir / f"{row['sample_id']}.json")
                affective = _cached_torch(affect_dir / f"{row['sample_id']}.pt", identities[row["sample_id"]])
                if asr["cache_identity"] != identities[row["sample_id"]] or asr.get("asr_status") != "completed" or affective is None:
                    raise ValueError("Intermediate feature identity mismatch")
                encoded = encoder.encode(asr["transcript"])
                values = _load_checked_window(row, data_root, processing["audio"])
                item = {"waveform": torch.from_numpy(values.copy()), "affective": affective["affective"],
                        "text": encoded["text"], "text_present": encoded["text_present"],
                        "token_ids": encoded["token_ids"], "transcript": asr["transcript"],
                        "sample_id": row["sample_id"], "cache_identity": identities[row["sample_id"]], "cache_complete": True}
                _atomic_torch_save(cache_dir / f"{row['sample_id']}.pt", item)
                report["completed_count"] += 1
            except Exception as error:
                failure("bert", row, error)
                if encoder is None:
                    break
    finally:
        del encoder
        _release_models()
    finish_stage("bert")
    report["completed_count"] += report["cached_count"]
    report["empty_transcript_count"] += sum(not has_text_content(read_json(asr_dir / f"{row['sample_id']}.json")["transcript"]) for row in pending)
    report["status"] = "completed"
    write_json(report_path, report)
    return report
