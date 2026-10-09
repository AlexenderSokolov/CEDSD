"""Traceable corpus audit, deterministic audio windows, and feature batches.

Source/generator are reporting axes, never connectivity edges.  Primary split
checks cover FAD AISHELL3 identities; external identity overlap is exposed rather
than misrepresented as speaker-independent generalization.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path, PurePosixPath
import re
import wave

import numpy as np

from .common import config_digest, file_sha256, read_json, read_jsonl, write_json, write_jsonl

PRIMARY_ROLES = {"train", "validation", "test_seen", "test_unseen"}
SSB = re.compile(r"(?<![A-Za-z0-9])(SSB\d{8})(?!\d)", re.I)
BAC = re.compile(r"(?<![A-Za-z0-9])(BAC\d{3}S\d{4}W\d+)(?!\d)", re.I)
LABELS = {"0": 0, "real": 0, "bonafide": 0, "bona_fide": 0, "bona fide": 0,
          "1": 1, "fake": 1, "spoof": 1}


def _read_audio(path):
    """Decode to float32 [frames, channels], preserving original rate."""
    try:
        import soundfile as sf
    except ImportError:
        with wave.open(str(path), "rb") as stream:
            rate, channels, width = stream.getframerate(), stream.getnchannels(), stream.getsampwidth()
            raw = stream.readframes(stream.getnframes())
        if width == 1:
            values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
        elif width == 2:
            values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
        elif width == 3:
            octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            ints = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
            ints = (ints ^ 0x800000) - 0x800000
            values = ints.astype(np.float32) / 8388608
        elif width == 4:
            values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
        else:
            raise ValueError(f"Unsupported PCM sample width: {width}")
        values = values.reshape(-1, channels)
    else:
        values, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate <= 0 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty, non-finite, or invalid-rate audio")
    return np.asarray(values, dtype=np.float32), int(rate)


def _window_from_audio(values, rate, sample_rate, max_seconds):
    if sample_rate <= 0 or max_seconds <= 0:
        raise ValueError("Audio sample_rate and max_seconds must be positive")
    # Cut before resampling so the filter never sees content beyond the window.
    count = min(len(values), int(math.floor(rate * max_seconds)))
    mono = values[:count].mean(axis=1, dtype=np.float32)
    if rate != sample_rate:
        from scipy.signal import resample_poly
        divisor = math.gcd(rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, rate // divisor)
    mono = np.ascontiguousarray(mono[:int(sample_rate * max_seconds)], dtype=np.float32)
    if not len(mono) or not np.isfinite(mono).all():
        raise ValueError("Window is empty or non-finite")
    return mono


def load_window(path: Path, sample_rate: int = 16000, max_seconds: float = 5) -> np.ndarray:
    """First max_seconds, mono, target-rate float32; never pad short clips."""
    values, rate = _read_audio(path)
    return _window_from_audio(values, rate, sample_rate, max_seconds)


def window_digest(values, sample_rate=16000):
    digest = hashlib.sha256(f"mono-float32:{sample_rate}:".encode())
    digest.update(np.asarray(values, dtype="<f4").tobytes())
    return digest.hexdigest()


def resolve_audio_path(data_root, filename):
    """Never allow CSV paths or symlinks to escape the declared corpus root."""
    root = Path(data_root).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest audio path escapes data_root")
    return path


def _labels_path(data_root, config):
    chosen = config.get("data", {}).get("labels_file")
    if chosen:
        path = Path(chosen)
        return path if path.is_absolute() else data_root / path
    paths = sorted(data_root.glob("*.csv"))
    if len(paths) != 1:
        raise ValueError("Set data.labels_file: data_root must otherwise contain exactly one CSV")
    return paths[0]


def _partition(role):
    return "test" if role in {"test_seen", "test_unseen"} else role


def _path_record(filename, original_label):
    filename = str(filename).replace("\\", "/")
    while filename.startswith("./"):
        filename = filename[2:]
    parts = list(PurePosixPath(filename).parts)
    lower = [part.lower() for part in parts]
    label = LABELS.get(str(original_label).strip().lower())
    source = parts[0] if parts else None
    fad = "fad" in lower
    mdpe = any(part == "mdpe" for part in lower)
    cv = any(part in {"commonvoice", "common_voice", "common-voice", "cv"} for part in lower)
    if fad:
        source = "FAD"
    elif mdpe:
        source = "MDPE"
    elif cv:
        source = "CommonVoice"
    ssbs = list(dict.fromkeys(token.upper() for token in SSB.findall(filename)))
    bacs = list(dict.fromkeys(token.upper() for token in BAC.findall(filename)))
    identities = [("aishell3", token, f"aishell3:{token[:7]}") for token in ssbs]
    identities += [("aishell1", token, f"aishell1:{re.search(r'S\d{4}', token).group(0)}") for token in bacs]
    speakers = sorted({item[2] for item in identities})
    parents = sorted({f"{item[0]}:{item[1]}" for item in identities})
    stem = PurePosixPath(filename).stem
    source_tokens = [item for item in identities if item[1] in stem.upper()]
    main_id = source_tokens[-1] if source_tokens else (identities[-1] if identities else None)
    base_corpus = main_id[0] if main_id else next((p for p in lower if p in {"aishell1", "aishell3", "thchs30", "magicread"}), None)
    original_partition = None
    for part in lower:
        if part in {"train", "train_noise"}:
            original_partition = "train"
        elif part in {"dev", "dev_noise", "validation", "val"}:
            original_partition = "validation"
        elif part in {"test", "test_noise", "test_seen", "test_unseen"}:
            original_partition = "test"
    seen = "unseen" not in lower and "test_unseen" not in lower
    primary_role = ("test_seen" if seen else "test_unseen") if original_partition == "test" else original_partition
    generator = parts[-2] if fad and label == 1 and len(parts) > 1 else (None if label == 0 else source)
    role = "quarantine"
    reasons = []
    if label is None:
        reasons.append("unknown_label")
    elif mdpe or cv:
        if label == 1:
            reasons.append("unexpected_label_for_real_source")
        else:
            role = "stress_mdpe" if mdpe else "stress_commonvoice"
    elif fad:
        expected = {0 if p in {"real", "real_noise", "bonafide"} else 1 for p in lower if p in {"real", "real_noise", "bonafide", "fake", "fake_noise", "spoof"}}
        if len(expected) != 1 or label not in expected:
            reasons.append("path_label_conflict")
        if any("replaceonce" in part.lower() or "partial" in part.lower() for part in parts):
            reasons.append("partial_spoof_untraceable_parent")
        if base_corpus == "aishell3" and not ssbs:
            reasons.append("untraceable_primary_identity")
        if ssbs:
            if len(ssbs) != 1 or original_partition is None:
                reasons.append("untraceable_primary_identity")
            else:
                role = primary_role
        else:
            role = "external_bonafide" if label == 0 else "external_spoof"
    else:
        role = "external_bonafide" if label == 0 else "external_spoof"
    if reasons:
        role = "quarantine"
    condition = "noise" if any(p == "noise" or p.endswith("_noise") for p in lower) else "clean"
    return {
        "sample_id": hashlib.sha256(filename.encode("utf-8")).hexdigest()[:24],
        "file": filename, "label": label, "original_label": original_label,
        "role": role, "source": source, "base_corpus": base_corpus,
        "speaker_id": main_id[2] if main_id else None,
        "utterance_id": main_id[1] if main_id else None,
        "parent_id": f"{main_id[0]}:{main_id[1]}" if main_id else None,
        "speaker_ids": speakers, "parent_ids": parents,
        "identity_roles": [{"utterance_id": item[1], "speaker_id": item[2],
                            "role": "source" if item in source_tokens else "reference_or_target",
                            "provenance": "path_inferred"} for item in identities],
        "generator_id": generator, "condition": condition,
        "original_partition": original_partition,
        "window_start": 0.0, "window_end": None,
        "provenance": {"label": "provided_csv", "identity": "path_inferred" if identities else "unknown",
                       "split": "FAD_original_path" if fad else None},
        "exclusion_reason": ";".join(reasons) if reasons else None,
        "content_sha256": None, "pcm_sha256": None, "window_sha256": None,
    }


def _exclude(record, reason):
    existing = set(filter(None, (record.get("exclusion_reason") or "").split(";")))
    existing.add(reason)
    record["exclusion_reason"] = ";".join(sorted(existing))
    record["role"] = "quarantine"


def audit_dataset(data_root: Path, output_dir: Path, config: dict) -> dict:
    """Write an all-row manifest, a reproducible audit report, and quarantine CSV."""
    data_root, output_dir = Path(data_root), Path(output_dir)
    labels_path = _labels_path(data_root, config)
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"file", "label"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain file,label columns")
        records = [_path_record(row["file"], row["label"]) for row in reader]
    if not records:
        raise ValueError("Empty label manifest")
    audio = config.get("audio", {})
    sample_rate, max_seconds = int(audio.get("sample_rate", 16000)), float(audio.get("max_seconds", 5))
    for row_index, record in enumerate(records):
        record["manifest_row"] = row_index + 2
        try:
            path = resolve_audio_path(data_root, record["file"])
            record["content_sha256"] = file_sha256(path)
            values, rate = _read_audio(path)
            digest = hashlib.sha256(f"decoded-float32:{rate}:{values.shape[1]}:".encode())
            digest.update(np.asarray(values, dtype="<f4").tobytes())
            record["pcm_sha256"] = digest.hexdigest()
            window = _window_from_audio(values, rate, sample_rate, max_seconds)
            record.update(window_sha256=window_digest(window, sample_rate), window_end=len(window) / sample_rate,
                          original_sample_rate=rate, original_channels=values.shape[1],
                          duration_seconds=len(values) / rate, window_num_samples=len(window))
        except Exception as error:
            _exclude(record, f"audio_read_failure:{type(error).__name__}")
            record["audio_error"] = str(error)

    # Identity graph excludes external reference overlap: external evaluations
    # explicitly disclose that overlap instead of claiming primary disjointness.
    parents = list(range(len(records)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    def union(first, second):
        parents[find(second)] = find(first)
    primary = [i for i, row in enumerate(records) if row["source"] == "FAD" and row["base_corpus"] == "aishell3" and row["parent_id"] and row["original_partition"]]
    edge_groups = defaultdict(list)
    for index in primary:
        row = records[index]
        for edge_type, values in [("speaker", row["speaker_ids"]), ("parent", row["parent_ids"]),
                                  ("duplicate", [row[k] for k in ("content_sha256", "pcm_sha256", "window_sha256") if row[k]])]:
            for value in values:
                edge_groups[(edge_type, value)].append(index)
    violations = Counter()
    for (edge_type, _), indices in edge_groups.items():
        for other in indices[1:]:
            union(indices[0], other)
        if len({records[i]["original_partition"] for i in indices}) > 1:
            violations[edge_type] += 1
            for index in indices:
                _exclude(records[index], f"cross_partition_{edge_type}")
    components = defaultdict(list)
    for index in primary:
        components[find(index)].append(index)
    for indices in components.values():
        reasons = {reason for index in indices for reason in (records[index]["exclusion_reason"] or "").split(";") if reason.startswith("cross_partition_")}
        if reasons:
            for index in indices:
                for reason in reasons:
                    _exclude(records[index], reason)

    duplicate_groups = defaultdict(list)
    file_groups = defaultdict(list)
    for index, record in enumerate(records):
        file_groups[record["file"]].append(index)
        if record["pcm_sha256"]:
            duplicate_groups[record["pcm_sha256"]].append(index)
    for indices in file_groups.values():
        if len(indices) > 1:
            for index in indices:
                _exclude(records[index], "duplicate_manifest_file")
    for indices in duplicate_groups.values():
        if len({records[index]["label"] for index in indices}) > 1:
            for index in indices:
                _exclude(records[index], "duplicate_label_conflict")
        for index in indices:
            records[index]["pcm_duplicate_group_size"] = len(indices)
    # Compute overlaps only against admitted train records, after quarantine.
    train = [row for row in records if row["role"] == "train"]
    train_speakers = {v for row in train for v in row["speaker_ids"]}
    train_parents = {v for row in train for v in row["parent_ids"]}
    train_generators = {row["generator_id"] for row in train if row["generator_id"]}
    train_hashes = {row[k] for row in train for k in ("content_sha256", "pcm_sha256", "window_sha256") if row[k]}
    for row in records:
        row["train_speaker_overlap"] = bool(set(row["speaker_ids"]) & train_speakers) if row["speaker_ids"] else None
        row["train_parent_overlap"] = bool(set(row["parent_ids"]) & train_parents) if row["parent_ids"] else None
        row["train_content_overlap"] = any(row[k] in train_hashes for k in ("content_sha256", "pcm_sha256", "window_sha256") if row[k])
        row["train_generator_name_overlap"] = row["generator_id"] in train_generators if row["generator_id"] else None
        if row["role"] not in PRIMARY_ROLES | {"quarantine"} and row["train_content_overlap"]:
            _exclude(row, "external_duplicate_of_train")
    role_counts = dict(sorted(Counter(row["role"] for row in records).items()))
    cross_counts = Counter((row["role"], row["source"], row["label"]) for row in records)
    detailed_counts = Counter((row["role"], row["source"], row["base_corpus"], row["generator_id"], row["condition"], row["label"]) for row in records)
    primary_labels = {partition: sorted({row["label"] for row in records if row["role"] in PRIMARY_ROLES and _partition(row["role"]) == partition}) for partition in ("train", "validation", "test")}
    audit = {"status": "completed", "record_count": len(records), "role_counts": role_counts,
             "source_label_counts": [{"role": key[0], "source": key[1], "label": key[2], "count": count} for key, count in sorted(cross_counts.items(), key=lambda item: str(item[0]))],
             "group_label_counts": [{"role": key[0], "source": key[1], "base_corpus": key[2], "generator_id": key[3], "condition": key[4], "label": key[5], "count": count} for key, count in sorted(detailed_counts.items(), key=lambda item: str(item[0]))],
             "primary_partition_labels": primary_labels,
             "primary_ready": all(labels == [0, 1] for labels in primary_labels.values()),
             "cross_partition_violations": dict(violations),
             "primary_component_count": len(components), "primary_largest_component": max(map(len, components.values()), default=0),
             "unknown_identity_count": sum(not row["speaker_ids"] for row in records),
             "quarantine_reasons": dict(Counter(reason for row in records for reason in (row["exclusion_reason"] or "").split(";") if reason)),
             "labels_sha256": file_sha256(labels_path), "audio_fingerprint": config_digest({"sample_rate": sample_rate, "max_seconds": max_seconds, "window": "first-unpadded-v1"}),
             "claim_limits": ["Path-inferred speaker identity is not independently verified cross-corpus identity.",
                              "Text, device, source, and generator disjointness are not established by this audit.",
                              "Full-PCM/window hashes do not exclude transformed near-duplicates.",
                              "Pure-label external groups require class-conditional rates; EER is undefined."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "manifest.jsonl", records)
    write_json(output_dir / "audit.json", audit)
    with (output_dir / "quarantine.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "file", "original_label", "exclusion_reason"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in records if row["role"] == "quarantine")
    return audit


class FeatureDataset:
    """Cached frozen features. Overrides alter explicit feature keys per sample."""
    def __init__(self, manifest_path, cache_dir, roles, overrides=None):
        self.records = [row for row in read_jsonl(manifest_path) if row["role"] in set(roles)]
        self.cache_dir = Path(cache_dir)
        self.overrides = overrides or {}
        metadata = self.cache_dir / "preprocess.json"
        if not metadata.exists():
            raise ValueError("Feature cache processing metadata is missing")
        report = read_json(metadata)
        self.processing = report.get("processing")
        self.processing_fingerprint = report.get("processing_fingerprint")
        if not isinstance(self.processing, dict) or config_digest(self.processing) != self.processing_fingerprint:
            raise ValueError("Feature cache processing metadata fingerprint mismatch")
        if "models" not in self.processing or "sample_rate" not in self.processing.get("audio", {}):
            raise ValueError("Feature cache processing metadata lacks model/audio identity")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        record = self.records[index]
        item = torch.load(self.cache_dir / f"{record['sample_id']}.pt", map_location="cpu", weights_only=True)
        if not item.get("cache_complete", False):
            raise ValueError("Feature cache is not complete")
        if item.get("sample_id") != record["sample_id"]:
            raise ValueError("Feature sample_id mismatch")
        expected_identity = {"processing_fingerprint": self.processing_fingerprint,
                             "window_sha256": record.get("window_sha256"),
                             "content_sha256": record.get("content_sha256"),
                             "models": self.processing["models"]}
        if not all(expected_identity[key] for key in ("window_sha256", "content_sha256")) or item.get("cache_identity") != expected_identity:
            raise ValueError("Feature cache identity does not match manifest and processing metadata")
        waveform = torch.as_tensor(item["waveform"], dtype=torch.float32).cpu().numpy()
        if waveform.ndim != 1 or window_digest(waveform, self.processing["audio"]["sample_rate"]) != record["window_sha256"]:
            raise ValueError("Cached waveform does not match audited window digest")
        override = self.overrides.get(record["sample_id"], {})
        if set(override) - {"text", "token_ids", "text_present", "transcript"}:
            raise ValueError("Text diagnostic override may not change audio, label, or sample identity")
        item.update(override)
        item.update(label=record["label"], record=record)
        return item


def collate_features(items):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    if not items:
        raise ValueError("Cannot collate an empty batch")
    waveforms = [torch.as_tensor(row["waveform"], dtype=torch.float32) for row in items]
    affective = [torch.as_tensor(row["affective"], dtype=torch.float32) for row in items]
    texts = [torch.as_tensor(row["text"], dtype=torch.float32) for row in items]
    for row, waveform, affect, text in zip(items, waveforms, affective, texts):
        if waveform.ndim != 1 or not len(waveform) or affect.ndim != 2 or not len(affect) or text.ndim != 2 or not len(text):
            raise ValueError(f"Invalid feature shape for {row['sample_id']}")
        if not all(torch.isfinite(value).all() for value in (waveform, affect, text)):
            raise ValueError(f"Non-finite features for {row['sample_id']}")
    wave_lengths = torch.tensor([len(value) for value in waveforms], dtype=torch.long)
    audio_lengths = torch.tensor([len(value) for value in affective], dtype=torch.long)
    text_lengths = torch.tensor([len(value) for value in texts], dtype=torch.long)
    return {"waveforms": pad_sequence(waveforms, batch_first=True), "waveform_lengths": wave_lengths,
            "affective_features": pad_sequence(affective, batch_first=True),
            "audio_mask": torch.arange(int(audio_lengths.max()))[None, :] < audio_lengths[:, None],
            "text_features": pad_sequence(texts, batch_first=True),
            "text_mask": torch.arange(int(text_lengths.max()))[None, :] < text_lengths[:, None],
            "text_present": torch.tensor([bool(row["text_present"]) for row in items], dtype=torch.bool),
            "labels": torch.tensor([row["label"] for row in items], dtype=torch.long),
            "sample_ids": [row["sample_id"] for row in items], "records": [row["record"] for row in items]}
