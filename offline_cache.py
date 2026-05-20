import csv
import hashlib
import os
from pathlib import Path

import torch


def normalize_audio_path(path: str) -> str:
    """Normalize cache keys so CSV, single-GPU, and DDP paths resolve consistently."""
    if path is None:
        return ""
    return os.path.normpath(str(path).strip())


class _BasePathCache:
    def __init__(self, cache_path: str, rank: int = 0, world_size: int = 1):
        raw_path = str(cache_path).strip()
        if len(raw_path) == 0:
            raise ValueError("cache_path must not be empty")

        self.base_path = Path(raw_path)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)

        self.read_paths = [self.base_path]
        stem = self.base_path.stem
        suffix = self.base_path.suffix
        shard_pattern = f"{stem}.rank*{suffix}"
        for p in sorted(self.base_path.parent.glob(shard_pattern)):
            if p not in self.read_paths:
                self.read_paths.append(p)

        if self.world_size > 1:
            self.write_path = self.base_path.with_name(f"{stem}.rank{self.rank}{suffix}")
            if self.write_path not in self.read_paths:
                self.read_paths.append(self.write_path)
        else:
            self.write_path = self.base_path

        self.data = {}
        self._load_all()

    def _load_all(self):
        self.data = {}
        for path in self.read_paths:
            if not path.exists():
                continue
            try:
                self._load_single(path)
            except Exception:
                continue

    def _load_single(self, path: Path):
        raise NotImplementedError()

    def _append_rows(self, rows):
        if not rows:
            return
        file_exists = self.write_path.exists()
        with open(self.write_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self._header())
            writer.writerows(rows)

    def _header(self):
        raise NotImplementedError()


class OfflineAsrTextCache(_BasePathCache):
    def _header(self):
        return ["audio_path", "text"]

    def _load_single(self, path: Path):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return
            for row in reader:
                key = normalize_audio_path(row.get("audio_path", ""))
                if len(key) == 0:
                    continue
                text = row.get("text", "")
                if text is None:
                    text = ""
                self.data[key] = str(text)

    def get(self, audio_path: str):
        key = normalize_audio_path(audio_path)
        if len(key) == 0:
            return None
        return self.data.get(key, None)

    def get_many(self, audio_paths):
        return [self.get(p) for p in audio_paths]

    def put_many(self, audio_paths, texts):
        rows = []
        for path, text in zip(audio_paths, texts):
            key = normalize_audio_path(path)
            if len(key) == 0:
                continue
            text_val = "" if text is None else str(text)
            if self.data.get(key, None) == text_val:
                continue
            self.data[key] = text_val
            rows.append([key, text_val])
        self._append_rows(rows)


class OfflineFapiStatsCache(_BasePathCache):
    def _header(self):
        return ["audio_path", "ppl", "topk_mean"]

    def _load_single(self, path: Path):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return
            for row in reader:
                key = normalize_audio_path(row.get("audio_path", ""))
                if len(key) == 0:
                    continue
                try:
                    ppl = float(row.get("ppl", "nan"))
                    topk = float(row.get("topk_mean", "nan"))
                except Exception:
                    continue
                if not (ppl == ppl and topk == topk):
                    continue
                self.data[key] = (ppl, topk)

    def get(self, audio_path: str):
        key = normalize_audio_path(audio_path)
        if len(key) == 0:
            return None
        return self.data.get(key, None)

    def get_many(self, audio_paths):
        return [self.get(p) for p in audio_paths]

    def put_many(self, audio_paths, ppls, topk_means):
        rows = []
        for path, ppl, topk in zip(audio_paths, ppls, topk_means):
            key = normalize_audio_path(path)
            if len(key) == 0:
                continue
            try:
                ppl_val = float(ppl)
                topk_val = float(topk)
            except Exception:
                continue
            if not (ppl_val == ppl_val and topk_val == topk_val):
                continue

            old_val = self.data.get(key, None)
            if old_val is not None:
                if abs(float(old_val[0]) - ppl_val) < 1e-12 and abs(float(old_val[1]) - topk_val) < 1e-12:
                    continue

            self.data[key] = (ppl_val, topk_val)
            rows.append([key, f"{ppl_val:.10f}", f"{topk_val:.10f}"])
        self._append_rows(rows)


class OfflineE2VFeatureCache:
    def __init__(self, cache_dir: str, rank: int = 0, world_size: int = 1):
        raw_path = str(cache_dir).strip()
        if len(raw_path) == 0:
            raise ValueError("cache_dir must not be empty")

        self.base_dir = Path(raw_path)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Read root caches first, then DDP rank shards, so offline prewarm and multi-rank runs interoperate.
        self.read_dirs = [self.base_dir]
        for i in range(max(1, self.world_size)):
            shard_dir = self.base_dir / f"rank{i}"
            if shard_dir not in self.read_dirs:
                self.read_dirs.append(shard_dir)

        if self.world_size > 1:
            self.write_dir = self.base_dir / f"rank{self.rank}"
        else:
            self.write_dir = self.base_dir
        self.write_dir.mkdir(parents=True, exist_ok=True)

    def _file_stem(self, audio_path: str) -> str:
        normalized = normalize_audio_path(audio_path)
        if len(normalized) == 0:
            raise ValueError("audio_path must not be empty")
        digest_hex = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]
        base_name = Path(normalized).stem
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_name)
        return f"{safe_name}_{digest_hex}"

    def get_path(self, audio_path: str) -> Path:
        return self.write_dir / f"{self._file_stem(audio_path)}.pt"

    def _all_candidate_paths(self, audio_path: str) -> list[Path]:
        file_name = f"{self._file_stem(audio_path)}.pt"
        candidates = []
        for d in self.read_dirs:
            candidates.append(d / file_name)
        return candidates

    def exists(self, audio_path: str) -> bool:
        for cache_path in self._all_candidate_paths(audio_path):
            if cache_path.exists():
                return True
        return False

    def get(self, audio_path: str):
        for cache_path in self._all_candidate_paths(audio_path):
            if not cache_path.exists():
                continue
            try:
                return torch.load(cache_path, map_location="cpu")
            except Exception:
                continue
        return None

    def get_many(self, audio_paths):
        return [self.get(p) for p in audio_paths]

    def put(self, audio_path: str, payload: dict):
        cache_path = self.get_path(audio_path)
        tmp_path = cache_path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)

    def put_many(self, audio_paths, payloads):
        for path, payload in zip(audio_paths, payloads):
            self.put(path, payload)


class OfflineAcousticFeatureCache:
    def __init__(self, cache_dir: str, rank: int = 0, world_size: int = 1):
        raw_path = str(cache_dir).strip()
        if len(raw_path) == 0:
            raise ValueError("cache_dir must not be empty")

        self.base_dir = Path(raw_path)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Read root caches first, then DDP rank shards, matching the E2V cache behavior.
        self.read_dirs = [self.base_dir]
        for i in range(max(1, self.world_size)):
            shard_dir = self.base_dir / f"rank{i}"
            if shard_dir not in self.read_dirs:
                self.read_dirs.append(shard_dir)

        if self.world_size > 1:
            self.write_dir = self.base_dir / f"rank{self.rank}"
        else:
            self.write_dir = self.base_dir
        self.write_dir.mkdir(parents=True, exist_ok=True)

    def _file_stem(self, audio_path: str) -> str:
        normalized = normalize_audio_path(audio_path)
        if len(normalized) == 0:
            raise ValueError("audio_path must not be empty")
        digest_hex = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]
        base_name = Path(normalized).stem
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_name)
        return f"{safe_name}_{digest_hex}"

    def get_path(self, audio_path: str) -> Path:
        return self.write_dir / f"{self._file_stem(audio_path)}.pt"

    def _all_candidate_paths(self, audio_path: str) -> list[Path]:
        file_name = f"{self._file_stem(audio_path)}.pt"
        candidates = []
        for d in self.read_dirs:
            candidates.append(d / file_name)
        return candidates

    def exists(self, audio_path: str) -> bool:
        for cache_path in self._all_candidate_paths(audio_path):
            if cache_path.exists():
                return True
        return False

    def get(self, audio_path: str):
        for cache_path in self._all_candidate_paths(audio_path):
            if not cache_path.exists():
                continue
            try:
                # Acoustic payloads contain numpy arrays; PyTorch >= 2.6 rejects them with weights_only=True.
                # These are local trusted caches, so read directly with weights_only=False.
                return torch.load(cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                # Older PyTorch versions do not support the weights_only argument.
                try:
                    return torch.load(cache_path, map_location="cpu")
                except Exception:
                    continue
            except Exception:
                continue
        return None

    def get_many(self, audio_paths):
        return [self.get(p) for p in audio_paths]

    def put(self, audio_path: str, payload: dict):
        cache_path = self.get_path(audio_path)
        tmp_path = cache_path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)

    def put_many(self, audio_paths, payloads):
        for path, payload in zip(audio_paths, payloads):
            self.put(path, payload)
