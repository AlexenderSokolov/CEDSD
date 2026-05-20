import argparse
import os
import sys
from pathlib import Path

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Prewarm offline Emotion2Vec feature caches.")
    parser.add_argument("--train-csv", default=os.environ.get("FAPI_TRAIN_CSV", "data/train/train.csv"))
    parser.add_argument("--train-root", default=os.environ.get("FAPI_TRAIN_ROOT", "data/train"))
    parser.add_argument("--val-csv", default=os.environ.get("FAPI_VAL_CSV", "data/val/val.csv"))
    parser.add_argument("--val-root", default=os.environ.get("FAPI_VAL_ROOT", "data/val"))
    parser.add_argument("--test-csv", default=os.environ.get("FAPI_TEST_CSV", "data/test/test.csv"))
    parser.add_argument("--test-root", default=os.environ.get("FAPI_TEST_ROOT", "data/test"))
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--raw-duration", type=float, default=5.0)
    parser.print_help()
    raise SystemExit(0)

import librosa
import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from Audio_united.Emotion2vec.e2v import Emotion2VecExtractor
from Audio_united.MFCASTDA.MFCASTDA import Stage1_Dual_Stream
from config import TrainConfig
from offline_cache import OfflineE2VFeatureCache, normalize_audio_path


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    return base_dir / path


def _load_audio_paths(csv_path: Path, data_root: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    if "file" not in df.columns:
        raise KeyError(f"CSV is missing the 'file' column: {csv_path}")

    paths = []
    for file_name in df["file"].astype(str).tolist():
        audio_path = normalize_audio_path(data_root / file_name)
        paths.append(audio_path)
    return paths


def _chunked(values: list[str], chunk_size: int):
    if chunk_size <= 0:
        chunk_size = 1
    for start in range(0, len(values), chunk_size):
        yield values[start:start + chunk_size]


def _pad_or_trim_waveform(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    if waveform.numel() >= target_samples:
        return waveform[:target_samples]
    pad = target_samples - waveform.numel()
    return torch.nn.functional.pad(waveform, (0, pad))


def _infer_target_num_frames(stage1: Stage1_Dual_Stream, sample_path: str, sample_rate: int, raw_duration: float) -> int:
    waveform, _ = librosa.load(sample_path, sr=sample_rate, mono=True, duration=raw_duration)
    waveform_tensor = torch.from_numpy(waveform.astype("float32"))
    waveform_tensor = _pad_or_trim_waveform(waveform_tensor, int(sample_rate * raw_duration))
    waveform_tensor = waveform_tensor.unsqueeze(0)
    try:
        stage1_device = next(stage1.parameters()).device
    except StopIteration:
        stage1_device = torch.device("cpu")
    waveform_tensor = waveform_tensor.to(stage1_device)
    with torch.no_grad():
        stage1_feats = stage1(waveform_tensor)
    if stage1_feats is None or stage1_feats.ndim != 3:
        raise RuntimeError(f"Invalid Stage1 output; cannot infer target_num_frames: {sample_path}")
    return int(stage1_feats.size(-1))


def _build_payload(feats, mask, scores, lengths=None, frame_time=None):
    payload = {
        "e2v_feats": feats.detach().cpu(),
        "e2v_mask": mask.detach().cpu(),
        "e2v_scores": scores.detach().cpu(),
    }
    if lengths is not None:
        payload["lengths"] = lengths.detach().cpu()
    if frame_time is not None:
        payload["frame_time"] = frame_time.detach().cpu() if torch.is_tensor(frame_time) else frame_time
    return payload


def _prewarm_paths(audio_paths: list[str], extractor: Emotion2VecExtractor, cache: OfflineE2VFeatureCache, target_num_frames: int, chunk_size: int):
    """Populate frame-aligned Emotion2Vec cache files for later batch-level reuse."""
    pending_paths = [path for path in audio_paths if not cache.exists(path)]
    if len(pending_paths) == 0:
        print(f"[E2V] cache already complete; skipping | total={len(audio_paths)}")
        return 0

    print(f"[E2V] prewarming {len(pending_paths)}/{len(audio_paths)} items | target_num_frames={target_num_frames}")
    written = 0
    for chunk in _chunked(pending_paths, chunk_size):
        try:
            feats, mask, scores, lengths, frame_time = extractor.extract(chunk, target_num_frames=target_num_frames, return_meta=True)
        except Exception as exc:
            print(f"[E2V] batch extraction failed; falling back to single-item extraction | chunk_size={len(chunk)} | error={exc}")
            for audio_path in chunk:
                try:
                    single_feats, single_mask, single_scores, single_lengths, single_frame_time = extractor.extract([audio_path], target_num_frames=target_num_frames, return_meta=True)
                    payload = _build_payload(single_feats.squeeze(0), single_mask.squeeze(0), single_scores.squeeze(0), single_lengths, single_frame_time)
                    cache.put(audio_path, payload)
                    written += 1
                except Exception as single_exc:
                    print(f"[E2V] skipping failed sample: {audio_path} | error={single_exc}")
            continue

        for offset, audio_path in enumerate(chunk):
            payload = _build_payload(
                feats[offset],
                mask[offset],
                scores[offset],
                lengths[offset:offset + 1],
                frame_time,
            )
            cache.put(audio_path, payload)
            written += 1

        print(f"[E2V] written {written}/{len(pending_paths)}")

    return written


def main():
    parser = argparse.ArgumentParser(description="Prewarm offline Emotion2Vec feature caches.")
    parser.add_argument("--train-csv", default=os.environ.get("FAPI_TRAIN_CSV", "data/train/train.csv"))
    parser.add_argument("--train-root", default=os.environ.get("FAPI_TRAIN_ROOT", "data/train"))
    parser.add_argument("--val-csv", default=os.environ.get("FAPI_VAL_CSV", "data/val/val.csv"))
    parser.add_argument("--val-root", default=os.environ.get("FAPI_VAL_ROOT", "data/val"))
    parser.add_argument("--test-csv", default=os.environ.get("FAPI_TEST_CSV", "data/test/test.csv"))
    parser.add_argument("--test-root", default=os.environ.get("FAPI_TEST_ROOT", "data/test"))
    parser.add_argument("--skip-test", action="store_true", help="Skip the test split")
    parser.add_argument("--device", default="cuda:0", help="Device used by Emotion2Vec and Stage1")
    parser.add_argument("--chunk-size", type=int, default=8, help="Emotion2Vec batch size")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--raw-duration", type=float, default=5.0)
    args = parser.parse_args()

    cfg = TrainConfig()
    cache_root = CURRENT_DIR / "offline_cache_store"
    cache_root.mkdir(parents=True, exist_ok=True)

    e2v_cache_dir = _resolve_path(CURRENT_DIR, cfg.e2v_offline_cache_dir)
    e2v_cache_dir.mkdir(parents=True, exist_ok=True)
    cache = OfflineE2VFeatureCache(str(e2v_cache_dir), rank=0, world_size=1)

    audio_paths = []
    audio_paths.extend(_load_audio_paths(Path(args.train_csv), Path(args.train_root)))
    audio_paths.extend(_load_audio_paths(Path(args.val_csv), Path(args.val_root)))
    if not args.skip_test:
        audio_paths.extend(_load_audio_paths(Path(args.test_csv), Path(args.test_root)))

    unique_audio_paths = list(dict.fromkeys(audio_paths))
    if len(unique_audio_paths) == 0:
        raise RuntimeError("No audio paths were found")

    stage1 = Stage1_Dual_Stream(
        sample_rate=args.sample_rate,
        n_mels=80,
        duration=args.raw_duration,
        cutoff_freq=4000,
        split_mode="mel_mask",
        transition_bins=2,
    ).to(args.device)
    stage1.eval()

    target_num_frames = _infer_target_num_frames(stage1, unique_audio_paths[0], args.sample_rate, args.raw_duration)
    print(f"[INFO] audio items to prewarm: {len(unique_audio_paths)}")
    print(f"[INFO] E2V cache directory: {e2v_cache_dir}")
    print(f"[INFO] inferred target_num_frames={target_num_frames}")

    extractor = Emotion2VecExtractor(device=args.device)
    _prewarm_paths(unique_audio_paths, extractor, cache, target_num_frames, args.chunk_size)

    print("[DONE] Emotion2Vec offline cache prewarm completed")


if __name__ == "__main__":
    main()
