import argparse
import os
import sys
from pathlib import Path

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Prewarm offline MFCC/F0 acoustic feature caches.")
    parser.add_argument("--train-csv", default=os.environ.get("FAPI_TRAIN_CSV", "data/train/train.csv"))
    parser.add_argument("--train-root", default=os.environ.get("FAPI_TRAIN_ROOT", "data/train"))
    parser.add_argument("--val-csv", default=os.environ.get("FAPI_VAL_CSV", "data/val/val.csv"))
    parser.add_argument("--val-root", default=os.environ.get("FAPI_VAL_ROOT", "data/val"))
    parser.add_argument("--include-test", dest="include_test", action="store_true")
    parser.add_argument("--no-include-test", dest="include_test", action="store_false")
    parser.add_argument("--test-csv", default=os.environ.get("FAPI_TEST_CSV", "data/test/test.csv"))
    parser.add_argument("--test-root", default=os.environ.get("FAPI_TEST_ROOT", "data/test"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--raw-duration", type=float, default=5.0)
    parser.add_argument("--acoustic-max-len", type=int, default=512)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.set_defaults(include_test=True)
    parser.print_help()
    raise SystemExit(0)

import librosa
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from Dataset_all import _sanitize_array, extract_crepe_features, extract_mfcc_with_deltas
from config import TrainConfig
from offline_cache import OfflineAcousticFeatureCache, normalize_audio_path


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


def _load_waveform(audio_path: str, sample_rate: int, raw_duration: float, raw_target_samples: int):
    try:
        waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=raw_duration)
    except Exception as exc:
        print(f"[ACOUSTIC] read failed; using zeros: {audio_path} | error={exc}")
        waveform = [0.0] * raw_target_samples

    if waveform is None or len(waveform) == 0:
        waveform = [0.0] * raw_target_samples

    waveform = _sanitize_array(
        waveform,
        name=f"waveform_np({audio_path})",
        fallback_shape=None,
    )
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    return waveform


def _prewarm_paths(
    audio_paths: list[str],
    cache: OfflineAcousticFeatureCache,
    sample_rate: int,
    raw_duration: float,
    acoustic_max_len: int,
    n_mfcc: int,
    chunk_size: int,
):
    """Populate CPU-computed MFCC/F0 cache files before model training or evaluation."""
    pending_paths = [path for path in audio_paths if not cache.exists(path)]
    if len(pending_paths) == 0:
        print(f"[ACOUSTIC] cache already complete; skipping | total={len(audio_paths)}")
        return 0

    raw_target_samples = int(sample_rate * raw_duration)
    print(
        f"[ACOUSTIC] prewarming {len(pending_paths)}/{len(audio_paths)} items | "
        f"sr={sample_rate} | duration={raw_duration} | max_len={acoustic_max_len} | n_mfcc={n_mfcc}"
    )

    written = 0
    for chunk in _chunked(pending_paths, chunk_size):
        for audio_path in chunk:
            try:
                waveform = _load_waveform(audio_path, sample_rate, raw_duration, raw_target_samples)
                mfcc = extract_mfcc_with_deltas(
                    waveform,
                    sample_rate=sample_rate,
                    n_mfcc=n_mfcc,
                    max_len=acoustic_max_len,
                )
                f0, voiced_probs, mask = extract_crepe_features(
                    waveform,
                    sample_rate=sample_rate,
                    max_len=acoustic_max_len,
                )
                payload = {
                    "mfcc": mfcc,
                    "f0": f0,
                    "voiced_probs": voiced_probs,
                    "mask": mask,
                }
                cache.put(audio_path, payload)
                written += 1
            except Exception as exc:
                print(f"[ACOUSTIC] skipping failed sample: {audio_path} | error={exc}")

        print(f"[ACOUSTIC] written {written}/{len(pending_paths)}")

    return written


def main():
    parser = argparse.ArgumentParser(description="Prewarm offline MFCC/F0 acoustic feature caches.")
    parser.add_argument("--train-csv", default=os.environ.get("FAPI_TRAIN_CSV", "data/train/train.csv"))
    parser.add_argument("--train-root", default=os.environ.get("FAPI_TRAIN_ROOT", "data/train"))
    parser.add_argument("--val-csv", default=os.environ.get("FAPI_VAL_CSV", "data/val/val.csv"))
    parser.add_argument("--val-root", default=os.environ.get("FAPI_VAL_ROOT", "data/val"))
    parser.add_argument(
        "--include-test",
        dest="include_test",
        action="store_true",
        help="Also prewarm the test split (default)",
    )
    parser.add_argument(
        "--no-include-test",
        dest="include_test",
        action="store_false",
        help="Disable test split prewarm and process only train/val",
    )
    parser.add_argument("--test-csv", default=os.environ.get("FAPI_TEST_CSV", "data/test/test.csv"))
    parser.add_argument("--test-root", default=os.environ.get("FAPI_TEST_ROOT", "data/test"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--raw-duration", type=float, default=5.0)
    parser.add_argument("--acoustic-max-len", type=int, default=512)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=8, help="Progress chunk size")
    parser.set_defaults(include_test=True)
    args = parser.parse_args()

    cfg = TrainConfig()
    cache_root = CURRENT_DIR / "offline_cache_store"
    cache_root.mkdir(parents=True, exist_ok=True)

    acoustic_cache_dir = _resolve_path(
        CURRENT_DIR,
        str(getattr(cfg, "acoustic_offline_cache_dir", "offline_cache_store/acoustic_cache")),
    )
    acoustic_cache_dir.mkdir(parents=True, exist_ok=True)
    cache = OfflineAcousticFeatureCache(str(acoustic_cache_dir), rank=0, world_size=1)

    audio_paths = []
    audio_paths.extend(_load_audio_paths(Path(args.train_csv), Path(args.train_root)))
    audio_paths.extend(_load_audio_paths(Path(args.val_csv), Path(args.val_root)))
    if args.include_test:
        audio_paths.extend(_load_audio_paths(Path(args.test_csv), Path(args.test_root)))

    unique_audio_paths = list(dict.fromkeys(audio_paths))
    if len(unique_audio_paths) == 0:
        raise RuntimeError("No audio paths were found")

    print(f"[INFO] audio items to prewarm: {len(unique_audio_paths)}")
    print(f"[INFO] acoustic cache directory: {acoustic_cache_dir}")

    _prewarm_paths(
        unique_audio_paths,
        cache,
        sample_rate=int(args.sample_rate),
        raw_duration=float(args.raw_duration),
        acoustic_max_len=int(args.acoustic_max_len),
        n_mfcc=int(args.n_mfcc),
        chunk_size=int(args.chunk_size),
    )

    print("[DONE] MFCC/F0 offline cache prewarm completed")


if __name__ == "__main__":
    main()
