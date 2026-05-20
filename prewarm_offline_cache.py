import argparse
import os
import sys
from pathlib import Path

if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Prewarm offline ASR text caches and FAPI statistics caches.")
    parser.add_argument("--train-csv", default=os.environ.get("FAPI_TRAIN_CSV", "data/train/train.csv"))
    parser.add_argument("--train-root", default=os.environ.get("FAPI_TRAIN_ROOT", "data/train"))
    parser.add_argument("--val-csv", default=os.environ.get("FAPI_VAL_CSV", "data/val/val.csv"))
    parser.add_argument("--val-root", default=os.environ.get("FAPI_VAL_ROOT", "data/val"))
    parser.add_argument("--mode", choices=["train_val", "test", "all"], default="test")
    parser.add_argument("--test-csv", default=os.environ.get("FAPI_TEST_CSV", "data/test/test.csv"))
    parser.add_argument("--test-root", default=os.environ.get("FAPI_TEST_ROOT", "data/test"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--skip-fapi", action="store_true")
    parser.print_help()
    raise SystemExit(0)

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import TrainConfig
from offline_cache import OfflineAsrTextCache, OfflineFapiStatsCache, normalize_audio_path
from Text_encoder.FunASR.ASR import asr_infer
from FinalPart import FAPILMScorer, compute_gamma_lambda_from_stats


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


def _prewarm_asr_cache(audio_paths: list[str], asr_cache: OfflineAsrTextCache, device: str, chunk_size: int):
    """Populate ASR text cache entries without running a full training/evaluation pass."""
    pending_paths = [p for p in audio_paths if asr_cache.get(p) is None]
    if len(pending_paths) == 0:
        print(f"[ASR] cache already complete; skipping | total={len(audio_paths)}")
        return 0

    print(f"[ASR] prewarming {len(pending_paths)}/{len(audio_paths)} items")
    written = 0
    for chunk in _chunked(pending_paths, chunk_size):
        texts = asr_infer(chunk, device=device)
        asr_cache.put_many(chunk, texts)
        written += len(chunk)
        print(f"[ASR] written {written}/{len(pending_paths)}")
    return written


def _prewarm_fapi_cache(audio_paths: list[str], asr_cache: OfflineAsrTextCache, fapi_cache: OfflineFapiStatsCache, cfg: TrainConfig, lm_device: str):
    """Populate FAPI statistics from cached ASR text, keeping expensive LM scoring offline."""
    scorer = FAPILMScorer(cfg.lm_name, device=os.environ.get("FAPI_LM_DEVICE", lm_device), precision=cfg.lm_precision)
    pending_paths = [p for p in audio_paths if fapi_cache.get(p) is None]
    if len(pending_paths) == 0:
        print(f"[FAPI] cache already complete; skipping | total={len(audio_paths)}")
        return 0

    print(f"[FAPI] prewarming {len(pending_paths)}/{len(audio_paths)} items")
    written = 0
    for audio_path in pending_paths:
        text = asr_cache.get(audio_path)
        if text is None:
            text = ""
        ppl, topk_mean = scorer.ppl_and_topk(text, topk=cfg.topk)
        gamma_t, lam = compute_gamma_lambda_from_stats(
            ppl=ppl,
            topk_mean=topk_mean,
            p_threshold=cfg.p_threshold,
            gamma_base=cfg.gamma_base,
        )
        fapi_cache.put_many([audio_path], [ppl], [topk_mean])
        written += 1
        if written % 50 == 0 or written == len(pending_paths):
            print(f"[FAPI] written {written}/{len(pending_paths)} | gamma_t={gamma_t:.4f} | lambda={lam:.4f}")
    return written


def _export_asr_csv(audio_paths: list[str], asr_cache: OfflineAsrTextCache, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = 0
    for audio_path in audio_paths:
        text = asr_cache.get(audio_path)
        if text is None:
            text = ""
            missing += 1
        rows.append({
            "audio_path": normalize_audio_path(audio_path),
            "text": str(text),
        })

    pd.DataFrame(rows, columns=["audio_path", "text"]).to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[ASR-EXPORT] exported: {output_csv} | rows={len(rows)} | empty_text={missing}")


def main():
    parser = argparse.ArgumentParser(description="Prewarm offline ASR text caches and FAPI statistics caches.")
    parser.add_argument("--train-csv", default=os.environ.get("FAPI_TRAIN_CSV", "data/train/train.csv"))
    parser.add_argument("--train-root", default=os.environ.get("FAPI_TRAIN_ROOT", "data/train"))
    parser.add_argument("--val-csv", default=os.environ.get("FAPI_VAL_CSV", "data/val/val.csv"))
    parser.add_argument("--val-root", default=os.environ.get("FAPI_VAL_ROOT", "data/val"))
    parser.add_argument(
        "--mode",
        choices=["train_val", "test", "all"],
        default="test",
        help="Cache mode: train_val, test, or all",
    )
    parser.add_argument("--test-csv", default=os.environ.get("FAPI_TEST_CSV", "data/test/test.csv"))
    parser.add_argument("--test-root", default=os.environ.get("FAPI_TEST_ROOT", "data/test"))
    parser.add_argument("--device", default="cuda:0", help="Device used for ASR inference")
    parser.add_argument("--chunk-size", type=int, default=8, help="ASR prewarm batch size")
    parser.add_argument("--skip-fapi", action="store_true", help="Only prewarm ASR text cache")
    args = parser.parse_args()

    cfg = TrainConfig()
    cache_root = CURRENT_DIR / "offline_cache_store"
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "test":
        asr_cache_file = str(getattr(cfg, "asr_offline_cache_file_test", "offline_cache_store/offline_asr_cache_test.csv"))
        fapi_cache_file = str(getattr(cfg, "fapi_offline_cache_file_test", "offline_cache_store/offline_fapi_stats_cache_test.csv"))
    else:
        asr_cache_file = str(getattr(cfg, "asr_offline_cache_file", "offline_cache_store/offline_asr_cache.csv"))
        fapi_cache_file = str(getattr(cfg, "fapi_offline_cache_file", "offline_cache_store/offline_fapi_stats_cache.csv"))

    asr_cache_path = _resolve_path(CURRENT_DIR, asr_cache_file)
    fapi_cache_path = _resolve_path(CURRENT_DIR, fapi_cache_file)
    asr_cache = OfflineAsrTextCache(str(asr_cache_path), rank=0, world_size=1)
    fapi_cache = OfflineFapiStatsCache(str(fapi_cache_path), rank=0, world_size=1)

    train_val_audio_paths = []
    test_audio_paths = []
    if args.mode in ("train_val", "all"):
        train_val_audio_paths.extend(_load_audio_paths(Path(args.train_csv), Path(args.train_root)))
        train_val_audio_paths.extend(_load_audio_paths(Path(args.val_csv), Path(args.val_root)))
    if args.mode in ("test", "all"):
        test_audio_paths.extend(_load_audio_paths(Path(args.test_csv), Path(args.test_root)))

    audio_paths = train_val_audio_paths + test_audio_paths

    unique_audio_paths = list(dict.fromkeys(audio_paths))
    print(f"[INFO] audio items to prewarm: {len(unique_audio_paths)}")
    print(f"[INFO] mode: {args.mode}")
    print(f"[INFO] ASR cache: {asr_cache_path}")
    print(f"[INFO] FAPI cache: {fapi_cache_path}")
    print(f"[INFO] skip_fapi: {args.skip_fapi}")

    _prewarm_asr_cache(unique_audio_paths, asr_cache, device=args.device, chunk_size=args.chunk_size)
    if not args.skip_fapi:
        _prewarm_fapi_cache(unique_audio_paths, asr_cache, fapi_cache, cfg, lm_device=cfg.lm_device)

    if args.mode == "test" and len(test_audio_paths) > 0:
        _export_asr_csv(list(dict.fromkeys(test_audio_paths)), asr_cache, asr_cache_path)

    print("[DONE] offline cache prewarm completed")


if __name__ == "__main__":
    main()
