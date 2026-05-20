import argparse as _early_argparse
import sys as _early_sys


def _print_dependency_light_help():
    parser = _early_argparse.ArgumentParser(description="FAPI speech forgery detection CLI.")
    parser.add_argument("--mode", choices=["train", "test", "predict", "predict_iam"], default="train")
    parser.add_argument("--train-csv", default="data/train/train.csv")
    parser.add_argument("--train-root", default="data/train")
    parser.add_argument("--val-csv", default="data/val/val.csv")
    parser.add_argument("--val-root", default="data/val")
    parser.add_argument("--test-csv", default="data/test/test.csv")
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--audio", action="append", default=[])
    parser.add_argument("--audio-list", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--offline-cache-root", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--disable-feature-cache", action="store_true")
    parser.add_argument("--disable-offline-cache", action="store_true")
    parser.add_argument("--shared-artifacts", action="store_true")
    parser.add_argument("--skip-final-test", action="store_true")
    parser.print_help()


if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in _early_sys.argv[1:]):
    _print_dependency_light_help()
    raise SystemExit(0)

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from Audio_united.Emotion2vec.e2v import Emotion2VecExtractor
from Audio_united.MFCASTDA.MFCASTDA import Stage1_Dual_Stream
from Text_encoder.BERT import TextEncoder
from Text_encoder.FunASR.ASR import asr_infer
from spectrum.Acoustic_models import CNN_MFCC, Transformer_F0, Fusion_MLP
from inverse_attention_runtime import InverseAttentionRuntime
from attention_interpretability import InterpretabilityScorer, AttentionVisualization, InterpretabilityLogger
from FinalPart import FAPIVizSuite
import pandas as pd
from Model_all import MultimodalDeepfakeDetector
# Import the unified dataset entry point.
import sys
import os
import logging
import argparse
from datetime import datetime, timedelta
from config import TrainConfig # Training parameters.
from losses_multitask import MultiTaskLossComputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    ) # Common evaluation metrics.


def _resolve_amp_dtype(cfg):
    """Map the configured AMP dtype string to a torch dtype."""
    amp_dtype_str = str(getattr(cfg, "amp_dtype", "bf16")).strip().lower()
    if amp_dtype_str == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_num_workers(requested_workers: int, world_size: int = 1) -> int:
    """Cap DataLoader workers per rank according to local CPU resources."""
    requested_workers = int(max(0, requested_workers))
    if requested_workers == 0:
        return 0
    cpu_total = int(os.cpu_count() or requested_workers)
    reserve = 2 if cpu_total > 4 else 0
    per_rank_cap = max(1, (cpu_total - reserve) // max(1, int(world_size)))
    return int(max(1, min(requested_workers, per_rank_cap)))


def _build_loader_perf_kwargs(cfg, num_workers: int) -> dict:
    """Build DataLoader performance options in one place."""
    kwargs = {
        "pin_memory": torch.cuda.is_available(),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg, "dataloader_persistent_workers", True))
        kwargs["prefetch_factor"] = int(max(1, getattr(cfg, "dataloader_prefetch_factor", 2)))
    return kwargs


def _resolve_inference_temperature(cfg) -> float:
    """Resolve and validate inference temperature, falling back to 1.0 for invalid values."""
    try:
        temperature = float(getattr(cfg, "inference_temperature", 1.0))
    except Exception:
        temperature = 1.0
    if (not np.isfinite(temperature)) or temperature <= 0.0:
        return 1.0
    return float(temperature)


def _resolve_temperature_candidates(cfg):
    """Resolve temperature-grid candidates from a tuple/list or comma-separated string."""
    raw_candidates = getattr(cfg, "temperature_candidates", (1.0, 1.2, 1.5, 2.0, 2.5, 3.0))
    if isinstance(raw_candidates, str):
        raw_candidates = raw_candidates.replace("，", ",").split(",")

    candidates = []
    for item in raw_candidates:
        try:
            t = float(item)
        except Exception:
            continue
        if np.isfinite(t) and t > 0.0:
            candidates.append(float(t))

    if len(candidates) == 0:
        return [1.0]
    return sorted(set(candidates))


def _sigmoid_with_temperature_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return sigmoid probabilities."""
    t = max(float(temperature), 1e-6)
    z = np.asarray(logits, dtype=np.float32).reshape(-1) / t
    z = np.clip(z, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _compute_eer_from_probs(y_true: np.ndarray, y_prob: np.ndarray):
    """Compute EER and its corresponding threshold."""
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        fnr = 1.0 - tpr
        idx = int(np.nanargmin(np.absolute(fpr - fnr)))
        return float(fpr[idx]), float(thresholds[idx])
    except Exception:
        return 0.0, 0.5


def _compute_binary_metrics_from_probs(all_labels: np.ndarray, all_preds: np.ndarray, threshold: float = 0.5):
    """Compute binary metrics from probabilities, including EER."""
    all_labels = np.asarray(all_labels, dtype=np.int32).reshape(-1)
    all_preds = np.asarray(all_preds, dtype=np.float32).reshape(-1)
    preds_binary = (all_preds > float(threshold)).astype(int)

    accuracy = accuracy_score(all_labels, preds_binary)
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    f1 = f1_score(all_labels, preds_binary, zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, preds_binary)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.0

    try:
        pr_auc = average_precision_score(all_labels, all_preds)
    except ValueError:
        pr_auc = 0.0

    try:
        mcc = matthews_corrcoef(all_labels, preds_binary)
    except ValueError:
        mcc = 0.0

    eer, eer_threshold = _compute_eer_from_probs(all_labels, all_preds)

    try:
        cm = confusion_matrix(all_labels, preds_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "mcc": float(mcc),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_true": all_labels.tolist(),
        "y_pred": preds_binary.tolist(),
        "y_prob": all_preds.tolist(),
    }


def _search_best_temperature_by_eer(y_true, logits, candidates):
    """Grid-search candidate temperatures for the minimum EER."""
    labels = np.asarray(y_true, dtype=np.int32).reshape(-1)
    logits_arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if labels.size == 0 or logits_arr.size == 0 or labels.size != logits_arr.size:
        return None

    grid_results = []
    for t in candidates:
        temp = max(float(t), 1e-6)
        probs = _sigmoid_with_temperature_np(logits_arr, temp)
        eer, eer_threshold = _compute_eer_from_probs(labels, probs)
        grid_results.append({
            "temperature": float(temp),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
        })

    if len(grid_results) == 0:
        return None

    best = min(grid_results, key=lambda x: (x["eer"], x["temperature"]))
    return {
        "best_temperature": float(best["temperature"]),
        "best_eer": float(best["eer"]),
        "best_eer_threshold": float(best["eer_threshold"]),
        "all_results": grid_results,
    }

# Current file directory.
current_dir = os.path.dirname(__file__)

# Parent directory, kept importable for legacy entry points.
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Keep historical direct-script imports working.
sys.path.append(parent_dir)

# Dataset import kept after the compatibility path setup.
from Dataset_all import UnifiedMultimodalDataset


def _build_logger(log_dir: Path, log_file: str, rank: int | None = None):
    logger = logging.getLogger("fapi_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(fmt)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # In DDP, add rank-specific logs so the first failing worker is easy to find.
    if rank is not None:
        log_path = Path(log_file)
        rank_file_name = f"{log_path.stem}.rank{rank}{log_path.suffix or '.log'}"
        rank_file_handler = logging.FileHandler(log_dir / rank_file_name, encoding="utf-8")
        rank_file_handler.setFormatter(fmt)
        logger.addHandler(rank_file_handler)
    return logger


def _resolve_artifact_paths(cfg):
    """Resolve output directory, log filename, and best-checkpoint filename.

    Supports environment variable overrides:
    - TRAIN_RUN_ID
    - TRAIN_AUTO_TIMESTAMP_RUN_ID (1/true/yes/on)
    - TRAIN_ISOLATE_RUN_ARTIFACTS (1/true/yes/on)
    - TRAIN_LOG_FILE
    - TRAIN_BEST_MODEL_FILE
    """
    output_dir_name = str(getattr(cfg, "output_dir_name", "outputs_fapi_viz")).strip() or "outputs_fapi_viz"
    base_dir = Path(current_dir) / output_dir_name

    env_run_id = str(os.environ.get("TRAIN_RUN_ID", "")).strip()
    cfg_run_id = str(getattr(cfg, "run_id", "")).strip()
    run_id = env_run_id or cfg_run_id
    auto_ts_env = str(os.environ.get("TRAIN_AUTO_TIMESTAMP_RUN_ID", "")).strip().lower()
    auto_ts = bool(getattr(cfg, "auto_timestamp_run_id", True))
    if auto_ts_env in {"1", "true", "yes", "on"}:
        auto_ts = True
    if auto_ts_env in {"0", "false", "no", "off"}:
        auto_ts = False

    if (len(run_id) == 0) and auto_ts:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    isolate_env = str(os.environ.get("TRAIN_ISOLATE_RUN_ARTIFACTS", "")).strip().lower()
    isolate = bool(getattr(cfg, "isolate_run_artifacts", True))
    if isolate_env in {"1", "true", "yes", "on"}:
        isolate = True
    if isolate:
        safe_run_id = run_id if len(run_id) > 0 else "run_default"
        viz_dir = base_dir / safe_run_id
    else:
        safe_run_id = run_id if len(run_id) > 0 else "shared"
        viz_dir = base_dir

    train_log_file = str(os.environ.get("TRAIN_LOG_FILE", getattr(cfg, "train_log_file", "train.log"))).strip() or "train.log"
    best_model_file = str(os.environ.get("TRAIN_BEST_MODEL_FILE", getattr(cfg, "best_model_file", "best_model.pth"))).strip() or "best_model.pth"
    best_model_path = viz_dir / best_model_file
    return viz_dir, safe_run_id, train_log_file, best_model_path


def _apply_quick_slice(df: pd.DataFrame, max_rows: int):
    """Subsample a DataFrame for quick mode.

    Prefer label-stratified sampling so the quick subset stays close to the
    original class distribution and does not over-amplify rare hard cases.
    """
    if df is None:
        return df
    if max_rows is None or max_rows <= 0:
        return df
    if len(df) <= max_rows:
        return df

    if "label" not in df.columns:
        return df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    sampled_parts = []
    grouped = list(df.groupby("label", dropna=False))
    if len(grouped) <= 1:
        return df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    group_sizes = {str(label): len(group_df) for label, group_df in grouped}
    total_size = float(len(df))

    # Allocate proportional quotas while keeping at least one sample per class.
    quotas = {}
    allocated = 0
    fractional_parts = []
    for label, group_df in grouped:
        raw_quota = (len(group_df) / total_size) * max_rows
        quota_floor = min(len(group_df), max(1, int(raw_quota)))
        quotas[label] = quota_floor
        allocated += quota_floor
        fractional_parts.append((raw_quota - int(raw_quota), label))

    # If quotas exceed the target, reclaim from larger groups first.
    if allocated > max_rows:
        overflow = allocated - max_rows
        for _, label in sorted(fractional_parts, key=lambda x: (len(df[df["label"] == x[1]]), x[0]), reverse=True):
            if overflow <= 0:
                break
            if quotas[label] > 1:
                quotas[label] -= 1
                overflow -= 1

    # Fill any remaining quota by descending fractional remainder.
    elif allocated < max_rows:
        remain = max_rows - allocated
        for _, label in sorted(fractional_parts, key=lambda x: x[0], reverse=True):
            if remain <= 0:
                break
            group_len = len(df[df["label"] == label])
            if quotas[label] < group_len:
                quotas[label] += 1
                remain -= 1

    for label, group_df in grouped:
        quota = min(quotas[label], len(group_df))
        sampled_parts.append(group_df.sample(n=quota, random_state=42))

    sampled_df = pd.concat(sampled_parts, axis=0)
    if len(sampled_df) > max_rows:
        sampled_df = sampled_df.sample(n=max_rows, random_state=42)
    return sampled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def _check_loss_sanity(total_loss, loss_dict, cfg):
    if total_loss is None or loss_dict is None:
        return False, "loss is empty"
    if not torch.is_tensor(total_loss):
        return False, "total_loss is not a tensor"
    if not torch.isfinite(total_loss).all():
        return False, "total_loss is non-finite"

    required_keys = ("L_CE", "L_KL", "L_conflict", "L_FAPI", "L_task_sum", "UW_ratio")
    for key in required_keys:
        if key not in loss_dict or loss_dict[key] is None:
            return False, f"loss_dict is missing key: {key}"

    for key, value in loss_dict.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return False, f"{key} is non-finite"

    loss_empty_eps = float(getattr(cfg, "loss_empty_eps", 1e-8))
    task_sum_value = loss_dict["L_task_sum"]
    if torch.is_tensor(task_sum_value):
        task_sum_value = float(task_sum_value.detach().abs().item())
    else:
        task_sum_value = abs(float(task_sum_value))

    if task_sum_value <= loss_empty_eps:
        return False, "L_task_sum is empty"

    primary_total = 0.0
    for key in ("L_CE", "L_KL", "L_conflict", "L_FAPI"):
        value = loss_dict[key]
        if torch.is_tensor(value):
            value = float(value.detach().abs().item())
        else:
            value = abs(float(value))
        primary_total += value

    if primary_total <= loss_empty_eps:
        return False, "all primary losses are empty"

    return True, "ok"


def _flag_to_int(value, default: bool = True) -> int:
    if value is None:
        return int(bool(default))
    if torch.is_tensor(value):
        return int(float(value.detach().item()) > 0.5)
    return int(bool(value))


def _skip_stage_label(stage: str) -> str:
    """Translate internal skip-stage identifiers into readable log labels."""
    label_map = {
        "forward": "data/forward failure, None model output, or invalid text quality",
        "loss": "empty/non-finite loss, missing loss key, or intercepted empty-text batch",
        "backward": "backward pass or gradient synchronization failure",
    }
    return label_map.get(stage, "unknown stage")


def _format_skip_stats(skip_stats: dict) -> str:
    """Format skip statistics with readable stage descriptions."""
    return (
        f"forward={skip_stats.get('forward', 0)}({_skip_stage_label('forward')}) | "
        f"loss={skip_stats.get('loss', 0)}({_skip_stage_label('loss')}) | "
        f"backward={skip_stats.get('backward', 0)}({_skip_stage_label('backward')})"
    )


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _format_forward_diag_for_log(diag: dict) -> str:
    if not isinstance(diag, dict) or len(diag) == 0:
        return ""
    sample_idx = diag.get("sample_idx", [])
    sample_paths = diag.get("sample_paths", [])
    return (
        f" | root_cause_code={diag.get('root_cause_code', '')}"
        f" | diag_batch_id={diag.get('batch_id', '')}"
        f" | diag_epoch_id={diag.get('epoch_id', '')}"
        f" | diag_step_id={diag.get('step_id', '')}"
        f" | diag_rank={diag.get('rank', '')}"
        f" | diag_sample_idx={sample_idx[:8]}"
        f" | diag_sample_paths={sample_paths[:3]}"
    )


def _safe_div(numerator: float, denominator: float) -> float:
    denom = float(denominator)
    if abs(denom) <= 1e-12:
        return 0.0
    return float(numerator) / denom


def _log_eval_metrics_panel(logger, stage_name: str, metrics: dict):
    """Log a validation/test metric panel instead of a single headline metric."""
    tn = int(metrics.get("tn", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tp = int(metrics.get("tp", 0))

    tpr = _safe_div(tp, tp + fn)  # recall / sensitivity
    tnr = _safe_div(tn, tn + fp)  # specificity
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)

    logger.info(
        f"[{stage_name}] Loss={float(metrics.get('loss', 0.0)):.4f} | "
        f"Acc={float(metrics.get('accuracy', 0.0)):.4f} | "
        f"Precision={float(metrics.get('precision', 0.0)):.4f} | "
        f"Recall={float(metrics.get('recall', 0.0)):.4f} | "
        f"F1={float(metrics.get('f1', 0.0)):.4f}"
    )
    logger.info(
        f"[{stage_name}] BalAcc={float(metrics.get('balanced_accuracy', 0.0)):.4f} | "
        f"MCC={float(metrics.get('mcc', 0.0)):.4f} | "
        f"AUC={float(metrics.get('auc', 0.0)):.4f} | "
        f"PR-AUC={float(metrics.get('pr_auc', 0.0)):.4f} | "
        f"EER={float(metrics.get('eer', 0.0)):.4f} | "
        f"EER-TH={float(metrics.get('eer_threshold', 0.5)):.4f}"
    )
    logger.info(
        f"[{stage_name}] ConfMat TN={tn}, FP={fp}, FN={fn}, TP={tp} | "
        f"TPR={tpr:.4f}, TNR={tnr:.4f}, FPR={fpr:.4f}, FNR={fnr:.4f}"
    )
    logger.info(
        f"[{stage_name}] valid_batches={int(metrics.get('valid_batches', 0))} | "
        f"skipped_batches={int(metrics.get('skipped_batches', 0))} | "
        f"temperature={float(metrics.get('temperature_used', 1.0)):.4f}"
    )


def _format_quick_label_stats(df: pd.DataFrame) -> str:
    """Format quick-mode label distribution for concise logs."""
    if df is None or len(df) == 0 or "label" not in df.columns:
        return "label_stats=NA"
    counts = df["label"].astype(str).value_counts(dropna=False).to_dict()
    total = float(len(df))
    parts = []
    for key, value in sorted(counts.items(), key=lambda item: item[0]):
        parts.append(f"{key}:{value}/{int(total)}={value/total:.3f}")
    return "label_stats={" + ", ".join(parts) + "}"


def _build_convergence_row(epoch_idx: int, train_loss: float, val_metrics: dict, prev_row: dict | None, cfg: TrainConfig):
    """Build one convergence-analysis row with current metrics and previous-row deltas."""
    val_auc = float(val_metrics.get("auc", 0.0))
    val_f1 = float(val_metrics.get("f1", 0.0))
    val_eer = float(val_metrics.get("eer", 0.0))
    val_loss = float(val_metrics.get("loss", 0.0))

    if prev_row is None:
        delta_auc = 0.0
        delta_f1 = 0.0
        delta_eer = 0.0
        trend = "warmup"
    else:
        delta_auc = val_auc - float(prev_row["val_auc"])
        delta_f1 = val_f1 - float(prev_row["val_f1"])
        delta_eer = val_eer - float(prev_row["val_eer"])

        improve_votes = 0
        degrade_votes = 0

        if delta_auc > cfg.convergence_auc_tol:
            improve_votes += 1
        elif delta_auc < -cfg.convergence_auc_tol:
            degrade_votes += 1

        if delta_f1 > cfg.convergence_f1_tol:
            improve_votes += 1
        elif delta_f1 < -cfg.convergence_f1_tol:
            degrade_votes += 1

        if delta_eer < -cfg.convergence_eer_tol:
            improve_votes += 1
        elif delta_eer > cfg.convergence_eer_tol:
            degrade_votes += 1

        if improve_votes >= 2:
            trend = "improving"
        elif degrade_votes >= 2:
            trend = "degrading"
        else:
            trend = "plateau"

    return {
        "epoch": int(epoch_idx + 1),
        "train_loss": float(train_loss),
        "val_loss": val_loss,
        "val_auc": val_auc,
        "val_f1": val_f1,
        "val_eer": val_eer,
        "delta_auc": float(delta_auc),
        "delta_f1": float(delta_f1),
        "delta_eer": float(delta_eer),
        "trend": trend,
    }


def _collect_high_freq_empty_text_paths(log_file: Path, min_count: int) -> dict:
    """Collect high-frequency empty-text paths from training logs."""
    if log_file is None or (not log_file.exists()) or int(min_count) <= 0:
        return {}

    counts = {}
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                if ("ASR returned empty text" not in raw_line) and ("ASR 返回空文本" not in raw_line):
                    continue
                marker = "| path="
                idx = raw_line.find(marker)
                if idx < 0:
                    continue
                path_text = raw_line[idx + len(marker):].strip()
                if len(path_text) == 0:
                    continue
                normalized = os.path.normpath(path_text)
                counts[normalized] = counts.get(normalized, 0) + 1
    except Exception:
        return {}

    return {k: v for k, v in counts.items() if v >= int(min_count)}


def _filter_train_df_by_empty_text_logs(df_train: pd.DataFrame, train_data_path: str, cfg: TrainConfig, viz_dir: Path, logger) -> pd.DataFrame:
    """Filter high-frequency empty-text samples from previous logs."""
    if df_train is None or len(df_train) == 0:
        return df_train
    if not bool(getattr(cfg, "empty_text_offline_filter_enable", True)):
        return df_train
    if "file" not in df_train.columns:
        logger.warning("离线空文本过滤跳过：df_train 缺少 file 列。")
        return df_train

    log_file = viz_dir / cfg.train_log_file
    min_count = int(getattr(cfg, "empty_text_offline_min_count", 3))
    bad_path_counts = _collect_high_freq_empty_text_paths(log_file, min_count=min_count)
    if not bad_path_counts:
        logger.info("离线空文本过滤：未发现达到阈值的高频空文本路径。")
        return df_train

    bad_abs_paths = set(os.path.normpath(p) for p in bad_path_counts.keys())
    bad_basenames = set(os.path.basename(p) for p in bad_abs_paths)

    abs_paths = df_train["file"].astype(str).map(lambda x: os.path.normpath(os.path.join(train_data_path, x)))
    basenames = df_train["file"].astype(str).map(lambda x: os.path.basename(x))
    remove_mask = abs_paths.isin(bad_abs_paths) | basenames.isin(bad_basenames)

    removed = int(remove_mask.sum())
    if removed <= 0:
        logger.info("离线空文本过滤：日志中有高频路径，但未命中当前训练集。")
        return df_train

    removed_df = df_train.loc[remove_mask].copy()
    kept_df = df_train.loc[~remove_mask].reset_index(drop=True)

    if len(kept_df) == 0:
        logger.warning("离线空文本过滤后训练集为空，已回退为原始训练集。")
        return df_train

    audit_file = str(getattr(cfg, "empty_text_offline_audit_file", "filtered_empty_text_samples.csv")).strip()
    audit_path = viz_dir / audit_file
    try:
        removed_df.to_csv(audit_path, index=False, encoding="utf-8")
    except Exception as e:
        logger.warning(f"离线空文本审计文件写入失败: {audit_path} | {e}")

    top_items = sorted(bad_path_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_desc = ", ".join([f"{os.path.basename(k)}:{v}" for k, v in top_items])
    logger.warning(
        f"离线空文本过滤生效 | min_count={min_count} | removed={removed}/{len(df_train)} | "
        f"kept={len(kept_df)} | top={top_desc}"
    )
    logger.info(f"离线空文本过滤审计文件: {audit_path}")
    return kept_df




def train(df_train, train_data_path, df_val, val_data_path, cfg=None):
    print("Starting train function")
    print("\ninitializing training run")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = world_size > 1
    if use_ddp:
        # Rank 0 may validate alone while other ranks wait; a long timeout avoids false NCCL failure reports.
        ddp_timeout = timedelta(minutes=120)
        dist.init_process_group(
            backend="nccl",
            rank=global_rank,
            world_size=world_size,
            device_id=local_rank,
            timeout=ddp_timeout,
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    print(f"Device set to {device}")
    viz_dir = Path(current_dir) / "outputs_fapi_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------- input validation start --------------------
    if df_train is None or len(df_train) == 0:
        raise ValueError("训练数据 df 为空")
    if not train_data_path:
        raise ValueError("train_data_path 不能为空")
    if df_val is None or len(df_val) == 0:
        raise ValueError("验证数据 df 为空")
    if not val_data_path:
        raise ValueError("val_data_path 不能为空")
    # -------------------- input validation end --------------------
    
    cfg = cfg or TrainConfig()
    viz_dir, run_id, train_log_file, best_model_path = _resolve_artifact_paths(cfg)
    viz_dir.mkdir(parents=True, exist_ok=True)
    cfg.train_log_file = train_log_file
    logger = _build_logger(viz_dir, cfg.train_log_file, rank=global_rank)
    mode_name = "Quick" if cfg.quick_mode else "Formal"
    logger.info(
        f"Run隔离信息 | run_id={run_id} | viz_dir={viz_dir} | "
        f"train_log={cfg.train_log_file} | best_model={best_model_path.name}"
    )
    if use_ddp:
        logger.info("DDP timeout 已设置为 120 分钟（防止长时间验证阶段触发默认超时）")

    if cfg.quick_mode:
        # Quick mode uses small stratified subsets for end-to-end smoke checks.
        df_train = _apply_quick_slice(df_train, cfg.quick_train_rows)
        df_val = _apply_quick_slice(df_val, cfg.quick_val_rows)
        logger.info(
            f"QuickMode ON | train_rows={len(df_train)} | val_rows={len(df_val)} | "
            f"workers={cfg.quick_num_workers} | {_format_quick_label_stats(df_train)} | {_format_quick_label_stats(df_val)}"
        )
    else:
        logger.info(f"FormalMode ON | train_rows={len(df_train)} | val_rows={len(df_val)}")

    # Filter high-frequency empty-text samples from previous logs to reduce repeated bad-sample noise.
    df_train = _filter_train_df_by_empty_text_logs(
        df_train=df_train,
        train_data_path=train_data_path,
        cfg=cfg,
        viz_dir=viz_dir,
        logger=logger,
    )
    logger.info(f"过滤后训练集行数: {len(df_train)}")

    # ===== UQ CONTROL START: log controller parameters for traceability =====
    logger.info(
        "UQ控制参数 | "
        f"uw_lr_ratio={cfg.uw_lr_ratio:.4f}, uw_s=[{cfg.uw_s_min:.3f},{cfg.uw_s_max:.3f}], "
        f"uw_reg_coef={cfg.uw_reg_coef:.3f}, ratio_target=[{cfg.uw_ratio_target_low:.2f},{cfg.uw_ratio_target_high:.2f}], "
        f"warn={cfg.uw_ratio_warn:.2f}, critical={cfg.uw_ratio_critical:.2f}, "
        f"ema_beta={cfg.uw_ratio_ema_beta:.2f}, patience={cfg.uw_ratio_patience}, "
        f"auto_enable={int(cfg.uw_auto_enable)}, auto_decay={cfg.uw_auto_decay:.2f}, "
        f"auto_floor={cfg.uw_auto_floor:.2f}, auto_cooldown={cfg.uw_auto_cooldown_steps}, "
        f"auto_s_raise={cfg.uw_auto_s_raise:.3f}, "
        f"recover_enable={int(cfg.uw_auto_recover_enable)}, recover_growth={cfg.uw_auto_recover_growth:.2f}, "
        f"recover_ceiling={cfg.uw_auto_recover_ceiling:.2f}, recover_patience={cfg.uw_auto_recover_patience}"
    )
    # ===== UQ CONTROL END =====
    
    # Build independent data loaders; num_workers controls CPU file-reading workers.
    cache_dir = str(viz_dir / cfg.feature_cache_dir) if cfg.use_feature_cache else None
    e2v_cache_dir = str(Path(__file__).resolve().parent / cfg.e2v_offline_cache_dir) if getattr(cfg, "e2v_offline_cache_enable", True) else None
    acoustic_cache_dir = str(Path(__file__).resolve().parent / cfg.acoustic_offline_cache_dir) if getattr(cfg, "acoustic_offline_cache_enable", True) else None
    # ===== PERF OPT #3: bound workers by CPU resources and enable persistent/prefetch loading =====
    req_train_workers = cfg.quick_num_workers if cfg.quick_mode else int(getattr(cfg, "train_num_workers", 8))
    req_val_workers = cfg.quick_num_workers if cfg.quick_mode else int(getattr(cfg, "val_num_workers", 6))
    train_workers = _resolve_num_workers(req_train_workers, world_size if use_ddp else 1)
    val_workers = _resolve_num_workers(req_val_workers, world_size if use_ddp else 1)
    print("Creating datasets")
    
    
    # Build the multimodal training dataset from the CSV rows, audio root, sampling rate, duration, and caches.
    train_dataset = UnifiedMultimodalDataset(
        df_train,
        train_data_path,
        sr=16000,
        raw_duration=5.0,
        cache_dir=cache_dir,
        e2v_cache_dir=e2v_cache_dir,
        e2v_cache_enable=getattr(cfg, "e2v_offline_cache_enable", True),
        acoustic_cache_dir=acoustic_cache_dir,
        acoustic_cache_enable=getattr(cfg, "acoustic_offline_cache_enable", True),
    )
    # DDP uses a DistributedSampler so each process sees a distinct shard.
    train_sampler = DistributedSampler(train_dataset,shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None), # Shuffle only for single-process training.
        sampler=train_sampler,
        num_workers=train_workers,
        **_build_loader_perf_kwargs(cfg, train_workers),
    )
    # DataLoader handles batching, optional shuffling, and worker parallelism.
    if local_rank == 0 or not use_ddp:
        val_dataset = UnifiedMultimodalDataset(
            df_val,
            val_data_path,
            sr=16000,
            raw_duration=5.0,
            cache_dir=cache_dir,
            e2v_cache_dir=e2v_cache_dir,
            e2v_cache_enable=getattr(cfg, "e2v_offline_cache_enable", True),
            acoustic_cache_dir=acoustic_cache_dir,
            acoustic_cache_enable=getattr(cfg, "acoustic_offline_cache_enable", True),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=val_workers,
            **_build_loader_perf_kwargs(cfg, val_workers),
        )
    else:
        val_loader = None
    
    print("Initializing model")
    model = MultimodalDeepfakeDetector(cfg=cfg,device=device).to(device)
    print("Model initialized")
    loss_computer = MultiTaskLossComputer(cfg).to(device)
    fapi_loss_enabled = bool(getattr(cfg, "use_fapi_loss", True))
    logger.info(
        f"[LossSwitch] use_fapi_loss={int(fapi_loss_enabled)} | "
        f"fapi_penalty_every_n_steps={int(getattr(cfg, 'fapi_penalty_every_n_steps', 1))}"
    )
    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    
    # Initialize the loss computer and optimizer.
    # Use a smaller learning rate for uncertainty parameters so they do not dominate early training.
    uw_lr_ratio = float(getattr(cfg, "uw_lr_ratio", 0.1))
    loss_named_params = dict(loss_computer.named_parameters())
    uw_param_names = {"s1", "s2", "s3"} if fapi_loss_enabled else {"s1", "s2"}
    uw_params = [loss_named_params[n] for n in uw_param_names if n in loss_named_params]
    if not fapi_loss_enabled and hasattr(loss_computer, "s3"):
        loss_computer.s3.requires_grad_(False)
    other_loss_params = [p for n, p in loss_named_params.items() if n not in uw_param_names and (fapi_loss_enabled or n != "s3")]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.parameters()) + other_loss_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": uw_params, "lr": cfg.lr * uw_lr_ratio, "weight_decay": 0.0},
        ],
    )
    history = {"train_loss": [], "val_loss": []}
    convergence_rows = []

    # ===== PERF OPT #1: automatic mixed precision training =====
    amp_enabled = bool(getattr(cfg, "use_amp", True)) and torch.cuda.is_available()
    amp_dtype = _resolve_amp_dtype(cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and amp_dtype == torch.float16))

    logger.info(f"开始训练 [{mode_name}] | 使用设备: {device} | 批大小: {cfg.batch_size} | 最大步数/epoch: {cfg.max_steps_per_epoch}")
    logger.info(
        "跳过统计说明 | "
        "forward=取数阶段失败/前向返回None/文本质量无效 | "
        "loss=损失为空、非有限或空文本批次被拦截 | "
        "backward=反向传播失败或梯度同步异常"
    )
    
    epoches=cfg.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epoches),
        eta_min=float(getattr(cfg, "min_lr", 1e-6)),
    )
    
    best_auc = 0.0
    stop_after_epoch = False
    val_metrics = {
        "y_true": [],
        "y_pred": [],
        "ppl": [],
        "loss": 0.0,
        "auc": 0.0,
    }
    # ===== UQ AUTO-FIX START: keep state across epochs for a stable control loop =====
    # These states are not reset every epoch so the controller sees trends rather than single-batch noise.
    uw_ratio_ema = None
    uw_ratio_warn_streak = 0
    uw_ratio_critical_streak = 0
    uw_auto_cooldown_left = 0
    uw_auto_adjust_count = 0
    uw_ratio_target_streak = 0
    uw_auto_recover_count = 0
    # ===== UQ AUTO-FIX END =====
    for epoch in range(epoches): 
        # --- training phase ---
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        if local_rank == 0:
            logger.info(f"========== Epoch [{epoch+1}/{epoches}] ==========")
        epoch_loss_total = 0.0
        epoch_loss_steps = 0
        epoch_skip_stats = {"forward": 0, "loss": 0, "backward": 0}
        consecutive_skip_batches = 0
       
        for i, batch_data in enumerate(train_loader):
            if epoch_loss_steps >= cfg.max_steps_per_epoch:
                logger.info(f"达到最大步数限制，提前结束本轮: {cfg.max_steps_per_epoch}")
                stop_after_epoch = True
                break
            # Per-rank batch validity flag: 1.0 means usable, 0.0 means skip.
            batch_valid = torch.tensor(1.0, device=device)
            outputs = None
            total_loss = None
            loss_dict = None
            forward_done = False
            loss_done = False
            skip_stage = "forward"
            skip_reason = ""
            forward_diag = {}
            # -------------------- training-batch exception handling start --------------------
            try:
                if not isinstance(batch_data, dict):
                    raise ValueError("batch_data 不是 dict")
                # Validate key inputs before entering the model; missing keys usually indicate a broken dataset row.
                required_keys = ["mfcc", "f0", "raw_waveform", "audio_path","labels","F_emo"]
                for k in required_keys:
                    if k not in batch_data:
                        raise KeyError(f"batch_data 缺少键: {k}")
                labels = batch_data["labels"].to(device)

                model_ref = _unwrap_model(model)
                if hasattr(model_ref, "set_forward_context"):
                    model_ref.set_forward_context(
                        batch_id=i,
                        epoch_id=epoch + 1,
                        step_id=epoch_loss_steps,
                        rank_id=global_rank,
                        audio_paths=batch_data.get("audio_path", []),
                    )
                
                # ===== FAPI policy: compute every step by default for backward compatibility =====
                fapi_every_n = max(1, int(getattr(cfg, "fapi_penalty_every_n_steps", 1)))
                compute_fapi_penalty = fapi_loss_enabled and ((epoch_loss_steps % fapi_every_n) == 0)

                # Forward pass with AMP and optional FAPI downsampling.
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(
                        batch_data,
                        is_train=True,
                        labels=labels,
                        compute_fapi_penalty=compute_fapi_penalty,
                    )
                if outputs is None:
                    raise ValueError("模型输出了 None")
                forward_done = True

                text_empty_ratio = float(outputs.get("empty_text_ratio", torch.tensor(0.0, device=device)).item()) if torch.is_tensor(outputs.get("empty_text_ratio", None)) else float(outputs.get("empty_text_ratio", 0.0))
                text_batch_invalid = bool(float(outputs.get("text_batch_invalid", torch.tensor(0.0, device=device)).item()) > 0.5) if torch.is_tensor(outputs.get("text_batch_invalid", None)) else bool(outputs.get("text_batch_invalid", False))
                text_batch_warn = bool(float(outputs.get("text_batch_warn", torch.tensor(0.0, device=device)).item()) > 0.5) if torch.is_tensor(outputs.get("text_batch_warn", None)) else bool(outputs.get("text_batch_warn", False))
                if text_batch_warn and local_rank == 0:
                    logger.warning(
                        f"Batch {i} 文本质量告警 | empty_ratio={text_empty_ratio:.2f} | "
                        f"empty_count={int(float(outputs.get('empty_text_count', torch.tensor(0.0, device=device)).item()) if torch.is_tensor(outputs.get('empty_text_count', None)) else float(outputs.get('empty_text_count', 0.0)))}"
                    )
                if text_batch_invalid:
                    raise ValueError(
                        f"空文本批次占比过高或重复空路径 | empty_ratio={text_empty_ratio:.2f}"
                    )
                
                lambda_penalty_vec = outputs["lambda_penalty"]
                
                # Compute multitask loss.
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    total_loss, loss_dict = loss_computer(
                        outputs=outputs,
                        batch=batch_data,
                        lambda_penalty_vec=lambda_penalty_vec
                    )
                loss_done = True
                loss_ok, loss_reason = _check_loss_sanity(total_loss, loss_dict, cfg)
                if not loss_ok:
                    raise ValueError(loss_reason)
            except Exception as e:
                err_msg = str(e)
                model_ref = _unwrap_model(model)
                if hasattr(model_ref, "get_last_forward_error_info"):
                    forward_diag = model_ref.get_last_forward_error_info() or {}
                if "NCCL" in err_msg or "DistBackendError" in err_msg:
                    logger.exception(f"Rank {global_rank} 在 Batch {i} 遇到分布式致命错误，立即终止本轮: {e}")
                    raise
                if not forward_done:
                    skip_stage = "forward"
                elif not loss_done:
                    skip_stage = "loss"
                else:
                    skip_stage = "loss"
                skip_reason = err_msg
                logger.warning(
                    f"Rank {global_rank} 在 Batch {i} 发生错误({skip_stage}): {e}"
                    f"{_format_forward_diag_for_log(forward_diag)}"
                )
                batch_valid *= 0.0  # Implementation detail.
            # Synchronize batch validity across all ranks.
            if use_ddp:
                # MIN reduction makes any failed rank force the whole batch to be skipped.
                dist.all_reduce(batch_valid, op=dist.ReduceOp.MIN)
            
            if batch_valid.item() == 0.0:
                # If any rank failed, every rank skips the same batch.
                epoch_skip_stats[skip_stage] = epoch_skip_stats.get(skip_stage, 0) + 1
                consecutive_skip_batches += 1
                if local_rank == 0:
                    logger.warning(
                        f"Batch {i} 存在异常卡，全体跳过 | stage={skip_stage}({_skip_stage_label(skip_stage)}) | "
                        f"consecutive_skip={consecutive_skip_batches} | reason={skip_reason}"
                        f"{_format_forward_diag_for_log(forward_diag)}"
                    )
                    if consecutive_skip_batches >= int(cfg.skip_batch_warn_patience):
                        logger.warning(
                            f"连续跳过 {consecutive_skip_batches} 个 batch，建议优先检查采样、空文本和损坏样本。"
                        )
                optimizer.zero_grad(set_to_none=True)
                continue

            # ===== SAFETY FIX START: ensure loss tensors exist before backward =====
            if total_loss is None or loss_dict is None:
                if local_rank == 0:
                    logger.warning(f"Batch {i} 状态异常(total_loss/loss_dict 为空)，全体跳过该 batch。")
                optimizer.zero_grad(set_to_none=True)
                continue
            # ===== SAFETY FIX END =====

            # -------------------- backward pass and gradient synchronization --------------------
            # This block is reached only after every rank completes the forward pass.
            optimizer.zero_grad()
            backward_valid = torch.tensor(1.0, device=device)
            backward_done = False

            try:
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                backward_done = True
            except Exception as e:
                err_msg = str(e)
                if "NCCL" in err_msg or "DistBackendError" in err_msg:
                    logger.exception(f"Rank {global_rank} Batch {i} 反向传播遇到分布式致命错误，立即终止本轮: {e}")
                    raise
                logger.exception(f"Rank {global_rank} Batch {i} 反向传播失败: {e}")
                backward_valid *= 0.0

            if use_ddp:
                # If any rank fails during backward, all ranks skip the batch before the next synchronization point.
                dist.all_reduce(backward_valid, op=dist.ReduceOp.MIN)

            if backward_valid.item() == 0.0:
                epoch_skip_stats["backward"] += 1
                consecutive_skip_batches += 1
                if local_rank == 0:
                    logger.warning(
                        f"Batch {i} 反向传播存在异常卡，全体跳过该 batch | stage=backward({_skip_stage_label('backward')}) | "
                        f"consecutive_skip={consecutive_skip_batches}"
                    )
                    if consecutive_skip_batches >= int(cfg.skip_batch_warn_patience):
                        logger.warning(
                            f"连续跳过 {consecutive_skip_batches} 个 batch，反向阶段异常频繁。"
                        )
                optimizer.zero_grad(set_to_none=True)
                continue

            consecutive_skip_batches = 0

            if dist.is_initialized():
                # Manually average loss-computer gradients so every rank has tensors for each parameter.
                for param in loss_computer.parameters():
                    if param.grad is None:
                        param.grad = torch.zeros_like(param.data)
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= dist.get_world_size()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # ===== UQ AUTO-FIX START: global ratio statistics and automatic correction =====
            # UW_ratio = |L_UW_reg| / (L_CE + L_KL + L_Conflict + L_FAPI)
            # Use the global step mean plus EMA to reduce single-rank and single-step noise.
            uw_ratio_now = float(loss_dict["UW_ratio"].item())
            if use_ddp:
                # In DDP, aggregate before deciding so every rank applies the same control action.
                uw_ratio_t = torch.tensor(uw_ratio_now, device=device, dtype=torch.float32)
                dist.all_reduce(uw_ratio_t, op=dist.ReduceOp.SUM)
                uw_ratio_now = float((uw_ratio_t / float(world_size)).item())

            if uw_ratio_ema is None:
                uw_ratio_ema = uw_ratio_now
            else:
                uw_ratio_ema = float(cfg.uw_ratio_ema_beta) * uw_ratio_ema + (1.0 - float(cfg.uw_ratio_ema_beta)) * uw_ratio_now

            if uw_ratio_ema >= float(cfg.uw_ratio_critical):
                uw_ratio_critical_streak += 1
                uw_ratio_warn_streak += 1
            elif uw_ratio_ema >= float(cfg.uw_ratio_warn):
                uw_ratio_warn_streak += 1
                uw_ratio_critical_streak = 0
            else:
                uw_ratio_warn_streak = 0
                uw_ratio_critical_streak = 0

            # ===== UQ AUTO-RECOVER START: count stable steps inside the target interval =====
            if float(cfg.uw_ratio_target_low) <= uw_ratio_ema <= float(cfg.uw_ratio_target_high):
                uw_ratio_target_streak += 1
            else:
                uw_ratio_target_streak = 0
            # ===== UQ AUTO-RECOVER END =====

            if uw_auto_cooldown_left > 0:
                # During cooldown, avoid repeated corrections that would cause oscillation.
                uw_auto_cooldown_left -= 1

            if (
                bool(cfg.uw_auto_enable)
                and uw_auto_cooldown_left == 0
                and uw_ratio_critical_streak >= int(cfg.uw_ratio_patience)
            ):
                # loss_computer is not wrapped by DDP, so the local instance is enough.
                loss_computer_for_fix = loss_computer
                old_coef = float(getattr(loss_computer_for_fix, "uw_reg_coef", 1.0))
                new_coef = max(float(cfg.uw_auto_floor), old_coef * float(cfg.uw_auto_decay))
                with torch.no_grad():
                    # Action 1: reduce the UQ regularization coefficient to suppress dominance.
                    loss_computer_for_fix.uw_reg_coef = float(new_coef)
                    for s_name in ("s1", "s2", "s3"):
                        s_param = getattr(loss_computer_for_fix, s_name)
                        # Action 2: gently raise s to reduce the exp(-s) weighting strength.
                        s_param.add_(float(cfg.uw_auto_s_raise))
                        s_param.data.clamp_(min=float(loss_computer_for_fix.s_min), max=float(loss_computer_for_fix.s_max))

                uw_auto_adjust_count += 1
                uw_auto_cooldown_left = int(cfg.uw_auto_cooldown_steps)
                uw_ratio_warn_streak = 0
                uw_ratio_critical_streak = 0

                if local_rank == 0:
                    logger.warning(
                        f"[UQ-AUTO-FIX] trigger={uw_auto_adjust_count} | "
                        f"uw_reg_coef: {old_coef:.4f} -> {new_coef:.4f} | "
                        f"ema={uw_ratio_ema:.4f} | cooldown={uw_auto_cooldown_left}"
                    )

            # ===== UQ AUTO-RECOVER START: slowly restore uw_reg_coef after stabilization =====
            if (
                bool(cfg.uw_auto_recover_enable)
                and uw_auto_cooldown_left == 0
                and uw_ratio_target_streak >= int(cfg.uw_auto_recover_patience)
            ):
                # loss_computer is not wrapped by DDP, so the local instance is enough.
                loss_computer_for_fix = loss_computer
                old_coef = float(getattr(loss_computer_for_fix, "uw_reg_coef", 1.0))
                new_coef = min(float(cfg.uw_auto_recover_ceiling), old_coef * float(cfg.uw_auto_recover_growth))
                if new_coef > old_coef + 1e-12:
                    with torch.no_grad():
                        # Restore only the coefficient; do not pull s backward, keeping recovery smoother.
                        loss_computer_for_fix.uw_reg_coef = float(new_coef)
                    uw_auto_recover_count += 1
                    uw_auto_cooldown_left = int(cfg.uw_auto_cooldown_steps)
                    uw_ratio_target_streak = 0
                    if local_rank == 0:
                        logger.info(
                            f"[UQ-AUTO-RECOVER] trigger={uw_auto_recover_count} | "
                            f"uw_reg_coef: {old_coef:.4f} -> {new_coef:.4f} | "
                            f"ema={uw_ratio_ema:.4f} | cooldown={uw_auto_cooldown_left}"
                        )
            # ===== UQ AUTO-RECOVER END =====
            # ===== UQ AUTO-FIX END =====

            epoch_loss_total += float(total_loss.item())
            epoch_loss_steps += 1
            
            # -------------------------------------------------------
            # Emit monitoring logs.
            if i % max(1, cfg.print_every_n_steps) == 0 and ((not use_ddp) or local_rank == 0):
                # loss_computer is not wrapped by DDP, so the local instance is enough.
                loss_computer_for_report = loss_computer
                lambda_report = loss_computer_for_report.get_lambda_report()

                logger.info(
                    f"Step [{i}/{len(train_loader)}] | Total={total_loss.item():.4f} | "
                    f"L_CE={loss_dict['L_CE'].item():.4f} | L_KL={loss_dict['L_KL'].item():.4f} | "
                    f"L_Conflict={loss_dict['L_conflict'].item():.6e} | "
                    f"fapi_enabled={_flag_to_int(loss_dict.get('fapi_enabled', fapi_loss_enabled), fapi_loss_enabled)} | "
                    f"fapi_compute={int(compute_fapi_penalty)} | "
                    f"L_FAPI={loss_dict['L_FAPI'].item():.6e} | L_FAPI_raw={loss_dict['L_FAPI_raw'].item():.6e} | "
                    f"fapi_scale={loss_dict['fapi_scale'].item():.1f} | "
                    f"ftext_req_grad={loss_dict['ftext_requires_grad'].item():.0f} | "
                    f"fapi_grads_none={loss_dict['fapi_grads_is_none'].item():.0f} | "
                    f"fapi_grad_rms={loss_dict['fapi_grad_rms_mean'].item():.6e} | "
                    f"lambda_raw_mean={loss_dict['lambda_penalty_mean'].item():.4f} | "
                    f"lambda_clamped_mean={loss_dict['lambda_penalty_clamped_mean'].item():.4f} | "
                    f"g_mean={loss_dict['gate_mean'].item():.4f} | "
                    f"fake_score_mean={loss_dict['fake_score_mean'].item():.4f} | "
                    f"eff_margin={loss_dict['effective_margin'].item():.4f} | "
                    f"conf_active={loss_dict['conflict_active_ratio'].item():.4f} | "
                    f"num_fakes={loss_dict['num_fakes'].item():.0f} | "
                    f"L_task_sum={loss_dict['L_task_sum'].item():.4f} | "
                    f"L_UW_reg={loss_dict['L_UW_reg'].item():.4f} | "
                    f"L_UW_reg_raw={loss_dict['L_UW_reg_raw'].item():.4f} | "
                    f"uw_reg_coef={lambda_report['uw_reg_coef']:.3f} | "
                    # uw_ratio is instantaneous, uw_ratio_ema is trend, and target_streak tracks recovery progress.
                    f"uw_ratio={uw_ratio_now:.4f} | uw_ratio_ema={uw_ratio_ema:.4f} | cooldown={uw_auto_cooldown_left} | "
                    f"target_streak={uw_ratio_target_streak} | "
                    f"s1={lambda_report['s1']:.4f}, s2={lambda_report['s2']:.4f}, s3={lambda_report['s3']:.4f} | "
                    f"lambda1={lambda_report['lambda1']:.6f}, lambda2={lambda_report['lambda2']:.6f}, lambda3={lambda_report['lambda3']:.6f}, "
                    f"lambda3_active={lambda_report.get('lambda3_active', lambda_report['lambda3']):.6f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                # ===== UQ CONTROL START: warn on repeated threshold violations =====
                if uw_ratio_critical_streak >= int(cfg.uw_ratio_patience):
                    logger.warning(
                        f"[UQ-CRITICAL] uw_ratio_ema={uw_ratio_ema:.4f} >= {cfg.uw_ratio_critical:.2f} "
                        f"连续{uw_ratio_critical_streak}次。建议立即降低 uw_lr_ratio 或上调 uw_s_min。"
                    )
                elif uw_ratio_warn_streak >= int(cfg.uw_ratio_patience):
                    logger.warning(
                        f"[UQ-WARN] uw_ratio_ema={uw_ratio_ema:.4f} >= {cfg.uw_ratio_warn:.2f} "
                        f"连续{uw_ratio_warn_streak}次。建议观察并准备收紧 UQ 参数。"
                    )
                # ===== UQ CONTROL END =====

        if epoch_loss_steps == 0:
            if local_rank == 0:
                logger.error(
                    f"Epoch [{epoch+1}/{epoches}] 没有任何有效 batch，已停止训练 | "
                    f"skip_stats={epoch_skip_stats}"
                )
            stop_after_epoch = True
            break

        history["train_loss"].append(epoch_loss_total / max(1, epoch_loss_steps))
        if local_rank == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epoches}] 跳过统计 | {_format_skip_stats(epoch_skip_stats)} | "
                f"valid_steps={epoch_loss_steps}"
            )
        
        # --- validation phase at the end of each epoch ---
        if use_ddp:
            dist.barrier()
        if local_rank == 0:
            # When only rank 0 validates, use the unwrapped model to avoid extra DDP synchronization.
            eval_model = model.module if use_ddp and hasattr(model, "module") else model
            val_metrics = evaluate(eval_model, val_loader, loss_computer, device, cfg, temperature=1.0)
            history["val_loss"].append(val_metrics["loss"])
            logger.info(f"Epoch {epoch+1} 验证指标汇总:")
            _log_eval_metrics_panel(logger, f"VAL-E{epoch+1}", val_metrics)
            if val_metrics.get("skipped_batches", 0) > 0:
                logger.warning(
                    f"Epoch {epoch+1} 验证跳过 batch 数: {val_metrics['skipped_batches']} | "
                    f"有效 batch 数: {val_metrics.get('valid_batches', 0)} | "
                    f"说明: 多数情况是验证 batch 的模型输出或 loss 不可用"
                )

            prev_row = convergence_rows[-1] if len(convergence_rows) > 0 else None
            current_row = _build_convergence_row(
                epoch_idx=epoch,
                train_loss=history["train_loss"][-1],
                val_metrics=val_metrics,
                prev_row=prev_row,
                cfg=cfg,
            )
            convergence_rows.append(current_row)
            logger.info(
                f"[Convergence] epoch={current_row['epoch']} | trend={current_row['trend']} | "
                f"dAUC={current_row['delta_auc']:+.4f} | dF1={current_row['delta_f1']:+.4f} | "
                f"dEER={current_row['delta_eer']:+.4f}"
            )
            
            # --- save the best model ---
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                state_dict = model.module.state_dict() if use_ddp else model.state_dict()
                torch.save(state_dict, best_model_path)
                logger.info(f"发现更好的模型，已保存权重: {best_model_path}")

        scheduler.step()

        # ==================== convergence logging ====================
        if use_ddp:
            dist.barrier() # Implementation detail.
        # ======================================================
        if stop_after_epoch:
            logger.info("本轮已完成验证，随后停止继续训练。")
            break

    if local_rank == 0:
        if len(history["val_loss"]) == 0:
            history["val_loss"] = [0.0 for _ in history["train_loss"]]

        if convergence_rows:
            summary_path = viz_dir / cfg.convergence_summary_file
            pd.DataFrame(convergence_rows).to_csv(summary_path, index=False, encoding="utf-8")
            logger.info(f"收敛摘要已保存: {summary_path}")

        FAPIVizSuite.plot_loss_curves(history, save_path=str(viz_dir / "loss_curve.png"), show=False)
    if use_ddp:
        dist.destroy_process_group()
    if local_rank == 0 and val_metrics.get("y_true") is not None and val_metrics.get("y_pred") is not None:
        FAPIVizSuite.plot_confusion_matrix(
            y_true=val_metrics["y_true"],
            y_pred=val_metrics["y_pred"],
            save_path=str(viz_dir / "val_confusion_matrix.png"),
            show=False,
        )
    if local_rank == 0 and val_metrics.get("ppl") is not None and val_metrics.get("y_true") is not None:
        y_true_arr = np.asarray(val_metrics["y_true"], dtype=np.int32)
        ppl_arr = np.asarray(val_metrics["ppl"], dtype=np.float32)
        real_ppl = ppl_arr[y_true_arr == 0].tolist() if (y_true_arr == 0).any() else []
        fake_ppl = ppl_arr[y_true_arr == 1].tolist() if (y_true_arr == 1).any() else []
        if len(real_ppl) > 0 or len(fake_ppl) > 0:
            FAPIVizSuite.plot_ppl_threshold(
                scores_real=real_ppl,
                scores_fake=fake_ppl,
                threshold=cfg.p_threshold,
                save_path=str(viz_dir / "val_ppl_threshold.png"),
                show=False,
            )
            
# Run validation at epoch boundaries.
def evaluate(model, dataloader, loss_computer, device, cfg, threshold=0.5, temperature=None):
    """
    通用的评估函数，可用于 Validation 或 Test
    """
    model.eval() # Implementation detail.
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []
    all_ppl = []
    valid_batches = 0
    skipped_batches = 0
    amp_enabled = bool(getattr(cfg, "use_amp", True)) and torch.cuda.is_available()
    amp_dtype = _resolve_amp_dtype(cfg)
    if temperature is None:
        temperature = _resolve_inference_temperature(cfg)
    else:
        try:
            temperature = float(temperature)
        except Exception:
            temperature = _resolve_inference_temperature(cfg)
    if (not np.isfinite(temperature)) or temperature <= 0.0:
        temperature = 1.0
    
    with torch.no_grad(): # Implementation detail.
        for batch_data in dataloader:
            # print("Debug Labels:", batch_data["labels"])  # Debug label mapping.
            try:
                # 1. Forward pass.
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(batch_data, is_train=False)
                if outputs is None:
                    continue

                text_empty_ratio = float(outputs.get("empty_text_ratio", torch.tensor(0.0, device=device)).item()) if torch.is_tensor(outputs.get("empty_text_ratio", None)) else float(outputs.get("empty_text_ratio", 0.0))
                text_batch_invalid = bool(float(outputs.get("text_batch_invalid", torch.tensor(0.0, device=device)).item()) > 0.5) if torch.is_tensor(outputs.get("text_batch_invalid", None)) else bool(outputs.get("text_batch_invalid", False))
                text_batch_warn = bool(float(outputs.get("text_batch_warn", torch.tensor(0.0, device=device)).item()) > 0.5) if torch.is_tensor(outputs.get("text_batch_warn", None)) else bool(outputs.get("text_batch_warn", False))
                if text_batch_warn:
                    logging.getLogger("fapi_train").warning(
                        f"评估批次文本质量告警 | empty_ratio={text_empty_ratio:.2f} | invalid={text_batch_invalid}"
                    )
                
                # 2. Normalize labels.
                #batch_data["labels"] = batch_data["label"] 
                
                # 3. Compute loss.
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    loss, loss_dict = loss_computer(
                        outputs=outputs, 
                        batch=batch_data, 
                        lambda_penalty_vec=outputs["lambda_penalty"]
                    )
                loss_ok, loss_reason = _check_loss_sanity(loss, loss_dict, cfg)
                if not loss_ok:
                    raise ValueError(loss_reason)
                total_loss += loss.item()
                valid_batches += 1
                
                # 4. Collect predicted probabilities and labels for metrics.
                # fake_logit is converted to probabilities with sigmoid.
                fake_logit = outputs["fake_logit"]
                # PERF HOTFIX: NumPy cannot directly consume low-precision CPU tensors from AMP; cast to float32 first.
                logits = fake_logit.detach().float().cpu().numpy().reshape(-1)
                probs = torch.sigmoid(fake_logit / float(temperature)).detach().float().cpu().numpy().reshape(-1)
                labels = batch_data["labels"].detach().cpu().numpy().reshape(-1)
                if "ppl" in outputs and outputs["ppl"] is not None:
                    all_ppl.extend(torch.as_tensor(outputs["ppl"]).detach().float().cpu().numpy().reshape(-1).tolist())
                
                all_preds.extend(probs)
                all_labels.extend(labels)
                all_logits.extend(logits)
                
            except Exception as e:
                logging.getLogger("fapi_train").warning(f"评估批次出错，跳过: {e}")
                skipped_batches += 1
                continue
                
    # --- compute global evaluation metrics ---
    if valid_batches == 0 or len(all_preds) == 0:
        if skipped_batches > 0:
            logging.getLogger("fapi_train").warning(
                f"评估阶段没有有效 batch | skipped_batches={skipped_batches}"
            )
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
            "mcc": 0.0,
            "auc": 0.0,
            "pr_auc": 0.0,
            "eer": 0.0,
            "eer_threshold": 0.5,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "y_true": [],
            "y_pred": [],
            "y_prob": [],
            "y_logit": [],
            "temperature_used": float(temperature),
            "ppl": [],
            "valid_batches": 0,
            "skipped_batches": skipped_batches,
        }

    avg_loss = total_loss / valid_batches
    if skipped_batches > 0:
        logging.getLogger("fapi_train").warning(
            f"评估阶段跳过 batch 数: {skipped_batches} | 有效 batch 数: {valid_batches}"
        )
    
    all_preds = np.array(all_preds, dtype=np.float32).reshape(-1)
    all_labels = np.array(all_labels, dtype=np.int32).reshape(-1)
    
    metrics = _compute_binary_metrics_from_probs(all_labels, all_preds, threshold=threshold)
    metrics.update(
        {
            "loss": float(avg_loss),
            "y_logit": np.asarray(all_logits, dtype=np.float32).reshape(-1).tolist(),
            "temperature_used": float(temperature),
            "ppl": all_ppl,
            "valid_batches": int(valid_batches),
            "skipped_batches": int(skipped_batches),
        }
    )
    return metrics

def test(df_test, test_data_path, model_path, cfg=None):
    cfg = cfg or TrainConfig()
    cfg.asr_offline_cache_file = str(getattr(cfg, "asr_offline_cache_file_test", cfg.asr_offline_cache_file))
    cfg.fapi_offline_cache_file = str(getattr(cfg, "fapi_offline_cache_file_test", cfg.fapi_offline_cache_file))
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device("cuda:0" if gpu_count > 0 else "cpu")
    viz_dir, run_id, train_log_file, _ = _resolve_artifact_paths(cfg)
    viz_dir.mkdir(parents=True, exist_ok=True)
    cfg.train_log_file = train_log_file
    logger = _build_logger(viz_dir, cfg.train_log_file)
    mode_name = "Quick" if cfg.quick_mode else "Formal"

    if cfg.quick_mode:
        df_test = _apply_quick_slice(df_test, cfg.quick_test_rows)
        logger.info(f"QuickMode ON | test_rows={len(df_test)} | workers={cfg.quick_num_workers} | {_format_quick_label_stats(df_test)}")
    else:
        logger.info(f"FormalMode ON | test_rows={len(df_test)}")
    
    e2v_cache_dir = str(Path(__file__).resolve().parent / cfg.e2v_offline_cache_dir) if getattr(cfg, "e2v_offline_cache_enable", True) else None
    test_dataset = UnifiedMultimodalDataset(df_test, test_data_path, e2v_cache_dir=e2v_cache_dir, e2v_cache_enable=getattr(cfg, "e2v_offline_cache_enable", True))
    # ===== PERF OPT #3: enable persistent workers and prefetch for the test DataLoader =====
    req_test_workers = cfg.quick_num_workers if cfg.quick_mode else int(getattr(cfg, "test_num_workers", 4))
    test_workers = _resolve_num_workers(req_test_workers, 1)
    test_loader = DataLoader(
        test_dataset,
        batch_size=5,
        shuffle=False,
        num_workers=test_workers,
        **_build_loader_perf_kwargs(cfg, test_workers),
    )

    model = MultimodalDeepfakeDetector(cfg=cfg,device=device).to(device)
    
    # Resolve checkpoint path.
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device),strict=False) # Implementation detail.
        print("成功加载最佳模型权重进行测试")
    else:
        print("没有找到 best_model.pth，请检查文件路径")
        return
    
    model.eval()
    # Initialize the loss computer.
    loss_computer = MultiTaskLossComputer(cfg).to(device)
    
    infer_temperature = _resolve_inference_temperature(cfg)
    test_results = evaluate(model, test_loader, loss_computer, device, cfg=cfg, temperature=infer_temperature)
    test_results["temperature_used"] = float(test_results.get("temperature_used", infer_temperature))

    if bool(getattr(cfg, "temperature_grid_search_enable", True)):
        candidates = _resolve_temperature_candidates(cfg)
        search_report = _search_best_temperature_by_eer(
            test_results.get("y_true", []),
            test_results.get("y_logit", []),
            candidates,
        )
        if search_report is not None:
            search_desc = ", ".join(
                [f"T={item['temperature']:.2f}:EER={item['eer']:.4f}" for item in search_report["all_results"]]
            )
            logger.info(f"Temperature 网格搜索 | {search_desc}")
            logger.info(
                f"Temperature 最优 | T={search_report['best_temperature']:.2f} | "
                f"EER={search_report['best_eer']:.4f} | "
                f"EER-TH={search_report['best_eer_threshold']:.4f}"
            )

            if bool(getattr(cfg, "temperature_grid_search_apply_best", True)):
                best_temp = float(search_report["best_temperature"])
                best_probs = _sigmoid_with_temperature_np(
                    np.asarray(test_results.get("y_logit", []), dtype=np.float32),
                    best_temp,
                )
                calibrated_metrics = _compute_binary_metrics_from_probs(
                    np.asarray(test_results.get("y_true", []), dtype=np.int32),
                    best_probs,
                    threshold=0.5,
                )
                test_results.update(calibrated_metrics)
                test_results["temperature_used"] = best_temp
                logger.info(f"已应用最优温度到最终测试指标 | T={best_temp:.2f}")

    logger.info("-" * 30)
    logger.info("测试完成")
    _log_eval_metrics_panel(logger, "TEST", test_results)
    if test_results.get("y_true") is not None and test_results.get("y_pred") is not None:
        FAPIVizSuite.plot_confusion_matrix(
            y_true=test_results["y_true"],
            y_pred=test_results["y_pred"],
            save_path=str(viz_dir / "test_confusion_matrix.png"),
            show=False,
        )
    if test_results.get("ppl") is not None and test_results.get("y_true") is not None:
        y_true_arr = np.asarray(test_results["y_true"], dtype=np.int32)
        ppl_arr = np.asarray(test_results["ppl"], dtype=np.float32)
        real_ppl = ppl_arr[y_true_arr == 0].tolist() if (y_true_arr == 0).any() else []
        fake_ppl = ppl_arr[y_true_arr == 1].tolist() if (y_true_arr == 1).any() else []
        if len(real_ppl) > 0 or len(fake_ppl) > 0:
            FAPIVizSuite.plot_ppl_threshold(
                scores_real=real_ppl,
                scores_fake=fake_ppl,
                threshold=test_results.get("eer_threshold", cfg.p_threshold),
                save_path=str(viz_dir / "test_ppl_threshold.png"),
                show=False,
            )
    logger.info("-" * 30)


def predict(path_list, model_path=None, return_feature_pack=False, cfg=None):
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device("cuda:0" if gpu_count > 0 else "cpu")
    use_data_parallel = gpu_count > 1
    cfg = cfg or TrainConfig()
    infer_temperature = _resolve_inference_temperature(cfg)
    viz_dir, run_id, train_log_file, default_best_model_path = _resolve_artifact_paths(cfg)
    viz_dir.mkdir(parents=True, exist_ok=True)
    cfg.train_log_file = train_log_file
    logger = _build_logger(viz_dir, cfg.train_log_file)
    mode_name = "Quick" if cfg.quick_mode else "Formal"

    if cfg.quick_mode and len(path_list) > cfg.quick_predict_rows:
        # Quick mode limits prediction to the first N files to avoid long waits.
        path_list = list(path_list)[: cfg.quick_predict_rows]
        logger.info(f"QuickMode ON | predict_rows={len(path_list)} | workers={cfg.quick_num_workers}")
    elif not cfg.quick_mode:
        logger.info(f"FormalMode ON | predict_rows={len(path_list)}")
    logger.info(f"推理温度缩放启用 | T={infer_temperature:.4f}")

    # -------------------- input validation start --------------------
    if path_list is None:
        raise ValueError("path_list 不能为空")
    if not isinstance(path_list, (list, tuple)):
        raise ValueError("path_list 必须是 list 或 tuple")
    if len(path_list) == 0:
        logger.warning("path_list 为空，已跳过推理")
        empty_ret = {"predictions": [], "feature_pack": None}
        if return_feature_pack:
            empty_ret["feature_pack"] = {
                "F_AE": None,
                "F_text": None,
                "text_mask": None,
                "audio_mask": None,
                "F_emo": None,
                "ppl": None,
                "topk_mean": None,
                "gamma_t": None,
                "lambda_penalty": None,
                "fake_prob": None,
                "emotion_probs": None,
                "audio_path": [],
                "labels": np.asarray([], dtype=np.int32),
            }
        return empty_ret
    # -------------------- input validation end --------------------

    # Build a minimal CSV-like DataFrame for dataset compatibility.
    df_temp = pd.DataFrame({
        "file": path_list,
        "label": [0] * len(path_list),
        "angry": [0.0] * len(path_list),
        "disgusted": [0.0] * len(path_list),
        "fearful": [0.0] * len(path_list),
        "happy": [0.0] * len(path_list),
        "neutral": [0.0] * len(path_list),
        "other": [0.0] * len(path_list),
        "sad": [0.0] * len(path_list),
        "surprised": [0.0] * len(path_list),
        "unknown": [0.0] * len(path_list),
    })
    e2v_cache_dir = str(Path(__file__).resolve().parent / cfg.e2v_offline_cache_dir) if getattr(cfg, "e2v_offline_cache_enable", True) else None
    dataset = UnifiedMultimodalDataset(df=df_temp, root_dir="", sr=16000, raw_duration=5.0, e2v_cache_dir=e2v_cache_dir, e2v_cache_enable=getattr(cfg, "e2v_offline_cache_enable", True))
    # ===== PERF OPT #3: enable persistent workers and prefetch for prediction =====
    req_pred_workers = cfg.quick_num_workers if cfg.quick_mode else int(getattr(cfg, "pred_num_workers", 4))
    pred_workers = _resolve_num_workers(req_pred_workers, 1)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=pred_workers,
        **_build_loader_perf_kwargs(cfg, pred_workers),
    )

    model = MultimodalDeepfakeDetector(cfg=cfg, device=device).to(device)
    if model_path is None:
        model_path = str(default_best_model_path)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"成功加载模型权重: {model_path}")
    else:
        raise FileNotFoundError(f"未找到模型权重文件: {model_path}")

    model.eval()

    predictions = []
    feature_pack = {
        "F_AE": [],
        "F_text": [],
        "text_mask": [],
        "audio_mask": [],
        "F_emo": [],
        "ppl": [],
        "topk_mean": [],
        "gamma_t": [],
        "lambda_penalty": [],
        "fake_prob": [],
        "emotion_probs": [],
        "audio_path": [],
        "labels": [],
    }

    logger.info("开始推理")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            try:
                outputs = model(batch_data, is_train=False)
                if outputs is None:
                    continue
                
                # Convert logits to detached CPU probabilities for NumPy processing.
                probs = torch.sigmoid(outputs["fake_logit"] / float(infer_temperature)).detach().cpu().numpy().reshape(-1)
                # Normalize probabilities.
                emotion_probs_tensor = torch.softmax(outputs["emo_logit"], dim=-1).detach().cpu()
                emotion_scores = emotion_probs_tensor.numpy()
                # Align predictions with file names.
                paths = batch_data["audio_path"]
                labels = batch_data["labels"].detach().cpu().numpy().reshape(-1)

                # Align each audio path with its prediction result.
                for idx, audio_path in enumerate(paths):
                    prob = float(probs[idx]) if idx < len(probs) else None
                    if idx < len(emotion_scores):
                        emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
                        emotions = {label: float(score) for label, score in zip(emotion_labels, emotion_scores[idx])}
                    else: # Implementation detail.
                        emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
                        emotions = {label: 0.0 for label in emotion_labels}
                    logger.info(f"[PRED] {audio_path} -> prob={prob:.4f}, emotions={emotions}")
                    predictions.append({
                        "file": audio_path,
                        "probability": prob,
                        "prediction": int(prob > 0.5) if prob is not None else None,
                        "emotion_scores": emotions,
                    })

                if return_feature_pack:
                    feature_pack["F_AE"].append(outputs["F_AE"].detach().cpu())
                    feature_pack["F_text"].append(outputs["F_text"].detach().cpu())
                    feature_pack["text_mask"].append(outputs["text_mask"].detach().cpu())
                    feature_pack["audio_mask"].append(outputs["audio_mask"].detach().cpu())
                    feature_pack["F_emo"].append(outputs["F_emo"].detach().cpu())
                    feature_pack["ppl"].append(torch.as_tensor(outputs["ppl"]).detach().cpu())
                    feature_pack["topk_mean"].append(torch.as_tensor(outputs["topk_mean"]).detach().cpu())
                    feature_pack["gamma_t"].append(torch.as_tensor(outputs["gamma_t"]).detach().cpu())
                    feature_pack["lambda_penalty"].append(torch.as_tensor(outputs["lambda_penalty"]).detach().cpu())
                    feature_pack["fake_prob"].append(torch.as_tensor(probs).detach().cpu())
                    feature_pack["emotion_probs"].append(emotion_probs_tensor)
                    feature_pack["audio_path"].extend(list(paths))
                    feature_pack["labels"].extend(labels.tolist())

            except Exception as e:
                logger.warning(f"推理 batch {batch_idx} 失败，已跳过: {e}")
                continue

    if return_feature_pack:
        merged_pack = {
            "F_AE": torch.cat(feature_pack["F_AE"], dim=0) if feature_pack["F_AE"] else None,
            "F_text": torch.cat(feature_pack["F_text"], dim=0) if feature_pack["F_text"] else None,
            "text_mask": torch.cat(feature_pack["text_mask"], dim=0) if feature_pack["text_mask"] else None,
            "audio_mask": torch.cat(feature_pack["audio_mask"], dim=0) if feature_pack["audio_mask"] else None,
            "F_emo": torch.cat(feature_pack["F_emo"], dim=0) if feature_pack["F_emo"] else None,
            "ppl": torch.cat(feature_pack["ppl"], dim=0) if feature_pack["ppl"] else None,
            "topk_mean": torch.cat(feature_pack["topk_mean"], dim=0) if feature_pack["topk_mean"] else None,
            "gamma_t": torch.cat(feature_pack["gamma_t"], dim=0) if feature_pack["gamma_t"] else None,
            "lambda_penalty": torch.cat(feature_pack["lambda_penalty"], dim=0) if feature_pack["lambda_penalty"] else None,
            "fake_prob": torch.cat(feature_pack["fake_prob"], dim=0) if feature_pack["fake_prob"] else None,
            "emotion_probs": torch.cat(feature_pack["emotion_probs"], dim=0) if feature_pack["emotion_probs"] else None,
            "audio_path": feature_pack["audio_path"],
            "labels": np.asarray(feature_pack["labels"], dtype=np.int32),
        }
        return {"predictions": predictions, "feature_pack": merged_pack}

    return predictions


# -------------------- interpretability helper block start --------------------
def run_and_visualize_inverse_attention(path_list):
    runtime = InverseAttentionRuntime(hidden_dim=512, num_heads=8, dropout=0.1)
    ret = predict(path_list, return_feature_pack=True)
    pack = ret.get("feature_pack")
    if pack is None:
        raise RuntimeError("未能生成逆向注意力输入特征，请检查上游日志。")

    device = "cuda"
    F_AE = pack["F_AE"].to(device)
    F_text = pack["F_text"].to(device)
    text_mask = pack["text_mask"].to(dtype=torch.bool, device=device)
    audio_mask = pack["audio_mask"].to(dtype=torch.bool, device=device)

    outputs = runtime.run(F_AE, F_text, text_mask, audio_mask=audio_mask)
    real_inputs = runtime.make_real_inputs(pack)

    print(f"输入文本语义矩阵 F_text 维度: {F_text.shape}")
    print(f"输入声学联合矩阵 F_A-E 维度: {F_AE.shape}")
    print(f"输出跨模态冲突特征 F_A-T 维度: {outputs['F_AT'].shape}")
    print(f"软对齐矩阵维度: {outputs['align_weights'].shape}")
    print(f"标准交叉注意力矩阵维度: {outputs['cross_weights'].shape}")
    print(f"逆向注意力矩阵维度: {outputs['att_inversed_weights'].shape}")
    print(f"当前自适应 beta(sigmoid): {outputs['beta'].item():.4f}")

    # Unified interpretability score: RCI(0.6) + TPM(0.4).
    score_report = InterpretabilityScorer.evaluate(
        outputs=outputs,
        attention_mask=text_mask,
        rci_mode="global",
        tpm_mode="basic",
        topk_ratio=0.1,
        rci_weight=0.6,
        tpm_weight=0.4,
    )
    print("[INFO] 解释性评分汇总")
    print(f"- RCI mean: {score_report['components']['rci']['score_mean']:.4f}")
    print(f"- TPM mean: {score_report['components']['tpm']['score_mean']:.4f}")
    print(f"- Total mean: {score_report['total']['score_mean']:.4f}")
    print(f"- Total level: {score_report['total']['level']}")

    # Three visualization views: all-head aggregation, most-conflicting heads, and head disagreement.
    viz_ret = AttentionVisualization.plot_three_tier(
        outputs=outputs,
        sample_idx=0,
        attention_mask=text_mask[0],
        topk_heads=3,
        show=True,
    )
    print(f"[INFO] 三层可视化文件: {viz_ret['save_path']}")
    print(f"[INFO] 最冲突头索引(Top-k): {viz_ret['top_head_indices']}")

    # Persist interpretability scores and visualization metadata as JSON and TXT.
    log_ret = InterpretabilityLogger.save(
        score_report=score_report,
        viz_report=viz_ret,
        outputs=outputs,
    )
    print(f"[INFO] 解释性报告JSON: {log_ret['json_path']}")
    print(f"[INFO] 解释性报告TXT: {log_ret['txt_path']}")

    if pack.get("emotion_probs") is not None and pack["emotion_probs"].size(0) > 0:
        radar_dir = Path(current_dir) / "outputs_fapi_viz"
        radar_dir.mkdir(parents=True, exist_ok=True)
        FAPIVizSuite.plot_emotion_radar(
            pack["emotion_probs"][0],
            labels=[
                "Angry", "Disgusted", "Fearful", "Happy", "Neutral",
                "Other", "Sad", "Surprised", "Unknown",
            ],
            title="Emotion Radar (Sample 0)",
            save_path=str(radar_dir / "emotion_radar_sample0.png"),
            show=False,
        )

    return outputs, real_inputs
# -------------------- main entrypoint helpers --------------------

def _env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def _read_csv_required(path: str, role: str) -> pd.DataFrame:
    if not path:
        raise ValueError(f"{role} CSV path is required")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{role} CSV does not exist: {path}")
    return pd.read_csv(path)


def _read_audio_paths(audio_args, audio_list_file):
    paths = []
    if audio_args:
        paths.extend([str(p) for p in audio_args if str(p).strip()])
    if audio_list_file:
        if not os.path.exists(audio_list_file):
            raise FileNotFoundError(f"Audio list file does not exist: {audio_list_file}")
        with open(audio_list_file, "r", encoding="utf-8") as f:
            paths.extend([line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")])
    return paths


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and run inference for emotion-driven audio deepfake detection."
    )
    parser.add_argument("--mode", choices=["train", "test", "predict", "predict_iam"], default="train")
    parser.add_argument("--train-csv", default=_env_default("FAPI_TRAIN_CSV", "data/train/train.csv"))
    parser.add_argument("--train-root", default=_env_default("FAPI_TRAIN_ROOT", "data/train"))
    parser.add_argument("--val-csv", default=_env_default("FAPI_VAL_CSV", "data/val/val.csv"))
    parser.add_argument("--val-root", default=_env_default("FAPI_VAL_ROOT", "data/val"))
    parser.add_argument("--test-csv", default=_env_default("FAPI_TEST_CSV", "data/test/test.csv"))
    parser.add_argument("--test-root", default=_env_default("FAPI_TEST_ROOT", "data/test"))
    parser.add_argument("--model-path", default=_env_default("FAPI_MODEL_PATH", ""))
    parser.add_argument("--audio", action="append", default=[], help="Audio file for prediction; repeat for multiple files.")
    parser.add_argument("--audio-list", default="", help="UTF-8 text file containing one audio path per line.")
    parser.add_argument("--output-dir", default="", help="Override TrainConfig.output_dir_name.")
    parser.add_argument("--run-id", default="", help="Stable run id for output isolation.")
    parser.add_argument("--feature-cache-dir", default="", help="Override in-run feature cache directory name.")
    parser.add_argument("--offline-cache-root", default="", help="Override offline cache root directory.")
    parser.add_argument("--quick", action="store_true", help="Run with the quick-mode row limits from TrainConfig.")
    parser.add_argument("--disable-feature-cache", action="store_true")
    parser.add_argument("--disable-offline-cache", action="store_true")
    parser.add_argument("--shared-artifacts", action="store_true", help="Do not isolate outputs into a run-specific folder.")
    parser.add_argument("--skip-final-test", action="store_true", help="In train mode, skip the rank-0 final test pass.")
    return parser


def _config_from_args(args) -> TrainConfig:
    cfg = TrainConfig()
    if args.output_dir:
        cfg.output_dir_name = args.output_dir
    if args.run_id:
        cfg.run_id = args.run_id
        cfg.auto_timestamp_run_id = False
    if args.shared_artifacts:
        cfg.isolate_run_artifacts = False
    if args.quick:
        cfg.quick_mode = True
    if args.feature_cache_dir:
        cfg.feature_cache_dir = args.feature_cache_dir
    if args.disable_feature_cache:
        cfg.use_feature_cache = False
    if args.disable_offline_cache:
        cfg.asr_offline_cache_enable = False
        cfg.fapi_offline_cache_enable = False
        cfg.e2v_offline_cache_enable = False
        cfg.acoustic_offline_cache_enable = False
    if args.offline_cache_root:
        root = Path(args.offline_cache_root)
        cfg.asr_offline_cache_file = str(root / "offline_asr_cache.csv")
        cfg.asr_offline_cache_file_test = str(root / "offline_asr_cache_test.csv")
        cfg.fapi_offline_cache_file = str(root / "offline_fapi_stats_cache.csv")
        cfg.fapi_offline_cache_file_test = str(root / "offline_fapi_stats_cache_test.csv")
        cfg.e2v_offline_cache_dir = str(root / "e2v_cache")
        cfg.acoustic_offline_cache_dir = str(root / "acoustic_cache")
    if args.model_path:
        cfg.best_model_file = args.model_path
    return cfg


def _default_model_path(cfg: TrainConfig, explicit_model_path: str = "") -> str:
    if explicit_model_path:
        return explicit_model_path
    _, _, _, default_best_model_path = _resolve_artifact_paths(cfg)
    return str(default_best_model_path)


@record
def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cfg = _config_from_args(args)
    model_path = _default_model_path(cfg, args.model_path)

    if args.mode == "train":
        train(
            df_train=_read_csv_required(args.train_csv, "train"),
            train_data_path=args.train_root,
            df_val=_read_csv_required(args.val_csv, "validation"),
            val_data_path=args.val_root,
            cfg=cfg,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if (not args.skip_final_test) and local_rank == 0:
            test(
                df_test=_read_csv_required(args.test_csv, "test"),
                test_data_path=args.test_root,
                model_path=model_path,
                cfg=cfg,
            )
    elif args.mode == "test":
        test(
            df_test=_read_csv_required(args.test_csv, "test"),
            test_data_path=args.test_root,
            model_path=model_path,
            cfg=cfg,
        )
    elif args.mode == "predict":
        predict_path_list = _read_audio_paths(args.audio, args.audio_list)
        if not predict_path_list:
            parser.error("predict mode requires --audio or --audio-list")
        preds = predict(predict_path_list, model_path=model_path, return_feature_pack=True, cfg=cfg)
        pack = preds.get("feature_pack")
        if pack is not None and pack.get("emotion_probs") is not None and pack["emotion_probs"].size(0) > 0:
            radar_dir = Path(current_dir) / cfg.output_dir_name
            radar_dir.mkdir(parents=True, exist_ok=True)
            FAPIVizSuite.plot_emotion_radar(
                pack["emotion_probs"][0],
                labels=[
                    "Angry", "Disgusted", "Fearful", "Happy", "Neutral",
                    "Other", "Sad", "Surprised", "Unknown",
                ],
                title="Emotion Radar (Predict Sample 0)",
                save_path=str(radar_dir / "predict_emotion_radar_sample0.png"),
                show=False,
            )
    elif args.mode == "predict_iam":
        predict_path_list = _read_audio_paths(args.audio, args.audio_list)
        if not predict_path_list:
            parser.error("predict_iam mode requires --audio or --audio-list")
        assert InverseAttentionRuntime.smoke_test(), "InverseAttentionRuntime smoke test failed"
        run_and_visualize_inverse_attention(predict_path_list)
    

if __name__ == "__main__":
    main()
