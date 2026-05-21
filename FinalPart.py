import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import numpy as np
from typing import Any, cast
from config import TrainConfig
from losses_multitask import MultiTaskLosses

def build_lm(lm_name="Qwen/Qwen2.5-3B", device=None, precision="fp16"):
    """Build the causal language model used for next-token FAPI scoring."""
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device_obj.type == "cuda" and device_obj.index is None:
        # Bind the LM to the rank-local CUDA device under DDP.
        device_obj = torch.device(f"cuda:{torch.cuda.current_device()}")
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    precision_cfg = str(precision).strip().lower()
    dtype = None
    if device_obj.type == "cuda":
        if precision_cfg == "bf16":
            dtype = torch.bfloat16
        elif precision_cfg == "fp32":
            dtype = torch.float32
        else:
            dtype = torch.float16
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            lm_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ),
    )
    model = model.to(device_obj)
    model.eval()
    return tokenizer, model, str(device_obj)

def compute_text_ppl_and_topk_mean(text, tokenizer, model, device, topk=5):
    """Compute PPL and top-k confidence statistics for one ASR transcript.

    Lower PPL means the transcript is more predictable under the LM. Higher
    top-k mean indicates stronger next-token confidence.
    """
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    # Single-token inputs do not provide a next-token target.
    if input_ids.size(1) < 2:
        return 1.0, 0.0

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

        # Align logits at position t with the token target at t + 1.
        logits = outputs.logits[:, :-1, :].float()  # [1, T-1, V]
        labels = input_ids[:, 1:]  # [1, T-1]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # Mean negative log likelihood gives sentence-level perplexity.
        nll = -token_log_probs.mean()
        ppl = torch.exp(nll).item()

        # Average top-k probability is used as the confidence proxy in gamma.
        probs = torch.softmax(logits, dim=-1)
        k = min(topk, probs.size(-1))
        topk_mean = torch.topk(probs, k=k, dim=-1).values.mean().item()

    return float(ppl), float(topk_mean)

def compute_gamma_lambda(ppl, topk_mean, p_threshold=385.4104, gamma_base=10.0):
    """Compute the FAPI dynamic sensitivity and penalty strength."""
    gamma_t = float(gamma_base * topk_mean)
    x = gamma_t * (float(p_threshold) - float(ppl))
    # Numerically stable sigmoid.
    if x >= 0:
        lambda_penalty = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        lambda_penalty = exp_x / (1.0 + exp_x)
    return gamma_t, float(lambda_penalty)



class FAPILMScorer:
    """Small wrapper around the LM scorer used by FAPI."""

    def __init__(self, lm_name: str, device: torch.device, precision: str = "fp16"):
        super().__init__()
        self.device = torch.device(device)
        
        tokenizer, model, actual_device = build_lm(lm_name=lm_name, device=str(device), precision=precision)
        self.device = torch.device(actual_device)
        self.tokenizer = tokenizer
        self.model = model

    @torch.no_grad()
    def ppl_and_topk(self, text: str, topk: int = 5):
        """Compute text perplexity and top-k mean confidence."""
        ppl, topk_mean = compute_text_ppl_and_topk_mean(
            text=text,
            tokenizer=self.tokenizer,
            model=self.model,
            device=str(self.device),
            topk=topk,
        )
        return float(ppl), float(topk_mean)


def compute_gamma_lambda_from_stats(ppl, topk_mean, p_threshold, gamma_base=10.0):
    """Compute gamma_t and lambda_penalty from cached or freshly scored FAPI stats."""
    gamma_t, lambda_penalty = compute_gamma_lambda(
        ppl=ppl,
        topk_mean=topk_mean,
        p_threshold=p_threshold,
        gamma_base=gamma_base,
    )
    return float(gamma_t), float(lambda_penalty)
def temporal_align(x: torch.Tensor, target_len: int):
    """x: [B, T, D] -> [B, target_len, D]"""
    if x.size(1) == target_len:
        return x
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {x.dim()}D")
    x_t = x.transpose(1, 2)
    x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
    return x_t.transpose(1, 2)

def FAPI(
    F_A,
    F_S2,
    F_text,
    audio_mask,
    text_mask,
    texts,
    U_audio,
    F_emo,
    lm_scorer,
    cfg: TrainConfig,
    audio_paths=None,
    fapi_stats_cache=None,
):
    """Merge the three streams and FAPI statistics into a detector batch.

    Returns audio, fused acoustic-emotion, text, masks, uncertainty, emotion
    labels, and the PPL/gamma/lambda tensors needed by the downstream model.
    """
    current_device = F_A.device
    
    # Align text sequence length to the fused acoustic-emotion stream.
    t_s2 = F_S2.size(1)
    t_text = F_text.size(1)
    if t_text != t_s2:
        F_text = temporal_align(F_text, t_s2)
        text_mask_list = []
        for i in range(F_text.size(0)):
            mask = torch.ones(t_s2, dtype=torch.bool, device=current_device)
            text_mask_list.append(mask)
        text_mask = torch.stack(text_mask_list)
    
    # Score text with the LM, reusing offline cache entries when available.
    ppl_list, topk_list, gamma_list, lam_list = [], [], [], []
    if audio_paths is None:
        audio_paths = [None for _ in range(len(texts))]

    cache_miss_paths = []
    cache_miss_ppls = []
    cache_miss_topks = []

    for idx, text in enumerate(texts):
        path_key = audio_paths[idx] if idx < len(audio_paths) else None
        cache_hit = None
        if fapi_stats_cache is not None and path_key is not None:
            cache_hit = fapi_stats_cache.get(path_key)

        if cache_hit is not None:
            ppl = float(cache_hit[0])
            topk_mean = float(cache_hit[1])
        else:
            ppl, topk_mean = lm_scorer.ppl_and_topk(text, topk=cfg.topk)
            if path_key is not None:
                cache_miss_paths.append(path_key)
                cache_miss_ppls.append(float(ppl))
                cache_miss_topks.append(float(topk_mean))

        gamma_t, lam = compute_gamma_lambda_from_stats(
            ppl=ppl,
            topk_mean=topk_mean,
            p_threshold=cfg.p_threshold,
            gamma_base=cfg.gamma_base,
        )
        ppl_list.append(ppl)
        topk_list.append(topk_mean)
        gamma_list.append(gamma_t)
        lam_list.append(lam)

    if fapi_stats_cache is not None and len(cache_miss_paths) > 0:
        fapi_stats_cache.put_many(cache_miss_paths, cache_miss_ppls, cache_miss_topks)
    
    batch = {
        # Stream 1: acoustic physical basis.
        "F_A": F_A,  # [B, 512]
        "audio_mask": audio_mask,  # [B, T]
        "U_audio": U_audio,  # [B]
        
        # Stream 2: emotion and time-frequency fusion.
        "F_S2": F_S2,  # [B, T, 896]
        "F_emo": F_emo,  # [B, 9]
        
        # Stream 3: ASR-derived text semantics.
        "texts": texts,
        "F_text": F_text,  # [B, T, 768]
        "text_mask": text_mask,  # [B, T]
        
        # FAPI dynamic intervention statistics.
        "ppl": torch.tensor(ppl_list, dtype=torch.float32, device=current_device),
        "topk_mean": torch.tensor(topk_list, dtype=torch.float32, device=current_device),
        "gamma_t": torch.tensor(gamma_list, dtype=torch.float32, device=current_device),
        "lambda_penalty": torch.tensor(lam_list, dtype=torch.float32, device=current_device),
    }
    
    return batch

def compute_cross_entropy_from_att(att_cross_weights: torch.Tensor):
    """Compute cross-modal conflict entropy from attention weights."""
    p = torch.clamp(att_cross_weights, min=1e-12)
    entropy = -(p * torch.log(p)).sum(dim=-1) 
    if entropy.dim() == 3:
        return entropy.mean(dim=(1, 2))
    else:
        return entropy.mean(dim=-1)

# IACA gating and fusion step.
class IACAGate(nn.Module):
    """IACA gate: g = sigmoid(omega1 * U_audio + omega2 * H_cross - tau)."""

    def __init__(self, omega1_init=2.0, omega2_init=1.0, tau_init=2):
        super().__init__()
        self.omega1 = nn.Parameter(torch.tensor(float(omega1_init)))
        self.omega2 = nn.Parameter(torch.tensor(float(omega2_init)))
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

        
    def forward(self, u_audio, h_cross):
        return torch.sigmoid(self.omega1 * u_audio + self.omega2 * h_cross - self.tau)
class MultiTaskHead(nn.Module):
    """Shared prediction head for spoof detection and emotion classification."""

    def __init__(self, d_model=256, n_emotions=9):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
        )
        self.fake_head = nn.Linear(d_model, 1)
        self.emo_head = nn.Linear(d_model, n_emotions)

    def forward(self, x):
        pooled = x.mean(dim=1)
        z = self.shared(pooled)
        return self.fake_head(z).squeeze(-1), self.emo_head(z)


class FAPIVizSuite:
    """Visualization utilities for FAPI results.

    This class centralizes common train/validation figures:
    - loss curves
    - PPL distribution and threshold
    - confusion matrix
    - nine-class emotion radar chart
    - train/inference summary panel
    """

    EMOTION_LABELS = [
        "Angry",
        "Disgusted",
        "Fearful",
        "Happy",
        "Neutral",
        "Other",
        "Sad",
        "Surprised",
        "Unknown",
    ]

    @staticmethod
    def _to_numpy(values):
        if values is None:
            return np.asarray([], dtype=np.float32)
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    @staticmethod
    def _ensure_1d(values):
        arr = FAPIVizSuite._to_numpy(values)
        return arr.reshape(-1)

    @staticmethod
    def plot_loss_curves(history, title="Training Curves", save_path=None, show=True):
        """Plot training and validation loss curves."""
        train_loss = FAPIVizSuite._ensure_1d(history.get("train_loss", []))
        val_loss = FAPIVizSuite._ensure_1d(history.get("val_loss", history.get("val_ce", [])))

        fig, ax = plt.subplots(figsize=(8, 4))
        if train_loss.size > 0:
            ax.plot(train_loss, label="train_loss", linewidth=2)
        if val_loss.size > 0:
            ax.plot(val_loss, label="val_loss", linewidth=2)

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    @staticmethod
    def _compute_ppl_xlim(real_arr, fake_arr, threshold, zoom_quantiles=(0.05, 0.95), zoom_margin=0.15):
        """Compute a compact PPL axis range from the threshold and quantiles."""
        combined = np.concatenate([real_arr.reshape(-1), fake_arr.reshape(-1)])
        combined = combined[np.isfinite(combined)]
        if combined.size == 0:
            center = float(threshold)
            span = max(abs(center) * 0.5, 50.0)
            return max(0.0, center - span), center + span

        center = float(threshold)
        q_low, q_high = np.quantile(combined, zoom_quantiles)
        iqr = max(float(q_high) - float(q_low), 0.0)
        span = max(abs(center) * 0.5, 50.0, iqr * 0.75)
        span *= (1.0 + float(zoom_margin))
        x_min = max(0.0, center - span)
        x_max = center + span
        return x_min, x_max

    @staticmethod
    def plot_ppl_threshold(
        scores_real,
        scores_fake,
        threshold,
        title="PPL Distribution + Threshold",
        save_path=None,
        show=True,
        zoom=True,
        zoom_quantiles=(0.05, 0.95),
        zoom_margin=0.15,
    ):
        """Plot real/fake PPL distributions with the decision threshold."""
        real_arr = FAPIVizSuite._ensure_1d(scores_real)
        fake_arr = FAPIVizSuite._ensure_1d(scores_fake)

        fig, ax = plt.subplots(figsize=(8, 4))
        if real_arr.size > 0:
            ax.hist(real_arr, alpha=0.45, bins=20, label="real ppl", density=True)
        if fake_arr.size > 0:
            ax.hist(fake_arr, alpha=0.45, bins=20, label="fake ppl", density=True)
        ax.axvline(float(threshold), color="black", linestyle="--", label=f"threshold={float(threshold):.3f}")

        if zoom:
            x_min, x_max = FAPIVizSuite._compute_ppl_xlim(
                real_arr=real_arr,
                fake_arr=fake_arr,
                threshold=threshold,
                zoom_quantiles=zoom_quantiles,
                zoom_margin=zoom_margin,
            )
            ax.set_xlim(x_min, x_max)

        ax.set_title(title)
        ax.set_xlabel("PPL")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=("Real", "Fake"), normalize=False, title="Confusion Matrix", save_path=None, show=True):
        """Plot a binary confusion matrix."""
        y_true_arr = FAPIVizSuite._ensure_1d(y_true).astype(int)
        y_pred_arr = FAPIVizSuite._ensure_1d(y_pred).astype(int)

        labels = np.arange(len(class_names))
        cm = np.zeros((len(class_names), len(class_names)), dtype=np.float64)
        for t, p in zip(y_true_arr, y_pred_arr):
            if 0 <= t < len(class_names) and 0 <= p < len(class_names):
                cm[t, p] += 1

        if normalize and cm.sum() > 0:
            cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-12)

        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        fmt = ".2f" if normalize else ".0f"
        thresh = cm.max() * 0.55 if cm.size > 0 else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, cm

    @staticmethod
    def plot_emotion_radar(emotion_probs, labels=None, title="Emotion Radar", save_path=None, show=True):
        """Plot a nine-class emotion radar chart."""
        probs = FAPIVizSuite._ensure_1d(emotion_probs).astype(np.float64)
        if probs.size == 0:
            raise ValueError("emotion_probs 不能为空")

        if labels is None:
            labels = FAPIVizSuite.EMOTION_LABELS[: probs.size]
        if len(labels) != probs.size:
            raise ValueError("labels 长度必须与 emotion_probs 维度一致")

        angles = np.linspace(0, 2 * np.pi, probs.size, endpoint=False)
        probs_cycle = np.concatenate([probs, [probs[0]]])
        angles_cycle = np.concatenate([angles, [angles[0]]])

        fig = plt.figure(figsize=(6.4, 6.4))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles_cycle, probs_cycle, linewidth=2.2)
        ax.fill(angles_cycle, probs_cycle, alpha=0.22)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        ax.set_title(title, pad=18)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=220, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    @staticmethod
    def build_summary_panel(history=None, scores_real=None, scores_fake=None, threshold=None, y_true=None, y_pred=None, emotion_probs=None, emotion_labels=None, show=True, save_path=None):
        """Export the main visualization panels and return figures/statistics."""
        report = {}

        if history is not None:
            report["loss_fig"] = FAPIVizSuite.plot_loss_curves(history, show=show)

        if scores_real is not None and scores_fake is not None and threshold is not None:
            report["ppl_fig"] = FAPIVizSuite.plot_ppl_threshold(scores_real, scores_fake, threshold, show=show)

        if y_true is not None and y_pred is not None:
            report["cm_fig"], report["confusion_matrix"] = FAPIVizSuite.plot_confusion_matrix(y_true, y_pred, show=show)

        if emotion_probs is not None:
            report["radar_fig"] = FAPIVizSuite.plot_emotion_radar(emotion_probs, labels=emotion_labels, show=show)

        if save_path is not None:
            fig = plt.figure(figsize=(8, 6))
            fig.suptitle("FAPI Visualization Summary", fontsize=14)
            fig.text(0.05, 0.88, "See individual figures for detailed views.", fontsize=10)
            fig.tight_layout()
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            report["summary_path"] = save_path

        return report




'''

# Training and validation loop.
def compute_fapi_stats_for_batch(batch, lm_scorer, cfg: TrainConfig):
    """Compute FAPI statistics for a batch: PPL, gamma_t, and lambda_penalty."""
    texts = batch["texts"]
    ppl_list, topk_list, gamma_list, lam_list = [], [], [], []
    
    for text in texts:
        ppl, topk_mean = lm_scorer.ppl_and_topk(text, topk=cfg.topk)
        gamma_t, lam = compute_gamma_lambda_from_stats(
            ppl=ppl,
            topk_mean=topk_mean,
            p_threshold=cfg.p_threshold,
            gamma_base=cfg.gamma_base,
        )
        ppl_list.append(ppl)
        topk_list.append(topk_mean)
        gamma_list.append(gamma_t)
        lam_list.append(lam)
    
    return torch.tensor(lam_list, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, scaler, lm_scorer, cfg: TrainConfig, epoch: int):
    """Train one epoch with mixed precision and gradient clipping."""
    model.train()
    total = 0.0
    logs = []

    for step, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        
        # Compute FAPI statistics if they are not already present in the batch.
        lambda_vec = compute_fapi_stats_for_batch(batch, lm_scorer, cfg)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(batch, lambda_vec)
            loss, loss_dict = MultiTaskLosses.compute_total_loss(
                outputs=outputs,
                batch=batch,
                lambda_penalty_vec=lambda_vec,
                lambda1=cfg.lambda1,
                lambda2=cfg.lambda2,
                lambda3=cfg.lambda3,
                margin=cfg.margin,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.detach().item())
        logs.append({k: float(v.item()) for k, v in loss_dict.items()})

    avg = total / max(1, len(loader))
    print(f"[train][epoch={epoch}] loss={avg:.4f}")
    return avg, logs


@torch.no_grad()
def validate(model, loader, lm_scorer, cfg: TrainConfig, epoch: int):
    """Evaluate on the validation split."""
    model.eval()
    total = 0.0
    y_true, y_prob = [], []

    for batch in loader:
        lambda_vec = compute_fapi_stats_for_batch(batch, lm_scorer, cfg)
        outputs = model(batch, lambda_vec)

        labels = batch["labels"].to(outputs["fake_logit"].device, dtype=outputs["fake_logit"].dtype)
        loss = MultiTaskLosses.loss_ce_fake(outputs["fake_logit"], labels)
        total += float(loss.item())

        prob = torch.sigmoid(outputs["fake_logit"]).detach().cpu().numpy().tolist()
        y_prob.extend(prob)
        y_true.extend(labels.detach().cpu().numpy().tolist())

    avg = total / max(1, len(loader))
    print(f"[val][epoch={epoch}] ce_loss={avg:.4f}")
    return avg, y_true, y_prob


def save_checkpoint(model, optimizer, epoch, best_val, path):
    """Save a training checkpoint."""
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val,
        "config": asdict(cfg),
    }
    torch.save(ckpt, path)
    print(f"✓ 检查点已保存: {path}")
'''
'''Local test and plotting example.
# Initialize model, optimizer, and LM scorer.
model = FullDetector(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
lm_scorer = FAPILMScorer(cfg.lm_name, device)

best_val = float("inf")
history = {"train_loss": [], "val_ce": []}
ckpt_dir = repo_root / "outputs_fapi_train"
ckpt_dir.mkdir(exist_ok=True)

# Training loop.
for ep in range(1, cfg.num_epochs + 1):
    tr_loss, _ = train_one_epoch(model, train_loader, optimizer, scaler, lm_scorer, cfg, ep)
    va_loss, y_true, y_prob = validate(model, val_loader, lm_scorer, cfg, ep)

    history["train_loss"].append(tr_loss)
    history["val_ce"].append(va_loss)

    save_checkpoint(model, optimizer, ep, best_val, ckpt_dir / "last.pt")
    if va_loss < best_val:
        best_val = va_loss
        save_checkpoint(model, optimizer, ep, best_val, ckpt_dir / "best.pt")

print("✓ 训练结束，best_val:", best_val)

# Threshold calibration and inference output.
# Reuse the threshold-search function from the original FAPI notebook.
search_best_threshold = shared_funcs["search_best_threshold"]


def _collect_ppl_for_texts(texts, lm_scorer, cfg: TrainConfig, p_threshold: Optional[float] = None):
    """Collect PPL/top-k/gamma/lambda for a batch of texts without notebook helpers."""
    if p_threshold is None:
        p_threshold = cfg.p_threshold

    ppl_list, topk_list, gamma_list, lam_list = [], [], [], []
    for text in texts:
        ppl, topk_mean = lm_scorer.ppl_and_topk(text, cfg.topk)
        gamma_t, lam = compute_gamma_lambda_from_stats(ppl, topk_mean, p_threshold, cfg.gamma_base)
        ppl_list.append(ppl)
        topk_list.append(topk_mean)
        gamma_list.append(gamma_t)
        lam_list.append(lam)

    return {
        "ppl": torch.tensor(ppl_list, dtype=torch.float32, device=device),
        "topk_mean": torch.tensor(topk_list, dtype=torch.float32, device=device),
        "gamma_t": torch.tensor(gamma_list, dtype=torch.float32, device=device),
        "lambda_penalty": torch.tensor(lam_list, dtype=torch.float32, device=device),
    }


@torch.no_grad()
def infer_one_batch(model, batch, lm_scorer, cfg: TrainConfig, p_threshold: float):
    stats = _collect_ppl_for_texts(batch["texts"], lm_scorer, cfg, p_threshold=p_threshold)
    lam_vec = stats["lambda_penalty"]

    out = model(batch, lam_vec)

    fake_prob = torch.sigmoid(out["fake_logit"]).cpu().numpy()
    emo_prob = F.softmax(out["emo_logit"], dim=-1).cpu().numpy()

    rows = []
    for i in range(len(batch["texts"])):
        rows.append({
            "text": batch["texts"][i],
            "ppl": float(stats["ppl"][i].item()),
            "topk_mean": float(stats["topk_mean"][i].item()),
            "gamma_t": float(stats["gamma_t"][i].item()),
            "lambda_penalty": float(stats["lambda_penalty"][i].item()),
            "g": float(out["g"][i].cpu().item()),
            "fake_prob": float(fake_prob[i]),
            "emotion_probs": emo_prob[i].tolist(),
            "pred_label": int(fake_prob[i] >= 0.5),
            "beta": float(out["beta"].item()) if "beta" in out else None,
        })
    return rows


# Demonstration only: use a dedicated calibration set in real experiments.
calib_batch = next(iter(train_loader))
calib_stats = _collect_ppl_for_texts(calib_batch["texts"], lm_scorer, cfg)
ppl_arr = calib_stats["ppl"].cpu().numpy()
lab_arr = calib_batch["labels"].numpy()

scores_real = ppl_arr[lab_arr < 0.5].tolist() if (lab_arr < 0.5).any() else [float(ppl_arr.mean() + 1)]
scores_fake = ppl_arr[lab_arr >= 0.5].tolist() if (lab_arr >= 0.5).any() else [float(ppl_arr.mean() - 1)]

best_t = search_best_threshold(scores_real=scores_real, scores_generated=scores_fake)
print("threshold calibration:", best_t)

# Inference output.
infer_batch = next(iter(val_loader))
outputs_rows = infer_one_batch(model, infer_batch, lm_scorer, cfg, p_threshold=best_t["p_threshold"])
print("样本输出示例:")
for row in outputs_rows:
    print({k: row[k] for k in ["ppl", "gamma_t", "lambda_penalty", "g", "fake_prob", "pred_label", "beta"]})



# Result visualization.
import csv


def plot_training_curves(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_ce"], label="val_ce")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_ppl_distribution(scores_real, scores_fake, t):
    plt.figure(figsize=(8, 4))
    plt.hist(scores_real, alpha=0.45, bins=20, label="real ppl", density=True)
    plt.hist(scores_fake, alpha=0.45, bins=20, label="fake ppl", density=True)
    plt.axvline(t, color="black", linestyle="--", label=f"threshold={t:.3f}")

    x_min, x_max = FAPIVizSuite._compute_ppl_xlim(
        real_arr=np.asarray(scores_real, dtype=np.float64),
        fake_arr=np.asarray(scores_fake, dtype=np.float64),
        threshold=t,
    )
    plt.xlim(x_min, x_max)

    plt.title("PPL Distribution + Threshold")
    plt.xlabel("PPL")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_emotion_radar(emotion_probs, labels=None):
    probs = np.asarray(emotion_probs, dtype=np.float64)
    n = probs.shape[0]
    if labels is None:
        labels = [f"E{i}" for i in range(n)]

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    probs = np.concatenate([probs, [probs[0]]])
    angles = np.concatenate([angles, [angles[0]]])

    ax = plt.subplot(111, polar=True)
    ax.plot(angles, probs, linewidth=2)
    ax.fill(angles, probs, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Emotion Radar")
    plt.show()


def export_reports(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "inference_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "inference_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "ppl", "topk_mean", "gamma_t", "lambda_penalty", "g", "fake_prob", "pred_label"])
        w.writeheader()
        for r in rows:
            r2 = {k: r[k] for k in w.fieldnames}
            w.writerow(r2)

    print("saved:", json_path)
    print("saved:", csv_path)


# -------- run visualization and export --------
plot_training_curves(history)
plot_ppl_distribution(scores_real, scores_fake, best_t["t"])

if len(outputs_rows) > 0:
    plot_emotion_radar(outputs_rows[0]["emotion_probs"], labels=[
        "Angry", "Disgusted", "Fearful", "Happy", "Neutral",
        "Other", "Sad", "Surprised", "Unknown"
    ])

export_reports(outputs_rows, repo_root / "outputs_fapi_train")
'''
