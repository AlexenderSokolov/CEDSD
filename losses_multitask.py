import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class MultiTaskLosses:
    """Multitask loss collection: CE + KL + conflict regularization + FAPI."""

    @staticmethod
    def loss_ce_fake(fake_logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Binary cross entropy for real/fake classification."""
        labels = labels.to(device=fake_logit.device, dtype=fake_logit.dtype)
        return F.binary_cross_entropy_with_logits(fake_logit, labels)

    @staticmethod
    def loss_kl_emotion(emo_logit: torch.Tensor, soft_labels: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Emotion KL divergence against soft emotion-label distributions."""
        log_prob = F.log_softmax(emo_logit, dim=-1)
        target = torch.clamp(soft_labels.to(device=emo_logit.device, dtype=emo_logit.dtype), min=eps)
        target = target / target.sum(dim=-1, keepdim=True)
        return F.kl_div(log_prob, target, reduction="batchmean")

    @staticmethod
    def _reduce_conflict_signal(conflict_signal: torch.Tensor) -> torch.Tensor:
        """Reduce an arbitrary conflict signal to one score per sample."""
        if conflict_signal.dim() == 1:
            return conflict_signal
        return conflict_signal.reshape(conflict_signal.size(0), -1).mean(dim=1)

    @staticmethod
    def _compute_fake_scores(
        conflict_signal: torch.Tensor,
        labels: torch.Tensor,
        fake_label_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Select fake samples and return their conflict scores."""
        signal = MultiTaskLosses._reduce_conflict_signal(conflict_signal)
        fake_mask = labels.to(device=signal.device) >= fake_label_threshold
        if fake_mask.sum() == 0:
            return torch.empty(0, device=signal.device, dtype=signal.dtype)
        return signal[fake_mask]

    @staticmethod
    def loss_conflict_reg(
        conflict_signal: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 1.5,
        fake_label_threshold: float = 0.5,
        adaptive_margin: bool = False,
        adaptive_margin_ratio: float = 1.05,
    ) -> torch.Tensor:
        """Hinge-style conflict regularization applied only to fake samples."""
        fake_scores = MultiTaskLosses._compute_fake_scores(
            conflict_signal=conflict_signal,
            labels=labels,
            fake_label_threshold=fake_label_threshold,
        )
        if fake_scores.numel() == 0:
            return conflict_signal.new_zeros(())

        effective_margin = margin
        if adaptive_margin:
            effective_margin = max(float(margin), float(fake_scores.mean().detach().item()) * float(adaptive_margin_ratio))

        # Apply ReLU(margin - score).
        loss_per_fake = F.relu(effective_margin - fake_scores)
        return loss_per_fake.mean()

    @staticmethod
    def conflict_debug_stats(
        conflict_signal: torch.Tensor,
        labels: torch.Tensor,
        margin: float,
        fake_label_threshold: float = 0.5,
        adaptive_margin: bool = False,
        adaptive_margin_ratio: float = 1.05,
    ) -> dict:
        """Return observable conflict statistics for diagnosing inactive regularization."""
        with torch.no_grad():
            fake_scores = MultiTaskLosses._compute_fake_scores(
                conflict_signal=conflict_signal,
                labels=labels,
                fake_label_threshold=fake_label_threshold,
            )
            if fake_scores.numel() == 0:
                zero = torch.tensor(0.0, device=conflict_signal.device, dtype=conflict_signal.dtype)
                return {
                    "num_fakes": zero,
                    "fake_score_mean": zero,
                    "fake_score_p90": zero,
                    "conflict_active_ratio": zero,
                    "effective_margin": torch.tensor(float(margin), device=conflict_signal.device, dtype=conflict_signal.dtype),
                }

            effective_margin = float(margin)
            if adaptive_margin:
                effective_margin = max(effective_margin, float(fake_scores.mean().detach().item()) * float(adaptive_margin_ratio))
            active_ratio = (fake_scores < effective_margin).to(dtype=conflict_signal.dtype).mean()
            p90 = torch.quantile(fake_scores, 0.9)
            return {
                "num_fakes": torch.tensor(float(fake_scores.numel()), device=conflict_signal.device, dtype=conflict_signal.dtype),
                "fake_score_mean": fake_scores.mean(),
                "fake_score_p90": p90,
                "conflict_active_ratio": active_ratio,
                "effective_margin": torch.tensor(effective_margin, device=conflict_signal.device, dtype=conflict_signal.dtype),
            }

    
    @staticmethod
    def compute_total_loss(
        outputs: dict,
        batch: dict,
        lambda_penalty_vec: torch.Tensor,
        s1: float | torch.Tensor = 0.0,
        s2: float | torch.Tensor = 0.0,
        s3: float | torch.Tensor = 0.0,
        s_min: float = -1.5,
        s_max: float = 1.5,
        uw_reg_coef: float | torch.Tensor = 1.0,
        use_fapi_loss: bool = True,
        loss_smooth_eps: float | torch.Tensor = 1e-4,
        fapi_penalty_scale: float | torch.Tensor = 1.0,
        margin: float = 0.5,
        fake_label_threshold: float = 0.5,
        adaptive_margin: bool = False,
        adaptive_margin_ratio: float = 1.05,
        debug_fapi_graph: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total loss and logging terms.

        Required outputs:
        - fake_logit, emo_logit, F_AT, F_text
        Required batch fields:
        - labels, F_emo
        """

        labels = batch["labels"].to(device=outputs["fake_logit"].device, dtype=outputs["fake_logit"].dtype)
        soft_labels = batch["F_emo"].to(device=outputs["emo_logit"].device, dtype=outputs["emo_logit"].dtype)

        l_ce = MultiTaskLosses.loss_ce_fake(outputs["fake_logit"], labels)
        l_kl = MultiTaskLosses.loss_kl_emotion(outputs["emo_logit"], soft_labels)
        conflict_signal = outputs["h_cross"] if ("h_cross" in outputs and outputs["h_cross"] is not None) else outputs["F_AT"]

        l_conf = MultiTaskLosses.loss_conflict_reg(
            conflict_signal,
            labels,
            margin=margin,
            fake_label_threshold=fake_label_threshold,
            adaptive_margin=adaptive_margin,
            adaptive_margin_ratio=adaptive_margin_ratio,
        )
        conf_stats = MultiTaskLosses.conflict_debug_stats(
            conflict_signal,
            labels,
            margin=margin,
            fake_label_threshold=fake_label_threshold,
            adaptive_margin=adaptive_margin,
            adaptive_margin_ratio=adaptive_margin_ratio,
        )
        # Evaluation usually runs under torch.no_grad(), so FAPI is enabled only when gradients are available.

        f_text_requires_grad = bool(outputs["F_text"].requires_grad)
        if torch.is_grad_enabled() and f_text_requires_grad:
            # Use the FAPI raw penalty computed inside the model and apply the configured scale.
            l_fapi_raw = outputs.get("l_fapi_raw", torch.zeros_like(l_ce))
            fapi_scale_t = torch.as_tensor(fapi_penalty_scale, device=l_ce.device, dtype=l_ce.dtype).clamp_min(1e-8)
            l_fapi = l_fapi_raw * fapi_scale_t

            # Keep FAPI monitoring metrics for training logs.
            fapi_grad_rms_mean = outputs.get("fapi_grad_rms_mean", torch.zeros_like(l_ce))
            fapi_grads_is_none = outputs.get("fapi_grads_is_none", True)
        else:
            l_fapi_raw = torch.zeros((), device=l_ce.device, dtype=l_ce.dtype)
            fapi_grad_rms_mean = torch.zeros((), device=l_ce.device, dtype=l_ce.dtype)
            fapi_grads_is_none = True

        # Optional probe to check whether CE gradients reach text and fused features.
        if debug_fapi_graph and torch.is_grad_enabled():
            probe_text = torch.autograd.grad(
                outputs=l_ce,
                inputs=outputs["F_text"],
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
                allow_unused=True,
            )[0]
            probe_fat = torch.autograd.grad(
                outputs=l_ce,
                inputs=outputs["F_AT"],
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
                allow_unused=True,
            )[0]
            ce_to_ftext_connected = 0.0 if probe_text is None else 1.0
            ce_to_fat_connected = 0.0 if probe_fat is None else 1.0
            ce_to_ftext_rms = torch.zeros((), device=l_ce.device, dtype=l_ce.dtype) if probe_text is None else probe_text.reshape(probe_text.size(0), -1).pow(2).mean(dim=1).sqrt().mean()
            ce_to_fat_rms = torch.zeros((), device=l_ce.device, dtype=l_ce.dtype) if probe_fat is None else probe_fat.reshape(probe_fat.size(0), -1).pow(2).mean(dim=1).sqrt().mean()
        else:
            ce_to_ftext_connected = 0.0
            ce_to_fat_connected = 0.0
            ce_to_ftext_rms = torch.zeros((), device=l_ce.device, dtype=l_ce.dtype)
            ce_to_fat_rms = torch.zeros((), device=l_ce.device, dtype=l_ce.dtype)

        if "lambda_penalty_clamped" in outputs and outputs["lambda_penalty_clamped"] is not None:
            lambda_penalty_clamped_mean = torch.as_tensor(
                outputs["lambda_penalty_clamped"],
                device=l_ce.device,
                dtype=l_ce.dtype,
            ).mean()
        else:
            lambda_penalty_clamped_mean = torch.zeros((), device=l_ce.device, dtype=l_ce.dtype)

        if "g" in outputs and outputs["g"] is not None:
            gate_mean = torch.as_tensor(outputs["g"], device=l_ce.device, dtype=l_ce.dtype).mean()
        else:
            gate_mean = torch.zeros((), device=l_ce.device, dtype=l_ce.dtype)

        fapi_scale_t = torch.as_tensor(fapi_penalty_scale, device=l_ce.device, dtype=l_ce.dtype).clamp_min(1e-8)
        if use_fapi_loss:
            l_fapi = l_fapi_raw * fapi_scale_t
        else:
            l_fapi = torch.zeros((), device=l_ce.device, dtype=l_ce.dtype)

        # Standard uncertainty weighting: sum(exp(-s_i) * (L_i + eps) + s_i).
        s1_t = torch.as_tensor(s1, device=l_ce.device, dtype=l_ce.dtype)
        s2_t = torch.as_tensor(s2, device=l_ce.device, dtype=l_ce.dtype)
        s3_t = torch.as_tensor(s3, device=l_ce.device, dtype=l_ce.dtype)
        smooth_eps_t = torch.as_tensor(loss_smooth_eps, device=l_ce.device, dtype=l_ce.dtype).clamp_min(0.0)

        # Use the learnable log-variance values directly; do not clamp s here.
        w1 = torch.exp(-s1_t)
        w2 = torch.exp(-s2_t)
        w3 = torch.exp(-s3_t)

        l_ce_smooth = l_ce + smooth_eps_t
        l_kl_smooth = l_kl + smooth_eps_t
        l_conf_smooth = l_conf + smooth_eps_t
        l_fapi_smooth = (l_fapi + smooth_eps_t) if use_fapi_loss else torch.zeros((), device=l_ce.device, dtype=l_ce.dtype)

        weighted_kl = w1 * l_kl_smooth
        weighted_conflict = w2 * l_conf_smooth
        weighted_fapi = w3 * l_fapi_smooth if use_fapi_loss else torch.zeros((), device=l_ce.device, dtype=l_ce.dtype)
        uw_reg = s1_t + s2_t + s3_t if use_fapi_loss else s1_t + s2_t
        # ===== UQ CONTROL START: scale the regularizer and expose ratio monitoring =====
        # 1) uw_reg_scaled is the term added to the total loss.
        # 2) task_sum contains task terms only, excluding UQ regularization.
        # 3) uw_ratio lets the controller detect when UQ starts dominating optimization.
        uw_reg_coef_t = torch.as_tensor(uw_reg_coef, device=l_ce.device, dtype=l_ce.dtype)
        uw_reg_scaled = uw_reg_coef_t * uw_reg
        task_sum = l_ce_smooth + l_kl_smooth + l_conf_smooth + (l_fapi_smooth if use_fapi_loss else torch.zeros((), device=l_ce.device, dtype=l_ce.dtype))
        uw_ratio = torch.abs(uw_reg_scaled) / torch.clamp(task_sum.detach(), min=1e-8)
        # ===== UQ CONTROL END =====

        total = l_ce_smooth + weighted_kl + weighted_conflict + weighted_fapi + uw_reg_scaled

        loss_dict = {
            "L_total": total,
            "L_CE": l_ce.detach(),
            "L_CE_smooth": l_ce_smooth.detach(),
            "L_KL": l_kl.detach(),
            "L_KL_smooth": l_kl_smooth.detach(),
            "L_conflict": l_conf.detach(),
            "L_conflict_smooth": l_conf_smooth.detach(),
            "L_FAPI": l_fapi.detach(),
            "L_FAPI_smooth": l_fapi_smooth.detach(),
            "L_FAPI_raw": l_fapi_raw.detach(),
            "fapi_enabled": torch.tensor(1.0 if use_fapi_loss else 0.0, device=l_ce.device, dtype=l_ce.dtype),
            "fapi_grad_rms_mean": fapi_grad_rms_mean.detach() if isinstance(fapi_grad_rms_mean, torch.Tensor) else fapi_grad_rms_mean,
            "fapi_grads_is_none": torch.tensor(1.0 if fapi_grads_is_none else 0.0, device=l_ce.device),
            "L_weighted_KL": weighted_kl.detach(),
            "L_weighted_conflict": weighted_conflict.detach(),
            "L_weighted_fapi": weighted_fapi.detach(),
            "L_UW_reg_raw": uw_reg.detach(),
            "L_UW_reg": uw_reg_scaled.detach(),
            "uw_reg_coef": uw_reg_coef_t.detach(),
            "loss_smooth_eps": smooth_eps_t.detach(),
            "L_task_sum": task_sum.detach(),
            "UW_ratio": uw_ratio.detach(),
            "fapi_scale": fapi_scale_t.detach(),
            "num_fakes": conf_stats["num_fakes"].detach(),
            "fake_score_mean": conf_stats["fake_score_mean"].detach(),
            "fake_score_p90": conf_stats["fake_score_p90"].detach(),
            "conflict_active_ratio": conf_stats["conflict_active_ratio"].detach(),
            "effective_margin": conf_stats["effective_margin"].detach(),
            "ftext_requires_grad": torch.tensor(
                1.0 if f_text_requires_grad else 0.0,
                device=l_ce.device,
                dtype=l_ce.dtype,
            ),
            "lambda_penalty_mean": lambda_penalty_vec.to(device=l_ce.device, dtype=l_ce.dtype).mean().detach(),
            "lambda_penalty_clamped_mean": lambda_penalty_clamped_mean.detach(),
            "gate_mean": gate_mean.detach(),
            "dbg_ce_to_ftext_connected": torch.tensor(ce_to_ftext_connected, device=l_ce.device, dtype=l_ce.dtype),
            "dbg_ce_to_fat_connected": torch.tensor(ce_to_fat_connected, device=l_ce.device, dtype=l_ce.dtype),
            "dbg_ce_to_ftext_rms": ce_to_ftext_rms.detach(),
            "dbg_ce_to_fat_rms": ce_to_fat_rms.detach(),
        }
        return total, loss_dict


class MultiTaskLossComputer(nn.Module):
    """Config-driven loss computer used by the training loop."""

    def __init__(self, cfg):
        super().__init__()
        # Retain historical fixed weights only as initialization hints.
        init_lambda1 = float(getattr(cfg, "lambda1", 0.2))
        init_lambda2 = float(getattr(cfg, "lambda2", 0.1))
        init_lambda3 = float(getattr(cfg, "lambda3", 0.05))

        # Learn log(sigma^2) parameters for uncertainty weighting: exp(-s_i) * L_i + s_i.
        self.s1 = nn.Parameter(self._init_log_variance(init_lambda1))
        self.s2 = nn.Parameter(self._init_log_variance(init_lambda2))
        self.s3 = nn.Parameter(self._init_log_variance(init_lambda3))
        
        self.margin = float(getattr(cfg, "margin", 0.5))
        self.fake_label_threshold = float(getattr(cfg, "fake_label_threshold", 0.5))
        self.adaptive_margin = bool(getattr(cfg, "adaptive_margin", True))
        self.adaptive_margin_ratio = float(getattr(cfg, "adaptive_margin_ratio", 1.05))
        self.fapi_penalty_scale = float(getattr(cfg, "fapi_penalty_scale", 100.0))
        self.use_fapi_loss = bool(getattr(cfg, "use_fapi_loss", True))
        self.debug_fapi_graph = bool(getattr(cfg, "debug_fapi_graph", False))
        self.s_min = float(getattr(cfg, "uw_s_min", -1.5))
        self.s_max = float(getattr(cfg, "uw_s_max", 1.5))
        # ===== UQ CONTROL START =====
        self.uw_reg_coef = float(getattr(cfg, "uw_reg_coef", 1.0))
        # ===== UQ CONTROL END =====
        self.loss_smooth_eps = float(getattr(cfg, "loss_smooth_eps", 1e-4))

    @staticmethod
    def _init_log_variance(init_weight: float) -> torch.Tensor:
        # Start at log-variance 0 so exp(-s)=1 and no task is favored initially.
        _ = init_weight
        return torch.tensor(0.0, dtype=torch.float32)

    def _effective_weight(self, s: torch.Tensor) -> torch.Tensor:
        return torch.exp(-s)

    def forward(self, outputs: dict, batch: dict, lambda_penalty_vec: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return MultiTaskLosses.compute_total_loss(
            outputs=outputs,
            batch=batch,
            lambda_penalty_vec=lambda_penalty_vec,
            s1=self.s1,
            s2=self.s2,
            s3=self.s3,
            s_min=self.s_min,
            s_max=self.s_max,
            uw_reg_coef=self.uw_reg_coef,
            loss_smooth_eps=self.loss_smooth_eps,
            fapi_penalty_scale=self.fapi_penalty_scale,
            use_fapi_loss=self.use_fapi_loss,
            margin=self.margin,
            fake_label_threshold=self.fake_label_threshold,
            adaptive_margin=self.adaptive_margin,
            adaptive_margin_ratio=self.adaptive_margin_ratio,
            debug_fapi_graph=self.debug_fapi_graph,
        )

    def get_lambda_report(self) -> dict:
        """Return current learnable lambda values for training logs."""
        with torch.no_grad():
            s1_raw = float(self.s1.item())
            s2_raw = float(self.s2.item())
            s3_raw = float(self.s3.item())
            s1 = s1_raw
            s2 = s2_raw
            s3 = s3_raw
            lambda1 = float(self._effective_weight(self.s1).item())
            lambda2 = float(self._effective_weight(self.s2).item())
            lambda3 = float(self._effective_weight(self.s3).item())
            return {
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "s1_raw": s1_raw,
                "s2_raw": s2_raw,
                "s3_raw": s3_raw,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "lambda3": lambda3,
                "lambda3_active": lambda3 if self.use_fapi_loss else 0.0,
                "fapi_enabled": 1.0 if self.use_fapi_loss else 0.0,
                "uw_reg_coef": float(self.uw_reg_coef),
                "loss_smooth_eps": float(self.loss_smooth_eps),
            }
