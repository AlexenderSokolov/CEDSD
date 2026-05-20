import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- inverse-attention runtime block start --------------------
class InverseAttentionRuntime(nn.Module):
    """Unified wrapper for cross-modal inverse attention, forward passes, and visualization hooks."""

    class _CrossModalSoftAlignment(nn.Module):
        """
        Stage 1: softly align raw audio and text features into a shared temporal space.
        """
        def __init__(self, audio_dim, text_dim, hidden_dim, dropout=0.1):
            super().__init__()
            self.hidden_dim = hidden_dim

            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.W_q_align = nn.Linear(hidden_dim, hidden_dim, bias=False)
            # In cross-modal alignment, Q comes from text and K/V from audio, so K/V can share one projection.
            self.W_kv_align = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)

            # Multi-scale temporal smoothing with parallel 3/5/7 convolutions followed by 1x1 fusion.
            self.audio_temporal_convs = nn.ModuleList(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
                ]
            )
            self.audio_temporal_fuse = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=False)
            self.temporal_dropout = nn.Dropout(dropout)
            self.conv_norm = nn.LayerNorm(hidden_dim)
            self.attn_dropout = nn.Dropout(dropout)

        def _masked_softmax(self, logits, mask, dim=-1, eps=1e-9):
            # Use the minimum finite value to avoid NaNs on fully masked rows, then renormalize valid entries.
            min_value = torch.finfo(logits.dtype).min
            masked_logits = logits.masked_fill(~mask, min_value)
            probs = torch.softmax(masked_logits, dim=dim)
            probs = probs * mask.to(dtype=probs.dtype)
            denom = probs.sum(dim=dim, keepdim=True).clamp_min(eps)
            return probs / denom

        def forward(self, F_AE, F_text, audio_mask=None):
            F_AE_prime = self.audio_proj(F_AE)
            F_text_prime = self.text_proj(F_text)

            audio_t = F_AE_prime.transpose(1, 2).contiguous()
            multi_scale_feats = [conv(audio_t) for conv in self.audio_temporal_convs]
            F_AE_conv = self.audio_temporal_fuse(torch.cat(multi_scale_feats, dim=1)).transpose(1, 2).contiguous()
            F_AE_prime = self.conv_norm(F_AE_prime + self.temporal_dropout(F_AE_conv))

            Q_align = self.W_q_align(F_text_prime)
            K_align, V_align = self.W_kv_align(F_AE_prime).chunk(2, dim=-1)

            S_align = torch.matmul(Q_align, K_align.transpose(-1, -2)) / (self.hidden_dim ** 0.5)
            # -------------------- explicit audio mask for soft alignment --------------------
            if audio_mask is not None:
                # audio_mask: [B, L_a] -> [B, 1, L_a], broadcastable to S_align [B, L_t, L_a].
                align_mask = audio_mask.unsqueeze(1).to(dtype=torch.bool, device=S_align.device)
                attention_weights = self._masked_softmax(S_align, align_mask, dim=-1)
            else:
                attention_weights = torch.softmax(S_align, dim=-1)
            attention_weights = self.attn_dropout(attention_weights)
            # -------------------- explicit audio mask for soft alignment --------------------
            F_AE_aligned = torch.matmul(attention_weights, V_align)

            return F_AE_aligned, F_text_prime, attention_weights

    class _InverseAttentionMechanism(nn.Module):
        """
          Stage 2: inverse attention mechanism over aligned audio and text features.
        """
        def __init__(self, d_model, num_heads=8, dropout=0.1, eps=1e-6):
            super().__init__()
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads")

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.eps = eps

            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)

            # Maintain global and per-head baselines in parallel for interpretability analysis.
            self.beta = nn.Parameter(torch.tensor([0.0]))
            self.beta_head = nn.Parameter(torch.zeros(num_heads))
            # Dynamic baseline branch: generate sample-level J from pooled cross-modal features.
            dyn_hidden = max(32, d_model // 2)
            self.dynamic_beta_mlp = nn.Sequential(
                nn.Linear(d_model * 2, dyn_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dyn_hidden, 1 + num_heads),
            )
            # Sigmoid bounds the static/dynamic baseline mix in (0, 1).
            self.dynamic_beta_mix_logit = nn.Parameter(torch.tensor(0.0))

            self.dropout = nn.Dropout(dropout)
            self.output_proj = nn.Linear(d_model, d_model)
            self.layer_norm = nn.LayerNorm(d_model)

        def _expand_mask(self, mask, B, device):
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() != 4:
                raise ValueError("mask must be 2D, 3D, or 4D")

            if mask.size(0) != B:
                raise ValueError("mask batch dimension does not match the input")

            return mask.to(device=device, dtype=torch.bool)

        def _masked_softmax(self, logits, mask, dim=-1, eps=1e-9):
            # Avoid extra where/zeros_like allocations while keeping fully masked rows numerically stable.
            min_value = torch.finfo(logits.dtype).min
            masked_logits = logits.masked_fill(~mask, min_value)
            probs = torch.softmax(masked_logits, dim=dim)
            probs = probs * mask.to(dtype=probs.dtype)
            denom = probs.sum(dim=dim, keepdim=True).clamp_min(eps)
            return probs / denom

        def forward(self, F_AE_aligned, F_text_prime, mask=None):
            B, L_t, _ = F_AE_aligned.size()
            
            
            # Project to multi-head format [B, L_t, H, d_k], then transpose to [B, H, L_t, d_k].
            Q = self.W_q(F_AE_aligned).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(F_text_prime).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(F_text_prime).view(B, L_t, self.num_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            # -------------------- robust cross-attention mask --------------------
            if mask is not None:
                # mask: [B, L_t] -> [B, 1, 1, L_t], broadcastable to scores [B, H, L_t, L_t].
                expanded_mask = self._expand_mask(mask, B, scores.device)
                att_cross_weights = self._masked_softmax(scores, expanded_mask, dim=-1, eps=self.eps)
            else:
                att_cross_weights = torch.softmax(scores, dim=-1)
            att_cross_weights = self.dropout(att_cross_weights)
            # -------------------- robust cross-attention mask --------------------

            static_beta = torch.sigmoid(self.beta)
            static_beta_head = torch.sigmoid(self.beta_head).view(1, self.num_heads, 1, 1)

            pooled_audio = F_AE_aligned.mean(dim=1)
            pooled_text = F_text_prime.mean(dim=1)
            dynamic_beta_logits = self.dynamic_beta_mlp(torch.cat([pooled_audio, pooled_text], dim=-1))
            dynamic_beta = torch.sigmoid(dynamic_beta_logits[:, :1]).view(B, 1, 1, 1)
            dynamic_beta_head = torch.sigmoid(dynamic_beta_logits[:, 1:]).view(B, self.num_heads, 1, 1)

            mix = torch.sigmoid(self.dynamic_beta_mix_logit)
            adaptive_beta = (1.0 - mix) * static_beta.view(1, 1, 1, 1).expand(B, 1, 1, 1) + mix * dynamic_beta
            adaptive_beta_head = (1.0 - mix) * static_beta_head.expand(B, -1, -1, -1) + mix * dynamic_beta_head

            J = adaptive_beta + self.eps
            # Headwise residuals are interpretability-only; skip them in training to avoid O(H*L^2) overhead.
            need_headwise = (not self.training)
            if need_headwise:
                J_head = adaptive_beta_head + self.eps
            else:
                J_head = None

            # Global-baseline variant used by the forward path.
            raw_inv_residual = F.relu(J - att_cross_weights)
            att_inversed_weights = raw_inv_residual / (raw_inv_residual.sum(dim=-1, keepdim=True) + self.eps)
            # Headwise-baseline variant used only for interpretability comparisons.
            if need_headwise:
                raw_inv_residual_head = F.relu(J_head - att_cross_weights)
            else:
                raw_inv_residual_head = None

            context = torch.matmul(att_inversed_weights, V)
            context = context.transpose(1, 2).contiguous().view(B, L_t, self.d_model)

            F_AT = self.layer_norm(F_AE_aligned + self.output_proj(context))
            return (
                F_AT, # Inverse-attention fused features.
                att_inversed_weights, # Inverse attention weights used by the final output path.
                att_cross_weights, # Cross-attention weights used by interpretability and conflict scoring.
                static_beta, # Global static baseline scalar.
                static_beta_head, # Per-head static baseline scalars.
                J, # Dynamic mixed global baseline.
                J_head, # Dynamic mixed headwise baseline.
                raw_inv_residual, # Global-baseline raw residual matrix for interpretability.
                raw_inv_residual_head, # Headwise-baseline raw residual matrix for interpretability.
            )

    class _DualStreamInverseBlock(nn.Module):
        """
        Top-level cross-modal inverse-analysis block.
        Inputs are audio/text features; outputs are fused features and attention evidence.
        """
        def __init__(self, audio_dim, text_dim, hidden_dim=512, num_heads=8, dropout=0.1):
            super().__init__()
            self.aligner = InverseAttentionRuntime._CrossModalSoftAlignment(
                audio_dim, text_dim, hidden_dim, dropout=dropout
            )
            self.iam = InverseAttentionRuntime._InverseAttentionMechanism(
                hidden_dim, num_heads=num_heads, dropout=dropout
            )

        def forward(self, F_AE, F_text, attention_mask=None, audio_mask=None):
            # attention_mask comes from text extraction and audio_mask from emotion2vec voiced probabilities.
            F_AE_aligned, F_text_prime, align_weights = self.aligner(F_AE, F_text, audio_mask=audio_mask)
            (
                F_AT,
                att_inversed_weights,  # [B, num_heads, L_t, L_t], inverse attention weights.
                att_cross_weights,     # [B, num_heads, L_t, L_t], cross-attention weights.
                beta,
                beta_head,
                J_global,
                J_headwise,
                raw_inv_residual,
                raw_inv_residual_headwise,
            ) = self.iam(
                F_AE_aligned, F_text_prime, mask=attention_mask
            )
            outputs = {
                "F_AT": F_AT,
                "att_inversed_weights": att_inversed_weights,
                "att_cross_weights": att_cross_weights,
                # Keep the global-baseline residual for backward compatibility.
                "inverse_raw_residual": raw_inv_residual,
                # Expose both baseline/residual variants for paired RCI calculations.
                "inverse_raw_residual_global": raw_inv_residual,
                "align_weights": align_weights,
                "beta": beta,
                "beta_head": beta_head,
                "baseline_global": J_global,
                "attention_mask": attention_mask,
            }
            if raw_inv_residual_headwise is not None:
                outputs["inverse_raw_residual_headwise"] = raw_inv_residual_headwise
            if J_headwise is not None:
                outputs["baseline_headwise"] = J_headwise
            return outputs

    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        # Lazily build and reuse the internal block once feature dimensions are known.
        self._model = None
        self._audio_dim = None
        self._text_dim = None

    def _ensure_model(self, F_AE, F_text):
        audio_dim = int(F_AE.size(-1))
        text_dim = int(F_text.size(-1))
        need_rebuild = (
            self._model is None
            or self._audio_dim != audio_dim
            or self._text_dim != text_dim
        )
        if need_rebuild:
            self._model = self._DualStreamInverseBlock(
                audio_dim=audio_dim,
                text_dim=text_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            ).to(F_AE.device)
            self._audio_dim = audio_dim
            self._text_dim = text_dim
        else:
            model_ref = self._model
            assert model_ref is not None
            self._model = model_ref.to(F_AE.device)
        assert self._model is not None
    
    # Public run interface.
    def run(self, F_AE, F_text, text_mask, audio_mask=None,is_train=False):
        """Build if needed, run forward, and return attention evidence."""
        self._ensure_model(F_AE, F_text)
        assert self._model is not None
        
        # Select train/eval mode from the caller's requested phase.
        if is_train:
            self._model.train()
        else:
            self._model.eval()
        with torch.set_grad_enabled(is_train):
            outputs = self._model(
                F_AE,
                F_text,
                attention_mask=text_mask,
                audio_mask=audio_mask,
            )
        return outputs

    @staticmethod
    def make_real_inputs(pack):
        return {
            "wav_list": pack["wav_list"],
            "texts": pack["texts"],
            "audio_mask": pack["audio_mask"],
            "e2v_scores": pack["e2v_scores"],
        }

    @staticmethod
    def plot(outputs, real_inputs=None, sample_idx=0, head_idx=0, save_path=None, show=True):
        """Compatibility wrapper around the three-tier interpretability visualization.

        The `head_idx` argument is retained for legacy callers but no longer controls plotting.
        """
        from attention_interpretability import AttentionVisualization

        viz_ret = AttentionVisualization.plot_three_tier(
            outputs=outputs,
            sample_idx=sample_idx,
            attention_mask=None,
            topk_heads=1,
            save_path=save_path,
            show=show,
        )
        return viz_ret["save_path"]

    @staticmethod
    def smoke_test(device=None):
        """Lightweight self-test that does not require external models."""
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        runtime = InverseAttentionRuntime(hidden_dim=128, num_heads=8, dropout=0.0)
        B, L_a, L_t, D_a, D_t = 2, 16, 10, 896, 768
        F_AE = torch.randn(B, L_a, D_a, device=dev)
        F_text = torch.randn(B, L_t, D_t, device=dev)
        text_mask = torch.ones(B, L_t, dtype=torch.bool, device=dev)
        audio_mask = torch.ones(B, L_a, dtype=torch.bool, device=dev)
        outputs = runtime.run(F_AE, F_text, text_mask, audio_mask=audio_mask)

        assert outputs["F_AT"].shape == (B, L_t, 128)
        assert outputs["align_weights"].shape == (B, L_t, L_a)
        assert outputs["att_cross_weights"].shape == (B, 8, L_t, L_t)
        assert outputs["att_inversed_weights"].shape == (B, 8, L_t, L_t)
        return True
# -------------------- inverse-attention runtime block end --------------------
