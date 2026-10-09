"""Uniform-referenced inverse relations with explicit frame and token masks."""

from __future__ import annotations

import math

import torch
from torch import nn


class UniformInverseRelation(nn.Module):
    """Apply Linear or Log U-ICA independently to each attention head.

    Padding never contributes to the uniform reference, inverse mass, TV, or
    pooling. Empty-content samples bypass softmax even if special tokens exist.
    The output is a concatenation of masked, mean-pooled head representations;
    there is no output bias, residual path, or normalization that erases TV.
    """

    def __init__(self, audio_dim: int, text_dim: int = 768,
                 hidden_dim: int = 256, num_heads: int = 8,
                 variant: str = "linear", kappa: float = 5.0):
        super().__init__()
        if variant not in {"linear", "log"}:
            raise ValueError("variant must be 'linear' or 'log'")
        if min(audio_dim, text_dim, hidden_dim, num_heads) <= 0 or hidden_dim % num_heads:
            raise ValueError("positive hidden_dim must be divisible by num_heads")
        if not math.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.variant = variant
        self.kappa = float(kappa)
        self.query = nn.Linear(audio_dim, hidden_dim, bias=False)
        self.key = nn.Linear(text_dim, hidden_dim, bias=False)
        self.value = nn.Linear(text_dim, hidden_dim, bias=False)

    def forward(self, affective_features: torch.Tensor, audio_mask: torch.Tensor,
                text_features: torch.Tensor, text_mask: torch.Tensor,
                text_present: torch.Tensor) -> dict[str, torch.Tensor]:
        if affective_features.ndim != 3 or text_features.ndim != 3:
            raise ValueError("affective_features and text_features must have shape [B,T,D]")
        batch, audio_steps, _ = affective_features.shape
        text_batch, text_steps, _ = text_features.shape
        if batch != text_batch or tuple(audio_mask.shape) != (batch, audio_steps):
            raise ValueError("audio_mask or feature batch shape mismatch")
        if tuple(text_mask.shape) != (batch, text_steps) or text_present.numel() != batch:
            raise ValueError("text_mask or text_present shape mismatch")
        device = affective_features.device
        am = audio_mask.to(device=device, dtype=torch.bool)
        tm = text_mask.to(device=device, dtype=torch.bool)
        present = text_present.to(device=device, dtype=torch.bool).reshape(batch)
        tm = tm & present[:, None]

        query = self.query(affective_features).reshape(batch, audio_steps, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(text_features).reshape(batch, text_steps, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(text_features).reshape(batch, text_steps, self.num_heads, self.head_dim).transpose(1, 2)

        # Even an entirely empty batch has a differentiable zero, with no 0/0 or
        # all-masked softmax in its graph. This also keeps projection grads defined.
        zero = (query.float().sum((1, 2, 3)) + key.float().sum((1, 2, 3))
                + value.float().sum((1, 2, 3))) * 0.0
        relation = zero[:, None].expand(batch, self.hidden_dim)
        strength = zero
        clipping_all = zero
        clipping_positive = zero
        positive_count = torch.zeros(batch, dtype=torch.long, device=device)
        clipped_count = torch.zeros_like(positive_count)
        valid_position_count = torch.zeros_like(positive_count)
        active = (tm.any(dim=1) & am.any(dim=1)).nonzero(as_tuple=True)[0]

        if active.numel():
            # Disable the surrounding autocast for logits and all reductions.
            # Casting tensors alone is insufficient: autocast would recast matmul.
            with torch.autocast(device_type=device.type, enabled=False):
                q = query[active].float()
                k = key[active].float()
                v = value[active].float()
                active_tm = tm[active]
                active_am = am[active]
                key_valid = active_tm[:, None, None, :]
                frame_valid = active_am[:, None, :, None]
                position_valid = (key_valid & frame_valid).expand(-1, self.num_heads, -1, -1)
                lengths = active_tm.sum(dim=1).float()[:, None, None, None]
                logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                masked_logits = logits.masked_fill(~key_valid, -torch.inf)
                attention = torch.softmax(masked_logits, dim=-1).masked_fill(~key_valid, 0.0)
                uniform = lengths.reciprocal()
                difference = (attention - uniform).masked_fill(~key_valid, 0.0)
                row_strength = 0.5 * difference.abs().sum(dim=-1, keepdim=True)
                deficit = (-difference).clamp_min(0.0)

                if self.variant == "linear":
                    # S * normalize(1-L*A) is algebraically deficit; evaluate it
                    # directly to avoid division near uniform attention.
                    context = torch.matmul(deficit, v)
                    raw_score = deficit * lengths
                    clipped = torch.zeros_like(position_valid)
                else:
                    baseline = torch.logsumexp(masked_logits, dim=-1, keepdim=True) - lengths.log()
                    raw_score = (baseline - logits).clamp_min(0.0).masked_fill(~key_valid, 0.0)
                    clipped = (raw_score > self.kappa) & position_valid
                    score = raw_score.clamp_max(self.kappa).masked_fill(~key_valid, 0.0)
                    mass = score.sum(dim=-1, keepdim=True)
                    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
                    weights = score / safe_mass
                    context = row_strength * torch.matmul(weights, v)

                context = context.masked_fill(~frame_valid, 0.0)
                frame_count = active_am.sum(dim=1).float()
                pooled_heads = context.sum(dim=2) / frame_count[:, None, None]
                active_relation = pooled_heads.reshape(active.numel(), self.hidden_dim)
                active_strength = row_strength.masked_fill(~frame_valid, 0.0).sum((1, 2, 3)) / (frame_count * self.num_heads)
                n_positive = ((raw_score > 0) & position_valid).sum((1, 2, 3))
                n_clipped = clipped.sum((1, 2, 3))
                n_valid = position_valid.sum((1, 2, 3))
                active_clipping_all = n_clipped.float() / n_valid.clamp_min(1)
                active_clipping_positive = n_clipped.float() / n_positive.clamp_min(1)

            relation = relation.index_copy(0, active, active_relation)
            strength = strength.index_copy(0, active, active_strength)
            clipping_all = clipping_all.index_copy(0, active, active_clipping_all)
            clipping_positive = clipping_positive.index_copy(0, active, active_clipping_positive)
            positive_count = positive_count.index_copy(0, active, n_positive)
            clipped_count = clipped_count.index_copy(0, active, n_clipped)
            valid_position_count = valid_position_count.index_copy(0, active, n_valid)

        return {"z_rel": relation, "strength": strength,
                "clipping_all": clipping_all, "clipping_positive": clipping_positive,
                "positive_count": positive_count, "clipped_count": clipped_count,
                "valid_position_count": valid_position_count,
                "text_present": present, "valid_text_tokens": tm.sum(dim=1),
                "valid_audio_frames": am.sum(dim=1)}
