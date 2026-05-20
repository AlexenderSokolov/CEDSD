import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import os
import json


class RCIMetrics:
    """RCI 相关计算集合类。

    目标：将“原始冲突强度 RCI”计算逻辑集中管理，便于审计、复现与扩展。
    核心思路：
    1) 先从 cross attention 得到原始残差 R = ReLU(J - P)
    2) 再按有效 token 掩码做区域约束
    3) 最后做长度归一化得到可比较分数
    """

    @staticmethod
    def _expand_query_key_mask(attention_mask: torch.Tensor, heads: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Build [B, H, L_q, L_k] valid mask from [B, L_t].

        attention_mask: [B, L_t]
        -> q_mask: [B, 1, L_t, 1]
        -> k_mask: [B, 1, 1, L_t]
        -> valid:  [B, H, L_t, L_t]
        """
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must be [B, L_t]")
        # Implementation detail.
        q_mask = attention_mask.to(device=device, dtype=torch.bool).unsqueeze(1).unsqueeze(-1)
        k_mask = attention_mask.to(device=device, dtype=torch.bool).unsqueeze(1).unsqueeze(1)
        valid = q_mask & k_mask
        return valid.expand(-1, heads, -1, -1).to(dtype=dtype)

    @staticmethod
    def _broadcast_baseline(
        baseline: torch.Tensor,
        B: int,
        H: int,
        L_q: int,
        L_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """将任意可广播基线张量统一展开为 [B, H, L_q, L_k]。"""
        J = baseline.detach().to(device=device, dtype=dtype)
        if J.numel() == 1:
            return J.reshape(1, 1, 1, 1).expand(B, H, L_q, L_k)
        if J.dim() == 1 and J.size(0) == B:
            return J.view(B, 1, 1, 1).expand(B, H, L_q, L_k)
        if J.dim() == 2 and J.size(0) == B and J.size(1) == H:
            return J.view(B, H, 1, 1).expand(B, H, L_q, L_k)
        if J.dim() == 4:
            return J.expand(B, H, L_q, L_k)
        raise ValueError(f"unsupported baseline shape: {tuple(J.shape)}")

    @staticmethod
    def _resolve_baseline(cross: torch.Tensor, outputs: dict, baseline_mode: str, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return baseline J and raw residual R with shape [B, H, L_q, L_k].

        两种 baseline_mode 的用途：
        1) global:
            - 使用单一全局基线 J（所有 head 共享一个阈值）。
            - 适合做稳定主指标、跨批次对比、论文主表报告。
        2) headwise:
            - 每个 head 有独立基线 J_h。
            - 适合做多头差异分析、诊断某些头是否过于敏感/迟钝。

        设计上优先读取 runtime 已输出的残差与基线，保证“前向证据”和“解释指标”一致；
        若缺失这些字段，再按 beta/beta_head 回退重算。
        """
        B, H, L_q, L_k = cross.shape

        if baseline_mode == "global":
            if "inverse_raw_residual_global" in outputs and "baseline_global" in outputs:
                R = outputs["inverse_raw_residual_global"].detach().float().to(device=cross.device)
                J = RCIMetrics._broadcast_baseline(
                    baseline=outputs["baseline_global"],
                    B=B,
                    H=H,
                    L_q=L_q,
                    L_k=L_k,
                    device=cross.device,
                    dtype=cross.dtype,
                )
                return J, R

            beta = outputs.get("beta", None)
            if beta is None:
                raise KeyError("global baseline requires 'beta' or ('baseline_global' + 'inverse_raw_residual_global')")
            if not torch.is_tensor(beta):
                beta = torch.tensor(beta, device=cross.device, dtype=cross.dtype)
            # Implementation detail.
            J_scalar = torch.sigmoid(beta.detach().float().to(device=cross.device)).reshape(1, 1, 1, 1) + eps
            J = J_scalar.expand(B, H, L_q, L_k)
            # Implementation detail.
            R = torch.relu(J - cross)
            return J, R

        if baseline_mode == "headwise":
            if "inverse_raw_residual_headwise" in outputs and "baseline_headwise" in outputs:
                R = outputs["inverse_raw_residual_headwise"].detach().float().to(device=cross.device)
                J = RCIMetrics._broadcast_baseline(
                    baseline=outputs["baseline_headwise"],
                    B=B,
                    H=H,
                    L_q=L_q,
                    L_k=L_k,
                    device=cross.device,
                    dtype=cross.dtype,
                )
                return J, R

            beta_head = outputs.get("beta_head", None)
            if beta_head is None:
                raise KeyError("headwise baseline requires 'beta_head' or ('baseline_headwise' + 'inverse_raw_residual_headwise')")
            if not torch.is_tensor(beta_head):
                beta_head = torch.tensor(beta_head, device=cross.device, dtype=cross.dtype)
            # Implementation detail.
            J_head = torch.sigmoid(beta_head.detach().float().to(device=cross.device)).view(1, -1, 1, 1) + eps
            if J_head.size(1) != H:
                raise ValueError("beta_head size does not match num_heads")
            J = J_head.expand(B, H, L_q, L_k)
            R = torch.relu(J - cross)
            return J, R

        raise ValueError("baseline_mode must be one of: 'global', 'headwise'")

    @staticmethod
    def evaluate(
        outputs: dict,
        attention_mask: torch.Tensor | None = None,
        baseline_mode: str = "global",
        eps: float = 1e-9,
    ) -> dict:
        """Compute per-sample Raw Conflict Intensity (RCI).

        RCI definition:
        RCI = (1 / (H * Lq_eff * Lk_eff)) * sum_{h,t,i} (r_{h,t,i} / J)
        where r = ReLU(J - p_cross), and J comes from global/headwise baseline.

        计算流程（逐样本）：
        1) 取标准交叉注意力 P = outputs["att_cross_weights"]。
        2) 按 baseline_mode 获得基线 J 与原始残差 R。
        3) 若有 attention_mask，则只在有效 query-key 区域累加，避免 padding 污染。
        4) 计算分子：sum(R / J)；计算分母：H * Lq_eff * Lk_eff。
        5) 输出每个样本的 RCI 与 batch 平均值。

        两种模式的推荐使用场景：
        - baseline_mode="global":
            用于主流程评分、版本回归、跨模型横向对比（更稳、波动更小）。
        - baseline_mode="headwise":
            用于解释性诊断与研究分析（更细粒度，可发现头间异质性）。
        """
        if "att_cross_weights" not in outputs:
            raise KeyError("outputs must contain 'att_cross_weights'")

        cross = outputs["att_cross_weights"].detach().float()
        B, H, L_q, L_k = cross.shape

        J, raw_residual = RCIMetrics._resolve_baseline(cross, outputs, baseline_mode=baseline_mode, eps=eps)

        if attention_mask is None:
            # Implementation detail.
            attention_mask = outputs.get("attention_mask", None)

        if attention_mask is not None:
            valid = RCIMetrics._expand_query_key_mask(attention_mask, H, dtype=cross.dtype, device=cross.device)
            # Implementation detail.
            raw_residual = raw_residual * valid
            # Implementation detail.
            q_eff = attention_mask.to(device=cross.device, dtype=cross.dtype).sum(dim=1).clamp_min(1.0)
            k_eff = q_eff
        else:
            valid = torch.ones_like(raw_residual)
            q_eff = torch.full((B,), float(L_q), device=cross.device, dtype=cross.dtype)
            k_eff = torch.full((B,), float(L_k), device=cross.device, dtype=cross.dtype)

        # Implementation detail.
        numerator = (raw_residual / (J + eps) * valid).sum(dim=(1, 2, 3))
        denom = (float(H) * q_eff * k_eff).clamp_min(1.0)
        rci = numerator / denom

        return {
            "name": "RCI",
            "mode": baseline_mode,
            "score_per_sample": rci,
            "score_mean": float(rci.mean().item()),
            "meta": {
                "batch_size": int(B),
                "num_heads": int(H),
                "L_q": int(L_q),
                "L_k": int(L_k),
            },
        }

    @staticmethod
    def evaluate_dual(outputs: dict, attention_mask: torch.Tensor | None = None, eps: float = 1e-9) -> dict:
        """Compute both global/headwise RCI in one call.

        该接口用于直接做“双基线对照实验”：
        - global: 稳定主指标
        - headwise: 诊断细粒度差异
        - delta_mean: 两者平均差值，可快速判断“头级个性化阈值”是否显著改变冲突强度估计。
        """
        global_res = RCIMetrics.evaluate(outputs, attention_mask=attention_mask, baseline_mode="global", eps=eps)
        headwise_res = RCIMetrics.evaluate(outputs, attention_mask=attention_mask, baseline_mode="headwise", eps=eps)
        return {
            "global": global_res,
            "headwise": headwise_res,
            "delta_mean": headwise_res["score_mean"] - global_res["score_mean"],
        }


class TPMMetrics:
    """TPM 计算类（准备阶段）。

    先提供可运行的基础版 Top-k 峰值统计，后续可扩展到完整的 value-aware TPM。
    """

    @staticmethod
    def _resolve_att_inversed_weights(outputs: dict) -> torch.Tensor:
        if "att_inversed_weights" not in outputs:
            raise KeyError("outputs must contain 'att_inversed_weights' for TPM")
        return outputs["att_inversed_weights"].detach().float()  # [B, H, L_q, L_k]

    @staticmethod
    def _resolve_valid_mask(weights: torch.Tensor, outputs: dict, attention_mask: torch.Tensor | None) -> torch.Tensor:
        # Implementation detail.
        # Implementation detail.
        B, H, _, _ = weights.shape
        if attention_mask is None:
            attention_mask = outputs.get("attention_mask", None)
        if attention_mask is None:
            return torch.ones_like(weights)
        return RCIMetrics._expand_query_key_mask(attention_mask, H, dtype=weights.dtype, device=weights.device)

    @staticmethod
    def evaluate(
        outputs: dict,
        attention_mask: torch.Tensor | None = None,
        topk_ratio: float = 0.1,
        mode: str = "basic",
        value_norms: torch.Tensor | None = None,
    ) -> dict:
        """统一 TPM 入口。

        mode:
        - basic: 仅使用 att_inversed_weights 的 Top-k 峰值统计。
        - value_aware: 使用 a=q*||v|| 的 Value-aware 统计。
        """
        if mode == "basic":
            return TPMMetrics._evaluate_basic(
                outputs=outputs,
                attention_mask=attention_mask,
                topk_ratio=topk_ratio,
            )

        if mode == "value_aware":
            if value_norms is None:
                raise ValueError("value_norms is required when mode='value_aware'")
            return TPMMetrics._evaluate_value_aware(
                outputs=outputs,
                value_norms=value_norms,
                attention_mask=attention_mask,
                topk_ratio=topk_ratio,
            )

        raise ValueError("mode must be one of: 'basic', 'value_aware'")

    @staticmethod
    def _evaluate_basic(
        outputs: dict,
        attention_mask: torch.Tensor | None = None,
        topk_ratio: float = 0.1,
    ) -> dict:
        """基础 TPM：基于 att_inversed_weights 的 Top-k 质量和。"""
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be in (0, 1]")

        inv = TPMMetrics._resolve_att_inversed_weights(outputs)
        B, H, L_q, L_k = inv.shape
        valid = TPMMetrics._resolve_valid_mask(inv, outputs, attention_mask)
        inv = inv * valid

        # Implementation detail.
        k = max(1, int(topk_ratio * L_k))
        topk_vals = torch.topk(inv, k=k, dim=-1).values  # [B, H, L_q, k]
        # Implementation detail.
        tpm = topk_vals.sum(dim=-1).mean(dim=(1, 2))

        return {
            "name": "TPM",
            "mode": "basic",
            "score_per_sample": tpm,
            "score_mean": float(tpm.mean().item()),
            "topk_ratio": float(topk_ratio),
            "topk_k": int(k),
            "meta": {
                "batch_size": int(B),
                "num_heads": int(H),
                "L_q": int(L_q),
                "L_k": int(L_k),
            },
        }

    @staticmethod
    def _evaluate_value_aware(
        outputs: dict,
        value_norms: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        topk_ratio: float = 0.1,
    ) -> dict:
        """Value-aware TPM 入口。

        参数 `value_norms` 期望形状为 [B, H, L_k] 或 [B, 1, L_k]，
        其值应为对应 key 位置的 Value 范数（例如 L1/L2）。
        """
        inv = TPMMetrics._resolve_att_inversed_weights(outputs)
        B, H, _, L_k = inv.shape

        if value_norms.dim() == 3 and value_norms.size(0) == B and value_norms.size(-1) == L_k:
            if value_norms.size(1) == 1:
                value_norms = value_norms.expand(B, H, L_k)
            elif value_norms.size(1) != H:
                raise ValueError("value_norms head dimension must be 1 or match num_heads")
        else:
            raise ValueError("value_norms must be [B,H,L_k] or [B,1,L_k]")

        # Implementation detail.
        score = inv * value_norms.unsqueeze(2).to(device=inv.device, dtype=inv.dtype)
        valid = TPMMetrics._resolve_valid_mask(score, outputs, attention_mask)
        score = score * valid

        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be in (0, 1]")
        k = max(1, int(topk_ratio * L_k))
        topk_vals = torch.topk(score, k=k, dim=-1).values
        tpm = topk_vals.sum(dim=-1).mean(dim=(1, 2))

        return {
            "name": "TPM",
            "mode": "value_aware",
            "score_per_sample": tpm,
            "score_mean": float(tpm.mean().item()),
            "topk_ratio": float(topk_ratio),
            "topk_k": int(k),
            "meta": {
                "batch_size": int(B),
                "num_heads": int(H),
                "L_k": int(L_k),
            },
        }


class InterpretabilityScorer:
    """解释性总评类：汇总 RCI 与 TPM 并输出总评分类。

    默认权重：
    - RCI: 0.6
    - TPM: 0.4
    可靠性分级：
    - High: [0.75, 1.0]
    - Medium: [0.45, 0.75)
    - Low: [0.0, 0.45)
    """

    @staticmethod
    def _to_unit_interval(x: torch.Tensor) -> torch.Tensor:
        # Implementation detail.
        return x.clamp(min=0.0, max=1.0)

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.75:
            return "High"
        if score >= 0.45:
            return "Medium"
        return "Low"

    @staticmethod
    def evaluate(
        outputs: dict,
        attention_mask: torch.Tensor | None = None,
        rci_mode: str = "global",
        tpm_mode: str = "basic",
        topk_ratio: float = 0.1,
        value_norms: torch.Tensor | None = None,
        rci_weight: float = 0.6,
        tpm_weight: float = 0.4,
        eps: float = 1e-9,
    ) -> dict:
        if rci_weight < 0 or tpm_weight < 0:
            raise ValueError("rci_weight and tpm_weight must be non-negative")
        weight_sum = rci_weight + tpm_weight
        if weight_sum <= 0:
            raise ValueError("sum of weights must be > 0")

        # Implementation detail.
        rw = rci_weight / weight_sum
        tw = tpm_weight / weight_sum

        rci_out = RCIMetrics.evaluate(
            outputs=outputs,
            attention_mask=attention_mask,
            baseline_mode=rci_mode,
            eps=eps,
        )
        tpm_out = TPMMetrics.evaluate(
            outputs=outputs,
            attention_mask=attention_mask,
            topk_ratio=topk_ratio,
            mode=tpm_mode,
            value_norms=value_norms,
        )

        rci_score = InterpretabilityScorer._to_unit_interval(rci_out["score_per_sample"])
        tpm_score = InterpretabilityScorer._to_unit_interval(tpm_out["score_per_sample"])
        total = rw * rci_score + tw * tpm_score

        per_sample_level = [InterpretabilityScorer._label(float(x.item())) for x in total]
        total_mean = float(total.mean().item())

        return {
            "weights": {"rci": rw, "tpm": tw},
            "components": {
                "rci": {
                    "mode": rci_out["mode"],
                    "score_mean": rci_out["score_mean"],
                    "score_per_sample": rci_out["score_per_sample"],
                },
                "tpm": {
                    "mode": tpm_out["mode"],
                    "score_mean": tpm_out["score_mean"],
                    "score_per_sample": tpm_out["score_per_sample"],
                },
            },
            "total": {
                "score_mean": total_mean,
                "score_per_sample": total,
                "level": InterpretabilityScorer._label(total_mean),
                "level_per_sample": per_sample_level,
            },
        }


class AttentionVisualization:
    """三层可视化口径：全头聚合、最冲突头、头间分歧。"""

    @staticmethod
    def _build_valid_mask(cross_map: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """返回 [H, L_q, L_k] 的有效区域掩码。"""
        H, _, _ = cross_map.shape
        if attention_mask is None:
            return torch.ones_like(cross_map)
        valid = RCIMetrics._expand_query_key_mask(
            attention_mask.unsqueeze(0),
            heads=H,
            dtype=cross_map.dtype,
            device=cross_map.device,
        )
        return valid.squeeze(0)

    @staticmethod
    def _head_conflict_scores(outputs: dict, sample_idx: int, valid_mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """按 head 计算冲突强度分数，用于挑选最冲突头。"""
        cross_map = outputs["att_cross_weights"][sample_idx].detach().float()

        if "inverse_raw_residual_headwise" in outputs:
            residual = outputs["inverse_raw_residual_headwise"][sample_idx].detach().float()
        elif "inverse_raw_residual_global" in outputs:
            residual = outputs["inverse_raw_residual_global"][sample_idx].detach().float()
        else:
            # Implementation detail.
            if "baseline_headwise" in outputs:
                j_raw = outputs["baseline_headwise"].detach().float().to(cross_map.device)
                if j_raw.dim() == 4 and j_raw.size(0) > 1:
                    j_raw = j_raw[sample_idx:sample_idx + 1]
                J = RCIMetrics._broadcast_baseline(
                    baseline=j_raw,
                    B=1,
                    H=cross_map.size(0),
                    L_q=cross_map.size(1),
                    L_k=cross_map.size(2),
                    device=cross_map.device,
                    dtype=cross_map.dtype,
                ).squeeze(0)
            elif "baseline_global" in outputs:
                j_raw = outputs["baseline_global"].detach().float().to(cross_map.device)
                if j_raw.dim() == 4 and j_raw.size(0) > 1:
                    j_raw = j_raw[sample_idx:sample_idx + 1]
                J = RCIMetrics._broadcast_baseline(
                    baseline=j_raw,
                    B=1,
                    H=cross_map.size(0),
                    L_q=cross_map.size(1),
                    L_k=cross_map.size(2),
                    device=cross_map.device,
                    dtype=cross_map.dtype,
                ).squeeze(0)
            else:
                beta = outputs.get("beta", torch.tensor([0.0], device=cross_map.device, dtype=cross_map.dtype))
                if not torch.is_tensor(beta):
                    beta = torch.tensor(beta, device=cross_map.device, dtype=cross_map.dtype)
                J = (torch.sigmoid(beta.detach().float()).reshape(1, 1, 1) + eps).expand_as(cross_map)
            residual = torch.relu(J - cross_map)

        residual = residual * valid_mask
        denom = valid_mask.sum(dim=(1, 2)).clamp_min(1.0)
        return residual.sum(dim=(1, 2)) / denom

    @staticmethod
    def plot_three_tier(
        outputs: dict,
        sample_idx: int = 0,
        attention_mask: torch.Tensor | None = None,
        topk_heads: int = 1,
        save_path: str | Path | None = None,
        show: bool = True,
    ) -> dict:
        """绘制三层可视化图并返回关键信息。

        第1层：全头聚合（mean over heads）
        第2层：最冲突头（按 head conflict score 选 Top-k）
        第3层：头间分歧（std over heads）
        """
        if "att_cross_weights" not in outputs or "att_inversed_weights" not in outputs or "align_weights" not in outputs:
            raise KeyError("outputs must contain: att_cross_weights, att_inversed_weights, align_weights")

        align_map = outputs["align_weights"][sample_idx].detach().float().cpu()      # [L_t, L_a]
        cross_map = outputs["att_cross_weights"][sample_idx].detach().float()         # [H, L_t, L_t]
        inverse_map = outputs["att_inversed_weights"][sample_idx].detach().float()      # [H, L_t, L_t]

        if attention_mask is None and "attention_mask" in outputs and outputs["attention_mask"] is not None:
            attention_mask = outputs["attention_mask"][sample_idx].detach().to(cross_map.device)

        valid_mask = AttentionVisualization._build_valid_mask(cross_map, attention_mask)
        head_scores = AttentionVisualization._head_conflict_scores(outputs, sample_idx, valid_mask)

        H = cross_map.size(0)
        topk = max(1, min(int(topk_heads), int(H)))
        top_idx = torch.topk(head_scores, k=topk).indices.tolist()
        lead_idx = int(top_idx[0])

        # Implementation detail.
        cross_mean = (cross_map * valid_mask).mean(dim=0).cpu()
        inverse_mean = (inverse_map * valid_mask).mean(dim=0).cpu()

        # Implementation detail.
        cross_top = (cross_map[lead_idx] * valid_mask[lead_idx]).cpu()
        inverse_top = (inverse_map[lead_idx] * valid_mask[lead_idx]).cpu()
        
        # Implementation detail.
        # Implementation detail.
        # Implementation detail.
        # Implementation detail.
        
        if "inverse_raw_residual_headwise" in outputs:
            residual_top = outputs["inverse_raw_residual_headwise"][sample_idx, lead_idx].detach().float().cpu()
        elif "inverse_raw_residual_global" in outputs:
            residual_top = outputs["inverse_raw_residual_global"][sample_idx, lead_idx].detach().float().cpu()
        else:
            # Implementation detail.
            cross_top_device = cross_map[lead_idx]
            if "baseline_headwise" in outputs:
                j_raw = outputs["baseline_headwise"].detach().float().to(cross_top_device.device)
                if j_raw.dim() == 4 and j_raw.size(0) > 1:
                    j_raw = j_raw[sample_idx:sample_idx + 1]
                j_full = RCIMetrics._broadcast_baseline(
                    baseline=j_raw,
                    B=1,
                    H=cross_map.size(0),
                    L_q=cross_map.size(1),
                    L_k=cross_map.size(2),
                    device=cross_top_device.device,
                    dtype=cross_top_device.dtype,
                ).squeeze(0)
                j_top = j_full[lead_idx]
            elif "baseline_global" in outputs:
                j_raw = outputs["baseline_global"].detach().float().to(cross_top_device.device)
                if j_raw.dim() == 4 and j_raw.size(0) > 1:
                    j_raw = j_raw[sample_idx:sample_idx + 1]
                j_full = RCIMetrics._broadcast_baseline(
                    baseline=j_raw,
                    B=1,
                    H=cross_map.size(0),
                    L_q=cross_map.size(1),
                    L_k=cross_map.size(2),
                    device=cross_top_device.device,
                    dtype=cross_top_device.dtype,
                ).squeeze(0)
                j_top = j_full[lead_idx]
            else:
                beta = outputs.get("beta", torch.tensor([0.0], device=cross_top_device.device, dtype=cross_top_device.dtype))
                if not torch.is_tensor(beta):
                    beta = torch.tensor(beta, device=cross_top_device.device, dtype=cross_top_device.dtype)
                j_top = (torch.sigmoid(beta.detach().float()).reshape(1) + 1e-9).view(1, 1)

            residual_top = torch.relu(j_top - cross_top_device).cpu()

        # Implementation detail.
        cross_std = (cross_map * valid_mask).std(dim=0).cpu()
        inverse_std = (inverse_map * valid_mask).std(dim=0).cpu()

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        # Row 1: Global Aggregation
        im00 = axes[0, 0].imshow(align_map, aspect="auto", origin="lower", cmap="viridis")
        axes[0, 0].set_title("Tier-1 Global: Soft Alignment")
        axes[0, 0].set_xlabel("Audio frame")
        axes[0, 0].set_ylabel("Text token")
        plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im01 = axes[0, 1].imshow(cross_mean, aspect="auto", origin="lower", cmap="Blues")
        axes[0, 1].set_title("Tier-1 Global: Cross(mean over heads)")
        axes[0, 1].set_xlabel("Key token")
        axes[0, 1].set_ylabel("Query token")
        plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im02 = axes[0, 2].imshow(inverse_mean, aspect="auto", origin="lower", cmap="magma")
        axes[0, 2].set_title("Tier-1 Global: Inverse(mean over heads)")
        axes[0, 2].set_xlabel("Key token")
        axes[0, 2].set_ylabel("Query token")
        plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # Row 2: Top Conflict Head
        im10 = axes[1, 0].imshow(cross_top, aspect="auto", origin="lower", cmap="Blues")
        axes[1, 0].set_title(f"Tier-2 Top Head: Cross(h={lead_idx})")
        axes[1, 0].set_xlabel("Key token")
        axes[1, 0].set_ylabel("Query token")
        plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im11 = axes[1, 1].imshow(inverse_top, aspect="auto", origin="lower", cmap="magma")
        axes[1, 1].set_title(f"Tier-2 Top Head: Inverse(h={lead_idx})")
        axes[1, 1].set_xlabel("Key token")
        axes[1, 1].set_ylabel("Query token")
        plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im12 = axes[1, 2].imshow(residual_top, aspect="auto", origin="lower", cmap="inferno")
        axes[1, 2].set_title(f"Tier-2 Top Head: Raw Residual(h={lead_idx})")
        axes[1, 2].set_xlabel("Key token")
        axes[1, 2].set_ylabel("Query token")
        plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # Row 3: Head Disagreement
        im20 = axes[2, 0].imshow(cross_std, aspect="auto", origin="lower", cmap="Greens")
        axes[2, 0].set_title("Tier-3 Disagreement: Cross(std over heads)")
        axes[2, 0].set_xlabel("Key token")
        axes[2, 0].set_ylabel("Query token")
        plt.colorbar(im20, ax=axes[2, 0], fraction=0.046, pad=0.04)

        im21 = axes[2, 1].imshow(inverse_std, aspect="auto", origin="lower", cmap="Greens")
        axes[2, 1].set_title("Tier-3 Disagreement: Inverse(std over heads)")
        axes[2, 1].set_xlabel("Key token")
        axes[2, 1].set_ylabel("Query token")
        plt.colorbar(im21, ax=axes[2, 1], fraction=0.046, pad=0.04)

        head_idx = torch.arange(H).cpu()
        axes[2, 2].bar(head_idx.numpy(), head_scores.detach().cpu().numpy())
        axes[2, 2].set_title("Tier-3 Disagreement: Head Conflict Scores")
        axes[2, 2].set_xlabel("Head index")
        axes[2, 2].set_ylabel("Conflict score")

        plt.tight_layout()

        if save_path is None:
            out_dir = Path.cwd() / "outputs" / "attention_maps"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = out_dir / f"three_tier_attention_{ts}.png"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[INFO] 三层可视化已保存: {save_path}")

        backend_name = str(plt.get_backend()).lower()
        is_inline_backend = ("inline" in backend_name) or ("nbagg" in backend_name)
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if show and (is_inline_backend or has_display):
            plt.show()
        else:
            plt.close(fig)

        return {
            "save_path": str(save_path),
            "top_head_indices": top_idx,
            "top_head": lead_idx,
            "head_scores": head_scores.detach().cpu(),
        }


class InterpretabilityLogger:
    """解释性结果日志器：保存评分与可视化元信息。"""

    @staticmethod
    def _to_jsonable(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {k: InterpretabilityLogger._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [InterpretabilityLogger._to_jsonable(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    @staticmethod
    def save(
        score_report: dict,
        viz_report: dict,
        outputs: dict,
        save_dir: str | Path | None = None,
        prefix: str = "interpretability_report",
    ) -> dict:
        if save_dir is None:
            save_dir = Path.cwd() / "outputs" / "attention_maps"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = save_dir / f"{prefix}_{ts}.json"
        txt_path = save_dir / f"{prefix}_{ts}.txt"

        payload = {
            "timestamp": ts,
            "score_report": InterpretabilityLogger._to_jsonable(score_report),
            "visualization_report": InterpretabilityLogger._to_jsonable(viz_report),
            "runtime_meta": {
                "F_AT_shape": list(outputs["F_AT"].shape) if "F_AT" in outputs else None,
                "align_weights_shape": list(outputs["align_weights"].shape) if "align_weights" in outputs else None,
                "att_cross_weights_shape": list(outputs["att_cross_weights"].shape) if "att_cross_weights" in outputs else None,
                "att_inversed_weights_shape": list(outputs["att_inversed_weights"].shape) if "att_inversed_weights" in outputs else None,
                "beta": float(outputs["beta"].item()) if "beta" in outputs and torch.is_tensor(outputs["beta"]) else None,
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        lines = [
            f"timestamp: {ts}",
            f"rci_mean: {payload['score_report']['components']['rci']['score_mean']:.6f}",
            f"tpm_mean: {payload['score_report']['components']['tpm']['score_mean']:.6f}",
            f"total_mean: {payload['score_report']['total']['score_mean']:.6f}",
            f"total_level: {payload['score_report']['total']['level']}",
            f"viz_path: {payload['visualization_report'].get('save_path', '')}",
            f"top_heads: {payload['visualization_report'].get('top_head_indices', [])}",
            f"beta: {payload['runtime_meta']['beta']}",
        ]
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[INFO] 解释性日志(JSON): {json_path}")
        print(f"[INFO] 解释性日志(TXT): {txt_path}")
        return {
            "json_path": str(json_path),
            "txt_path": str(txt_path),
        }

""" 调用示例：
from attention_interpretability import InterpretabilityScorer

report = InterpretabilityScorer.evaluate(
    outputs=outputs,
    attention_mask=text_mask,
    rci_mode="global",
    tpm_mode="basic",
    topk_ratio=0.1,
    rci_weight=0.6,
    tpm_weight=0.4,
)

print(report["components"]["rci"]["score_mean"])
print(report["components"]["tpm"]["score_mean"])
print(report["total"]["score_mean"], report["total"]["level"])
"""
"""
from attention_interpretability import AttentionVisualization

viz = AttentionVisualization.plot_three_tier(
    outputs=outputs,
    sample_idx=0,
    attention_mask=text_mask[0],  # 可选；不传则尝试用 outputs["attention_mask"]
    topk_heads=3,
    show=False,
)
print(viz["save_path"], viz["top_head_indices"])
"""