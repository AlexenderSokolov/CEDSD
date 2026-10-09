"""The six fixed complete-training arms of the original UICA closeout.

The acoustic path is inherited from the frozen original implementation.  This
module adds the prescribed capacity controls, plain CA and relation-component
interventions; it never changes the original Linear/Log full representation.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from .model import Detector, _initialization_seed
from .operators import UniformInverseRelation


class _CloseoutRelation(UniformInverseRelation):
    """Original Q/K/V projections, with the actual-weight decomposition."""

    def __init__(self, variant: str):
        if variant not in {"ca", "linear", "log"}:
            raise ValueError("unknown attention rule")
        super().__init__(1024, 768, 256, 8,
                         variant="linear" if variant == "ca" else variant,
                         kappa=5.0)
        self.variant = variant

    def forward(self, affective_features, audio_mask, text_features,
                text_mask, text_present):
        if affective_features.ndim != 3 or text_features.ndim != 3:
            raise ValueError("relation features must have shape [B,T,D]")
        batch, audio_steps, audio_dim = affective_features.shape
        text_batch, text_steps, text_dim = text_features.shape
        if audio_dim != 1024 or text_dim != 768:
            raise ValueError("closeout requires E2V=1024 and BERT=768")
        if batch != text_batch or tuple(audio_mask.shape) != (batch, audio_steps):
            raise ValueError("audio mask or feature batch shape mismatch")
        if tuple(text_mask.shape) != (batch, text_steps) or text_present.numel() != batch:
            raise ValueError("text mask or presence shape mismatch")
        device = affective_features.device
        if text_features.device != device:
            raise ValueError("audio and text features must share a device")
        am = audio_mask.to(device=device, dtype=torch.bool)
        present = text_present.to(device=device, dtype=torch.bool).reshape(batch)
        tm = text_mask.to(device=device, dtype=torch.bool) & present[:, None]
        query = self.query(affective_features).reshape(
            batch, audio_steps, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(text_features).reshape(
            batch, text_steps, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(text_features).reshape(
            batch, text_steps, self.num_heads, self.head_dim).transpose(1, 2)

        # Bypass all-masked softmax, including truly zero-length sequences.  The
        # zero retains defined projection gradients on an all-empty batch.
        zero = (query.float().sum((1, 2, 3)) + key.float().sum((1, 2, 3))
                + value.float().sum((1, 2, 3))) * 0.0
        relation = zero[:, None].expand(batch, self.hidden_dim)
        common, deviation = relation, relation
        mass_per_head = zero[:, None].expand(batch, self.num_heads)
        strength, clipping_all, clipping_positive = zero, zero, zero
        positive_count = torch.zeros(batch, dtype=torch.long, device=device)
        clipped_count, valid_position_count = positive_count.clone(), positive_count.clone()
        active = (tm.any(dim=1) & am.any(dim=1)).nonzero(as_tuple=True)[0]
        if active.numel():
            # Preserve the original FP32 softmax/reductions under BF16 training.
            with torch.autocast(device_type=device.type, enabled=False):
                q, k, v = query[active].float(), key[active].float(), value[active].float()
                active_tm, active_am = tm[active], am[active]
                key_valid = active_tm[:, None, None, :]
                frame_valid = active_am[:, None, :, None]
                position_valid = (key_valid & frame_valid).expand(-1, self.num_heads, -1, -1)
                lengths = active_tm.sum(dim=1).float()[:, None, None, None]
                logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                masked_logits = logits.masked_fill(~key_valid, -torch.inf)
                attention = torch.softmax(masked_logits, dim=-1).masked_fill(~key_valid, 0.0)
                difference = (attention - lengths.reciprocal()).masked_fill(~key_valid, 0.0)
                row_strength = 0.5 * difference.abs().sum(dim=-1, keepdim=True)
                deficit = (-difference).clamp_min(0.0)
                if self.variant == "linear":
                    weights = deficit
                    context = torch.matmul(deficit, v)
                    raw_score = deficit * lengths
                    clipped = torch.zeros_like(position_valid)
                elif self.variant == "log":
                    baseline = torch.logsumexp(masked_logits, dim=-1, keepdim=True) - lengths.log()
                    raw_score = (baseline - logits).clamp_min(0.0).masked_fill(~key_valid, 0.0)
                    clipped = (raw_score > self.kappa) & position_valid
                    score = raw_score.clamp_max(self.kappa).masked_fill(~key_valid, 0.0)
                    mass = score.sum(dim=-1, keepdim=True)
                    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
                    normalized = score / safe_mass
                    # Keep original operation order for the full Log output.
                    context = row_strength * torch.matmul(normalized, v)
                    weights = row_strength * normalized
                else:
                    weights = attention
                    context = torch.matmul(attention, v)
                    raw_score = attention
                    clipped = torch.zeros_like(position_valid)

                # mean(V) uses only the valid BERT tokens.  M is the actual
                # weight sum for every head/frame; no centering changes full.
                valid_values = v.masked_fill(~active_tm[:, None, :, None], 0.0)
                mean_value = valid_values.sum(dim=2, keepdim=True) / lengths
                row_mass = weights.sum(dim=-1, keepdim=True)
                common_context = row_mass * mean_value
                deviation_values = (v - mean_value).masked_fill(
                    ~active_tm[:, None, :, None], 0.0)
                deviation_context = torch.matmul(weights, deviation_values)
                frame_count = active_am.sum(dim=1).float()

                def pool(context_values):
                    context_values = context_values.masked_fill(~frame_valid, 0.0)
                    return (context_values.sum(dim=2) / frame_count[:, None, None]).reshape(
                        active.numel(), self.hidden_dim)

                active_relation = pool(context)
                active_common, active_deviation = pool(common_context), pool(deviation_context)
                active_mass = row_mass.masked_fill(~frame_valid, 0.0).sum(dim=2).squeeze(-1)
                active_mass = active_mass / frame_count[:, None]
                active_strength = row_strength.masked_fill(~frame_valid, 0.0).sum((1, 2, 3))
                active_strength = active_strength / (frame_count * self.num_heads)
                n_positive = ((raw_score > 0) & position_valid).sum((1, 2, 3))
                n_clipped, n_valid = clipped.sum((1, 2, 3)), position_valid.sum((1, 2, 3))
                active_clipping_all = n_clipped.float() / n_valid.clamp_min(1)
                active_clipping_positive = n_clipped.float() / n_positive.clamp_min(1)

            relation = relation.index_copy(0, active, active_relation)
            common, deviation = common.index_copy(0, active, active_common), deviation.index_copy(0, active, active_deviation)
            mass_per_head = mass_per_head.index_copy(0, active, active_mass)
            strength = strength.index_copy(0, active, active_strength)
            clipping_all = clipping_all.index_copy(0, active, active_clipping_all)
            clipping_positive = clipping_positive.index_copy(0, active, active_clipping_positive)
            positive_count = positive_count.index_copy(0, active, n_positive)
            clipped_count = clipped_count.index_copy(0, active, n_clipped)
            valid_position_count = valid_position_count.index_copy(0, active, n_valid)
        return {"z_rel": relation, "z_common": common, "z_deviation": deviation,
                "relation_mass": mass_per_head.mean(dim=1), "relation_mass_per_head": mass_per_head,
                "strength": strength, "clipping_all": clipping_all,
                "clipping_positive": clipping_positive, "positive_count": positive_count,
                "clipped_count": clipped_count, "valid_position_count": valid_position_count,
                "text_present": present, "valid_text_tokens": tm.sum(dim=1),
                "valid_audio_frames": am.sum(dim=1)}


class CloseoutDetector(Detector):
    """Fixed A/AE/PoolJoint/CA/Linear/Log arms with paired initialization.

    An optional ``acoustic`` tensor is a representation previously computed by
    this same checkpoint.  The runner must bind its cache to arm, seed,
    checkpoint and inference configuration; a tensor cannot verify provenance.
    """

    VARIANTS = ("acoustic", "ae", "pooljoint", "ca", "linear", "log")

    def __init__(self, config: dict, variant: str = "acoustic", seed: int = 17,
                 acoustic_backbone: nn.Module | None = None):
        if variant not in self.VARIANTS:
            raise ValueError("unknown closeout variant")
        settings = config.get("model", {})
        fixed = {"hidden_dim": 256, "affective_dim": 1024, "text_dim": 768,
                 "num_heads": 8, "unfreeze_last_layers": 4, "kappa": 5.0, "dropout": 0.1}
        for key, expected in fixed.items():
            if settings.get(key, expected) != expected:
                raise ValueError(f"closeout requires model.{key}={expected}")
        # The original acoustic constructor supplies normalization, masked
        # mean/std pooling, projection and exactly the last-four-layer policy.
        if acoustic_backbone is None:
            from .closeout_pretrained import load_published_backbone
            acoustic_backbone = load_published_backbone(config, seed)
        super().__init__(config, variant="acoustic", seed=seed,
                         acoustic_backbone=acoustic_backbone)
        if len(self.acoustic_backbone.encoder.layers) < 4:
            raise ValueError("backbone must expose at least four encoder layers")
        self.variant, self.seed = variant, int(seed)
        self.capacity_branch = None
        if variant in {"ca", "linear", "log"}:
            with _initialization_seed(seed + 300):
                self.relation = _CloseoutRelation(variant)
        elif variant in {"ae", "pooljoint"}:
            dimensions = (1024, 512, 256) if variant == "ae" else (1792, 320, 256)
            with _initialization_seed(seed + 300):
                self.capacity_branch = nn.Sequential(
                    nn.Linear(dimensions[0], dimensions[1], bias=False), nn.GELU(),
                    nn.Linear(dimensions[1], dimensions[2], bias=False))
        input_dim = 256 if variant == "acoustic" else 512
        with _initialization_seed(seed + 400):
            classifier_input = nn.Linear(input_dim, 256)
        # Bias shape is compatible across all arms; isolate its initialization
        # from the different number of first-layer weights drawn above.
        with _initialization_seed(seed + 401):
            nn.init.uniform_(classifier_input.bias, -1 / math.sqrt(256), 1 / math.sqrt(256))
        with _initialization_seed(seed + 402):
            classifier_output = nn.Linear(256, 2)
        self.classifier = nn.Sequential(classifier_input, nn.GELU(), nn.Dropout(0.1), classifier_output)
        self.train(True)

    @staticmethod
    def _pool_audio(features, mask):
        if features.ndim != 3 or tuple(mask.shape) != tuple(features.shape[:2]):
            raise ValueError("E2V feature/mask shape mismatch")
        valid = mask.to(device=features.device, dtype=torch.bool)
        with torch.autocast(device_type=features.device.type, enabled=False):
            values = features.float().masked_fill(~valid[:, :, None], 0.0)
            return values.sum(dim=1) / valid.sum(dim=1).clamp_min(1)[:, None]

    def forward(self, batch: dict, condition: str = "full", acoustic=None):
        if condition not in {"full", "common", "deviation", "zero"}:
            raise ValueError("unknown relation condition")
        if condition in {"common", "deviation"} and self.variant not in {"linear", "log", "ca"}:
            raise ValueError("common/deviation are defined only for attention arms")
        if acoustic is None:
            acoustic = self._encode_acoustic(batch["waveforms"], batch["waveform_lengths"])
        if acoustic.ndim != 2 or acoustic.size(1) != self.hidden_dim:
            raise ValueError("cached acoustic representation must have shape [B,256]")
        batch_size = acoustic.size(0)
        relation = acoustic.new_zeros(batch_size, self.hidden_dim)
        zero = acoustic.new_zeros(batch_size)
        diagnostics = {"z_rel": relation, "z_common": relation, "z_deviation": relation,
                       "relation_mass": zero, "relation_mass_per_head": zero[:, None].expand(-1, 8),
                       "strength": zero, "clipping_all": zero, "clipping_positive": zero,
                       "positive_count": zero.long(), "clipped_count": zero.long(),
                       "valid_position_count": zero.long()}
        if self.relation is not None:
            diagnostics = self.relation(batch["affective_features"], batch["audio_mask"],
                                        batch["text_features"], batch["text_mask"], batch["text_present"])
            relation = diagnostics["z_rel"]
        elif self.capacity_branch is not None:
            pooled_audio = self._pool_audio(batch["affective_features"], batch["audio_mask"])
            if self.variant == "pooljoint":
                pooled_text = self._pool_text(batch["text_features"], batch["text_mask"], batch["text_present"])
                pooled_audio = torch.cat([pooled_audio, pooled_text], dim=-1)
            relation = self.capacity_branch(pooled_audio)
        if relation.size(0) != batch_size or relation.device != acoustic.device:
            raise ValueError("acoustic and relation batch/device mismatch")
        diagnostics["z_rel_full"] = relation
        if condition == "zero":
            relation = torch.zeros_like(relation)
        elif condition == "common":
            relation = diagnostics["z_common"]
        elif condition == "deviation":
            relation = diagnostics["z_deviation"]
        diagnostics["z_rel"] = relation
        fused = acoustic if self.variant == "acoustic" else torch.cat([acoustic, relation], dim=-1)
        diagnostics["logits"] = self.classifier(fused)
        diagnostics["z_acoustic"] = acoustic
        return diagnostics

    def parameter_counts(self):
        branch = self.relation if self.relation is not None else self.capacity_branch
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "backbone_trainable": sum(p.numel() for p in self.acoustic_backbone.parameters() if p.requires_grad),
                "relation_branch": sum(p.numel() for p in branch.parameters()) if branch is not None else 0,
                "classifier": sum(p.numel() for p in self.classifier.parameters())}


def engineering_operator_checks(device: str = "cpu") -> dict:
    """Execute synthetic contract checks; no downloads or scientific fit.

    The tiny injected backbone exercises the trainable-layer boundary and model
    data flow only.  It provides no evidence about XLS-R training or detection.
    Exceptions are failures; returning this report requires every check to run.
    """
    device = torch.device(device)
    report = {"device": str(device), "dtype": "FP32", "checks": {},
              "scope": "synthetic operators, initialization and trainable-layer boundary"}

    def require(name, passed):
        if not bool(passed):
            raise AssertionError(f"closeout engineering check failed: {name}")
        report["checks"][name] = True

    def close(left, right, atol=2e-6, rtol=2e-5):
        return torch.allclose(left, right, atol=atol, rtol=rtol)

    generator = torch.Generator(device="cpu").manual_seed(1701)

    def random(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    features, text = random(3, 3, 1024), random(3, 4, 768)
    audio_mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]], device=device).bool()
    text_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]], device=device).bool()
    present = torch.tensor([True, True, False], device=device)
    for variant in ("ca", "linear", "log"):
        with _initialization_seed(317):
            relation = _CloseoutRelation(variant).to(device)
        result = relation(features, audio_mask, text, text_mask, present)
        require(f"{variant}_finite", all(torch.isfinite(value).all() for value in result.values()))
        require(f"{variant}_empty_text", torch.count_nonzero(result["z_rel"][2]) == 0)
        require(f"{variant}_decomposition", close(result["z_rel"], result["z_common"] + result["z_deviation"]))
        require(f"{variant}_capacity", sum(p.numel() for p in relation.parameters()) == 655360)
        padded_features = torch.cat([features, random(3, 2, 1024) * 1000], dim=1)
        padded_text = torch.cat([text, random(3, 2, 768) * 1000], dim=1)
        padded = relation(padded_features, torch.cat([audio_mask, audio_mask.new_zeros(3, 2)], dim=1),
                          padded_text, torch.cat([text_mask, text_mask.new_zeros(3, 2)], dim=1), present)
        require(f"{variant}_padding_invariance", close(result["z_rel"], padded["z_rel"]))
        if variant in {"linear", "log"}:
            original = UniformInverseRelation(1024, 768, 256, 8, variant, 5.0).to(device)
            original.load_state_dict(relation.state_dict(), strict=True)
            original_output = original(features, audio_mask, text, text_mask, present)
            require(f"{variant}_original_exact", torch.equal(result["z_rel"], original_output["z_rel"]))
            require(f"{variant}_original_diagnostics", all(
                torch.equal(result[name], original_output[name]) for name in
                ("strength", "clipping_all", "clipping_positive", "positive_count",
                 "clipped_count", "valid_position_count", "valid_text_tokens", "valid_audio_frames")))
            require(f"{variant}_single_token_zero", torch.count_nonzero(result["z_rel"][1]) == 0)
        else:
            projected = relation.value(text[1:2, :1]).reshape(1, 256)
            require("ca_single_token_value", close(result["z_rel"][1:2], projected))
            require("ca_mass_one", close(result["relation_mass"][:2], torch.ones(2, device=device)))
            require("ca_no_added_parameters", set(relation.state_dict()) == {"query.weight", "key.weight", "value.weight"})
        result["z_rel"].square().sum().backward()
        require(f"{variant}_finite_gradients", all(p.grad is not None and torch.isfinite(p.grad).all()
                                                  for p in relation.parameters()))
        # This known violating mask must be rejected, proving shape-check
        # sensitivity without altering the implementation or any user files.
        rejected = False
        try:
            relation(features, audio_mask[:, :1], text, text_mask, present)
        except ValueError:
            rejected = True
        require(f"{variant}_shape_negative_control", rejected)
        relation.zero_grad(set_to_none=True)
        empty = relation(features, audio_mask, text[:, :0], text_mask[:, :0], present)
        empty["z_rel"].sum().backward()
        require(f"{variant}_zero_token_gradient", torch.count_nonzero(empty["z_rel"]) == 0
                and all(p.grad is not None and torch.isfinite(p.grad).all() for p in relation.parameters()))
        with torch.no_grad():
            relation.query.weight.zero_()
        uniform = relation(features, audio_mask, text, text_mask, present)
        if variant in {"linear", "log"}:
            require(f"{variant}_uniform_zero", torch.count_nonzero(uniform["z_rel"]) == 0)
        else:
            require("ca_uniform_mean_value", close(uniform["z_rel"], uniform["z_common"]))

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.feature_projection = nn.Linear(1, 8)
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(6))

        def _get_feat_extract_output_lengths(self, lengths):
            return lengths

        def forward(self, inputs, attention_mask):
            values = self.feature_projection(inputs[:, :, None])
            for layer in self.encoder.layers:
                values = torch.tanh(layer(values))
            return SimpleNamespace(last_hidden_state=values)

    config = {"model": {"affective_dim": 1024}}
    batch = {"waveforms": random(3, 12), "waveform_lengths": torch.tensor([8, 12, 6], device=device),
             "affective_features": features, "audio_mask": audio_mask,
             "text_features": text, "text_mask": text_mask, "text_present": present}
    common_parameters, fusion_parameters, qkv_parameters = None, None, None
    counts = {}
    for variant in CloseoutDetector.VARIANTS:
        with _initialization_seed(117):
            backbone = TinyBackbone()
        model = CloseoutDetector(config, variant, 17, backbone).to(device).eval()
        counts[variant] = model.parameter_counts()
        common_state = {name: p.detach().clone() for name, p in model.named_parameters()
                        if name.startswith(("acoustic_backbone.", "acoustic_projection."))
                        or name in {"classifier.0.bias", "classifier.3.weight", "classifier.3.bias"}}
        if common_parameters is None:
            common_parameters = common_state
        else:
            require(f"{variant}_paired_common_initialization", all(torch.equal(p, common_parameters[name])
                                                                    for name, p in common_state.items()))
        if variant != "acoustic":
            input_weight = model.classifier[0].weight.detach().clone()
            if fusion_parameters is None:
                fusion_parameters = input_weight
            else:
                require(f"{variant}_paired_fusion_head", torch.equal(input_weight, fusion_parameters))
            require(f"{variant}_branch_capacity", counts[variant]["relation_branch"] == 655360)
        if model.relation is not None:
            qkv = {name: p.detach().clone() for name, p in model.relation.named_parameters()}
            if qkv_parameters is None:
                qkv_parameters = qkv
            else:
                require(f"{variant}_paired_qkv", all(torch.equal(p, qkv_parameters[name]) for name, p in qkv.items()))
        trainable_backbone = {name for name, p in model.acoustic_backbone.named_parameters() if p.requires_grad}
        expected = {f"encoder.layers.{layer}.{parameter}" for layer in range(2, 6) for parameter in ("weight", "bias")}
        require(f"{variant}_last_four_only", trainable_backbone == expected)
        model.train()
        require(f"{variant}_training_modes", not model.acoustic_backbone.training
                and all(not layer.training for layer in model.acoustic_backbone.encoder.layers[:-4])
                and all(layer.training for layer in model.acoustic_backbone.encoder.layers[-4:]))
        model.eval()
        groups = model.trainable_parameter_groups(1e-5, 1e-3)
        actual = [id(p) for group in groups for p in group["params"]]
        require(f"{variant}_optimizer_coverage", len(actual) == len(set(actual))
                and set(actual) == {id(p) for p in model.parameters() if p.requires_grad})
        output = model(batch)
        require(f"{variant}_forward_finite", torch.isfinite(output["logits"]).all())
        cached = model(batch, acoustic=output["z_acoustic"])
        require(f"{variant}_same_model_acoustic_cache", torch.equal(output["logits"], cached["logits"]))
        changed_batch = dict(batch)
        changed_waveforms = batch["waveforms"].clone()
        for index, length in enumerate(batch["waveform_lengths"]):
            changed_waveforms[index, int(length):] = 1000
        changed_batch["waveforms"] = changed_waveforms
        require(f"{variant}_waveform_padding", torch.equal(output["logits"], model(changed_batch)["logits"]))
        if variant in {"linear", "log"}:
            for condition in ("common", "deviation", "zero"):
                intervention = model(batch, condition=condition, acoustic=output["z_acoustic"])
                require(f"{variant}_{condition}_acoustic_unchanged", torch.equal(output["z_acoustic"], intervention["z_acoustic"]))
                expected_relation = torch.zeros_like(output["z_rel"]) if condition == "zero" else output[f"z_{condition}"]
                require(f"{variant}_{condition}_intervention", torch.equal(intervention["z_rel"], expected_relation))
        output["logits"].square().sum().backward()
        require(f"{variant}_complete_model_gradients", all(p.grad is not None and torch.isfinite(p.grad).all()
                                                          for p in model.parameters() if p.requires_grad))
    report["synthetic_parameter_counts"] = counts
    report["status"] = "passed"
    return report
