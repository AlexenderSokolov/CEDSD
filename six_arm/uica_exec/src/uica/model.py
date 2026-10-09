"""Small acoustic/text detectors and U-ICA fusion without runtime preprocessing."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from .operators import UniformInverseRelation


@contextmanager
def _initialization_seed(seed: int):
    # Isolate CPU parameter initialization without changing the training RNG or
    # initializing CUDA. Each common subsystem gets its own stable seed.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        yield


class Detector(nn.Module):
    def __init__(self, config: dict, variant: str = "linear", seed: int = 17,
                 acoustic_backbone: nn.Module | None = None):
        super().__init__()
        if variant not in {"acoustic", "linear", "log", "text"}:
            raise ValueError("unknown detector variant")
        self.variant = variant
        model_config = config.get("model", {})
        hidden = int(model_config.get("hidden_dim", 256))
        text_dim = int(model_config.get("text_dim", 768))
        self.hidden_dim = hidden
        # The selected XLS-R-300m preprocessor_config.json specifies true.
        # A pinned preprocessor with normalization disabled can override this.
        self.xlsr_do_normalize = bool(model_config.get("xlsr_do_normalize", True))
        self.unfreeze_last_layers = int(model_config.get("unfreeze_last_layers", 4))
        if self.unfreeze_last_layers <= 0:
            raise ValueError("unfreeze_last_layers must be positive")
        self.acoustic_backbone = None
        self.acoustic_projection = None
        self.relation = None

        if variant != "text":
            if acoustic_backbone is None:
                from transformers import Wav2Vec2Model
                models = config.get("models", {})
                identifier = models.get("xlsr")
                if not identifier:
                    raise ValueError("models.xlsr is required without an injected acoustic_backbone")
                load_kwargs = {}
                revision = models.get("revisions", {}).get("xlsr", models.get("xlsr_revision"))
                if isinstance(identifier, dict):
                    revision = identifier.get("revision", revision)
                    identifier = identifier.get("path", identifier.get("name"))
                if revision is not None:
                    load_kwargs["revision"] = revision
                with _initialization_seed(seed + 100):
                    acoustic_backbone = Wav2Vec2Model.from_pretrained(identifier, **load_kwargs)
            self.acoustic_backbone = acoustic_backbone
            self._configure_backbone()
            acoustic_dim = int(acoustic_backbone.config.hidden_size)
            with _initialization_seed(seed + 200):
                self.acoustic_projection = nn.Sequential(nn.Linear(2 * acoustic_dim, hidden), nn.GELU())

        if variant in {"linear", "log"}:
            if "affective_dim" not in model_config:
                raise ValueError("model.affective_dim must be populated from the feature cache")
            with _initialization_seed(seed + 300):
                self.relation = UniformInverseRelation(
                    audio_dim=int(model_config["affective_dim"]),
                    text_dim=text_dim, hidden_dim=hidden,
                    num_heads=int(model_config.get("num_heads", 8)),
                    variant=variant, kappa=float(model_config.get("kappa", 5.0)))

        classifier_dim = text_dim if variant == "text" else hidden * (2 if self.relation is not None else 1)
        with _initialization_seed(seed + 400):
            self.classifier = nn.Sequential(
                nn.Linear(classifier_dim, hidden), nn.GELU(),
                nn.Dropout(float(model_config.get("dropout", 0.1))), nn.Linear(hidden, 2))
        self.train(True)

    def _configure_backbone(self) -> None:
        backbone = self.acoustic_backbone
        backbone.requires_grad_(False)
        backbone.config.layerdrop = 0.0
        backbone.config.apply_spec_augment = False
        backbone.config.mask_time_prob = 0.0
        backbone.config.mask_feature_prob = 0.0
        for layer in backbone.encoder.layers[-self.unfreeze_last_layers:]:
            layer.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.acoustic_backbone is not None:
            # eval() on the whole backbone keeps frozen CNN, feature projection,
            # positional convolution, prefix layers and top-level dropout fixed.
            self.acoustic_backbone.eval()
            if mode:
                for layer in self.acoustic_backbone.encoder.layers[-self.unfreeze_last_layers:]:
                    layer.train(True)
        return self

    def trainable_parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        backbone_parameters = []
        head_parameters = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                destination = backbone_parameters if name.startswith("acoustic_backbone.") else head_parameters
                destination.append(parameter)
        groups = []
        if backbone_parameters:
            groups.append({"params": backbone_parameters, "lr": backbone_lr, "name": "backbone"})
        if head_parameters:
            groups.append({"params": head_parameters, "lr": head_lr, "name": "heads"})
        return groups

    def _encode_acoustic(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2 or lengths.ndim != 1 or lengths.numel() != waveforms.size(0):
            raise ValueError("waveform length shape mismatch")
        lengths = lengths.to(device=waveforms.device, dtype=torch.long)
        if torch.any(lengths <= 0) or torch.any(lengths > waveforms.size(1)):
            raise ValueError("waveform lengths must be positive and within the padded waveform")
        attention_mask = torch.arange(waveforms.size(1), device=waveforms.device)[None, :] < lengths[:, None]
        frame_lengths = self.acoustic_backbone._get_feat_extract_output_lengths(lengths)
        if torch.any(frame_lengths <= 0):
            raise ValueError("waveform length is too short for the acoustic convolution stack")
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            acoustic_input = waveforms.float().masked_fill(~attention_mask, 0.0)
            if self.xlsr_do_normalize:
                sample_count = lengths.float()[:, None]
                sample_mean = acoustic_input.sum(dim=1, keepdim=True) / sample_count
                centered = (acoustic_input - sample_mean).masked_fill(~attention_mask, 0.0)
                sample_variance = centered.square().sum(dim=1, keepdim=True) / sample_count
                acoustic_input = centered / torch.sqrt(sample_variance + 1e-7)
        outputs = self.acoustic_backbone(acoustic_input, attention_mask=attention_mask.long())
        features = outputs.last_hidden_state
        if torch.any(frame_lengths > features.size(1)):
            raise ValueError("acoustic frame length exceeds backbone output")
        valid = torch.arange(features.size(1), device=features.device)[None, :] < frame_lengths[:, None]
        with torch.autocast(device_type=features.device.type, enabled=False):
            values = features.float().masked_fill(~valid[:, :, None], 0.0)
            count = frame_lengths.float()[:, None]
            mean = values.sum(dim=1) / count
            deviations = (values - mean[:, None, :]).masked_fill(~valid[:, :, None], 0.0)
            variance = deviations.square().sum(dim=1) / count
            std = torch.sqrt(variance.clamp_min(0.0) + 1e-5)
            pooled = torch.cat([mean, std], dim=-1)
        return self.acoustic_projection(pooled)

    @staticmethod
    def _pool_text(features: torch.Tensor, text_mask: torch.Tensor,
                   text_present: torch.Tensor) -> torch.Tensor:
        valid = text_mask.bool() & text_present.bool().reshape(-1, 1)
        with torch.autocast(device_type=features.device.type, enabled=False):
            values = features.float().masked_fill(~valid[:, :, None], 0.0)
            return values.sum(dim=1) / valid.sum(dim=1).clamp_min(1)[:, None]

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        if self.variant == "text":
            fused = self._pool_text(batch["text_features"], batch["text_mask"], batch["text_present"])
            acoustic = None
        else:
            acoustic = self._encode_acoustic(batch["waveforms"], batch["waveform_lengths"])
            fused = acoustic
        zero = fused.new_zeros(fused.size(0))
        diagnostics = {"z_rel": fused.new_zeros((fused.size(0), self.hidden_dim)),
                       "strength": zero, "clipping_all": zero, "clipping_positive": zero,
                       "positive_count": zero.long(), "clipped_count": zero.long(),
                       "valid_position_count": zero.long()}
        if self.relation is not None:
            diagnostics = self.relation(batch["affective_features"], batch["audio_mask"],
                                        batch["text_features"], batch["text_mask"], batch["text_present"])
            fused = torch.cat([acoustic, diagnostics["z_rel"]], dim=-1)
        diagnostics["logits"] = self.classifier(fused)
        if acoustic is not None:
            diagnostics["z_acoustic"] = acoustic
        return diagnostics
