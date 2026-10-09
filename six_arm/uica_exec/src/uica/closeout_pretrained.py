"""Strict CPU loading of the pinned published XLS-R backbone.

Transformers 4.44.2's generic from_pretrained conversion misses the two old
weight-normalization keys under Torch 2.5.1.  This loader reads the local locked
state directly and permits only that exact rename and the official prefix.
Every backbone parameter and persistent buffer must originate in that file.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .common import file_sha256
from .model import _initialization_seed


WEIGHTS_SHA256 = "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"
CONFIG_SHA256 = "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac"
REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
PREFIX = "wav2vec2."
WEIGHT_NORM_RENAMES = {
    "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}
# These seven tensors were observed in the exact pinned official file.  They
# belong to Wav2Vec2ForPreTraining, and are not part of Wav2Vec2Model.
PRETRAINING_HEAD_SHAPES = {
    "quantizer.codevectors": (1, 640, 384),
    "quantizer.weight_proj.weight": (640, 512),
    "quantizer.weight_proj.bias": (640,),
    "project_hid.weight": (768, 1024),
    "project_hid.bias": (768,),
    "project_q.weight": (768, 768),
    "project_q.bias": (768,),
}


def _map_published_state(published, expected):
    """Validate all keys/shapes/dtypes before any model tensor is overwritten.

    Exposed separately so missing, shape, dtype and unknown-key controls can
    fail independently without allocating a complete 300M parameter model.
    No values are created, cast, reshaped, or substituted during conversion.
    """
    if not isinstance(published, Mapping) or not isinstance(expected, Mapping) or not expected:
        raise ValueError("published and expected backbone states must be nonempty mappings")
    mapped, sources, renamed, omitted = {}, {}, [], []
    for source, value in published.items():
        if not isinstance(source, str) or not isinstance(value, torch.Tensor):
            raise ValueError("published state must contain only string/tensor entries")
        if value.device.type != "cpu":
            raise ValueError("published tensors must be read on CPU")
        if source in PRETRAINING_HEAD_SHAPES:
            if tuple(value.shape) != PRETRAINING_HEAD_SHAPES[source] or value.dtype != torch.float32:
                raise ValueError("official pretraining head shape/dtype differs: " + source)
            omitted.append(source)
            continue
        target = source[len(PREFIX):] if source.startswith(PREFIX) else source
        if target not in expected and target in WEIGHT_NORM_RENAMES:
            target = WEIGHT_NORM_RENAMES[target]
        if target not in expected:
            raise ValueError("unknown published backbone key: " + source)
        if target in mapped:
            raise ValueError("multiple published keys target one backbone tensor: " + target)
        destination = expected[target]
        if not isinstance(destination, torch.Tensor):
            raise ValueError("expected backbone state must contain tensors")
        if tuple(value.shape) != tuple(destination.shape):
            raise ValueError("published backbone shape mismatch: " + source + " -> " + target)
        if value.dtype != destination.dtype:
            raise ValueError("published backbone dtype mismatch: " + source + " -> " + target)
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite published backbone tensor: " + source)
        mapped[target], sources[target] = value, source
        stripped = source[len(PREFIX):] if source.startswith(PREFIX) else source
        if target != stripped:
            renamed.append({"source": source, "target": target, "shape": list(value.shape), "dtype": str(value.dtype)})
    missing = sorted(set(expected) - set(mapped))
    if missing:
        raise ValueError("published backbone parameters/buffers missing: " + ", ".join(missing))
    return mapped, {"expected_tensor_count": len(expected), "loaded_tensor_count": len(mapped),
                    "weight_norm_renames": renamed, "omitted_pretraining_heads": sorted(omitted),
                    "tensor_sources": sources}


def load_published_backbone(config: dict, seed: int):
    """Return an entirely published FP32 Wav2Vec2Model with a loading audit.

    This function only constructs and loads CPU tensors.  The original caller
    remains responsible for the last-four-layer trainability policy and device
    placement.  CPU RNG isolation keeps all other paired initializations intact.
    """
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    identifier = config.get("models", {}).get("xlsr")
    if isinstance(identifier, dict):
        identifier = identifier.get("path")
    if not isinstance(identifier, (str, Path)) or not identifier:
        raise ValueError("strict published loading requires a local models.xlsr snapshot directory")
    folder = Path(identifier).resolve()
    weights_path, config_path = folder / "pytorch_model.bin", folder / "config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise ValueError("locked local XLS-R weights/config are unavailable")
    revision = config.get("models", {}).get("revisions", {}).get("xlsr")
    if revision is not None and revision != REVISION:
        raise ValueError("XLS-R revision differs from the published closeout lock")
    if file_sha256(weights_path) != WEIGHTS_SHA256 or file_sha256(config_path) != CONFIG_SHA256:
        raise ValueError("local XLS-R weights/config SHA256 differs from the immutable publication lock")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("published backbone construction requires the established FP32 default dtype")
    specification = Wav2Vec2Config.from_json_file(str(config_path))
    with _initialization_seed(int(seed) + 100), torch.device("cpu"):
        backbone = Wav2Vec2Model(specification)
    published = torch.load(weights_path, map_location="cpu", weights_only=True)
    mapped, audit = _map_published_state(published, backbone.state_dict())
    # The real file is required to contain all seven observed official heads.
    # Allowlisting alone must not hide a different or incomplete state export.
    if set(audit["omitted_pretraining_heads"]) != set(PRETRAINING_HEAD_SHAPES):
        raise ValueError("published file pretraining-head inventory differs from the locked snapshot")
    loaded = backbone.load_state_dict(mapped, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("strict published load returned missing/unexpected tensors")
    state = backbone.state_dict()
    for name, source_tensor in mapped.items():
        if not torch.equal(state[name], source_tensor):
            raise ValueError("loaded backbone tensor differs from its published source: " + name)
    if file_sha256(weights_path) != WEIGHTS_SHA256 or file_sha256(config_path) != CONFIG_SHA256:
        raise ValueError("published local files changed while loading")
    audit.update({"status": "strict_published_backbone_loaded", "source_path": str(weights_path),
                  "weights_sha256": WEIGHTS_SHA256, "config_path": str(config_path), "config_sha256": CONFIG_SHA256,
                  "revision": REVISION, "construction_device": "cpu", "construction_dtype": "float32",
                  "initialization_seed": int(seed) + 100, "seed_policy": "isolated_CPU_seed_plus_100",
                  "strict_state_load": True, "all_loaded_tensors_equal_published": True,
                  "missing_keys": [], "unexpected_keys": []})
    backbone.published_initialization = audit
    return backbone


def published_state_negative_controls():
    """Small CPU controls with real weight-norm names; no fit or checkpoint."""
    old_g, old_v = tuple(WEIGHT_NORM_RENAMES)
    new_g, new_v = (WEIGHT_NORM_RENAMES[old_g], WEIGHT_NORM_RENAMES[old_v])
    expected = {new_g: torch.zeros(1, 1, 2), new_v: torch.zeros(4, 2, 2)}
    source = {PREFIX + old_g: torch.ones(1, 1, 2), PREFIX + old_v: torch.ones(4, 2, 2)}
    mapped, audit = _map_published_state(source, expected)
    if len(audit["weight_norm_renames"]) != 2 or mapped[new_g] is not source[PREFIX + old_g]:
        raise AssertionError("exact weight-normalization map did not retain published tensors")
    cases = {
        "missing": {PREFIX + old_g: source[PREFIX + old_g]},
        "shape": {**source, PREFIX + old_v: torch.ones(4, 2, 3)},
        "dtype": {**source, PREFIX + old_v: source[PREFIX + old_v].double()},
        "unknown_backbone": {**source, PREFIX + "unrecognized.weight": torch.ones(1)},
        "duplicate_target": {**source, new_g: source[PREFIX + old_g]},
    }
    checked = {"exact_published_map": True}
    for name, changed in cases.items():
        try:
            _map_published_state(changed, expected)
        except ValueError:
            checked[name + "_rejected"] = True
        else:
            raise AssertionError("published-state negative control accepted: " + name)
    return checked
