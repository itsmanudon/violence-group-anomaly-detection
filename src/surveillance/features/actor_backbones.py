"""Shared, strictly local feature-export contracts for actor backbone adapters."""

import json
import math
from pathlib import Path

import torch
from torch import nn

from surveillance.features.geometry import normalized_boxes_to_pixels, validate_actor_layout

METADATA_FILENAME = "actor_backbone.json"


def _checkpoint_error(message: str) -> ValueError:
    return ValueError(
        f"Incompatible actor backbone checkpoint: {message}. "
        "Export a feature-only TorchScript archive; see docs/actor-backbones.md"
    )


def _validate_metadata(metadata: object, architecture: str, endpoint: str, channels: int) -> dict:
    if not isinstance(metadata, dict):
        raise _checkpoint_error("metadata must be a JSON object")
    expected = {
        "schema_version": 1,
        "architecture": architecture,
        "endpoint": endpoint,
        "output_channels": channels,
        "feature_only": True,
    }
    for key, value in expected.items():
        if type(metadata.get(key)) is not type(value) or metadata[key] != value:
            raise _checkpoint_error(f"metadata {key} must be {value!r}")
    if not isinstance(metadata.get("provenance"), str) or not metadata["provenance"].strip():
        raise _checkpoint_error("a nonempty provenance declaration is required")
    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise _checkpoint_error("preprocessing must be an object")
    if preprocessing.get("color_order") != "rgb" or preprocessing.get("input_range") != [0, 1]:
        raise _checkpoint_error("preprocessing must declare rgb and input_range [0,1]")
    for name in ("mean", "std"):
        values = preprocessing.get(name)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or not all(type(v) in (int, float) and math.isfinite(v) for v in values)
        ):
            raise _checkpoint_error(f"preprocessing {name} needs three finite numbers")
    if any(v <= 0 for v in preprocessing["std"]):
        raise _checkpoint_error("preprocessing std must be positive")
    size = preprocessing.get("input_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(type(v) is int and v > 0 for v in size)
    ):
        raise _checkpoint_error("preprocessing input_size must be positive [height,width]")
    if architecture == "pose_hrnet_w32" and size != [256, 192]:
        raise _checkpoint_error("HRNet input_size must be [256,192]")
    if architecture == "i3d" and (
        type(metadata.get("input_frames")) is not int or metadata["input_frames"] != 10
    ):
        raise _checkpoint_error("I3D input_frames must be 10")
    return metadata


class LocalActorBackbone(nn.Module):
    """Load a vetted local archive. Metadata is a declaration, not provenance proof."""

    def __init__(
        self,
        checkpoint: Path | None,
        *,
        architecture: str,
        endpoint: str,
        channels: int,
        frozen: bool,
        clip_outside: bool,
    ):
        super().__init__()
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"{architecture} checkpoint missing: supply a vetted local feature-only "
                "TorchScript archive with actor_backbone.json; see docs/actor-backbones.md. "
                "Precomputed features can be used without backbone checkpoints."
            )
        extra = {METADATA_FILENAME: ""}
        try:
            self.backbone = torch.jit.load(str(checkpoint), map_location="cpu", _extra_files=extra)
        except (RuntimeError, ValueError, OSError) as error:
            raise _checkpoint_error("cannot load local TorchScript archive") from error
        try:
            metadata = json.loads(extra[METADATA_FILENAME])
        except (ValueError, TypeError) as error:
            raise _checkpoint_error(f"missing or invalid embedded {METADATA_FILENAME}") from error
        self.metadata = _validate_metadata(metadata, architecture, endpoint, channels)
        self.frozen = frozen
        self.clip_outside = clip_outside
        preprocessing = self.metadata["preprocessing"]
        self.register_buffer("mean", torch.tensor(preprocessing["mean"], dtype=torch.float32))
        self.register_buffer("std", torch.tensor(preprocessing["std"], dtype=torch.float32))
        self.input_size = tuple(preprocessing["input_size"])
        if not frozen and not list(self.backbone.parameters()):
            raise _checkpoint_error(
                "fine-tuning requires preserved parameters; use a non-frozen scripted export"
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not frozen)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def _validate_inputs(
        self, frames: torch.Tensor, boxes: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        if not frames.is_floating_point() or not torch.isfinite(frames).all():
            raise ValueError("frames must be finite floating-point RGB in [0,1]")
        if ((frames < 0) | (frames > 1)).any():
            raise ValueError("frames must be floating-point RGB in [0,1]")
        if frames.shape[-1] < 1 or frames.shape[-2] < 1:
            raise ValueError("frames must have positive spatial size")
        validate_actor_layout(boxes, valid_mask, frames.shape[0], frames.device)
        normalized_boxes_to_pixels(
            boxes[valid_mask], frames.shape[-1], frames.shape[-2], clip_outside=self.clip_outside
        )

    def _normalize(self, frames: torch.Tensor) -> torch.Tensor:
        return (frames - self.mean[None, :, None, None]) / self.std[None, :, None, None]

    @staticmethod
    def _pad(features: torch.Tensor, valid_mask: torch.Tensor, feature_dim: int) -> torch.Tensor:
        output = features.new_zeros((*valid_mask.shape, feature_dim))
        output[valid_mask] = features.flatten(1)
        return output
