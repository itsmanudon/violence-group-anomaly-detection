"""Local I3D Mixed_4f actor features with explicit temporal/spatial pooling."""

from pathlib import Path

import torch
from torch.nn import functional as F

from surveillance.features.actor_backbones import LocalActorBackbone
from surveillance.features.geometry import roi_align_actors


class I3DActorExtractor(LocalActorBackbone):
    """Ten RGB frames -> Mixed_4f -> temporal mean -> 90x160 -> 5x5 actor RoI."""

    feature_dim = 832 * 5 * 5

    def __init__(self, checkpoint: Path | None, frozen: bool = True, *, clip_outside: bool = False):
        super().__init__(
            checkpoint,
            architecture="i3d",
            endpoint="Mixed_4f",
            channels=832,
            frozen=frozen,
            clip_outside=clip_outside,
        )

    def forward(
        self, frames: torch.Tensor, boxes: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:3] != (10, 3):
            raise ValueError("I3D expects frames [B,10,3,H,W] RGB in [0,1]")
        self._validate_inputs(frames, boxes, valid_mask)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            if not valid_mask.any():
                return frames.new_zeros((*valid_mask.shape, self.feature_dim))
            batch, time, channels, height, width = frames.shape
            resized = F.interpolate(
                frames.reshape(batch * time, channels, height, width),
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            inputs = self._normalize(resized).reshape(batch, time, channels, *self.input_size)
            features = self.backbone(inputs.permute(0, 2, 1, 3, 4).contiguous())
            if (
                not isinstance(features, torch.Tensor)
                or features.ndim != 5
                or features.shape[:2] != (batch, 832)
                or min(features.shape[2:]) < 1
            ):
                raise ValueError(
                    "I3D checkpoint must return [B,832,t,h,w] Mixed_4f features; "
                    "see docs/actor-backbones.md"
                )
            if not torch.isfinite(features).all():
                raise ValueError("I3D checkpoint produced nonfinite features")
            spatial = F.interpolate(
                features.mean(dim=2), size=(90, 160), mode="bilinear", align_corners=False
            )
            actors = roi_align_actors(
                spatial, boxes, valid_mask, (5, 5), clip_outside=self.clip_outside
            )
            return self._pad(actors, valid_mask, self.feature_dim)
