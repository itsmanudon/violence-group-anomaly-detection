"""Local HRNet-W32 pre-heatmap features for annotated actor crops."""

from contextlib import nullcontext
from pathlib import Path

import torch

from surveillance.features.actor_backbones import LocalActorBackbone
from surveillance.features.geometry import roi_align_actors


class HRNetPoseExtractor(LocalActorBackbone):
    """Crop actors to 256x192; flatten the 32x64x48 tensor before final_layer."""

    feature_dim = 32 * 64 * 48

    def __init__(self, checkpoint: Path | None, frozen: bool = True, *, clip_outside: bool = False):
        super().__init__(
            checkpoint,
            architecture="pose_hrnet_w32",
            endpoint="pre_final_layer",
            channels=32,
            frozen=frozen,
            clip_outside=clip_outside,
        )

    def forward(
        self, frames: torch.Tensor, boxes: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("HRNet expects frames [B,3,H,W] RGB in [0,1]")
        self._validate_inputs(frames, boxes, valid_mask)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            if not valid_mask.any():
                return frames.new_zeros((*valid_mask.shape, self.feature_dim))
            crops = roi_align_actors(
                frames, boxes, valid_mask, (256, 192), clip_outside=self.clip_outside
            )
            # GPU profiling can rewrite a trace after its cold call (e.g. fold
            # batch norms), changing rounding and cached actor features. Keep
            # inference-only exports on one graph from the first call onward.
            # Other scripted/fine-tunable backbones retain their caller policy.
            execution = (
                torch.jit.optimized_execution(False)
                if self.metadata.get("inference_only") is True
                else nullcontext()
            )
            with execution:
                features = self.backbone(self._normalize(crops))
            expected = (int(valid_mask.sum()), 32, 64, 48)
            if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected:
                raise ValueError(
                    "HRNet checkpoint must return [sum(valid),32,64,48] "
                    "pre_final_layer features; see docs/actor-backbones.md"
                )
            if not torch.isfinite(features).all():
                raise ValueError("HRNet checkpoint produced nonfinite features")
            return self._pad(features, valid_mask, self.feature_dim)
