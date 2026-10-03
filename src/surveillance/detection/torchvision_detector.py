"""Local COCO Faster R-CNN ResNet-50 FPN v1 adapter; never downloads weights."""

import hashlib
from dataclasses import asdict
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops.misc import FrozenBatchNorm2d

from surveillance.detection.config import DetectionConfig
from surveillance.detection.person_detector import DetectionResult, filter_detections
from surveillance.training.sultani_trainer import select_device


def _frozen_batch_norm(module):
    # The torchvision factory selects BatchNorm when both pretrained flags are None.
    # Official COCO v1 uses FrozenBatchNorm with eps=0; restore that architecture.
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            replacement = FrozenBatchNorm2d(child.num_features, eps=0.0)
            setattr(module, name, replacement)
        else:
            _frozen_batch_norm(child)


class TorchvisionPersonDetector:
    """COCO v1 detector with strict bare state_dict loading and no implicit downloads."""

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        config: DetectionConfig | None = None,
        device: str = "auto",
        *,
        allow_untrained: bool = False,
    ):
        self.config = config or DetectionConfig()
        if self.config.person_class_id != 1:
            raise ValueError("COCO Faster R-CNN requires person_class_id=1")
        if checkpoint is None and not allow_untrained:
            raise ValueError(
                "A local pretrained COCO Faster R-CNN checkpoint is required; "
                "automatic downloads and untrained CLI inference are disabled"
            )
        checkpoint = Path(checkpoint) if checkpoint is not None else None
        if checkpoint is not None and not checkpoint.is_file():
            raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint}")
        self.device = select_device(device)
        self.model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=91,
            box_score_thresh=0.0,
            box_nms_thresh=1.0,
            box_detections_per_img=1000,
        )
        _frozen_batch_norm(self.model.backbone)
        checkpoint_hash = None
        if checkpoint is not None:
            with checkpoint.open("rb") as stream:
                checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if (
                    not isinstance(state, dict)
                    or not state
                    or not all(
                        isinstance(key, str) and isinstance(value, torch.Tensor)
                        for key, value in state.items()
                    )
                ):
                    raise ValueError("Expected a bare tensor state_dict")
                # Reject extras explicitly: FrozenBatchNorm silently removes num_batches_tracked.
                expected = self.model.state_dict()
                if set(state) != set(expected):
                    raise ValueError(
                        "state_dict keys differ: "
                        f"missing={sorted(set(expected) - set(state))[:5]}, "
                        f"unexpected={sorted(set(state) - set(expected))[:5]}"
                    )
                self.model.load_state_dict(state, strict=True)
            except (RuntimeError, ValueError, TypeError) as error:
                raise ValueError(
                    f"Incompatible COCO Faster R-CNN ResNet50 FPN v1 checkpoint: {error}"
                ) from error
        self.model.to(self.device).eval()
        self.metadata = {
            "backend": "torchvision_fasterrcnn",
            "architecture": "fasterrcnn_resnet50_fpn",
            "num_classes": 91,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
            "untrained": checkpoint is None,
            "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__),
            "filter_config": asdict(self.config),
            "backend_candidate_cap": 1000,
            "backend_box_score_threshold": 0.0,
            "backend_box_nms_threshold": 1.0,
            "rpn_test_pre_nms_top_n": 1000,
            "rpn_test_post_nms_top_n": 1000,
            "rpn_nms_threshold": 0.7,
            "before_count_scope": "after_rpn_and_backend_candidate_cap",
        }

    def detect(self, image: torch.Tensor) -> DetectionResult:
        """Run eval inference at native RGB input size and return canonical person boxes."""
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 3
            or image.shape[0] != 3
            or min(image.shape[1:]) <= 0
            or not image.is_floating_point()
            or not torch.isfinite(image).all()
            or ((image < 0) | (image > 1)).any()
        ):
            raise ValueError("image must be finite RGB float [3,H,W] with values in [0,1]")
        with torch.inference_mode():
            output = self.model([image.to(self.device)])[0]
        result = filter_detections(
            output["boxes"], output["scores"], output["labels"], tuple(image.shape[1:]), self.config
        )
        return DetectionResult(
            result.boxes,
            result.scores,
            result.class_ids,
            result.image_size,
            {**self.metadata, **result.metadata},
        )
