"""Strict local-only person detector configuration."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class DetectionConfig:
    """Person selection, pixel-size limits, NMS and actor-cap settings."""

    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.5
    max_actors: int = 20
    min_box_width: float = 4.0
    min_box_height: float = 4.0
    min_box_area: float = 0.0
    person_class_id: int = 1

    def __post_init__(self):
        for name in ("confidence_threshold", "nms_iou_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")
        for name in ("min_box_width", "min_box_height", "min_box_area"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.max_actors) is not int or self.max_actors <= 0:
            raise ValueError("max_actors must be a positive integer")
        if type(self.person_class_id) is not int or self.person_class_id < 0:
            raise ValueError("person_class_id must be a nonnegative integer")


def validate_detection_config(config: dict) -> dict:
    """Merge defaults and reject unknown or invalid YAML settings."""
    if not isinstance(config, dict):
        raise ValueError("Detector config must be a YAML mapping")
    defaults = {
        "device": "auto",
        "detector": {
            "backend": "torchvision_fasterrcnn",
            "checkpoint": None,
            **asdict(DetectionConfig()),
        },
        "matching": {"iou_threshold": 0.5},
    }
    for key, value in config.items():
        if key not in defaults:
            raise ValueError(f"Unknown detector config key: {key}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            unknown = set(value) - set(defaults[key])
            if unknown:
                raise ValueError(f"Unknown {key} config keys: {sorted(unknown)}")
            defaults[key].update(value)
        else:
            defaults[key] = value
    device = defaults["device"]
    if not isinstance(device, str):
        raise ValueError("device must be auto or a torch device string")
    if device != "auto":
        try:
            torch.device(device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid device: {device}") from error
    detector = defaults["detector"]
    if detector["backend"] != "torchvision_fasterrcnn":
        raise ValueError("detector.backend must be torchvision_fasterrcnn")
    checkpoint = detector["checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("detector.checkpoint must be a nonempty local path or null")
    DetectionConfig(**{key: detector[key] for key in asdict(DetectionConfig())})
    threshold = defaults["matching"]["iou_threshold"]
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise ValueError("matching.iou_threshold must be finite and in (0,1]")
    return defaults


def load_detection_config(path: Path) -> dict:
    """Load validated YAML and resolve local checkpoint paths beside the YAML file."""
    path = Path(path)
    config = validate_detection_config(yaml.safe_load(path.read_text(encoding="utf-8")))
    checkpoint = config["detector"]["checkpoint"]
    if checkpoint is not None:
        config["detector"]["checkpoint"] = str((path.resolve().parent / checkpoint).resolve())
    return config
