"""Automatic person detection independent of ground-truth actor annotations."""

from surveillance.detection.config import (
    DetectionConfig,
    load_detection_config,
    validate_detection_config,
)
from surveillance.detection.person_detector import (
    DetectionResult,
    PersonDetector,
    filter_detections,
)
from surveillance.detection.records import (
    DetectionRecord,
    detection_fingerprint,
    read_detections,
    write_detections,
)

__all__ = [
    "DetectionConfig",
    "DetectionResult",
    "PersonDetector",
    "filter_detections",
    "DetectionRecord",
    "detection_fingerprint",
    "read_detections",
    "write_detections",
    "load_detection_config",
    "validate_detection_config",
]
