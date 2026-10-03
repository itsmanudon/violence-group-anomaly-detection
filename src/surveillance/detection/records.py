"""Versioned detection JSONL with explicit pixel coordinates and feature alignment."""

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from surveillance.detection.person_detector import DetectionResult, canonical_order


@dataclass(frozen=True)
class DetectionRecord:
    """One center frame and optional per-detection features relative to its JSONL."""

    dataset: str
    video_id: str
    source_video_id: str
    clip_id: str
    frame_index: int
    result: DetectionResult
    pose_feature_path: str | None = None
    rgb_feature_path: str | None = None
    feature_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for value in (self.dataset, self.video_id, self.source_video_id, self.clip_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("dataset/video/source/clip identifiers must be nonempty strings")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        if not isinstance(self.result, DetectionResult):
            raise ValueError("result must be a DetectionResult")
        person_class = self.result.metadata.get("person_class_id", 1)
        if type(person_class) is not int or person_class < 0:
            raise ValueError("metadata.person_class_id must be a nonnegative integer")
        if (self.result.class_ids != person_class).any():
            raise ValueError("Detection records must contain only the declared person class")
        for value in (self.pose_feature_path, self.rgb_feature_path):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Feature paths must be nonempty strings or null")
        if not isinstance(self.feature_metadata, dict):
            raise ValueError("feature_metadata must be a dict")

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset, self.clip_id


def detection_fingerprint(result: DetectionResult) -> str:
    """Hash ordered geometry, scores, class IDs and native image size for cache identity."""
    payload = {
        "boxes": result.boxes.tolist(),
        "scores": result.scores.tolist(),
        "class_ids": result.class_ids.tolist(),
        "image_size": list(result.image_size),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical(record):
    result = record.result
    order = canonical_order(result.boxes, result.scores, result.class_ids)
    if order != list(range(len(order))):
        if record.pose_feature_path is not None or record.rgb_feature_path is not None:
            raise ValueError("Unsorted detections with feature paths would break feature alignment")
        result = DetectionResult(
            result.boxes[order],
            result.scores[order],
            result.class_ids[order],
            result.image_size,
            result.metadata,
        )
        record = replace(record, result=result)
    return record


def _validated(records):
    if not records:
        raise ValueError("Detection file is empty")
    keys, centers = set(), set()
    output = []
    for record in records:
        if record.key in keys:
            raise ValueError(f"Duplicate detection clip: {record.key}")
        keys.add(record.key)
        center = record.source_video_id, record.frame_index
        if center in centers:
            raise ValueError(f"Duplicate source/center frame: {center}")
        centers.add(center)
        output.append(_canonical(record))
    return sorted(output, key=lambda row: row.key)


def write_detections(records: list[DetectionRecord], path: Path) -> None:
    """Validate and save deterministic v1 JSONL; zero-person scenes remain explicit."""
    rows = []
    for record in _validated(list(records)):
        result = record.result
        rows.append(
            {
                "version": 1,
                "coordinate_system": "absolute_xyxy",
                "dataset": record.dataset,
                "video_id": record.video_id,
                "source_video_id": record.source_video_id,
                "clip_id": record.clip_id,
                "frame_index": record.frame_index,
                "image_height": result.image_size[0],
                "image_width": result.image_size[1],
                "detections": [
                    {"box": box, "score": score, "class_id": class_id}
                    for box, score, class_id in zip(
                        result.boxes.tolist(),
                        result.scores.tolist(),
                        result.class_ids.tolist(),
                        strict=True,
                    )
                ],
                "metadata": result.metadata,
                "pose_feature_path": record.pose_feature_path,
                "rgb_feature_path": record.rgb_feature_path,
                "feature_metadata": record.feature_metadata,
            }
        )
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_detections(path: Path) -> list[DetectionRecord]:
    """Read v1 absolute-pixel records and canonicalize only uncached actor order."""
    path = Path(path)
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if type(row["version"]) is not int or row.pop("version") != 1:
                raise ValueError("Unsupported detection version (expected 1)")
            if row.pop("coordinate_system") != "absolute_xyxy":
                raise ValueError("coordinate_system must be absolute_xyxy")
            size = row.pop("image_height"), row.pop("image_width")
            detections = row.pop("detections")
            if not isinstance(detections, list):
                raise ValueError("detections must be a list")
            for detection in detections:
                if not isinstance(detection, dict) or set(detection) != {
                    "box",
                    "score",
                    "class_id",
                }:
                    raise ValueError("Each detection needs box, score and class_id")
                if type(detection["class_id"]) is not int:
                    raise ValueError("class_id must be an integer")
            boxes = (
                torch.tensor([d["box"] for d in detections]) if detections else torch.empty(0, 4)
            )
            result = DetectionResult(
                boxes,
                torch.tensor([d["score"] for d in detections]),
                torch.tensor([d["class_id"] for d in detections]),
                size,
                row.pop("metadata", {}),
            )
            records.append(DetectionRecord(result=result, **row))
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            raise ValueError(f"{path}: invalid detection line {number}: {error}") from error
    return _validated(records)
