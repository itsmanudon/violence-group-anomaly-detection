"""Join detector artifacts to annotated scenes without rewriting ground truth."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from surveillance.datasets.collective import ActorFeatureDataset, ActorRecord, read_actor_manifest
from surveillance.datasets.common import resolve_path
from surveillance.detection.matching import match_boxes, transfer_actor_labels
from surveillance.detection.records import detection_fingerprint, read_detections


def source_fingerprint(row: ActorRecord, manifest: Path) -> str:
    """Identify ordered source references, not image bytes; keep sources immutable."""
    identity = [
        row.dataset,
        row.video_id,
        row.source_video_id,
        row.clip_id,
        row.frame_indices,
        [str(resolve_path(p, manifest).resolve()) for p in row.frame_paths],
    ]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode()).hexdigest()


def normalized_detection_boxes(result) -> torch.Tensor:
    """Convert absolute xyxy to the model's normalized xyxy at an explicit boundary."""
    height, width = result.image_size
    return result.boxes / result.boxes.new_tensor([width, height, width, height])


def collate_actor_inputs(samples: list[dict]) -> dict:
    """Pad nonempty inference inputs without requiring or fabricating actor labels.

    Features are [N,D], boxes [N,4]. Returns [B,Nmax,D], [B,Nmax,4] and
    a True-valid [B,Nmax] mask. Labels, if present, remain evaluation metadata.
    Empty scenes must be handled by the pipeline before calling this function.
    """
    if not samples or any(len(s["actor_boxes"]) == 0 for s in samples):
        raise ValueError("collate_actor_inputs requires nonempty scenes")
    counts = [len(s["actor_boxes"]) for s in samples]
    batch = {
        "actor_boxes": pad_sequence([s["actor_boxes"] for s in samples], batch_first=True),
        "actor_valid_mask": torch.arange(max(counts))[None] < torch.tensor(counts)[:, None],
        "metadata": [s.get("metadata", {}) for s in samples],
    }
    for key in ("pose_features", "rgb_features", "frames"):
        present = [key in s for s in samples]
        if any(present) and not all(present):
            raise ValueError(f"Inconsistent {key} in actor input batch")
        if all(present):
            for s, count in zip(samples, counts, strict=True):
                values = s[key]
                if not values.is_floating_point() or not torch.isfinite(values).all():
                    raise ValueError(f"{key} must be finite floating point")
                if key != "frames" and (values.ndim != 2 or values.shape[0] != count):
                    raise ValueError(f"{key} must match the actor count [N,D]")
            batch[key] = (
                torch.stack([s[key] for s in samples])
                if key == "frames"
                else pad_sequence([s[key] for s in samples], batch_first=True)
            )
    return batch


class DetectedActorDataset(Dataset):
    """Cache-aware detector actors; unmatched action labels are -100, never invented.

    Empty samples retain source/group labels and bypass image/feature loading.
    Raw mode reuses the Milestone 2A decoder; cached mode verifies ordered boxes,
    source identity, feature shape and SHA256 before model inference.
    """

    def __init__(
        self,
        manifest: Path,
        detections: Path,
        split: str | None = "test",
        mode: str = "pose_only",
        pose_feature_dim: int = 98304,
        rgb_feature_dim: int = 20800,
        input_mode: str = "precomputed",
        image_size: tuple[int, int] = (480, 720),
        iou_threshold: float = 0.5,
    ):
        self.manifest, self.detection_path = Path(manifest), Path(detections)
        all_rows = read_actor_manifest(self.manifest)
        gt_keys = {(r.dataset, r.clip_id) for r in all_rows}
        det_rows = read_detections(self.detection_path)
        by_key = {r.key: r for r in det_rows}
        if set(by_key) - gt_keys:
            raise ValueError("Detection artifact contains scenes absent from the GT manifest")
        self.raw = ActorFeatureDataset(
            manifest, split, mode, pose_feature_dim, rgb_feature_dim, "raw", image_size
        )
        self.records = self.raw.records
        self.detections = []
        if input_mode not in {"raw", "precomputed"}:
            raise ValueError("input_mode must be raw or precomputed")
        self.input_mode, self.mode = input_mode, mode
        self.feature_dims = {"pose": pose_feature_dim, "rgb": rgb_feature_dim}
        self.iou_threshold = iou_threshold
        physical = {}
        splits = {(r.dataset, r.clip_id): r.split for r in all_rows}
        for det in det_rows:
            for path in (det.pose_feature_path, det.rgb_feature_path):
                if path is not None:
                    key = str(resolve_path(path, self.detection_path).resolve()).casefold()
                    if key in physical and physical[key] != splits[det.key]:
                        raise ValueError("Detected feature physical-path leakage across splits")
                    physical[key] = splits[det.key]
        for row in self.records:
            key = (row.dataset, row.clip_id)
            if key not in by_key:
                raise ValueError(
                    f"Missing detection record for {key}; record zero detections explicitly"
                )
            det = by_key[key]
            if (
                det.video_id != row.video_id
                or det.source_video_id != row.source_video_id
                or det.frame_index != row.frame_indices[5]
            ):
                raise ValueError(f"Detection source/reference frame mismatch for {key}")
            self.detections.append(det)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row, det = self.records[index], self.detections[index]
        result = det.result
        height, width = result.image_size
        gt = torch.tensor(row.actor_boxes, dtype=torch.float64) * torch.tensor(
            [width, height, width, height], dtype=torch.float64
        )
        matched = match_boxes(gt, result.boxes, self.iou_threshold)
        sample = {
            "actor_boxes": normalized_detection_boxes(result),
            "actor_labels": transfer_actor_labels(torch.tensor(row.actor_labels), matched),
            "group_label": row.group_label,
            "detection_result": result,
            "matching": matched,
            "metadata": {
                "dataset": row.dataset,
                "video_id": row.video_id,
                "source_video_id": row.source_video_id,
                "clip_id": row.clip_id,
                "split": row.split,
                "frame_indices": row.frame_indices,
                "box_source": "detections",
                "image_size": list(result.image_size),
            },
        }
        if not len(result.boxes):
            return sample
        if self.input_mode == "raw":
            center = resolve_path(row.frame_paths[5], self.manifest)
            image = cv2.imread(str(center))
            if image is None or tuple(image.shape[:2]) != tuple(result.image_size):
                raise ValueError(f"Detection image_size differs from source frame: {center}")
            sample["frames"] = self.raw[index]["frames"]
        else:
            modalities = ["pose", "rgb"] if "fusion" in self.mode else [self.mode.split("_")[0]]
            for modality in modalities:
                path = getattr(det, f"{modality}_feature_path")
                provenance = det.feature_metadata.get(modality, {})
                if "image_size" in provenance and provenance["image_size"] != list(
                    self.raw.image_size
                ):
                    raise ValueError(
                        f"{det.clip_id}: {modality} cache image_size differs from config"
                    )
                if path is None:
                    raise ValueError(
                        f"{det.clip_id}: missing detected {modality} features; run extraction"
                    )
                if provenance.get("detection_fingerprint") != detection_fingerprint(
                    result
                ) or provenance.get("source_fingerprint") != source_fingerprint(row, self.manifest):
                    raise ValueError(
                        f"{det.clip_id}: stale {modality} feature cache (boxes/source mismatch)"
                    )
                path = resolve_path(path, self.detection_path)
                if path.suffix.lower() != ".npy":
                    raise ValueError("Detected actor features must be .npy arrays")
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if provenance.get("feature_sha256") != digest:
                    raise ValueError(f"{path}: feature cache content hash mismatch")
                values = np.load(path, allow_pickle=False)
                expected = (len(result.boxes), self.feature_dims[modality])
                if (
                    values.shape != expected
                    or not np.issubdtype(values.dtype, np.floating)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(f"{path}: expected finite floating {expected} features")
                sample[f"{modality}_features"] = torch.from_numpy(values.astype(np.float32))
                if not torch.isfinite(sample[f"{modality}_features"]).all():
                    raise ValueError(f"{path}: feature values overflow float32")
        return sample
