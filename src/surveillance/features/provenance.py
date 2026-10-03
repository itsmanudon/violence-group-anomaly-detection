"""Versioned actor extraction identities and strict benchmark cache validation."""

import hashlib
import json
from pathlib import Path

import numpy as np

from surveillance.datasets.collective import ActorRecord, read_actor_manifest
from surveillance.datasets.collective_validation import DatasetValidationError
from surveillance.datasets.common import resolve_path
from surveillance.datasets.detected_actors import (
    DetectedActorDataset,
    normalized_detection_boxes,
    source_fingerprint,
)
from surveillance.detection.records import detection_fingerprint


def file_sha256(path: Path) -> str:
    """Hash file bytes without loading the complete artifact into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config_fingerprint(config: dict) -> str:
    """Canonical JSON SHA256, independent of mapping insertion order."""
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def extraction_config(
    modality: str,
    image_size: list[int] | tuple[int, int],
    checkpoint_hash: str,
    backbone_metadata: dict,
    box_source: str,
) -> dict:
    """Describe the actual adapters' preprocessing, actor order and RoI geometry."""
    if modality not in {"pose", "rgb"} or box_source not in {"ground_truth", "detections"}:
        raise ValueError("Expected pose/rgb modality and ground_truth/detections box source")
    if len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size):
        raise ValueError("image_size must be positive [height,width]")
    config = {
        "version": "actor_extraction_v2",
        "modality": modality,
        "image_size": list(image_size),
        "checkpoint_sha256": checkpoint_hash,
        "backbone": json.loads(json.dumps(backbone_metadata, allow_nan=False)),
        "box_source": box_source,
        "normalization": "backbone_only",
        "color_order": "rgb",
        "input_range": [0, 1],
        "box_convention": "normalized_xyxy_image_edges",
        "actor_order": "manifest_order" if box_source == "ground_truth" else "detection_order",
        "temporal_offsets": list(range(-5, 5)),
        "center_position": 5,
        "edge_policy": "replicate",
        "clip_outside": False,
        "roi_align": {"aligned": True, "sampling_ratio": 2, "spatial_scale": 1.0},
    }
    if modality == "pose":
        config.update(
            pose_crop_size=[256, 192], feature_shape=[32, 64, 48], pose_frame="annotation_center"
        )
    else:
        config.update(
            temporal_pooling="mean",
            feature_map_size=[90, 160],
            roi_size=[5, 5],
            feature_shape=[832, 5, 5],
            resize_mode="bilinear",
            resize_align_corners=False,
        )
    return config


def scene_fingerprint(row: ActorRecord) -> str:
    """Hash ordered source IDs, indices, normalized GT boxes and supervision.

    Feature paths and manifest location are deliberately excluded. Ordered image
    path identity is checked separately using the existing source fingerprint.
    """
    return config_fingerprint(
        {
            "dataset": row.dataset,
            "video_id": row.video_id,
            "source_video_id": row.source_video_id,
            "clip_id": row.clip_id,
            "frame_indices": row.frame_indices,
            "actor_boxes": row.actor_boxes,
            "actor_labels": row.actor_labels,
            "group_label": row.group_label,
        }
    )


def source_content_fingerprint(row: ActorRecord, manifest: Path) -> str:
    """Bind ordered source frame paths to their extraction-time SHA256 bytes.

    Edge-replicated frames retain their ordered entries, while each resolved file
    is read only once per scene. Missing files fail rather than weakening the
    identity to path references. No images or annotations are modified.
    """
    digests = {}
    ordered = []
    for value in row.frame_paths:
        path = resolve_path(value, Path(manifest)).resolve()
        if path not in digests:
            digests[path] = file_sha256(path)
        ordered.append({"path": str(path), "sha256": digests[path]})
    return config_fingerprint({"ordered_source_frames": ordered})


def validate_feature_caches(
    manifest: Path,
    actor_config: dict,
    expected: dict,
    detections: Path | None = None,
) -> dict:
    """Verify every selected modality across all splits for benchmark use.

    ``expected`` maps modalities to checkpoint_sha256, architecture, endpoint
    and optionally extraction_config_hash. Legacy cache acceptance elsewhere is
    unchanged; this benchmark API requires schema version 2 provenance.
    """
    manifest = Path(manifest)
    report = {
        "schema_version": 1,
        "kind": "actor_feature_caches",
        "valid": False,
        "box_source": "detections" if detections else "ground_truth",
        "checked_features": 0,
        "errors": [],
        "warnings": [],
        "features": [],
    }
    model = actor_config["model"]
    mode = model["mode"]
    modalities = ["pose", "rgb"] if "fusion" in mode else [mode.split("_")[0]]
    image_size = actor_config["data"]["image_size"]
    try:
        rows = read_actor_manifest(manifest)
        dataset = None
        if detections is not None:
            dataset = DetectedActorDataset(
                manifest,
                Path(detections),
                split=None,
                mode=mode,
                pose_feature_dim=model["pose_feature_dim"],
                rgb_feature_dim=model["rgb_feature_dim"],
                image_size=tuple(image_size),
                input_mode="precomputed",
            )
        for index, row in enumerate(rows):
            for modality in modalities:
                try:
                    spec = expected.get(modality)
                    if not isinstance(spec, dict) or any(
                        not spec.get(key)
                        for key in ("checkpoint_sha256", "architecture", "endpoint")
                    ):
                        raise ValueError(
                            f"Expected {modality} checkpoint/architecture/endpoint is required"
                        )
                    record = dataset.detections[index] if dataset else row
                    value = getattr(record, f"{modality}_feature_path")
                    if value is None:
                        raise ValueError(
                            f"Missing {modality} feature path; rerun feature extraction"
                        )
                    path = resolve_path(value, Path(detections) if dataset else manifest)
                    if path.suffix.lower() != ".npy":
                        raise ValueError("Actor feature cache must be a numeric .npy array")
                    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                    if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
                        raise ValueError("Missing/legacy provenance; rerun actor_extraction_v2")
                    if dataset and metadata != record.feature_metadata.get(modality):
                        raise ValueError("Detected sidecar differs from embedded feature_metadata")
                    boxes = (
                        normalized_detection_boxes(record.result).tolist()
                        if dataset
                        else row.actor_boxes
                    )
                    identities = {
                        "dataset": row.dataset,
                        "clip_id": row.clip_id,
                        "source_video_id": row.source_video_id,
                        "scene_fingerprint": scene_fingerprint(row),
                        "source_fingerprint": source_fingerprint(row, manifest),
                        "source_content_fingerprint": source_content_fingerprint(row, manifest),
                        "box_source": report["box_source"],
                        "actor_boxes": boxes,
                        "checkpoint_sha256": spec["checkpoint_sha256"],
                        "image_size": list(image_size),
                        "feature_sha256": file_sha256(path),
                    }
                    if dataset:
                        identities["detection_fingerprint"] = detection_fingerprint(record.result)
                    for key, value in identities.items():
                        if metadata.get(key) != value:
                            raise ValueError(f"Stale cache: {key} mismatch")
                    backbone = metadata.get("backbone", {})
                    if not isinstance(backbone, dict):
                        raise ValueError("Invalid backbone metadata")
                    for key in ("architecture", "endpoint"):
                        if backbone.get(key) != spec[key]:
                            raise ValueError(f"Stale cache: backbone {key} mismatch")
                    current = extraction_config(
                        modality,
                        image_size,
                        spec["checkpoint_sha256"],
                        backbone,
                        report["box_source"],
                    )
                    fingerprint = config_fingerprint(current)
                    if (
                        metadata.get("extraction_config") != current
                        or metadata.get("extraction_config_hash") != fingerprint
                    ):
                        raise ValueError("Stale extraction configuration; rerun feature extraction")
                    if spec.get("extraction_config_hash", fingerprint) != fingerprint:
                        raise ValueError("Expected extraction configuration hash mismatch")
                    values = np.load(path, allow_pickle=False)
                    shape = (len(boxes), model[f"{modality}_feature_dim"])
                    if metadata.get("shape") != list(shape):
                        raise ValueError("Provenance shape differs from actor feature contract")
                    if (
                        values.shape != shape
                        or not np.issubdtype(values.dtype, np.floating)
                        or not np.isfinite(values).all()
                        or not np.isfinite(values.astype(np.float32)).all()
                    ):
                        raise ValueError(f"Expected finite float32-compatible features {shape}")
                    report["features"].append(
                        {
                            "clip_id": row.clip_id,
                            "split": row.split,
                            "modality": modality,
                            "shape": list(shape),
                            "feature_sha256": identities["feature_sha256"],
                            "extraction_config_hash": fingerprint,
                        }
                    )
                    report["checked_features"] += 1
                except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                    report["errors"].append(f"{row.clip_id} {modality}: {error}")
            if dataset:
                try:
                    dataset[
                        index
                    ]  # Preserve the existing detector-source/content validation boundary.
                except (OSError, ValueError, TypeError) as error:
                    report["errors"].append(f"{row.clip_id}: {error}")
    except (OSError, ValueError, TypeError) as error:
        report["errors"].append(str(error))
    report["valid"] = not report["errors"]
    if not report["valid"]:
        raise DatasetValidationError(report)
    return report
