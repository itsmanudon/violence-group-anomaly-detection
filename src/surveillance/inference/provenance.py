"""Pinned offline deployment identity shared by live results and example caches."""

import json
import re
from dataclasses import asdict
from pathlib import Path

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.dcsass_protocol import CLASSES
from surveillance.detection.config import DetectionConfig

ASSET_KEYS = (
    "sultani_checkpoint",
    "c3d_checkpoint",
    "behavior_checkpoint",
    "detector_checkpoint",
    "i3d_checkpoint",
)


def pipeline_identity(config: dict) -> dict:
    """Require explicit pins without opening models, allowing offline cached playback."""
    hashes = config.get("model_sha256", {})
    if set(hashes) != set(ASSET_KEYS) or any(
        not isinstance(hashes[key], str) or not re.fullmatch("[0-9a-f]{64}", hashes[key])
        for key in ASSET_KEYS
    ):
        raise ValueError(
            "Demo model identities are not frozen; install completed training receipts"
        )
    return {
        "schema_version": 1,
        "models": dict(hashes),
        "training_status": "validation_selected_non_preflight",
        "training_population": config["training_population"],
        "preprocessing": {
            "c3d_mean": config["c3d_mean"],
            "c3d_channel_order": config["c3d_channel_order"],
            "c3d_temporal_unit_frames": 16,
            "num_anomaly_segments": 32,
            "detector": asdict(DetectionConfig(confidence_threshold=0.7)),
            "i3d": {
                "frames": 10,
                "image_size": [480, 720],
                "endpoint": "Mixed_4f",
                "temporal_pool": "mean",
                "spatial_resize": [90, 160],
                "roi_size": [5, 5],
            },
            "classes": list(CLASSES),
        },
        "policy": {
            "anomaly_threshold": config["anomaly_threshold"],
            "max_windows": config["max_behavior_windows"],
        },
    }


def verify_selected_checkpoint(path: Path) -> dict:
    """Only a completed validation selection can serve a learned subsystem."""
    receipt = path.parent / "selection.json"
    if not receipt.is_file():
        raise ValueError(f"Completed validation selection receipt missing: {receipt}")
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        saved.get("dry_run", False)
        or saved.get("selection_frozen") is not True
        or saved.get("checkpoint_sha256") != sha256(path)
    ):
        raise ValueError(f"Checkpoint selection is incomplete, preflight or changed: {path}")
    return saved
