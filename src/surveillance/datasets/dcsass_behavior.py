"""Validated cached detected actors for DCSASS group-only optimization."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from surveillance.datasets.dcsass_audit import sha256
from surveillance.experiments.dcsass_cache import (
    DETECTOR_SHA256,
    I3D_SHA256,
    validate_detection_cache,
)


def load_behavior_sample(row: dict, cache: Path) -> dict | None:
    """Load only genuine actors; validate byte/identity/sampling/feature contracts."""
    cache = Path(cache)
    payload = json.loads((cache / "detections" / f"{row['clip_id']}.json").read_text())
    result = validate_detection_cache(payload, row, DETECTOR_SHA256)
    if not len(result.boxes):
        return None
    path = cache / "rgb" / f"{row['clip_id']}.npy"
    metadata = json.loads(path.with_suffix(".json").read_text())
    expected = {
        "clip_id": row["clip_id"],
        "source_video_id": row["source_video_id"],
        "video_sha256": row["sha256"],
        "reference_frame": payload["reference_frame"],
        "frame_indices": payload["frame_indices"],
        "detection_fingerprint": payload["detection_fingerprint"],
        "detector_sha256": DETECTOR_SHA256,
        "i3d_sha256": I3D_SHA256,
        "shape": [len(result.boxes), 20800],
        "feature_sha256": sha256(path),
    }
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Stale DCSASS RGB feature identity/content: {row['clip_id']}")
    features = np.load(path, allow_pickle=False)
    if (
        features.shape != (len(result.boxes), 20800)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError("DCSASS RGB feature tensor must be finite float32 [N,20800]")
    height, width = result.image_size
    return {
        "actor_boxes": result.boxes / torch.tensor([width, height, width, height]),
        "rgb_features": torch.from_numpy(features),
        "group_label": row["group_label"],
        "metadata": {
            **row,
            "actor_count": len(result.boxes),
            "detection_scores": payload["scores"],
        },
    }


class BehaviorFeatureDataset(Dataset):
    """Preload validated covered features while retaining the complete population."""

    def __init__(self, manifest: Path, cache: Path, split: str, *, preflight: bool = False):
        manifest, cache = Path(manifest), Path(cache)
        registration = json.loads((cache / "registration.json").read_text())
        if registration["manifest_sha256"] != sha256(manifest):
            raise ValueError("Behavior cache/manifest identity changed")
        if registration["max_clips"] is not None and not preflight:
            raise ValueError("Bounded preflight features cannot define a full experiment")
        allowed = set(registration["clip_ids"])
        all_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        seen = {}
        for row in all_rows:
            source = row["source_video_id"]
            if source in seen and seen[source] != row["split"]:
                raise ValueError("Behavior source split leakage")
            seen[source] = row["split"]
        self.population = [r for r in all_rows if r["split"] == split and r["clip_id"] in allowed]
        self.samples, self.uncovered = [], []
        for row in self.population:
            sample = load_behavior_sample(row, cache)
            if sample is None:
                self.uncovered.append(row)
            else:
                self.samples.append(sample)
        if not self.samples:
            raise ValueError(f"No covered DCSASS clips in {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]
