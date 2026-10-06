"""Source-safe UCF baseline supporting external-drive installation."""

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from surveillance.datasets.common import check_leakage, split_records
from surveillance.datasets.dcsass_audit import sha256
from surveillance.experiments.dcsass_cache import write_json
from surveillance.video.segmentation import c3d_segment_frame_ranges

C3D_SHA256 = "beac4ff065de3663bf5c6bab36aedca4fe56ec59b231c66bf5374c23eda14831"


def remove_training_duplicates(records, identities: dict):
    """Keep the authors' full test set, remove training copies of identical content."""
    groups = defaultdict(list)
    for row in records:
        groups[identities[row.video_id]].append(row)
    kept, excluded = [], []
    for digest in sorted(groups):
        group = sorted(groups[digest], key=lambda r: r.video_id)
        if len({r.label for r in group}) != 1:
            raise ValueError("Conflicting binary labels for exact-content duplicate videos")
        tests = [r for r in group if r.split == "test"]
        if tests:
            kept.extend(tests)
            excluded.extend(r for r in group if r.split != "test")
        else:
            kept.append(group[0])
            excluded.extend(group[1:])
    return sorted(kept, key=lambda r: r.video_id), sorted(excluded, key=lambda r: r.video_id)


def protect_shared_sources(records, shared: dict[str, str], seed: int = 0):
    """Keep authors' test set; exclude train-list sources already held out in DCSASS.

    Existing shared validation sources stay validation. This deliberately changes
    the authors' training population to preserve independent cascade evaluation.
    """
    retained, excluded = [], []
    remaining = defaultdict(list)
    for row in records:
        assigned = shared.get(row.source_video_id)
        if row.split == "test":
            if assigned is not None and assigned != "test":
                raise ValueError("Shared protocol would expose an official test source")
            retained.append(row)
        elif assigned == "test":
            excluded.append(row)
        elif assigned:
            retained.append(replace(row, split=assigned))
        else:
            remaining[(row.anomaly_type or "Anomaly") if row.label else "Normal"].append(row)
    for category in sorted(remaining):
        retained.extend(split_records(remaining[category], seed, (0.85, 0.15, 0)))
    retained.sort(key=lambda r: (r.dataset, r.video_id))
    excluded.sort(key=lambda r: (r.dataset, r.video_id))
    check_leakage(retained)
    return retained, excluded


def extract_cached_bag(row, video_sha: str, extractor, cache: Path):
    """Require video/feature/backbone identity before reusing a complete cache."""
    video = Path(row.path)
    if sha256(video) != video_sha:
        raise ValueError(f"Video changed: {row.video_id}")
    path = Path(cache) / "features" / f"{row.video_id}.npy"
    sidecar = path.with_suffix(".json")
    expected = {
        "video_id": row.video_id,
        "source_video_id": row.source_video_id,
        "video_sha256": video_sha,
        "c3d_sha256": C3D_SHA256,
        "mean": [104, 117, 128],
        "channel_order": "rgb",
        "unit_frames": 16,
        "shape": [32, 4096],
        "segment_frame_ranges": [list(pair) for pair in c3d_segment_frame_ranges(row.num_frames)],
    }
    if path.exists() and sidecar.exists():
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in expected.items()) or saved.get(
            "feature_sha256"
        ) != sha256(path):
            raise ValueError(f"Stale C3D cache: {row.video_id}")
        values = np.load(path, allow_pickle=False)
        reused = True
    else:
        values = extractor.extract_video(video, 32)
        if values.shape != (32, 4096) or not np.isfinite(values).all():
            raise ValueError(f"Invalid C3D cache values: {row.video_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, values)
        temporary.replace(path)
        write_json(
            sidecar,
            {
                **expected,
                "feature_sha256": sha256(path),
                "preprocessing": extractor.metadata,
                "timings": getattr(extractor, "last_timings", {}),
            },
        )
        reused = False
    if values.shape != (32, 4096) or not np.isfinite(values).all():
        raise ValueError(f"Invalid C3D cache values: {row.video_id}")
    return path, reused
