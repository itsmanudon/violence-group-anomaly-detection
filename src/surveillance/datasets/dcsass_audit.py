"""Decode and fingerprint every installed DCSASS clip without changing annotations."""

import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from surveillance.datasets.dcsass_protocol import CLASSES, behavior_label, read_raw_labels
from surveillance.datasets.preparation import VIDEO_SUFFIXES, source_identity


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path) -> dict:
    """Bind all file names, byte sizes and write times for copy/mutation detection."""
    return {
        p.relative_to(root).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def audit_video(path: Path) -> dict:
    """Check metadata, full decode, stable geometry and content identity."""
    row = {"sha256": sha256(path), "bytes": path.stat().st_size, "readable": False}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("cannot_open")
        fps, reported = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if not np.isfinite(fps) or fps <= 0 or min(reported, width, height) < 1:
            raise ValueError("invalid_metadata")
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError("changing_resolution")
            count += 1
        row.update(
            fps=fps,
            num_frames=count,
            reported_num_frames=reported,
            width=width,
            height=height,
            duration_sec=count / fps,
        )
        if not count or count != reported:
            raise ValueError("incomplete_decode_or_frame_count_mismatch")
        row["readable"] = True
    except (ValueError, cv2.error) as error:
        row["error"] = str(error)
    finally:
        cap.release()
    return row


def audit_dcsass(root: Path, workers: int = 4, max_clips: int | None = None) -> dict:
    """Audit one explicit export root; duplicate installations are not populations."""
    root = Path(root).resolve()
    before = inventory(root)
    labels, issues = read_raw_labels(root / "Labels")
    videos = sorted(root / name for name in before if Path(name).suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise ValueError("No DCSASS video files")
    stems = [p.stem for p in videos]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate clip identifiers: select one dataset installation root")
    if max_clips is not None:
        videos = videos[:max_clips]
    start = time.perf_counter()
    rows, hashes = [], defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (path, decoded) in enumerate(
            zip(videos, pool.map(audit_video, videos), strict=True), 1
        ):
            category = path.relative_to(root).parts[0]
            source = source_identity(path, {})
            parent_source = source_identity(Path(path.parent.name), {})
            if source != parent_source:
                raise ValueError(f"Clip/parent source disagreement: {path}")
            annotation = labels.get(path.stem)
            label = annotation["label"] if annotation else None
            group = behavior_label(category, label) if annotation else None
            if annotation and annotation["category"] != category:
                raise ValueError(f"Annotation/category disagreement: {path}")
            selected = group is not None and decoded["readable"]
            reason = (
                "unreadable_video"
                if not decoded["readable"]
                else "missing_or_invalid_annotation"
                if not annotation
                else "excluded_from_actor_transformer_human_centric_v1"
                if group is None
                else None
            )
            row = dict(
                clip_id=path.stem,
                source_video_id=source,
                path=str(path),
                category=category,
                binary_label=label,
                group_label=group,
                selected=selected,
                exclusion_reason=reason,
                **decoded,
            )
            rows.append(row)
            hashes[row["sha256"]].append(row["clip_id"])
            if i % 500 == 0:
                print(f"Audited {i}/{len(videos)} clips", flush=True)
    after = inventory(root)
    if before != after:
        raise ValueError("DCSASS changed during audit; wait for copy completion and rerun")
    good = [r for r in rows if r["readable"]]
    return {
        "schema_version": 1,
        "status": "audited",
        "dataset_root": str(root),
        "bounded_preflight": max_clips is not None,
        "inventory_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        "label_sha256": {p.name: sha256(p) for p in sorted((root / "Labels").glob("*.csv"))},
        "total_clips": len(rows),
        "readable_clips": len(good),
        "unreadable_clips": len(rows) - len(good),
        "source_count": len({r["source_video_id"] for r in rows}),
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "selected_class_counts": dict(
            Counter(CLASSES[r["group_label"]] for r in rows if r["selected"])
        ),
        "binary_label_counts": dict(Counter(str(r["binary_label"]) for r in rows)),
        "exclusion_counts": dict(Counter(r["exclusion_reason"] for r in rows if not r["selected"])),
        "missing_video_annotations": sorted(set(labels) - set(stems)),
        "annotation_issues": issues,
        "resolution_counts": dict(Counter(f"{r['width']}x{r['height']}" for r in good)),
        "fps_counts": dict(Counter(str(r["fps"]) for r in good)),
        "duration_quantiles_sec": np.quantile(
            [r["duration_sec"] for r in good], [0, 0.25, 0.5, 0.75, 1]
        ).tolist()
        if good
        else [],
        "duplicates": [ids for ids in hashes.values() if len(ids) > 1],
        "seconds": time.perf_counter() - start,
        "clips": rows,
    }
