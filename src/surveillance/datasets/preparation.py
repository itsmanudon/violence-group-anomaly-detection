"""Shared preparation helpers with explicit source identity recovery."""

import csv
import json
import os
import re
from pathlib import Path

from surveillance.datasets.common import Record
from surveillance.video.decode import probe_video

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def source_identity(path: Path, source_map: dict[str, str]) -> str:
    """Recover UCF basename before _x264/clip suffix; otherwise require mapping.

    Recognizes e.g. Fighting001_x264_12 and Normal_Videos001_x264.
    It never pretends unknown filenames identify independent original sources.
    """
    key = path.as_posix()
    if key in source_map:
        return source_map[key].casefold()
    match = re.fullmatch(r"(.+\d+)_x264(?:[_-].*)?", path.stem, re.IGNORECASE)
    if match:
        return match[1].casefold()
    raise ValueError(
        f"Cannot recover source identity for {key}. Supply --source-map JSON "
        "mapping relative video paths to original UCF source IDs."
    )


def read_source_map(path: Path | None) -> dict[str, str]:
    """Load an explicit relative-path -> canonical source ID JSON mapping."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in data.items()
    ):
        raise ValueError("Source map must be a JSON object of nonempty string IDs")
    return data


def discover_videos(root: Path) -> list[Path]:
    """Enumerate installed videos; do not download anything."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}. See data/README.md.")
    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise ValueError(f"No videos found under {root}; expected mp4/avi/mov/mkv files")
    return videos


def make_record(
    dataset: str,
    path: Path,
    root: Path,
    output: Path,
    label: int,
    anomaly_type: str,
    source_map: dict[str, str],
    source: str | None = None,
) -> Record:
    """Probe metadata and construct a portable manifest-relative record."""
    relative = path.relative_to(root)
    metadata = probe_video(path)
    return Record(
        dataset,
        relative.with_suffix("").as_posix(),
        source.casefold() if source else source_identity(relative, source_map),
        Path(os.path.relpath(path.resolve(), output.resolve().parent)).as_posix(),
        "train",
        label,
        anomaly_type,
        metadata.duration_sec,
        metadata.fps,
        metadata.num_frames,
    )


def read_label_csv(path: Path) -> dict[str, dict[str, str]]:
    """Read normalized DCSASS CSV with path,label and optional source_video_id/type.

    Local exports vary: explicitly normalize their columns instead of inferring
    clip labels from anomaly-category directories (normal clips can occur there).
    """
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label"} <= set(reader.fieldnames or []):
            raise ValueError("Labels CSV requires header path,label; see data/README.md")
        result = {}
        for row in reader:
            key = Path(row["path"]).as_posix()
            if key in result or row["label"] not in {"0", "1"}:
                raise ValueError(f"Duplicate path or invalid binary label: {key}")
            result[key] = row
        return result
