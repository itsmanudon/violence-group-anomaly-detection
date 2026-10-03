"""DCSASS clip preparation, preserving per-clip binary annotations."""

from pathlib import Path

from surveillance.datasets.common import Record
from surveillance.datasets.preparation import discover_videos, make_record, read_label_csv


def prepare_dcsass(
    root: Path, labels: Path, output: Path, source_map: dict[str, str] | None = None
) -> list[Record]:
    """Prepare installed DCSASS using explicit clip labels, never folder guesses."""
    videos = discover_videos(root)
    annotations = read_label_csv(labels)
    records = []
    for path in videos:
        key = path.relative_to(root).as_posix()
        if key not in annotations:
            raise ValueError(f"Missing DCSASS clip label for {key}; add it to {labels}")
        row = annotations[key]
        records.append(
            make_record(
                "dcsass",
                path,
                root,
                output,
                int(row["label"]),
                row.get("anomaly_type") or path.parent.name,
                source_map or {},
                row.get("source_video_id"),
            )
        )
    unused = set(annotations) - {p.relative_to(root).as_posix() for p in videos}
    if unused:
        raise ValueError(f"Labels reference missing videos: {sorted(unused)[:5]}")
    return records
