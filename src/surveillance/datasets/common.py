"""JSONL manifests and deterministic source-aware splits."""

import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Record:
    """Paths are relative to the manifest directory or absolute.

    Temporal intervals use seconds, [start, end). None means unknown annotations;
    [] means explicitly annotated with no anomalous intervals.
    source_video_id uses shared UCF identity across derived datasets.
    """

    dataset: str
    video_id: str
    source_video_id: str
    path: str
    split: str
    label: int
    anomaly_type: str | None = None
    duration_sec: float | None = None
    fps: float | None = None
    num_frames: int | None = None
    temporal_annotations: list[list[float]] | None = None
    feature_path: str | None = None
    feature_sha256: str | None = None

    def __post_init__(self) -> None:
        if not all((self.dataset, self.video_id, self.source_video_id, self.path)):
            raise ValueError("dataset, video_id, source_video_id and path must be nonempty")
        if self.split not in {"train", "val", "test"} or self.label not in {0, 1}:
            raise ValueError("split must be train/val/test and label must be 0/1")
        if self.feature_sha256 is not None and (
            not isinstance(self.feature_sha256, str)
            or len(self.feature_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.feature_sha256)
        ):
            raise ValueError("feature_sha256 must be a lowercase SHA256 identity")
        for value in (self.duration_sec, self.fps, self.num_frames):
            if value is not None and (value <= 0 or not float("-inf") < value < float("inf")):
                raise ValueError("Video metadata must be positive and finite")
        if self.temporal_annotations is not None:
            for interval in self.temporal_annotations:
                if len(interval) != 2 or not 0 <= interval[0] < interval[1] < float("inf"):
                    raise ValueError("Temporal intervals must be finite [start,end) seconds")
                if self.duration_sec is not None and interval[1] > self.duration_sec + 1e-6:
                    raise ValueError("Annotation extends beyond video duration")


def read_manifest(path: Path) -> list[Record]:
    """Read and validate JSONL, reporting invalid line numbers."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(Record(**json.loads(line)))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}: invalid manifest line {line_number}: {error}") from error
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    keys = [(r.dataset, r.video_id) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Manifest contains duplicate dataset/video_id pairs")
    return records


def write_manifest(records: list[Record], path: Path) -> None:
    """Write records as UTF-8 JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(r)) + "\n" for r in records), encoding="utf-8")


def resolve_path(value: str, manifest: Path) -> Path:
    """Resolve paths consistently for all manifest consumers."""
    path = Path(value)
    return path if path.is_absolute() else Path(manifest).resolve().parent / path


def check_leakage(records: list[Record]) -> None:
    """Reject sources crossing splits, including across dataset names."""
    seen: dict[str, str] = {}
    for record in records:
        if record.source_video_id in seen and seen[record.source_video_id] != record.split:
            raise ValueError(f"Source-video leakage: {record.source_video_id} crosses splits")
        seen[record.source_video_id] = record.split


def split_records(
    records: list[Record], seed: int = 7, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> list[Record]:
    """Shuffle unique sources, assign whole groups; independent of record order.

    Ratios refer to source counts, not clips. This utility is not an official
    UCF split or stratification guarantee; inspect class counts before training.
    """
    if (
        len(ratios) != 3
        or not all(math.isfinite(r) for r in ratios)
        or min(ratios) < 0
        or abs(sum(ratios) - 1) > 1e-8
    ):
        raise ValueError("Three nonnegative split ratios must sum to one")
    sources = sorted({r.source_video_id for r in records})
    random.Random(seed).shuffle(sources)
    targets = [len(sources) * ratio for ratio in ratios]
    counts = [math.floor(target) for target in targets]
    remainder = len(sources) - sum(counts)
    priority = sorted(
        (i for i, ratio in enumerate(ratios) if ratio > 0),
        key=lambda i: (-(targets[i] - counts[i]), i),
    )
    for index in priority[:remainder]:
        counts[index] += 1
    first = counts[0]
    second = first + counts[1]
    assignments = {
        s: "train" if i < first else "val" if i < second else "test" for i, s in enumerate(sources)
    }
    result = [replace(r, split=assignments[r.source_video_id]) for r in records]
    return sorted(result, key=lambda r: (r.dataset, r.video_id))
