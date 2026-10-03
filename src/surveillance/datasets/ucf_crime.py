"""UCF-Crime categories, optional official splits and frame annotations."""

from dataclasses import replace
from pathlib import Path

from surveillance.datasets.common import Record, check_leakage
from surveillance.datasets.preparation import discover_videos, make_record

CATEGORIES = {
    "abuse",
    "arrest",
    "arson",
    "assault",
    "burglary",
    "explosion",
    "fighting",
    "roadaccidents",
    "robbery",
    "shooting",
    "shoplifting",
    "stealing",
    "vandalism",
}


def prepare_ucf_crime(
    root: Path, output: Path, source_map: dict[str, str] | None = None
) -> list[Record]:
    """Prepare category directories using video-level weak supervision."""
    records = []
    for path in discover_videos(root):
        category = next(
            (
                part
                for part in path.relative_to(root).parts[:-1]
                if part.casefold() in CATEGORIES
                or part.casefold() in {"normal", "normal_videos", "normalvideos"}
            ),
            None,
        )
        if category is None:
            raise ValueError(f"Unknown UCF category for {path}; use documented category folders")
        label = int(category.casefold() in CATEGORIES)
        records.append(
            make_record("ucf_crime", path, root, output, label, category, source_map or {})
        )
    return records


def apply_official_splits(records: list[Record], train_list: Path, test_list: Path) -> list[Record]:
    """Apply official filename lists (first whitespace-delimited field per line)."""

    def names(path: Path) -> set[str]:
        return {
            Path(line.split()[0].replace("\\", "/")).name.casefold()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }

    train_names, test_names = names(train_list), names(test_list)
    if train_names & test_names:
        raise ValueError("Official train/test lists overlap")
    result = []
    for row in records:
        name = Path(row.path).name.casefold()
        if name not in train_names | test_names:
            raise ValueError(f"Video absent from official lists: {name}")
        result.append(replace(row, split="train" if name in train_names else "test"))
    installed = {Path(row.path).name.casefold() for row in records}
    missing = (train_names | test_names) - installed
    if missing:
        raise ValueError(f"Official lists reference uninstalled videos: {sorted(missing)[:5]}")
    check_leakage(result)
    return result


def apply_frame_annotations(records: list[Record], annotation_file: Path) -> list[Record]:
    """Parse UCF: filename category start end [start end ...], 1-based inclusive.

    -1 -1 marks an absent interval. Convert to seconds [start-1,end)/fps.
    Only listed records acquire annotations; unknown abnormal intervals stay None.
    """
    annotations = {}
    for line in annotation_file.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4 or len(fields[2:]) % 2:
            raise ValueError(f"Invalid UCF annotation: {line}")
        values = list(map(int, fields[2:]))
        pairs = []
        for start, end in zip(values[::2], values[1::2]):
            if (start, end) == (-1, -1):
                continue
            if start < 1 or end < start:
                raise ValueError(f"Invalid 1-based inclusive frame interval: {line}")
            pairs.append((start, end))
        key = Path(fields[0].replace("\\", "/")).name.casefold()
        if key in annotations:
            raise ValueError(f"Duplicate annotation for {key}")
        annotations[key] = pairs
    result = []
    for row in records:
        name = Path(row.path).name.casefold()
        if name in annotations:
            if row.fps is None or row.num_frames is None:
                raise ValueError("Frame annotations require fps and num_frames")
            if any(end > row.num_frames for _, end in annotations[name]):
                raise ValueError(f"Annotation exceeds video frame count: {name}")
            row = replace(
                row,
                temporal_annotations=[
                    [(a - 1) / row.fps, b / row.fps] for a, b in annotations[name]
                ],
            )
        result.append(row)
    return result
