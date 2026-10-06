"""DCSASS observable-behavior labels and deterministic original-source splitting."""

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("Normal", "Abuse", "Assault", "Fighting", "Robbery", "Vandalism")
EXCLUDED = frozenset(
    (
        "Arrest",
        "Arson",
        "Accident",
        "RoadAccidents",
        "Burglary",
        "Explosion",
        "Shooting",
        "Stealing",
        "Shoplifting",
    )
)


def behavior_label(category: str, binary_label: int) -> int | None:
    """Map actual binary annotations; excluded categories never enter this task."""
    if type(binary_label) is not int or binary_label not in (0, 1):
        raise ValueError("DCSASS labels must be binary integers")
    if category in EXCLUDED:
        return None
    if category not in CLASSES:
        raise ValueError(f"Unknown DCSASS category: {category}")
    if category == "Normal" and binary_label:
        raise ValueError("Normal category cannot have an abnormal label")
    return 0 if binary_label == 0 else CLASSES.index(category)


def read_raw_labels(directory: Path) -> tuple[dict, dict]:
    """Read the observed headerless clip,category,binary CSV export.

    Identical duplicate rows are recorded and collapsed. Missing/invalid labels
    and conflicts are quarantined, never filled from folder names or neighbors.
    """
    grouped = defaultdict(list)
    issues = {"duplicate_rows": [], "invalid_rows": [], "conflicting_clip_ids": []}
    files = sorted(Path(directory).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No DCSASS category CSVs under {directory}")
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for line, row in enumerate(csv.reader(stream), 1):
                if (
                    len(row) != 3
                    or not row[0].strip()
                    or row[1] != path.stem
                    or row[2] not in ("0", "1")
                    or row[1] not in set(CLASSES) | EXCLUDED
                ):
                    issues["invalid_rows"].append({"file": path.name, "line": line, "row": row})
                    continue
                grouped[row[0]].append((row[1], int(row[2]), path.name, line))
    accepted = {}
    for clip_id, rows in sorted(grouped.items()):
        for row in rows[1:]:
            issues["duplicate_rows"].append({"clip_id": clip_id, "file": row[2], "line": row[3]})
        if len({(r[0], r[1]) for r in rows}) != 1:
            issues["conflicting_clip_ids"].append(clip_id)
            continue
        accepted[clip_id] = {"category": rows[0][0], "label": rows[0][1]}
    return accepted, issues


def split_sources(
    sources: dict[str, str], reserved_test: set[str], seed: int = 0
) -> dict[str, str]:
    """Stratify original category with ~70/15/15 whole sources, reserving UCF test.

    Class counts are derived before training. Reservations can exceed 15%; then
    test grows and training shrinks. No reservation is moved into optimization.
    """
    categories = defaultdict(list)
    for source, category in sorted(sources.items()):
        categories[category].append(source)
    rng = random.Random(seed)
    assignments = {}
    for category in sorted(categories):
        group = categories[category]
        n = len(group)
        targets = [n * ratio for ratio in (0.7, 0.15, 0.15)]
        counts = [math.floor(v) for v in targets]
        for i in sorted(range(3), key=lambda i: (-(targets[i] - counts[i]), i))[: n - sum(counts)]:
            counts[i] += 1
        if n >= 3:
            for index in (1, 2):
                if counts[index] == 0:
                    counts[0] -= 1
                    counts[index] = 1
        reserved = sorted(set(group) & reserved_test)
        counts[2] = max(counts[2], len(reserved))
        counts[1] = min(counts[1], n - counts[2])
        counts[0] = n - counts[1] - counts[2]
        remaining = sorted(set(group) - set(reserved))
        rng.shuffle(remaining)
        for source in reserved:
            assignments[source] = "test"
        for i, source in enumerate(remaining):
            assignments[source] = (
                "train" if i < counts[0] else "val" if i < counts[0] + counts[1] else "test"
            )
    return assignments


def build_human_split(audit: dict, reserved_test: set[str], seed: int = 0) -> tuple[list, dict]:
    """Validate supervision, assign whole selected sources and reject content leakage."""
    if audit.get("bounded_preflight") is not False:
        raise ValueError("Cannot freeze a bounded or unspecified dataset audit")
    selected = [row for row in audit["clips"] if row["selected"]]
    if not selected:
        raise ValueError("Human-centric population is empty")
    sources = {}
    for row in selected:
        if row["group_label"] != behavior_label(row["category"], row["binary_label"]):
            raise ValueError(f"Inconsistent behavior label: {row['clip_id']}")
        source = row["source_video_id"]
        if source in sources and sources[source] != row["category"]:
            raise ValueError("Original source has conflicting categories")
        sources[source] = row["category"]
    assignments = split_sources(sources, reserved_test, seed)
    targets = defaultdict(set)
    for row in selected:
        targets[row["source_video_id"]].add(row["group_label"])
    repairs = []
    # Folder categories alone do not imply positive clips. Before training,
    # repair missing positive validation support by swapping whole sources.
    for label, category in enumerate(CLASSES[1:], 1):
        if any(assignments[s] == "val" and label in t for s, t in targets.items()):
            continue
        donors = sorted(
            s
            for s, t in targets.items()
            if sources[s] == category and assignments[s] == "train" and label in t
        )
        receivers = sorted(
            s
            for s, t in targets.items()
            if sources[s] == category and assignments[s] == "val" and label not in t
        )
        if len(donors) >= 2 and receivers:
            donor, receiver = donors[0], receivers[0]
            assignments[donor], assignments[receiver] = "val", "train"
            repairs.append({"class": category, "train_to_val": donor, "val_to_train": receiver})
    rows = [dict(row, split=assignments[row["source_video_id"]]) for row in selected]
    content_splits = defaultdict(set)
    for row in rows:
        content_splits[row["sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in content_splits.values()):
        raise ValueError("Duplicate content leaks across source splits")
    for source in reserved_test & set(assignments):
        if assignments[source] != "test":
            raise ValueError("Official UCF test source enters optimization")
    report = {
        "protocol_id": "dcsass_human_centric_v1",
        "seed": seed,
        "classes": list(CLASSES),
        "split_unit": "original_source_video_id",
        "target_ratios": [0.7, 0.15, 0.15],
        "pretraining_class_support_repairs": repairs,
        "source_overlap": [],
        "duplicate_content_overlap": [],
        "source_assignments": assignments,
        "ucf_test_sources_present": sorted(reserved_test & set(assignments)),
        "clip_counts": dict(Counter(r["split"] for r in rows)),
        "source_counts": dict(Counter(assignments.values())),
        "class_counts": {
            split: {
                name: sum(r["split"] == split and r["group_label"] == i for r in rows)
                for i, name in enumerate(CLASSES)
            }
            for split in ("train", "val", "test")
        },
    }
    return rows, report
