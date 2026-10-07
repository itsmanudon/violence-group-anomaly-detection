"""Strict, report-preserving validation of locally installed Collective data.

The sequence IDs are from the related authors' released code. They do not
establish exact agreement with the Actor-Transformers paper's unreleased split.
No image or annotation is changed by these checks.
"""

import json
import re
from collections import Counter
from pathlib import Path

import cv2

from surveillance.datasets.collective import (
    CLASSES,
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
    ActorRecord,
    _validate_manifest,
    parse_collective_annotations,
    temporal_frame_indices,
)
from surveillance.datasets.common import resolve_path


class DatasetValidationError(ValueError):
    """Structural failure retaining the complete JSON-serializable report."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("; ".join(report["errors"]))


def _report(kind: str) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "valid": False,
        "protocol": "related-author-code 32/12; paper-exact source IDs unverified",
        "classes": list(CLASSES),
        "errors": [],
        "warnings": [],
        "invalid_boxes": [],
        "missing_frames": [],
        "duplicate_ids": [],
        "counts": {"sequences": 0, "scenes": 0, "actors": 0},
        "splits": {},
        "sequences": [],
    }


def _finish(report: dict) -> dict:
    report["valid"] = not report["errors"]
    if not report["valid"]:
        raise DatasetValidationError(report)
    return report


def _holdout(validation_sequences, report):
    values = list(validation_sequences)
    if any(type(v) is not int for v in values) or len(set(values)) != len(values):
        report["errors"].append("Validation sequence IDs must be unique integers")
        return set()
    if not set(values) <= set(TRAIN_SEQUENCES):
        report["errors"].append(
            "Validation sequences must be a subset of official training sources"
        )
    if set(values) == set(TRAIN_SEQUENCES):
        report["errors"].append("Validation holdout must leave training sources")
    return set(values)


def _counts(rows, report, actor_count_warning):
    report["counts"] = {
        "sequences": len({r.source_video_id for r in rows}),
        "scenes": len(rows),
        "actors": sum(len(r.actor_labels) for r in rows),
    }
    for split in ("train", "val", "test"):
        selected = [r for r in rows if r.split == split]
        actors = Counter(label for r in selected for label in r.actor_labels)
        groups = Counter(r.group_label for r in selected)
        report["splits"][split] = {
            "sequences": len({r.source_video_id for r in selected}),
            "source_ids": sorted({r.source_video_id for r in selected}),
            "scenes": len(selected),
            "actors": sum(len(r.actor_labels) for r in selected),
            "actor_class_distribution": {name: actors[i] for i, name in enumerate(CLASSES)},
            "group_class_distribution": {name: groups[i] for i, name in enumerate(CLASSES)},
        }
    for row in rows:
        if len(row.actor_labels) > actor_count_warning:
            report["warnings"].append(
                f"{row.clip_id}: actor count {len(row.actor_labels)} exceeds warning threshold "
                f"{actor_count_warning}"
            )


def _sequence_id(row):
    match = re.fullmatch(r"seq(\d{2})", row.video_id)
    if match and row.source_video_id == f"collective:{row.video_id}":
        return int(match[1])
    return None


def validate_manifest(
    manifest: Path,
    validation_sequences=(1, 2, 3),
    require_full_split: bool = True,
    check_images: bool = True,
    actor_count_warning: int = 30,
) -> dict:
    """Validate all splits before any benchmark consumer filters scenes.

    Subset mode permits generic synthetic identities; Collective identities still
    obey the released sequence split and explicitly selected validation sources.
    """
    manifest = Path(manifest)
    report = _report("collective_manifest")
    report["manifest"] = str(manifest.resolve())
    validation = _holdout(validation_sequences, report)
    rows = []
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        report["errors"].append(str(error))
        return _finish(report)
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(ActorRecord(**json.loads(line)))
        except (ValueError, TypeError, KeyError) as error:
            diagnostic = f"{manifest}: line {number}: {error}"
            report["errors"].append(diagnostic)
            if "box" in str(error).lower():
                report["invalid_boxes"].append(diagnostic)
    try:
        _validate_manifest(rows, manifest)
    except ValueError as error:
        report["errors"].append(str(error))
    source_lengths = {}
    seen, centers, sequence_splits, decoded, dimensions, index_paths = set(), set(), {}, {}, {}, {}
    for row in rows:
        key, center = (row.dataset, row.clip_id), (row.source_video_id, row.frame_indices[5])
        for identity, identities in ((key, seen), (center, centers)):
            if identity in identities:
                report["duplicate_ids"].append(list(identity))
            identities.add(identity)
        if any(v >= len(CLASSES) for v in [*row.actor_labels, row.group_label]):
            report["errors"].append(f"{row.clip_id}: five-class labels must lie in 0..4")
        majority = min(Counter(row.actor_labels), key=lambda v: (-row.actor_labels.count(v), v))
        if row.group_label != majority:
            report["errors"].append(f"{row.clip_id}: group label differs from actor majority")
        if len({tuple(b) for b in row.actor_boxes}) != len(row.actor_boxes):
            report["errors"].append(f"{row.clip_id}: duplicate actor boxes")
            report["duplicate_ids"].append([row.clip_id, "actor_boxes"])
        sid = _sequence_id(row)
        if row.dataset == "collective" or require_full_split:
            if row.dataset != "collective" or sid not in range(1, 45):
                report["errors"].append(f"{row.clip_id}: invalid Collective source identity")
            else:
                expected = (
                    "test" if sid in TEST_SEQUENCES else "val" if sid in validation else "train"
                )
                if row.split != expected:
                    report["errors"].append(
                        f"{row.clip_id}: source belongs to {expected}, not {row.split}"
                    )
                sequence_splits[sid] = row.split
            if min(row.frame_indices) < 1 or row.frame_indices[5] % 10 != 1:
                report["errors"].append(f"{row.clip_id}: Collective centers must be 1,11,21,...")
            if any(b - a not in {0, 1} for a, b in zip(row.frame_indices, row.frame_indices[1:])):
                report["errors"].append(
                    f"{row.clip_id}: frame indices must be consecutive with edge replication"
                )
            if check_images and sid is not None:
                folder = resolve_path(row.frame_paths[5], manifest).resolve().parent
                if folder not in source_lengths:
                    indices = [
                        int(p.stem.removeprefix("frame"))
                        for p in folder.glob("frame*.jpg")
                        if p.stem.removeprefix("frame").isdigit()
                    ]
                    source_lengths[folder] = max(indices, default=0)
                length = source_lengths[folder]
                center_index = row.frame_indices[5]
                if length and 1 <= center_index <= length:
                    if row.frame_indices != temporal_frame_indices(center_index, length):
                        report["errors"].append(
                            f"{row.clip_id}: frame window differs from offsets -5..+4 "
                            "with edge replication"
                        )
        for index, value in zip(row.frame_indices, row.frame_paths, strict=True):
            path = resolve_path(value, manifest).resolve()
            reference = (row.source_video_id, index)
            if reference in index_paths and index_paths[reference] != path:
                report["errors"].append(
                    f"{row.clip_id}: source frame {index} has inconsistent paths"
                )
            index_paths[reference] = path
            if not check_images:
                continue
            if path not in decoded:
                image = cv2.imread(str(path))
                decoded[path] = None if image is None else tuple(image.shape[:2])
                if image is None:
                    report["errors"].append(f"Missing or undecodable frame: {path}")
                    report["missing_frames"].append(str(path))
            size = decoded[path]
            if size is not None:
                previous = dimensions.setdefault(row.source_video_id, size)
                if previous != size:
                    report["errors"].append(f"{path}: image dimensions vary within source sequence")
    if require_full_split:
        actual = {s for s, split in sequence_splits.items() if split != "test"}
        test = {s for s, split in sequence_splits.items() if split == "test"}
        if actual != set(TRAIN_SEQUENCES) or test != set(TEST_SEQUENCES):
            report["errors"].append(
                "Full manifest must cover released 32 training/validation and 12 test sources"
            )
    _counts(rows, report, actor_count_warning)
    return _finish(report)


def validate_collective(
    root: Path,
    validation_sequences=(1, 2, 3),
    manifest: Path | None = None,
    require_full_split: bool = True,
    actor_count_warning: int = 30,
    *,
    box_policy: str = "strict",
) -> dict:
    """Decode every frame; audit explicitly allowed visible-boundary corrections."""
    if box_policy not in {"strict", "clip_to_image"}:
        raise ValueError("box_policy must be strict or clip_to_image")
    root = Path(root).resolve()
    report = _report("collective_root")
    report["box_policy"] = box_policy
    report["box_corrections"] = []
    report["root"] = str(root)
    validation = _holdout(validation_sequences, report)
    folders = {sid: root / f"seq{sid:02d}" for sid in range(1, 45)}
    present = {sid for sid, folder in folders.items() if folder.is_dir()}
    if require_full_split and present != set(range(1, 45)):
        report["errors"].append(
            f"missing Collective sequence directories: {sorted(set(range(1, 45)) - present)}"
        )
    if not present:
        report["errors"].append("No Collective sequence directories found")
    rows = []
    for sid in sorted(present):
        folder, frames = folders[sid], {}
        for path in sorted(folder.glob("frame*.jpg")):
            suffix = path.stem.removeprefix("frame")
            if not suffix.isdigit() or int(suffix) < 1:
                report["errors"].append(f"{path}: invalid one-based frame filename")
                continue
            index = int(suffix)
            if index in frames:
                report["errors"].append(f"{folder}: duplicate frame index {index}")
                report["duplicate_ids"].append([folder.name, index])
            frames[index] = path
        if not frames:
            report["errors"].append(f"{folder}: missing source frames")
            continue
        missing = sorted(set(range(1, max(frames) + 1)) - set(frames))
        if missing:
            report["errors"].append(f"{folder}: missing consecutive source frames {missing}")
            report["missing_frames"].extend(str(folder / f"frame{i:04d}.jpg") for i in missing)
        image_size = None
        for path in frames.values():
            image = cv2.imread(str(path))
            if image is None:
                report["errors"].append(f"Cannot decode frame: {path}")
                report["missing_frames"].append(str(path))
                continue
            size = tuple(image.shape[:2])
            if image_size is None:
                image_size = size
            elif size != image_size:
                report["errors"].append(f"{path}: image dimensions vary within sequence")
        annotation = folder / "annotations.txt"
        scenes = {}
        if image_size is not None:
            try:
                scenes = parse_collective_annotations(
                    annotation,
                    image_size,
                    box_policy=box_policy,
                    corrections=report["box_corrections"],
                )
                # Check every referenced frame, including nonselected centers and NA actors.
                for number, line in enumerate(
                    annotation.read_text(encoding="utf-8").splitlines(), 1
                ):
                    if line.strip() and int(line.split()[0]) not in frames:
                        report["errors"].append(
                            f"{annotation}: line {number} references missing frame"
                        )
                        report["missing_frames"].append(f"{folder.name}:{line.split()[0]}")
            except (ValueError, OSError) as error:
                report["errors"].append(str(error))
                if "box" in str(error).lower():
                    report["invalid_boxes"].append(str(error))
        report["sequences"].append(
            {
                "sequence_id": sid,
                "source_video_id": f"collective:{folder.name}",
                "frames": len(frames),
                "image_size": list(image_size) if image_size else None,
                "centers": sorted(scenes),
                "scenes": len(scenes),
                "actors": sum(len(s["actor_labels"]) for s in scenes.values()),
            }
        )
        for center, scene in sorted(scenes.items()):
            if center not in frames:
                continue
            indices = temporal_frame_indices(center, max(frames))
            if any(i not in frames for i in indices):
                continue
            rows.append(
                ActorRecord(
                    dataset="collective",
                    video_id=folder.name,
                    source_video_id=f"collective:{folder.name}",
                    clip_id=f"{folder.name}:{center:04d}",
                    split="test"
                    if sid in TEST_SEQUENCES
                    else "val"
                    if sid in validation
                    else "train",
                    frame_paths=[str(frames[i]) for i in indices],
                    frame_indices=indices,
                    **scene,
                )
            )
    _counts(rows, report, actor_count_warning)
    if report["box_corrections"]:
        selected = sum(c["selected_supervised"] for c in report["box_corrections"])
        report["warnings"].append(
            f"Clipped {len(report['box_corrections'])} boundary boxes under {box_policy}; "
            f"{selected} belong to selected supervised actors. Raw annotations were not changed."
        )
    if manifest is not None:
        try:
            report["manifest_validation"] = validate_manifest(
                manifest, validation_sequences, require_full_split, True, actor_count_warning
            )
        except DatasetValidationError as error:
            report["manifest_validation"] = error.report
            report["errors"].extend(error.report["errors"])
        # Independently verify manifest supervision/source references against parsed originals.
        expected = {(r.dataset, r.clip_id): r for r in rows}
        actual = {}
        try:
            from surveillance.datasets.collective import read_actor_manifest

            actual = {(r.dataset, r.clip_id): r for r in read_actor_manifest(Path(manifest))}
            if set(actual) != set(expected):
                report["errors"].append("Manifest scenes differ from parsed local annotations")
            for key in actual.keys() & expected.keys():
                a, b = actual[key], expected[key]
                values = (
                    "source_video_id",
                    "video_id",
                    "split",
                    "frame_indices",
                    "actor_boxes",
                    "actor_labels",
                    "group_label",
                )
                if any(getattr(a, name) != getattr(b, name) for name in values):
                    report["errors"].append(
                        f"{a.clip_id}: manifest differs from parsed source annotation"
                    )
                paths = [resolve_path(p, Path(manifest)).resolve() for p in a.frame_paths]
                if paths != [Path(p).resolve() for p in b.frame_paths]:
                    report["errors"].append(
                        f"{a.clip_id}: manifest source frame paths differ from local sequence"
                    )
        except (ValueError, OSError) as error:
            report["errors"].append(str(error))
    return _finish(report)
