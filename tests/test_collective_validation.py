"""Structural checks on tiny local Collective sequences, never benchmark scores."""

import json
from dataclasses import asdict, replace

import cv2
import numpy as np
import pytest

from surveillance.datasets.collective import ActorRecord, temporal_frame_indices


def sequence(root, sid=1, count=11):
    folder = root / f"seq{sid:02d}"
    folder.mkdir()
    for index in range(1, count + 1):
        assert cv2.imwrite(str(folder / f"frame{index:04d}.jpg"), np.zeros((12, 20, 3), np.uint8))
    (folder / "annotations.txt").write_text("1 2 1 10 10 2\n11 2 1 10 10 6\n")
    return folder


def row(root, sid=1, split="val"):
    folder = root / f"seq{sid:02d}"
    indices = temporal_frame_indices(1, 11)
    return ActorRecord(
        "collective",
        folder.name,
        f"collective:{folder.name}",
        f"{folder.name}:0001",
        split,
        [str(folder / f"frame{i:04d}.jpg") for i in indices],
        indices,
        [[0.1, 1 / 12, 0.6, 11 / 12]],
        [0],
        0,
    )


def manifest(path, rows):
    path.write_text("".join(json.dumps(asdict(r)) + "\n" for r in rows))
    return path


def test_root_reports_actual_parser_centers_and_dimensions(tmp_path):
    from surveillance.datasets.collective_validation import validate_collective

    sequence(tmp_path)
    report = validate_collective(tmp_path, (1,), require_full_split=False)
    assert report["valid"]
    assert report["counts"]["scenes"] == 2
    assert report["sequences"][0]["centers"] == [1, 11]
    assert report["sequences"][0]["image_size"] == [12, 20]
    assert report["splits"]["val"]["actors"] == 2


@pytest.mark.parametrize("fault", ["gap", "corrupt", "dimensions", "box", "action", "duplicate"])
def test_root_fails_with_recoverable_json_report(tmp_path, fault):
    from surveillance.datasets.collective_validation import (
        DatasetValidationError,
        validate_collective,
    )

    folder = sequence(tmp_path)
    if fault == "gap":
        (folder / "frame0002.jpg").unlink()
    elif fault == "corrupt":
        (folder / "frame0002.jpg").write_text("broken")
    elif fault == "dimensions":
        cv2.imwrite(str(folder / "frame0002.jpg"), np.zeros((13, 20, 3), np.uint8))
    else:
        text = {
            "box": "1 -1 0 10 10 2\n",
            "action": "1 1 0 10 10 7\n",
            "duplicate": "1 1 0 10 10 2\n1 1 0 10 10 2\n",
        }[fault]
        (folder / "annotations.txt").write_text(text)
    with pytest.raises(DatasetValidationError) as caught:
        validate_collective(tmp_path, (1,), require_full_split=False)
    assert not caught.value.report["valid"]
    assert caught.value.report["errors"]
    json.dumps(caught.value.report, allow_nan=False)


def test_full_split_and_validation_holdout_are_enforced(tmp_path):
    from surveillance.datasets.collective_validation import (
        DatasetValidationError,
        validate_collective,
    )

    sequence(tmp_path)
    with pytest.raises(DatasetValidationError, match="missing"):
        validate_collective(tmp_path)
    with pytest.raises(DatasetValidationError, match="training"):
        validate_collective(tmp_path, (5,), require_full_split=False)


@pytest.mark.parametrize("fault", ["label", "majority", "duplicate", "source", "missing"])
def test_manifest_rejects_semantic_and_source_faults(tmp_path, fault):
    from surveillance.datasets.collective_validation import (
        DatasetValidationError,
        validate_manifest,
    )

    sequence(tmp_path)
    original = row(tmp_path)
    rows = [original]
    if fault == "label":
        rows = [replace(original, actor_labels=[5], group_label=5)]
    elif fault == "majority":
        rows = [replace(original, group_label=1)]
    elif fault == "duplicate":
        rows.append(original)
    elif fault == "source":
        rows.append(replace(original, clip_id="other", split="train"))
    else:
        (tmp_path / "seq01" / "frame0001.jpg").unlink()
    with pytest.raises(DatasetValidationError):
        validate_manifest(manifest(tmp_path / "rows.jsonl", rows), (1,), require_full_split=False)


def test_manifest_counts_all_splits_and_reports_actor_warning(tmp_path):
    from surveillance.datasets.collective_validation import validate_manifest

    sequence(tmp_path, 1)
    sequence(tmp_path, 4)
    sequence(tmp_path, 5)
    rows = [row(tmp_path, 1), row(tmp_path, 4, "train"), row(tmp_path, 5, "test")]
    report = validate_manifest(
        manifest(tmp_path / "rows.jsonl", rows),
        (1,),
        require_full_split=False,
        actor_count_warning=0,
    )
    assert report["counts"] == {"sequences": 3, "scenes": 3, "actors": 3}
    assert report["splits"]["test"]["group_class_distribution"]["crossing"] == 1
    assert len(report["warnings"]) == 3


def test_full_tree_has_exact_released_source_coverage(tmp_path):
    from surveillance.datasets.collective_validation import validate_collective

    for sid in range(1, 45):
        sequence(tmp_path, sid)
    report = validate_collective(tmp_path)
    assert report["counts"] == {"sequences": 44, "scenes": 88, "actors": 88}
    assert [report["splits"][s]["sequences"] for s in ("train", "val", "test")] == [29, 3, 12]


def test_nonselected_annotation_must_reference_existing_frame(tmp_path):
    from surveillance.datasets.collective_validation import (
        DatasetValidationError,
        validate_collective,
    )

    folder = sequence(tmp_path)
    with (folder / "annotations.txt").open("a") as stream:
        stream.write("12 2 1 10 10 2\n")
    with pytest.raises(DatasetValidationError, match="references missing"):
        validate_collective(tmp_path, (1,), require_full_split=False)


def test_incorrect_temporal_window_fails(tmp_path):
    from surveillance.datasets.collective_validation import (
        DatasetValidationError,
        validate_manifest,
    )

    sequence(tmp_path)
    original = row(tmp_path)
    incorrect = replace(
        original, frame_paths=[original.frame_paths[0]] * 10, frame_indices=[1] * 10
    )
    with pytest.raises(DatasetValidationError, match="window"):
        validate_manifest(
            manifest(tmp_path / "rows.jsonl", [incorrect]), (1,), require_full_split=False
        )


def test_cli_writes_failure_report_and_nonzero_exit(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "report.json"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/validate_collective.py",
            "--root",
            str(tmp_path),
            "--report",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1, process.stderr
    report = json.loads(output.read_text())
    assert not report["valid"] and report["errors"]
