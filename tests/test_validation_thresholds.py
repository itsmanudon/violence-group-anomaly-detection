"""Validation selection cannot consume test scenes or already filtered detections."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from surveillance.datasets.collective import ActorRecord, write_actor_manifest
from surveillance.detection.config import DetectionConfig
from surveillance.detection.person_detector import DetectionResult
from surveillance.detection.records import DetectionRecord, write_detections


def _row(split="val", sequence=2):
    name = f"seq{sequence:02d}"
    return ActorRecord(
        "collective",
        name,
        f"collective:{name}",
        f"{name}:0005",
        split,
        [f"{name}/{i}.jpg" for i in range(10)],
        list(range(10)),
        [[0.1, 0.1, 0.4, 0.9]],
        [0],
        0,
    )


def _candidate(row, metadata=None, scores=(0.8, 0.35)):
    broad = DetectionConfig(0, 1, 1000, 0, 0, 0)
    provenance = {
        "threshold_sweep_candidates": True,
        "filter_config": asdict(broad),
        "checkpoint_sha256": "a" * 64,
        "untrained": False,
        "backend_candidate_cap": 1000,
        "backend_box_score_threshold": 0.0,
        "backend_box_nms_threshold": 1.0,
        "before_count_scope": "after_rpn_and_backend_candidate_cap",
        "truncated": False,
        "filter_counts": {"max_actors": 0, "low_confidence": 0, "nms": 0, "too_small": 0},
    }
    provenance.update(metadata or {})
    result = DetectionResult(
        torch.tensor([[10, 10, 40, 90], [60, 10, 90, 90]]),
        torch.tensor(scores),
        torch.ones(2, dtype=torch.long),
        (100, 100),
        provenance,
    )
    return DetectionRecord(
        row.dataset,
        row.video_id,
        row.source_video_id,
        row.clip_id,
        row.frame_indices[5],
        result,
    )


def _files(tmp_path, rows=None, candidates=None):
    rows = rows or [_row(), _row("train", 1), _row("test", 5)]
    manifest, cached = tmp_path / "gt.jsonl", tmp_path / "candidates.jsonl"
    write_actor_manifest(rows, manifest)
    write_detections(candidates or [_candidate(rows[0])], cached)
    return manifest, cached, tmp_path / "receipt.json"


def test_selects_f1_then_recall_then_highest_confidence_without_test_population(tmp_path):
    from surveillance.experiments.thresholds import tune_detector_thresholds

    paths = _files(tmp_path)
    result = tune_detector_thresholds(*paths, [0.3, 0.5, 0.7], DetectionConfig())
    assert result["selected_confidence"] == 0.7
    assert result["selected_filters"]["confidence_threshold"] == 0.7
    assert [trial["f1"] for trial in result["trials"]] == pytest.approx([2 / 3, 1, 1])
    assert result["split"] == "val" and result["test_sources_used"] is False
    assert result["validation_source_ids"] == ["collective:seq02"]
    assert result["validation_population"][0]["clip_id"] == "seq02:0005"
    assert len(result["validation_population_sha256"]) == 64
    assert len(result["candidates_sha256"]) == 64
    assert json.loads(paths[2].read_text()) == result


@pytest.mark.parametrize("split,sequence", [("train", 1), ("test", 5)])
def test_rejects_nonvalidation_candidates_even_in_full_manifest(tmp_path, split, sequence):
    from surveillance.experiments.thresholds import tune_detector_thresholds

    rows = [_row(), _row(split, sequence)]
    paths = _files(tmp_path, rows, [_candidate(row) for row in rows])
    with pytest.raises(ValueError, match="validation-only"):
        tune_detector_thresholds(*paths, [0.5], DetectionConfig())
    assert not paths[2].exists()


@pytest.mark.parametrize(
    "metadata",
    [
        {"threshold_sweep_candidates": False},
        {"filter_config": asdict(DetectionConfig())},
        {"truncated": True},
        {"checkpoint_sha256": None},
        {"backend_candidate_cap": None},
        {"filter_counts": {"max_actors": 1}},
    ],
)
def test_rejects_irreversible_or_unknown_candidate_provenance(tmp_path, metadata):
    from surveillance.experiments.thresholds import tune_detector_thresholds

    row = _row()
    paths = _files(tmp_path, [row], [_candidate(row, metadata)])
    with pytest.raises(ValueError, match="candidate"):
        tune_detector_thresholds(*paths, [0.5], DetectionConfig())


def test_recall_constraint_fails_without_writing_frozen_receipt(tmp_path):
    from surveillance.experiments.thresholds import tune_detector_thresholds

    row = _row()
    paths = _files(tmp_path, [row], [_candidate(row, scores=(0.4, 0.9))])
    with pytest.raises(ValueError, match="recall"):
        tune_detector_thresholds(*paths, [0.5, 0.7], DetectionConfig(), min_recall=0.5)
    assert not paths[2].exists()


def test_selected_sequences_must_be_actual_validation_sources(tmp_path):
    from surveillance.experiments.thresholds import tune_detector_thresholds

    paths = _files(tmp_path)
    with pytest.raises(ValueError, match="validation"):
        tune_detector_thresholds(*paths, [0.5], DetectionConfig(), validation_sequences=[5])


def test_candidate_identity_must_match_validation_center(tmp_path):
    from surveillance.experiments.thresholds import tune_detector_thresholds

    row = _row()
    paths = _files(tmp_path, [row], [replace(_candidate(row), frame_index=6)])
    with pytest.raises(ValueError, match="identity"):
        tune_detector_thresholds(*paths, [0.5], DetectionConfig())


def test_cli_creates_validation_receipt_from_candidates(tmp_path):
    paths = _files(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/tune_collective_detector.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(paths[0]),
            "--candidates",
            str(paths[1]),
            "--output",
            str(paths[2]),
            "--confidences",
            "0.3",
            "0.5",
            "0.7",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(paths[2].read_text())["selected_confidence"] == 0.7


def test_equal_f1_prefers_higher_recall_before_higher_confidence(tmp_path):
    from surveillance.experiments.thresholds import tune_detector_thresholds

    row = replace(
        _row(),
        actor_boxes=[[0.1, 0.1, 0.4, 0.9], [0.5, 0.1, 0.7, 0.9]],
        actor_labels=[0, 0],
    )
    candidate = _candidate(row)
    candidate = replace(
        candidate,
        result=replace(
            candidate.result,
            boxes=torch.tensor([[10, 10, 40, 90], [50, 10, 70, 90], [0, 0, 9, 9], [80, 0, 99, 9]]),
            scores=torch.tensor([0.8, 0.4, 0.4, 0.4]),
            class_ids=torch.ones(4, dtype=torch.long),
        ),
    )
    paths = _files(tmp_path, [row], [candidate])
    receipt = tune_detector_thresholds(*paths, [0.3, 0.7], DetectionConfig())
    assert [trial["f1"] for trial in receipt["trials"]] == pytest.approx([2 / 3, 2 / 3])
    assert [trial["recall"] for trial in receipt["trials"]] == [1.0, 0.5]
    assert receipt["selected_confidence"] == 0.3


def test_test_gt_changes_cannot_change_validation_population_or_selection(tmp_path):
    from surveillance.experiments.thresholds import tune_detector_thresholds

    paths = _files(tmp_path)
    original = tune_detector_thresholds(*paths, [0.3, 0.7], DetectionConfig())
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    rows[-1]["actor_boxes"] = [[0.8, 0.8, 0.9, 0.9]]
    rows[-1]["actor_labels"] = [4]
    rows[-1]["group_label"] = 4
    paths[0].write_text("".join(json.dumps(row) + "\n" for row in rows))
    changed = tune_detector_thresholds(*paths, [0.3, 0.7], DetectionConfig())
    assert changed == original


def test_collection_reads_only_native_validation_center(tmp_path, monkeypatch):
    import cv2

    from surveillance.detection import torchvision_detector
    from surveillance.experiments.thresholds import collect_validation_candidates

    paths = _files(tmp_path)
    center = tmp_path / "seq02" / "5.jpg"
    center.parent.mkdir()
    # Only the validation center exists; reading other split frames fails.
    cv2.imwrite(str(center), np.full((100, 100, 3), [10, 20, 240], dtype=np.uint8))
    observed = []

    class LocalDetectorFixture:
        def __init__(self, checkpoint, config, device):
            assert config.confidence_threshold == 0 and config.nms_iou_threshold == 1

        def detect(self, image):
            observed.append(image)
            return _candidate(_row()).result

    monkeypatch.setattr(torchvision_detector, "TorchvisionPersonDetector", LocalDetectorFixture)
    records = collect_validation_candidates(
        paths[0], tmp_path / "collected.jsonl", tmp_path / "local.pt"
    )
    assert len(records) == 1 and records[0].source_video_id == "collective:seq02"
    assert observed[0].shape == (3, 100, 100)
    assert observed[0][0].mean() > observed[0][2].mean()


def test_collection_missing_checkpoint_fails_before_image_reads(tmp_path):
    from surveillance.experiments.thresholds import collect_validation_candidates

    paths = _files(tmp_path)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        collect_validation_candidates(
            paths[0], tmp_path / "collected.jsonl", tmp_path / "missing.pt"
        )
    assert not (tmp_path / "collected.jsonl").exists()


@pytest.mark.parametrize("target", ["manifest", "checkpoint"])
def test_collection_refuses_to_overwrite_inputs(tmp_path, target):
    from surveillance.experiments.thresholds import collect_validation_candidates

    paths = _files(tmp_path)
    checkpoint = tmp_path / "local.pt"
    checkpoint.write_bytes(b"local weights remain intact")
    output = paths[0] if target == "manifest" else checkpoint
    before = output.read_bytes()
    with pytest.raises(ValueError, match="output"):
        collect_validation_candidates(paths[0], output, checkpoint)
    assert output.read_bytes() == before


def test_corrupt_local_checkpoint_reports_actionable_cli_error(tmp_path):
    paths = _files(tmp_path)
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"corrupt local checkpoint")
    script = Path(__file__).resolve().parents[1] / "scripts/tune_collective_detector.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(paths[0]),
            "--output",
            str(paths[2]),
            "--collect-candidates",
            "--checkpoint",
            str(checkpoint),
            "--candidate-output",
            str(tmp_path / "collected.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Error:" in result.stderr and "checkpoint" in result.stderr
    assert "Traceback" not in result.stderr
    assert not paths[2].exists()
