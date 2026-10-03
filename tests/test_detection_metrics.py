"""Localization metrics preserve scene coverage and explicit undefined ratios."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from surveillance.datasets.collective import ActorRecord, write_actor_manifest
from surveillance.detection.matching import MatchResult, match_manifest
from surveillance.evaluation.detection_metrics import aggregate_matches, evaluate_detections


def test_aggregate_uses_micro_counts_and_reports_empty_frames():
    result = aggregate_matches(
        [
            MatchResult([0], [1], [0.75], [1], [0, 2], 2, 3),
            MatchResult([], [], [], [0], [], 1, 0),
        ]
    )
    assert result["matches"] == 1
    assert result["gt_count"] == result["detection_count"] == 3
    assert result["precision"] == result["recall"] == result["f1"] == pytest.approx(1 / 3)
    assert result["mean_matched_iou"] == 0.75
    assert result["missed_gt_actors"] == result["unmatched_detections"] == 2
    assert result["mean_actors_per_frame"] == 1.5
    assert result["frame_count"] == 2 and result["empty_frames"] == 1


@pytest.mark.parametrize(
    "gt_count,det_count,precision,recall,f1",
    [
        (0, 0, None, None, None),
        (1, 0, None, 0.0, 0.0),
        (0, 1, 0.0, None, 0.0),
    ],
)
def test_undefined_metrics(gt_count, det_count, precision, recall, f1):
    result = aggregate_matches(
        [
            MatchResult(
                [],
                [],
                [],
                list(range(gt_count)),
                list(range(det_count)),
                gt_count,
                det_count,
            )
        ]
    )
    assert result["precision"] == precision
    assert result["recall"] == recall
    assert result["f1"] == f1
    assert result["mean_matched_iou"] is None
    json.dumps(result, allow_nan=False)


def test_zero_frames_are_distinct_from_one_empty_frame():
    result = aggregate_matches([])
    assert result["frame_count"] == result["empty_frames"] == 0
    assert result["mean_actors_per_frame"] is None


def _gt(clip="scene", split="test"):
    return ActorRecord(
        "fixture",
        f"video-{clip}",
        f"source-{clip}",
        clip,
        split,
        [f"{clip}/{i}.jpg" for i in range(10)],
        list(range(10)),
        [[0.1, 0.2, 0.5, 0.8]],
        [3],
        3,
    )


def _detection(row, *, empty=False, **overrides):
    from surveillance.detection.person_detector import DetectionResult
    from surveillance.detection.records import DetectionRecord

    values = dict(
        dataset=row.dataset,
        video_id=row.video_id,
        source_video_id=row.source_video_id,
        clip_id=row.clip_id,
        frame_index=row.frame_indices[5],
        result=DetectionResult(
            boxes=torch.empty((0, 4)) if empty else torch.tensor([[20.0, 20.0, 100.0, 80.0]]),
            scores=torch.empty(0) if empty else torch.tensor([0.9]),
            class_ids=torch.empty(0, dtype=torch.long) if empty else torch.tensor([1]),
            image_size=(100, 200),
            metadata={"backend": "fixture"},
        ),
    )
    return DetectionRecord(**(values | overrides))


def _files(tmp_path, gt_rows, detections):
    from surveillance.detection.records import write_detections

    manifest, cached = tmp_path / "gt.jsonl", tmp_path / "detections.jsonl"
    write_actor_manifest(gt_rows, manifest)
    write_detections(detections, cached)
    return manifest, cached


def test_offline_join_converts_normalized_boxes_without_opening_images(tmp_path):
    row = _gt()
    paths = _files(tmp_path, [row], [_detection(row)])
    result = evaluate_detections(*paths)
    assert result["precision"] == result["recall"] == 1.0
    scene = result["scenes"][0]
    assert scene["gt_boxes"] == [[20.0, 20.0, 100.0, 80.0]]
    assert scene["actor_labels"] == [3]
    assert scene["scores"] == pytest.approx([0.9])
    json.dumps(result, allow_nan=False)


def test_explicit_empty_record_counts_as_missed_gt(tmp_path):
    row = _gt()
    result = evaluate_detections(*_files(tmp_path, [row], [_detection(row, empty=True)]))
    assert result["missed_gt_actors"] == result["empty_frames"] == 1
    assert result["recall"] == 0 and result["precision"] is None
    assert result["scenes"][0]["actor_labels"] == []


def test_other_known_splits_allowed_but_missing_selected_scene_rejected(tmp_path):
    test, train = _gt(), _gt("training", "train")
    paths = _files(tmp_path, [test, train], [_detection(test), _detection(train)])
    assert len(match_manifest(*paths)) == 1
    assert len(match_manifest(*paths, split=None)) == 2
    paths = _files(tmp_path, [test, train], [_detection(train)])
    with pytest.raises(ValueError, match="Missing detection"):
        match_manifest(*paths)


def test_unknown_detection_key_rejected(tmp_path):
    row = _gt()
    paths = _files(tmp_path, [row], [_detection(_gt("unknown"))])
    with pytest.raises(ValueError, match="Unknown detection"):
        match_manifest(*paths)


@pytest.mark.parametrize(
    "override",
    [
        {"source_video_id": "wrong-source"},
        {"video_id": "wrong-video"},
        {"frame_index": 6},
    ],
)
def test_inconsistent_scene_identity_rejected(tmp_path, override):
    row = _gt()
    paths = _files(tmp_path, [row], [_detection(row, **override)])
    with pytest.raises(ValueError, match="source/video/center"):
        match_manifest(*paths)


def test_invalid_split_and_threshold_fail_before_reading(tmp_path):
    with pytest.raises(ValueError, match="split"):
        match_manifest(tmp_path / "missing", tmp_path / "missing", split="invalid")
    with pytest.raises(ValueError, match="threshold"):
        match_manifest(tmp_path / "missing", tmp_path / "missing", iou_threshold=0)


@pytest.mark.parametrize("script", ["match_actor_boxes.py", "evaluate_detector.py"])
@pytest.mark.parametrize("override", [False, True])
def test_cli_writes_valid_json_with_scene_provenance(tmp_path, script, override):
    row = _gt()
    manifest, detections = _files(tmp_path, [row], [_detection(row)])
    output = tmp_path / "nested" / "report.json"
    config = tmp_path / "detector.yaml"
    config.write_text("matching:\n  iou_threshold: 0.75\n", encoding="utf-8")
    threshold_args = ["--iou-threshold", "0.5"] if override else []
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / script),
            "--manifest",
            str(manifest),
            "--detections",
            str(detections),
            "--split",
            "all",
            "--config",
            str(config),
            *threshold_args,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scenes"][0]["source_video_id"] == row.source_video_id
    assert report["scenes"][0]["actor_labels"] == [3]
    assert report["scenes"][0]["matches"]["detection_indices"] == [0]
    assert report["scenes"][0]["iou_threshold"] == (0.5 if override else 0.75)
