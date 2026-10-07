import json
from dataclasses import replace

import pytest
import torch

from surveillance.detection import (
    DetectionRecord,
    DetectionResult,
    detection_fingerprint,
    read_detections,
    write_detections,
)


def record(boxes=None, clip="a", frame=5):
    boxes = torch.tensor(boxes if boxes is not None else [[0, 0, 10, 10]]).reshape(-1, 4)
    result = DetectionResult(
        boxes, torch.ones(len(boxes)), torch.ones(len(boxes)), (20, 40), {"backend": "fixture"}
    )
    return DetectionRecord("collective", "seq1", "collective:seq1", clip, frame, result)


def test_roundtrip_empty_and_deterministic_records(tmp_path):
    path = tmp_path / "detections.jsonl"
    write_detections([record([], "z", 7), record()], path)
    before = path.read_bytes()
    rows = read_detections(path)
    assert [row.clip_id for row in rows] == ["a", "z"]
    assert rows[1].result.boxes.shape == (0, 4)
    write_detections(list(reversed(rows)), path)
    assert path.read_bytes() == before
    assert json.loads(before.splitlines()[0])["coordinate_system"] == "absolute_xyxy"


def test_feature_metadata_and_fingerprint_roundtrip(tmp_path):
    row = replace(
        record(), pose_feature_path="features/a.npy", feature_metadata={"pose": {"sha256": "abc"}}
    )
    path = tmp_path / "detections.jsonl"
    write_detections([row], path)
    actual = read_detections(path)[0]
    assert actual.pose_feature_path == row.pose_feature_path
    assert actual.feature_metadata == row.feature_metadata
    assert detection_fingerprint(actual.result) == detection_fingerprint(row.result)
    assert detection_fingerprint(row.result) != detection_fingerprint(record([]).result)


def test_writer_canonicalizes_without_features(tmp_path):
    row = record([[20, 0, 30, 10], [0, 0, 10, 10]])
    path = tmp_path / "d.jsonl"
    write_detections([row], path)
    assert read_detections(path)[0].result.boxes[:, 0].tolist() == [0, 20]
    with pytest.raises(ValueError, match="alignment"):
        write_detections([replace(row, rgb_feature_path="a.npy")], path)


def test_reader_canonicalizes_only_without_features(tmp_path):
    path = tmp_path / "d.jsonl"
    write_detections([record([[0, 0, 10, 10], [20, 0, 30, 10]])], path)
    data = json.loads(path.read_text())
    data["detections"].reverse()
    path.write_text(json.dumps(data))
    assert read_detections(path)[0].result.boxes[:, 0].tolist() == [0, 20]
    data["pose_feature_path"] = "a.npy"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="alignment"):
        read_detections(path)


@pytest.mark.parametrize("rows", [[], [record(), record()], [record(), record(clip="b")]])
def test_rejects_empty_duplicate_clip_and_center(tmp_path, rows):
    with pytest.raises(ValueError):
        write_detections(rows, tmp_path / "d.jsonl")


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("coordinate_system", "normalized_xyxy"),
        ("image_height", 0),
        ("frame_index", True),
    ],
)
def test_reader_rejects_invalid_schema(tmp_path, field, value):
    path = tmp_path / "d.jsonl"
    write_detections([record()], path)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="line 1"):
        read_detections(path)


def test_reader_rejects_fractional_class(tmp_path):
    path = tmp_path / "d.jsonl"
    write_detections([record()], path)
    data = json.loads(path.read_text())
    data["detections"][0]["class_id"] = 1.5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="class_id"):
        read_detections(path)


def test_records_reject_non_person_but_accept_declared_custom_person(tmp_path):
    row = record()
    non_person = replace(row.result, class_ids=torch.tensor([2]))
    with pytest.raises(ValueError, match="person class"):
        replace(row, result=non_person)
    custom = replace(non_person, metadata={"person_class_id": 2})
    path = tmp_path / "custom.jsonl"
    write_detections([replace(row, result=custom)], path)
    assert read_detections(path)[0].result.class_ids.tolist() == [2]
