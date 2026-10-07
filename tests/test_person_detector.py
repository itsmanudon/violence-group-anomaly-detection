import pytest
import torch

from surveillance.detection import (
    DetectionConfig,
    DetectionResult,
    filter_detections,
    load_detection_config,
    validate_detection_config,
)


def test_filter_stages_clamp_nms_and_counts():
    boxes = torch.tensor(
        [
            [-2, -2, 20, 20],
            [0, 0, 20, 20],
            [30, 0, 45, 20],
            [50, 0, 60, 20],
            [0, 0, 1, 1],
            [float("nan"), 0, 10, 10],
            [10, 0, 0, 20],
            [200, 0, 210, 10],
            [0, 0, 0, 20],
        ]
    )
    result = filter_detections(
        boxes,
        torch.tensor([0.9, 0.8, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),
        torch.tensor([1, 1, 1, 2, 1, 1, 1, 1, 1]),
        (100, 100),
    )
    assert result.boxes.tolist() == [[0, 0, 20, 20]]
    assert result.metadata["filter_counts"] == {
        "non_person": 1,
        "low_confidence": 1,
        "invalid_geometry": 3,
        "outside_image": 1,
        "too_small": 1,
        "nms": 1,
        "max_actors": 0,
    }
    assert result.metadata["before_count"] == 9
    assert result.metadata["after_count"] == 1
    assert result.metadata["num_detections_before_filter"] == 9
    assert result.metadata["num_detections_after_filter"] == 1
    assert result.metadata["filter_config"]["confidence_threshold"] == 0.5
    assert result.metadata["person_class_id"] == 1


def test_top_confidence_then_left_to_right_and_inclusive_threshold():
    result = filter_detections(
        torch.tensor([[50, 0, 60, 10], [0, 0, 10, 10], [25, 0, 35, 10]]),
        torch.tensor([0.8, 0.9, 0.5]),
        torch.ones(3),
        (100, 100),
        DetectionConfig(max_actors=2),
    )
    assert result.boxes[:, 0].tolist() == [0, 50]
    assert result.metadata["truncated"]
    assert result.metadata["filter_counts"]["max_actors"] == 1


def test_equal_scores_nms_and_truncation_are_permutation_invariant():
    boxes = torch.tensor([[1, 0, 21, 20], [0, 0, 20, 20], [50, 0, 60, 10]])
    outputs = [
        filter_detections(
            boxes[p], torch.ones(3), torch.ones(3), (100, 100), DetectionConfig(max_actors=1)
        ).boxes
        for p in ([0, 1, 2], [2, 1, 0], [1, 0, 2])
    ]
    assert all(torch.equal(outputs[0], other) for other in outputs)
    assert outputs[0].tolist() == [[0, 0, 20, 20]]


def test_empty_and_minimum_area():
    result = filter_detections(torch.empty(0, 4), torch.empty(0), torch.empty(0), (20, 30))
    assert result.boxes.shape == (0, 4)
    result = filter_detections(
        torch.tensor([[0, 0, 5, 5]]),
        torch.ones(1),
        torch.ones(1),
        (20, 30),
        DetectionConfig(min_box_area=26),
    )
    assert result.boxes.shape == (0, 4)


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [-0.1], [1.1]])
def test_invalid_scores_fail(scores):
    with pytest.raises(ValueError, match="scores"):
        filter_detections(torch.tensor([[0, 0, 10, 10]]), scores, [1], (20, 20))


@pytest.mark.parametrize(
    "boxes,scores,classes",
    [([1, 2, 3, 4], [1], [1]), ([[0, 0, 10, 10]], [], [1]), ([[0, 0, 10, 10]], [1], [1.5])],
)
def test_invalid_shapes_classes(boxes, scores, classes):
    with pytest.raises(ValueError):
        filter_detections(boxes, scores, classes, (20, 20))


@pytest.mark.parametrize(
    "box", [[0, 0, 21, 20], [0, 0, 0, 10], [-1, 0, 5, 10], [0, 0, float("inf"), 10]]
)
def test_result_rejects_invalid_final_geometry(box):
    with pytest.raises(ValueError, match="boxes"):
        DetectionResult(torch.tensor([box]), torch.ones(1), torch.ones(1), (20, 20))


@pytest.mark.parametrize(
    "settings",
    [
        {"max_actors": 0},
        {"max_actors": True},
        {"confidence_threshold": float("nan")},
        {"nms_iou_threshold": 1.1},
        {"min_box_width": -1},
        {"person_class_id": 1.5},
    ],
)
def test_config_invalid(settings):
    with pytest.raises(ValueError):
        DetectionConfig(**settings)


def test_yaml_relative_path_and_unknown_keys(tmp_path):
    path = tmp_path / "detector.yaml"
    path.write_text("detector:\n  checkpoint: weights.pth\n", encoding="utf-8")
    config = load_detection_config(path)
    assert config["detector"]["checkpoint"] == str(tmp_path / "weights.pth")
    assert config["matching"]["iou_threshold"] == 0.5
    with pytest.raises(ValueError, match="Unknown"):
        validate_detection_config({"detector": {"typo": 1}})
    with pytest.raises(ValueError, match="device"):
        validate_detection_config({"device": "bad-device"})
    with pytest.raises(ValueError, match="matching.iou_threshold"):
        validate_detection_config({"matching": {"iou_threshold": 0}})
