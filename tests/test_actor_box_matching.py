"""Threshold-aware assignment must preserve cardinality and label provenance."""

import json

import pytest
import torch

from surveillance.detection.matching import (
    MatchResult,
    match_boxes,
    pairwise_iou,
    transfer_actor_labels,
)


def boxes(rows):
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, 4)


def test_perfect_permuted_matching_and_label_transfer():
    gt = boxes([[0, 0, 10, 10], [20, 20, 30, 30]])
    result = match_boxes(gt, gt.flip(0))
    assert result.gt_indices == [0, 1]
    assert result.detection_indices == [1, 0]
    assert result.ious == [1.0, 1.0]
    assert not result.missed_gt and not result.unmatched_detections
    assert transfer_actor_labels(torch.tensor([3, 4]), result).tolist() == [4, 3]
    assert MatchResult(**json.loads(json.dumps(result.to_dict(), allow_nan=False))) == result


def test_partial_missing_extra_and_unmatched_labels():
    result = match_boxes(
        boxes([[0, 0, 10, 10], [20, 20, 30, 30]]),
        boxes([[0, 0, 10, 10], [40, 40, 50, 50], [60, 60, 70, 70]]),
    )
    assert result.gt_indices == result.detection_indices == [0]
    assert result.missed_gt == [1]
    assert result.unmatched_detections == [1, 2]
    assert transfer_actor_labels(torch.tensor([2, 3]), result).tolist() == [2, -100, -100]


@pytest.mark.parametrize("gt_count,detection_count", [(0, 0), (0, 2), (2, 0)])
def test_empty_inputs_have_no_fake_matches(gt_count, detection_count):
    gt = boxes([[0, 0, 10, 10]] * gt_count)
    detections = boxes([[0, 0, 10, 10]] * detection_count)
    assert pairwise_iou(gt, detections).shape == (gt_count, detection_count)
    result = match_boxes(gt, detections)
    assert result.gt_indices == result.detection_indices == result.ious == []
    assert result.missed_gt == list(range(gt_count))
    assert result.unmatched_detections == list(range(detection_count))
    labels = transfer_actor_labels(torch.zeros(gt_count, dtype=torch.long), result)
    assert labels.tolist() == [-100] * detection_count


def test_threshold_is_inclusive():
    result = match_boxes(boxes([[0, 0, 2, 2]]), boxes([[0, 0, 1, 2]]), 0.5)
    assert result.ious == [0.5]
    assert match_boxes(boxes([[0, 0, 2, 2]]), boxes([[0, 0, 1, 2]]), 0.50001).ious == []


def test_assignment_prioritizes_threshold_eligible_cardinality(monkeypatch):
    # Plain maximum-IoU picks the diagonal (1.48), then filtering leaves one
    # match. The off diagonal yields two threshold-eligible matches (1.02).
    monkeypatch.setattr(
        "surveillance.detection.matching.pairwise_iou",
        lambda *_: torch.tensor([[0.99, 0.51], [0.51, 0.49]], dtype=torch.float64),
    )
    result = match_boxes(boxes([]), boxes([]))
    assert result.gt_indices == [0, 1]
    assert result.detection_indices == [1, 0]


def test_assignment_uses_iou_to_break_cardinality_ties(monkeypatch):
    monkeypatch.setattr(
        "surveillance.detection.matching.pairwise_iou",
        lambda *_: torch.tensor([[0.7, 0.9], [0.8, 0.7]], dtype=torch.float64),
    )
    assert match_boxes(boxes([]), boxes([])).detection_indices == [1, 0]


def test_repeated_identical_boxes_are_deterministic():
    gt = boxes([[0, 0, 10, 10]] * 3)
    first = match_boxes(gt, gt)
    assert all(match_boxes(gt, gt) == first for _ in range(5))
    assert len(set(first.detection_indices)) == 3


@pytest.mark.parametrize("threshold", [0, -1, 1.1, float("nan"), float("inf"), True, None])
def test_invalid_thresholds_fail_even_on_empty_inputs(threshold):
    with pytest.raises(ValueError, match="threshold"):
        match_boxes(boxes([]), boxes([]), threshold)


@pytest.mark.parametrize(
    "invalid",
    [
        torch.ones(4),
        torch.zeros(1, 3),
        boxes([[0, 0, 0, 1]]),
        boxes([[0, 2, 1, 1]]),
        boxes([[0, 0, float("nan"), 1]]),
        boxes([[-1, 0, 1, 1]]),
        boxes([[0, 0, float("inf"), 1]]),
    ],
)
def test_invalid_boxes_fail(invalid):
    with pytest.raises(ValueError):
        pairwise_iou(invalid, boxes([]))


def test_large_coordinates_produce_finite_iou():
    huge = boxes([[0, 0, 1e200, 1e200]])
    assert pairwise_iou(huge, huge).item() == 1


@pytest.mark.parametrize("labels", [torch.tensor([1.0]), torch.tensor([1, 2]), torch.tensor([-1])])
def test_label_transfer_rejects_invalid_labels(labels):
    result = match_boxes(boxes([[0, 0, 1, 1]]), boxes([[0, 0, 1, 1]]))
    with pytest.raises(ValueError, match="labels"):
        transfer_actor_labels(labels, result)


def test_match_result_rejects_duplicate_or_missing_indices():
    with pytest.raises(ValueError, match="partition"):
        MatchResult([0, 0], [0, 1], [1, 1], [], [], 2, 2)
