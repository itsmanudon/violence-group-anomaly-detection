import pytest
import torch

from surveillance.datasets.dcsass_actor_inputs import coverage_report
from surveillance.video.actor_window import centered_frame_indices


def test_centered_sampling_uses_ten_frames_and_short_clip_edge_replication():
    assert centered_frame_indices(60) == list(range(25, 35))
    assert centered_frame_indices(3) == [0, 0, 0, 0, 0, 1, 2, 2, 2, 2]
    assert centered_frame_indices(1) == [0] * 10
    assert centered_frame_indices(100, center=20, start=18, end=23) == [
        18,
        18,
        18,
        18,
        19,
        20,
        21,
        22,
        22,
        22,
    ]
    with pytest.raises(ValueError):
        centered_frame_indices(0)
    with pytest.raises(ValueError):
        centered_frame_indices(10, center=10)


def test_detector_coverage_reports_uncovered_clips_without_gt_metrics():
    rows = [
        {"group_label": 0, "boxes": [[0, 0, 10, 10]], "scores": [0.8]},
        {"group_label": 3, "boxes": [], "scores": []},
        {"group_label": 3, "boxes": [[0, 0, 10, 10], [10, 0, 20, 10]], "scores": [0.9, 0.7]},
    ]
    report = coverage_report(rows)
    assert report["total_clips"] == 3
    assert report["covered_clips"] == 2
    assert report["no_actor_clips"] == 1
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["per_class"]["Fighting"]["no_actor_rate"] == 0.5
    assert not {"precision", "recall", "mean_iou"} & set(report)


def test_group_only_batch_does_not_fabricate_actor_targets():
    from surveillance.datasets.dcsass_actor_inputs import collate_behavior

    batch = collate_behavior(
        [
            {
                "actor_boxes": torch.tensor([[0.1, 0.1, 0.8, 0.9]]),
                "rgb_features": torch.ones(1, 4),
                "group_label": 3,
            },
            {
                "actor_boxes": torch.tensor([[0.2, 0.2, 0.5, 0.8], [0.5, 0.3, 0.9, 0.9]]),
                "rgb_features": torch.ones(2, 4),
                "group_label": 0,
            },
        ]
    )
    assert "actor_labels" not in batch
    assert batch["group_labels"].tolist() == [3, 0]
    assert batch["actor_valid_mask"].tolist() == [[True, False], [True, True]]


def test_clip_cache_binds_video_identity_and_reference_without_global_frame_guess():
    from surveillance.detection.person_detector import DetectionResult
    from surveillance.experiments.dcsass_cache import detection_payload, validate_detection_cache

    row = {
        "clip_id": "Fighting002_x264_1",
        "source_video_id": "fighting002",
        "sha256": "video-bytes",
        "num_frames": 3,
        "group_label": 3,
        "split": "train",
    }
    window = {
        "reference_frame": 1,
        "frame_indices": [0, 0, 0, 0, 0, 1, 2, 2, 2, 2],
        "image_size": [24, 32],
    }
    result = DetectionResult(
        torch.empty(0, 4),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        (24, 32),
        {"checkpoint_sha256": "detector", "filter_config": {"confidence_threshold": 0.7}},
    )
    payload = detection_payload(row, window, result)
    validate_detection_cache(payload, row, "detector")
    assert payload["reference_frame_coordinate_system"] == "zero_based_clip_local"
    assert payload["boxes"] == []
    payload["clip_id"] = "different-clip"
    with pytest.raises(ValueError, match="identity"):
        validate_detection_cache(payload, row, "detector")


def test_real_short_video_window_has_rgb_reference_and_replicated_model_frames(tmp_path):
    import cv2
    import numpy as np

    from surveillance.video.actor_window import read_actor_window

    path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((24, 32, 3), (10, 50, 200), dtype=np.uint8))
    writer.release()
    window = read_actor_window(path, 3)
    assert window["frames"].shape == (10, 3, 480, 720)
    assert window["reference_rgb"].shape == (3, 24, 32)
    assert window["reference_rgb"][0].mean() > window["reference_rgb"][2].mean()
    assert window["frame_indices"] == [0, 0, 0, 0, 0, 1, 2, 2, 2, 2]
    torch.testing.assert_close(window["frames"][0], window["frames"][4])
