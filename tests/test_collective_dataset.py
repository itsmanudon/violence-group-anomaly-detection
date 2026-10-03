from dataclasses import replace

import cv2
import numpy as np
import pytest

from surveillance.datasets.collective import (
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
    ActorFeatureDataset,
    ActorRecord,
    parse_collective_annotations,
    prepare_collective,
    read_actor_manifest,
    temporal_frame_indices,
    write_actor_manifest,
)


def record(**kwargs):
    fields = dict(
        dataset="collective",
        video_id="seq01",
        source_video_id="collective:seq01",
        clip_id="seq01:0001",
        split="train",
        frame_paths=["frame.jpg"] * 10,
        frame_indices=[1] * 6 + [2, 3, 4, 5],
        actor_boxes=[[0.1, 0.2, 0.5, 0.8]],
        actor_labels=[0],
        group_label=0,
    )
    fields.update(kwargs)
    return ActorRecord(**fields)


def test_actual_raw_columns_labels_and_majority(tmp_path):
    path = tmp_path / "annotations.txt"
    path.write_text(
        "1 10 20 30 40 2 7\n1 20 20 30 40 6 3\n"
        "1 30 20 30 40 6 1\n1 0 0 10 10 1 0\n"
        "2 10 20 30 40 3 0\n11 10 20 30 40 4 0\n"
    )
    scenes = parse_collective_annotations(path, (100, 200))
    assert sorted(scenes) == [1, 11]
    assert scenes[1]["actor_labels"] == [0, 4, 4]
    assert scenes[1]["group_label"] == 4
    assert scenes[1]["actor_boxes"][0] == [0.05, 0.2, 0.2, 0.6]
    assert scenes[11]["group_label"] == 2


@pytest.mark.parametrize("row", ["1 0 0 -1 20 2", "1 0 0 20 20 8", "1 0 0 20"])
def test_bad_raw_annotations_fail(tmp_path, row):
    path = tmp_path / "annotations.txt"
    path.write_text(row)
    with pytest.raises(ValueError):
        parse_collective_annotations(path, (100, 100))


def test_tie_policy_and_all_na_fail(tmp_path):
    path = tmp_path / "annotations.txt"
    path.write_text("1 0 0 10 10 6\n1 10 0 10 10 2")
    assert parse_collective_annotations(path, (100, 100))[1]["group_label"] == 0
    path.write_text("1 0 0 10 10 1\n")
    with pytest.raises(ValueError, match="no actors"):
        parse_collective_annotations(path, (100, 100))


def test_clamped_ten_frames_and_verified_split():
    assert temporal_frame_indices(1, 12) == [1] * 6 + [2, 3, 4, 5]
    assert temporal_frame_indices(11, 12) == [6, 7, 8, 9, 10, 11, 12, 12, 12, 12]
    assert len(TRAIN_SEQUENCES) == 32 and len(TEST_SEQUENCES) == 12
    assert not set(TRAIN_SEQUENCES) & set(TEST_SEQUENCES)


def test_manifest_roundtrip_duplicate_and_source_leakage(tmp_path):
    path = tmp_path / "manifest.jsonl"
    first = record()
    write_actor_manifest([first], path)
    assert read_actor_manifest(path) == [first]
    with pytest.raises(ValueError, match="duplicate"):
        write_actor_manifest([first, first], path)
    with pytest.raises(ValueError, match="leakage"):
        write_actor_manifest([first, replace(first, clip_id="other", split="test")], path)
    # Renaming a source cannot hide a reused physical frame across splits.
    with pytest.raises(ValueError, match="leakage"):
        write_actor_manifest(
            [
                first,
                replace(
                    first, video_id="seq05", clip_id="other", source_video_id="other", split="test"
                ),
            ],
            path,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(actor_boxes=[]),
        dict(actor_labels=[]),
        dict(actor_boxes=[[0.5, 0, 0.2, 1]]),
        dict(actor_boxes=[[0, 0, float("nan"), 1]]),
        dict(actor_labels=[-1]),
        dict(group_label=-1),
        dict(frame_indices=[1]),
    ],
)
def test_invalid_record(kwargs):
    with pytest.raises(ValueError):
        record(**kwargs)


def test_precomputed_features_and_shape_validation(tmp_path):
    np.save(tmp_path / "pose.npy", np.ones((1, 4), dtype=np.float32))
    path = tmp_path / "manifest.jsonl"
    write_actor_manifest([record(pose_feature_path="pose.npy")], path)
    ds = ActorFeatureDataset(path, "train", pose_feature_dim=4)
    assert ds[0]["pose_features"].shape == (1, 4)
    assert ds[0]["metadata"]["clip_id"] == "seq01:0001"
    with pytest.raises(ValueError, match="expected"):
        ActorFeatureDataset(path, "train", pose_feature_dim=5)[0]
    with pytest.raises(ValueError, match="rgb_feature_path"):
        ActorFeatureDataset(path, "train", mode="pose_rgb_early_fusion", pose_feature_dim=4)[0]


def test_rgb_fusion_nonfinite_and_actor_order(tmp_path):
    path = tmp_path / "manifest.jsonl"
    np.save(tmp_path / "pose.npy", np.full((1, 4), 2, dtype=np.float32))
    np.save(tmp_path / "rgb.npy", np.full((1, 6), 3, dtype=np.float32))
    write_actor_manifest([record(pose_feature_path="pose.npy", rgb_feature_path="rgb.npy")], path)
    ds = ActorFeatureDataset(
        path, None, mode="pose_rgb_late_fusion", pose_feature_dim=4, rgb_feature_dim=6
    )
    assert ds[0]["rgb_features"].tolist() == [[3] * 6]
    assert ds[0]["pose_features"].tolist() == [[2] * 4]
    np.save(tmp_path / "rgb.npy", np.full((1, 6), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        ds[0]


def test_read_checks_unselected_splits_and_duplicate_centers(tmp_path):
    import json
    from dataclasses import asdict

    path = tmp_path / "manifest.jsonl"
    first = record(pose_feature_path="unused.npy")
    second = replace(first, clip_id="new-name", split="test")
    path.write_text(json.dumps(asdict(first)) + "\n" + json.dumps(asdict(second)))
    with pytest.raises(ValueError, match="leakage"):
        ActorFeatureDataset(path, "train")
    with pytest.raises(ValueError, match="duplicate source/center"):
        write_actor_manifest([first, replace(first, clip_id="new-name")], path)


def test_raw_dataset_rgb_and_resize(tmp_path):
    img = np.zeros((8, 12, 3), dtype=np.uint8)
    img[:, :, 2] = 255
    cv2.imwrite(str(tmp_path / "frame.jpg"), img)
    path = tmp_path / "manifest.jsonl"
    write_actor_manifest([record()], path)
    item = ActorFeatureDataset(path, "train", input_mode="raw", image_size=(4, 6))[0]
    assert item["frames"].shape == (10, 3, 4, 6)
    assert item["frames"][0, 0].mean() > 0.98
    assert item["frames"][0, 2].mean() < 0.02


def test_prepare_local_sequences(tmp_path):
    for sid in (1, 5):
        folder = tmp_path / f"seq{sid:02d}"
        folder.mkdir()
        (folder / "annotations.txt").write_text("1 0 0 6 4 2 0\n")
        for i in range(1, 6):
            cv2.imwrite(str(folder / f"frame{i:04d}.jpg"), np.zeros((8, 12, 3), np.uint8))
    rows = prepare_collective(
        tmp_path, train_sequences=[1], test_sequences=[5], require_full_split=False
    )
    assert [r.split for r in rows] == ["train", "test"]
    assert rows[0].actor_boxes == [[0, 0, 0.5, 0.5]]
    with pytest.raises(ValueError, match="32/12"):
        prepare_collective(tmp_path, train_sequences=[1], test_sequences=[5])
