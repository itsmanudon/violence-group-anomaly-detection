from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from surveillance.datasets.dcsass import prepare_dcsass
from surveillance.datasets.preparation import discover_videos, source_identity
from surveillance.datasets.ucf_crime import (
    apply_frame_annotations,
    apply_official_splits,
    prepare_ucf_crime,
)
from surveillance.features.c3d import C3DFC6, C3DExtractor
from surveillance.video.decode import iter_clips, probe_video
from surveillance.video.transforms import c3d_transform


def make_video(path: Path, frames: int = 19) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    assert writer.isOpened()
    try:
        for i in range(frames):
            writer.write(np.full((24, 32, 3), i * 5, dtype=np.uint8))
    finally:
        writer.release()


def test_decode_padding_and_transform(tmp_path):
    path = tmp_path / "video.avi"
    make_video(path)
    metadata = probe_video(path)
    assert metadata.num_frames == 19
    assert metadata.duration_sec == pytest.approx(1.9)
    clips = list(iter_clips(path))
    assert len(clips) == 2
    np.testing.assert_array_equal(clips[1][-1], clips[1][2])
    assert c3d_transform(clips[0]).shape == (3, 16, 112, 112)
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "missing.avi")


def test_dcsass_uses_clip_labels_and_source_identity(tmp_path):
    root = tmp_path / "raw"
    make_video(root / "Fighting" / "Fighting001_x264_0.avi")
    make_video(root / "Fighting" / "Fighting001_x264_1.avi")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "path,label\nFighting/Fighting001_x264_0.avi,0\nFighting/Fighting001_x264_1.avi,1\n"
    )
    records = prepare_dcsass(root, labels, tmp_path / "manifest.jsonl")
    assert [r.label for r in records] == [0, 1]
    assert {r.source_video_id for r in records} == {"fighting001"}
    assert source_identity(Path("Fighting001_x264.mp4"), {}) == "fighting001"
    with pytest.raises(ValueError, match="source identity"):
        source_identity(Path("renamed.mp4"), {})


def test_ucf_official_split_and_annotations(tmp_path):
    root = tmp_path / "raw"
    make_video(root / "Fighting" / "Fighting001_x264.avi")
    make_video(root / "Normal_Videos" / "Normal_Videos001_x264.avi")
    rows = prepare_ucf_crime(root, tmp_path / "manifest.jsonl")
    training, testing = tmp_path / "train.txt", tmp_path / "test.txt"
    training.write_text("Normal_Videos001_x264.avi\n")
    testing.write_text("Fighting001_x264.avi\n")
    rows = apply_official_splits(rows, training, testing)
    annotations = tmp_path / "annotations.txt"
    annotations.write_text("Fighting001_x264.avi Fighting 2 5 -1 -1\n")
    rows = apply_frame_annotations(rows, annotations)
    fight = next(r for r in rows if r.label == 1)
    assert fight.split == "test"
    assert fight.temporal_annotations == [[0.1, 0.5]]


def test_c3d_missing_and_incompatible_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="C3D checkpoint missing"):
        C3DExtractor(None)
    path = tmp_path / "bad.pt"
    torch.save({"state_dict": {"wrong": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="Incompatible C3D"):
        C3DExtractor(path)


def test_c3d_fc6_shape_without_weights_or_allocation():
    with torch.device("meta"):
        model = C3DFC6()
        assert model(torch.empty(2, 3, 16, 112, 112)).shape == (2, 4096)


def test_discovery_ignores_source_directories_named_mp4(tmp_path):
    clip = tmp_path / "Fighting002_x264.mp4" / "Fighting002_x264_0.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"fixture")
    assert discover_videos(tmp_path) == [clip]
