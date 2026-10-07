from pathlib import Path

import numpy as np
import pytest

from surveillance.datasets.common import Record
from surveillance.datasets.ucf_crime import prepare_ucf_crime


def test_external_archive_normal_folders_and_event_exclusion(tmp_path, monkeypatch):
    root = tmp_path / "external"
    for relative in [
        "Training-Normal-Videos-Part-1/Normal_Videos001_x264.mp4",
        "Testing_Normal_Videos_Anomaly/Normal_Videos_001_x264.mp4",
        "Anomaly-Videos-Part-1/Fighting/Fighting001_x264.mp4",
        "Normal_Videos_for_Event_Recognition/Normal_Videos001_x264.mp4",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"metadata fixture")
    monkeypatch.setattr(
        "surveillance.datasets.ucf_crime.make_record",
        lambda dataset, path, root, output, label, category, mapping: Record(
            dataset, path.stem, path.stem, str(path), "train", label, category
        ),
    )
    rows = prepare_ucf_crime(root, tmp_path / "manifest.jsonl")
    assert len(rows) == 3
    assert sorted(r.label for r in rows) == [0, 0, 1]


def test_cross_volume_manifest_uses_absolute_video_path(tmp_path, monkeypatch):
    from surveillance.datasets.preparation import make_record
    from surveillance.video.decode import VideoMetadata

    root = tmp_path / "external"
    root.mkdir()
    path = root / "Fighting001_x264.mp4"
    monkeypatch.setattr(
        "surveillance.datasets.preparation.probe_video",
        lambda p: VideoMetadata(30, 60, 2),
    )
    monkeypatch.setattr(
        "surveillance.datasets.preparation.os.path.relpath",
        lambda *args: (_ for _ in ()).throw(ValueError("different drive mounts")),
    )
    row = make_record("ucf_crime", path, root, tmp_path / "manifest.jsonl", 1, "Fighting", {})
    assert Path(row.path) == path.resolve()


def test_shared_dcsass_held_out_sources_never_enter_ucf_training():
    from surveillance.experiments.sultani_ucf import protect_shared_sources

    rows = [Record("ucf_crime", str(i), str(i), f"{i}.mp4", "train", i % 2) for i in range(30)]
    rows.append(Record("ucf_crime", "official", "official", "test.mp4", "test", 1))
    shared = {"0": "test", "1": "val", "2": "train", "official": "test"}
    prepared, excluded = protect_shared_sources(rows, shared, seed=0)
    assert [r.source_video_id for r in excluded] == ["0"]
    splits = {r.source_video_id: r.split for r in prepared}
    assert splits["1"] == "val" and splits["2"] == "train" and splits["official"] == "test"
    assert prepared == protect_shared_sources(list(reversed(rows)), shared, seed=0)[0]
    with pytest.raises(ValueError, match="official test"):
        protect_shared_sources(rows, {"official": "train"}, seed=0)


@pytest.mark.parametrize("frames", [1, 17, 513, 1025])
def test_exact_c3d_frame_projection_matches_unit_partition(frames):
    from surveillance.evaluation.anomaly_metrics import project_scores
    from surveillance.video.segmentation import c3d_segment_frame_ranges

    scores = np.arange(32) / 32
    ranges = c3d_segment_frame_ranges(frames)
    projected = project_scores(scores, frames, "c3d_units")
    for bounds in set(ranges):
        indices = [i for i, pair in enumerate(ranges) if pair == bounds]
        np.testing.assert_allclose(projected[bounds[0] : bounds[1]], scores[indices].mean())
    assert len(projected) == frames and np.isfinite(projected).all()


def test_ucf_cache_reuses_only_identical_finite_bags(tmp_path):
    from surveillance.datasets.dcsass_audit import sha256
    from surveillance.experiments.sultani_ucf import extract_cached_bag

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video byte identity fixture")
    row = Record(
        "ucf_crime",
        "Normal/clip",
        "source",
        str(video),
        "train",
        0,
        fps=30,
        num_frames=60,
        duration_sec=2,
    )

    class Extractor:
        metadata = {"fixture": True}
        calls = 0

        def extract_video(self, *args):
            self.calls += 1
            return np.ones((32, 4096), dtype=np.float32)

    extractor = Extractor()
    digest = sha256(video)
    path, reused = extract_cached_bag(row, digest, extractor, tmp_path / "cache")
    assert not reused and path.is_file()
    assert extract_cached_bag(row, digest, extractor, tmp_path / "cache")[1]
    assert extractor.calls == 1
    values = np.load(path)
    values[0, 0] = np.nan
    np.save(path, values)
    with pytest.raises(ValueError, match="cache"):
        extract_cached_bag(row, digest, extractor, tmp_path / "cache")
    video.write_bytes(b"changed raw data")
    with pytest.raises(ValueError, match="Video changed"):
        extract_cached_bag(row, digest, extractor, tmp_path / "other")


def test_annotation_endpoint_intersection_is_explicit_and_bounded(tmp_path):
    from surveillance.datasets.ucf_crime import apply_frame_annotations

    row = Record(
        "ucf_crime",
        "Arson011",
        "arson011",
        "Arson011_x264.mp4",
        "test",
        1,
        num_frames=1266,
        fps=30,
        duration_sec=1266 / 30,
    )
    annotation = tmp_path / "annotation.txt"
    annotation.write_text("Arson011_x264.mp4 Arson 680 1267 -1 -1\n")
    with pytest.raises(ValueError, match="exceeds"):
        apply_frame_annotations([row], annotation)
    corrections = []
    prepared = apply_frame_annotations(
        [row], annotation, end_overrun_tolerance=2, boundary_adjustments=corrections
    )
    assert prepared[0].temporal_annotations == [[679 / 30, 1266 / 30]]
    assert corrections == [
        {
            "filename": "arson011_x264.mp4",
            "original_frames": [680, 1267],
            "effective_frames": [680, 1266],
            "available_frames": 1266,
        }
    ]
    annotation.write_text("Arson011_x264.mp4 Arson 680 1270 -1 -1\n")
    with pytest.raises(ValueError, match="exceeds"):
        apply_frame_annotations([row], annotation, end_overrun_tolerance=2)


def test_finite_feature_mutation_is_rejected_by_training_and_evaluation_loader(tmp_path):
    from surveillance.datasets.dcsass_audit import sha256
    from surveillance.datasets.features import record_features

    path = tmp_path / "bag.npy"
    np.save(path, np.ones((32, 4), dtype=np.float32))
    row = Record(
        "fixture",
        "clip",
        "source",
        "unused.mp4",
        "test",
        0,
        feature_path=str(path),
        feature_sha256=sha256(path),
    )
    assert record_features(row, tmp_path / "manifest.jsonl", 32, 4).shape == (32, 4)
    np.save(path, np.zeros((32, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="feature identity"):
        record_features(row, tmp_path / "manifest.jsonl", 32, 4)


def test_registered_frame_policy_rejects_projection_drift_before_test_access():
    from surveillance.experiments.sultani_selection import validate_evaluation_policy

    selected = {
        "evaluation_mode": "frame",
        "frame_projection": "c3d_units",
        "anomaly_threshold": 0.5,
    }
    assert validate_evaluation_policy(selected, "frame", None) == "c3d_units"
    with pytest.raises(ValueError, match="frozen evaluation"):
        validate_evaluation_policy(selected, "frame", "repeat")
    with pytest.raises(ValueError, match="frozen evaluation"):
        validate_evaluation_policy(selected, "bag", None)
    assert validate_evaluation_policy({"anomaly_threshold": 0.5}, "bag", None) == "repeat"


def test_exact_duplicate_train_copies_cannot_expose_test_or_cross_validation():
    from surveillance.experiments.sultani_ucf import remove_training_duplicates

    rows = [
        Record("ucf_crime", name, name, name + ".mp4", split, 0)
        for name, split in [
            ("test1", "test"),
            ("test2", "test"),
            ("clone", "train"),
            ("train1", "train"),
            ("train2", "train"),
        ]
    ]
    hashes = {
        name: "test-content" if name in {"test1", "test2", "clone"} else "train-content"
        for name in [r.video_id for r in rows]
    }
    kept, excluded = remove_training_duplicates(rows, hashes)
    assert {r.video_id for r in kept} == {"test1", "test2", "train1"}
    assert {r.video_id for r in excluded} == {"clone", "train2"}
    assert remove_training_duplicates(list(reversed(rows)), hashes) == (kept, excluded)
    conflicting = [
        Record("ucf_crime", "normal", "normal", "n.mp4", "train", 0),
        Record("ucf_crime", "positive", "positive", "p.mp4", "train", 1),
    ]
    with pytest.raises(ValueError, match="Conflicting binary"):
        remove_training_duplicates(conflicting, {"normal": "same", "positive": "same"})
