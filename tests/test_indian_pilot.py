from pathlib import Path

import numpy as np
import pytest

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_pilot import filename_key, phash, reconstruct_groups
from surveillance.datasets.indian_split import (
    assign_groups_stratified,
    freeze_pilot_split,
    verify_split,
)


def candidate(key, name, collection="Snatch-1.0", folder="Snatch Theft", label=1):
    return {
        "clip_id": key,
        "path": Path("/data") / folder / name,
        "collection": collection,
        "label": label,
        "sha256": key * 8,
    }


def test_phash_is_stable_and_separates_different_images():
    rng = np.random.default_rng(0)
    image = (rng.random((240, 320)) * 255).astype(np.uint8)
    other = (rng.random((240, 320)) * 255).astype(np.uint8)
    brighter = np.clip(image.astype(int) + 10, 0, 255).astype(np.uint8)
    assert bin(phash(image) ^ phash(brighter)).count("1") <= 4
    assert bin(phash(image) ^ phash(other)).count("1") > 16


def test_filename_keys_follow_observed_conventions():
    # Visual review showed Snatch N.mp4 / N_0.mp4 are different footage: no link.
    assert filename_key(candidate("a", "10.mp4")) is None
    assert filename_key(candidate("b", "10_0.mp4")) is None
    assert filename_key(candidate("a", "s_11.avi", folder="Normal")) is None
    walk = candidate("w", "temple-Scene-004-scene001-chunk002.mp4", "WalkIndia-50", "temple")
    assert filename_key(walk) == "walkindia-temple-Scene-004"
    hci = candidate("h", "krishna-clip-5.mp4", "HCI-User-Collection", "HCI")
    assert filename_key(hci) == "hci-krishna"


def test_groups_join_shared_frames_filenames_and_close_capture_times():
    hci = "HCI-User-Collection"
    clips = [
        candidate("a", "10.mp4"),
        candidate("b", "10_0.mp4"),
        candidate("c", "x1.mp4", "Archive", "Videos", 0),
        candidate("d", "x2.mp4", "Archive", "Videos", 0),
        candidate("e", "Screen_Recording_20261020_202523_Instagram.mp4", hci, "HCI"),
        candidate("f", "Screen_Recording_20261020_202543_Instagram.mp4", hci, "HCI"),
        candidate("g", "Screen_Recording_20261020_180608_YouTube.mp4", hci, "HCI"),
    ]
    shared = np.array([0x0F0F0F0F0F0F0F0F, 0x1234567812345678], dtype=np.uint64)
    prints = {c["clip_id"]: {"hashes": np.array([], dtype=np.uint64)} for c in clips}
    prints["c"]["hashes"] = shared
    prints["d"]["hashes"] = shared ^ np.uint64(1)  # one bit apart on both frames
    review = [{"name": "same-watermark", "clips": ["a", "b"], "evidence": "shared watermark"}]
    groups, reasons, _ = reconstruct_groups(
        clips, prints, hamming_threshold=4, min_frame_matches=2, review_merges=review
    )
    assert sorted(groups) == [["a", "b"], ["c", "d"], ["e", "f"], ["g"]]
    assert reasons["g"] == []
    assert reasons["a"] == ["review:same-watermark"]
    assert any(r.startswith("phash") for r in reasons["c"])
    assert reasons["e"] == ["capture_time<=5min"]
    with pytest.raises(ValueError, match="evidence"):
        reconstruct_groups(
            clips,
            prints,
            hamming_threshold=4,
            min_frame_matches=2,
            review_merges=[{"name": "x", "clips": ["a", "b"], "evidence": " "}],
        )


def test_stratified_assignment_meets_requirements_and_keeps_groups_whole():
    clips, groups = {}, []
    for g in range(40):
        label = int(g < 12)
        members = [f"g{g}c{i}" for i in range(1 + g % 2)]
        groups.append(members)
        for key in members:
            clips[key] = {"label": label, "collection": "A" if g % 3 else "B"}
    requirement = [("violence", {"label": 1}, 2)]
    assignment, report = assign_groups_stratified(
        clips,
        groups,
        7,
        (0.7, 0.15, 0.15),
        strata=("label", "collection"),
        requirements=requirement,
        forced_train={"g0c0"},
        trials=64,
    )
    for members in groups:
        assert len({assignment[k] for k in members}) == 1
    assert assignment["g0c0"] == "train"
    assert report["requirements_met"]["violence|val"] >= 2
    assert report["requirements_met"]["violence|test"] >= 2
    again, _ = assign_groups_stratified(
        clips,
        groups,
        7,
        (0.7, 0.15, 0.15),
        strata=("label", "collection"),
        requirements=requirement,
        forced_train={"g0c0"},
        trials=64,
    )
    assert again == assignment
    with pytest.raises(ValueError, match="requirements"):
        assign_groups_stratified(
            clips,
            groups,
            7,
            (0.7, 0.15, 0.15),
            strata=("label",),
            requirements=[("violence", {"label": 1}, 99)],
            trials=8,
        )


def test_pilot_freeze_is_verified_like_a_source_approved_split(tmp_path):
    clips, group_of = {}, {}
    for g in range(30):
        for i in range(2):
            key = f"clip{g:02d}{i}"
            path = tmp_path / "media" / f"{key}.mp4"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(key.encode())
            clips[key] = {
                "path": path,
                "sha256": sha256(path),
                "label": int(g < 10),
                "collection": "A",
                "geography": "unknown_geography",
                "fps": 25.0,
                "num_frames": 50,
            }
            group_of[key] = f"G{g:03d}"
    groups_csv = tmp_path / "groups.csv"
    groups_csv.write_text("clip,group_id\n")
    kwargs = dict(
        pilot_override="team_labels_unadjudicated",
        seed=7,
        requirements=[("violence", {"label": 1}, 2)],
        artifacts=[groups_csv],
        excluded={"x": "quality hold"},
    )
    with pytest.raises(ValueError, match="override"):
        freeze_pilot_split(clips, group_of, tmp_path / "bad", **{**kwargs, "pilot_override": "x"})
    receipt = freeze_pilot_split(clips, group_of, tmp_path / "split", **kwargs)
    rows, verified = verify_split(tmp_path / "split")
    assert verified["freeze_hash"] == receipt["freeze_hash"]
    assert receipt["pilot_override"] == "team_labels_unadjudicated"
    assert {r.split for r in rows} == {"train", "val", "test"}
    with pytest.raises(FileExistsError):
        freeze_pilot_split(clips, group_of, tmp_path / "split", **kwargs)
    groups_csv.write_text("clip,group_id\nchanged\n")
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify_split(tmp_path / "split")


def test_test_features_require_a_committed_protocol(tmp_path):
    from surveillance.training.indian_binary import cache_c3d, committed_protocol

    with pytest.raises(ValueError, match="test-protocol"):
        cache_c3d(tmp_path / "split", tmp_path / "c3d.pt", tmp_path / "cache", splits=("test",))
    root = Path(__file__).resolve().parents[1]
    draft = root / "runs" / "indian_pilot" / "uncommitted-protocol-fixture.md"
    draft.write_text("draft")
    try:
        with pytest.raises(ValueError, match="committed"):
            committed_protocol(draft)
    finally:
        draft.unlink()
    assert committed_protocol(root / "docs/handover/HANDOVER.md")["commit"]


def test_demo_pilot_head_is_hash_pinned_and_scores_the_given_bag(tmp_path):
    import json

    import torch

    from surveillance.inference.indian_pilot import IndianPilotHead, load_pilot_spec
    from surveillance.models.indian_binary import FrozenBagBinaryHead

    torch.manual_seed(0)
    head = FrozenBagBinaryHead(4)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "checkpoint_type": "indian_frozen_bag_binary_v1",
            "classes": ["non_violence", "violence"],
            "feature_dim": 4,
            "model_state": head.state_dict(),
        },
        checkpoint,
    )
    spec = {
        "checkpoint": "best.pt",
        "checkpoint_sha256": sha256(checkpoint),
        "threshold": 0.4,
        "split_freeze_hash": "f" * 64,
        "threshold_rule": "max validation F1 with recall >= 0.8",
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert load_pilot_spec(path) == spec
    bag = np.arange(8, dtype=np.float32).reshape(2, 4) / 8
    result = IndianPilotHead(spec, tmp_path).score(bag)
    expected = float(head(torch.from_numpy(bag)[None]).softmax(-1)[0, 1])
    assert result["violence_score"] == pytest.approx(expected)
    assert result["flagged"] == (expected >= 0.4)
    assert "human review" in result["scope"]
    with pytest.raises(ValueError, match="SHA256"):
        IndianPilotHead({**spec, "checkpoint_sha256": "0" * 64}, tmp_path)
    path.write_text(json.dumps({**spec, "threshold": 1.5}))
    with pytest.raises(ValueError, match="threshold"):
        load_pilot_spec(path)


def test_merged_splits_keep_assignments_and_never_merge_sources_across_datasets(tmp_path):
    from surveillance.datasets.indian_split import merge_frozen_splits

    records = []
    for dataset, split in (("a", "train"), ("b", "test")):
        for i in range(2):
            path = tmp_path / f"{dataset}{i}.mp4"
            path.write_bytes(f"{dataset}{i}".encode())
            records.append(
                {
                    "dataset": dataset,
                    "video_id": f"{dataset}{i}",
                    "source_video_id": "shared-name",
                    "path": path,
                    "split": split,
                    "label": i,
                    "sha256": sha256(path),
                    "fps": 25.0,
                    "num_frames": 50,
                }
            )
    receipt = merge_frozen_splits(
        records, tmp_path / "merged", artifacts=[], scope="test", excluded={}
    )
    rows, _ = verify_split(tmp_path / "merged")
    assert {r.video_id: r.split for r in rows} == {
        "a0": "train",
        "a1": "train",
        "b0": "test",
        "b1": "test",
    }
    assert {r.source_video_id for r in rows} == {"a:shared-name", "b:shared-name"}
    assert receipt["dataset_counts"]["b|test"] == 2
    records[0]["split"] = "val"
    with pytest.raises(ValueError, match="leakage"):
        merge_frozen_splits(records, tmp_path / "bad", artifacts=[], scope="test", excluded={})
