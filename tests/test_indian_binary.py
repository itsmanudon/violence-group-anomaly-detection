import copy
import json

import numpy as np
import pytest
import torch
from test_indian_split import split_fixture
from torch.nn import functional as F

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import write_new_json
from surveillance.datasets.indian_split import freeze_split, verify_split
from surveillance.evaluation.indian_binary import (
    calibrate_threshold,
    evaluate_abstention,
    evaluate_binary,
    temporal_alerts,
)
from surveillance.models.indian_binary import BinaryActorAdapter, FrozenBagBinaryHead
from surveillance.training.actor_transformer_trainer import ActorTransformerSystem
from surveillance.training.dcsass_behavior import load_behavior_checkpoint
from surveillance.training.indian_binary import (
    binary_class_weights,
    cache_c3d,
    calibrate_binary_checkpoint,
    load_binary_checkpoint,
    train_binary,
    training_indices,
)


def test_binary_shapes_and_six_class_representation_preserved():
    config = {
        "model": {
            "mode": "rgb_only",
            "rgb_feature_dim": 4,
            "embedding_dim": 8,
            "num_group_classes": 6,
            "num_actor_classes": 5,
            "transformer": {"num_heads": 1, "num_layers": 1, "feedforward_dim": 16, "dropout": 0.0},
        }
    }
    source = ActorTransformerSystem(config).eval()
    original = copy.deepcopy(source.state_dict())
    batch = {
        "rgb_features": torch.randn(2, 3, 4),
        "actor_boxes": torch.tensor([[[0.1, 0.1, 0.9, 0.9]] * 3] * 2),
        "actor_valid_mask": torch.tensor([[True, True, False], [True, False, False]]),
    }
    adapter = BinaryActorAdapter(source).train()
    assert not adapter.representation.training
    assert adapter(batch).shape == (2, 2)
    F.cross_entropy(adapter(batch), torch.tensor([0, 1])).backward()
    assert adapter.classifier.weight.grad is not None
    assert all(p.grad is None for p in adapter.representation.parameters())
    assert source(batch)["group_logits"].shape == (2, 6)
    assert all(torch.equal(value, original[key]) for key, value in source.state_dict().items())
    selective = BinaryActorAdapter(source, classifier_tuning=True).train()
    F.cross_entropy(selective(batch), torch.tensor([0, 1])).backward()
    assert selective.representation.model.rgb_projection.weight.grad is not None
    assert selective.representation.model.branches["rgb"].group_classifier.weight.grad is None


def test_bag_shapes_and_invalid_features():
    model = FrozenBagBinaryHead(4)
    assert model(torch.ones(2, 8, 4)).shape == (2, 2)
    for value in (torch.ones(2, 4), torch.empty(2, 0, 4), torch.full((2, 8, 4), float("nan"))):
        with pytest.raises(ValueError, match="finite"):
            model(value)


def test_metrics_and_recall_constrained_threshold():
    labels, scores = [0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9]
    report = evaluate_binary(labels, scores)
    assert report["confusion_matrix"] == [[1, 1], [1, 1]]
    assert report["f1"] == report["false_positive_rate"] == 0.5
    assert report["roc_auc"] == 0.75
    assert 0 <= report["pr_auc"] <= 1
    assert report["false_alarms_per_camera_hour"] is None
    result = calibrate_threshold(
        labels, scores, split="val", min_recall=1, candidates=[0.1, 0.4, 0.5, 0.9]
    )
    assert result["selected_threshold"] == 0.4
    with pytest.raises(ValueError, match="validation-only"):
        calibrate_threshold(labels, scores, split="test", min_recall=1, candidates=[0.4])
    with pytest.raises(ValueError, match="recall floor"):
        calibrate_threshold(labels, scores, split="val", min_recall=1, candidates=[0.9])
    with pytest.raises(ValueError, match="probabilities"):
        evaluate_binary([0, 1], [-1, 1])


def test_abstention_does_not_hide_missed_violence():
    result = evaluate_abstention([0, 0, 1, 1], [0.1, 0.5, 0.9, None], min_confidence=0.8)
    assert result["coverage"] == 0.5
    assert result["all_positive_recall"] == 0.5
    assert result["conditional_metrics"]["recall"] == 1
    assert result["confusion_matrix"] == [[1, 0, 1], [0, 1, 1]]


def test_temporal_persistence_hysteresis_gaps_and_causal_decision():
    intervals = [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6]]
    scores = [0.8, 0.6, 0.45, 0.2, 0.9]
    assert temporal_alerts(intervals, scores, on=0.7, off=0.4, min_duration=2) == [
        {"start": 0, "end": 3, "decision_time": 2}
    ]
    assert temporal_alerts([[0, 1], [2, 3]], [0.9, 0.9], min_duration=2) == []
    assert temporal_alerts([[0, 1], [1, 2]], [0.9, 0.1], alpha=0.1) == [
        {"start": 0, "end": 2, "decision_time": 1}
    ]
    with pytest.raises(ValueError, match="nonoverlapping"):
        temporal_alerts([[0, 2], [1, 3]], [0.9, 0.9])


def training_fixture(tmp_path):
    args = split_fixture(tmp_path)
    freeze_split(*args)
    rows, receipt = verify_split(args[-1])
    cache = tmp_path / "cache"
    cache.mkdir()
    entries = {}
    for i, row in enumerate(r for r in rows if r.split != "test"):
        path = cache / f"{i}.npy"
        np.save(path, np.full((2, 4), 1 if row.label else -1, dtype=np.float32))
        entries[row.video_id] = {"path": path.name, "sha256": sha256(path)}
    write_new_json(
        cache / "registration.json",
        {
            "schema_version": 1,
            "evidence_kind": "synthetic_software_fixture_only",
            "split_freeze_hash": receipt["freeze_hash"],
            "feature_dim": 4,
            "num_segments": 2,
            "features": entries,
        },
    )
    return args[-1], cache


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_smoke_save_resume_and_checkpoint_compatibility(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    split_dir, cache = training_fixture(tmp_path)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train_binary(split_dir, cache, full, steps=3, device=device)
    train_binary(split_dir, cache, resumed, steps=1, device=device)
    train_binary(split_dir, cache, resumed, steps=3, device=device, resume=resumed / "last.pt")
    _, a = load_binary_checkpoint(full / "last.pt")
    model, b = load_binary_checkpoint(resumed / "last.pt")
    assert a["step"] == b["step"] == 3
    assert a["history"] == b["history"]
    assert all(torch.equal(a["model_state"][key], value) for key, value in b["model_state"].items())
    assert model(torch.ones(1, 2, 4)).shape == (1, 2)
    with pytest.raises(ValueError, match="six-class"):
        load_behavior_checkpoint(full / "best.pt")
    with pytest.raises(FileExistsError):
        train_binary(split_dir, cache, full, steps=4, device=device)
    with pytest.raises(ValueError, match="changed"):
        train_binary(
            split_dir, cache, resumed, steps=4, device=device, seed=8, resume=resumed / "last.pt"
        )
    metadata = json.loads((resumed / "registration.json").read_text())
    assert metadata["test_used_for_selection"] is False
    assert metadata["feature_provenance"]["evidence_kind"] == "synthetic_software_fixture_only"
    if device == "cuda":
        assert metadata["environment"]["cuda_device"]


def test_training_rejects_mutated_features(tmp_path):
    split_dir, cache = training_fixture(tmp_path)
    (cache / "0.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        train_binary(split_dir, cache, tmp_path / "run", steps=1)


def test_training_rejects_incomplete_claimed_c3d_provenance(tmp_path):
    split_dir, cache = training_fixture(tmp_path)
    path = cache / "registration.json"
    registry = json.loads(path.read_text())
    registry["evidence_kind"] = "frozen_feature_cache_not_an_evaluation"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="C3D feature provenance"):
        train_binary(split_dir, cache, tmp_path / "run", steps=1)


def test_cache_excludes_protected_test_and_binds_extraction(tmp_path, monkeypatch):
    args = split_fixture(tmp_path)
    freeze_split(*args)
    rows, receipt = verify_split(args[-1])
    called = []

    class FixtureExtractor:
        def __init__(self, checkpoint, device, batch_size, mean, channel_order):
            assert tuple(mean) == (104, 117, 128) and channel_order == "rgb"

        def extract_video(self, path, segments):
            called.append(str(path))
            return np.ones((segments, 4096), dtype=np.float32)

    monkeypatch.setattr("surveillance.features.c3d.C3DExtractor", FixtureExtractor)
    checkpoint = tmp_path / "fixture-checkpoint.pt"
    checkpoint.write_bytes(b"synthetic extractor fixture; not real model weights")
    registry = cache_c3d(args[-1], checkpoint, tmp_path / "cache")
    expected = {row.video_id for row in rows if row.split != "test"}
    assert set(registry["features"]) == expected
    assert not set(called) & {row.path for row in rows if row.split == "test"}
    assert registry["split_freeze_hash"] == receipt["freeze_hash"]
    assert registry["checkpoint_sha256"] == sha256(checkpoint)
    with pytest.raises(FileExistsError):
        cache_c3d(args[-1], checkpoint, tmp_path / "cache")


def test_training_only_balancing_keeps_both_classes_in_the_pool():
    labels = [0] * 9 + [1]
    weights = binary_class_weights(labels)
    assert weights == pytest.approx([10 / 18, 5])
    indices = training_indices(labels, 1000, torch.Generator().manual_seed(7), "balanced")
    assert 0.4 < sum(labels[i] for i in indices) / len(indices) < 0.6
    assert indices == training_indices(labels, 1000, torch.Generator().manual_seed(7), "balanced")
    assert set(indices) == set(range(10))
    with pytest.raises(ValueError, match="Both training classes"):
        binary_class_weights([0, 0])


@pytest.mark.parametrize(
    "device,precision,sampling,weighting",
    [
        ("cpu", "fp32", "balanced", "none"),
        ("cpu", "fp32", "uniform", "balanced"),
        ("cuda", "amp_fp16", "balanced", "none"),
    ],
)
def test_imbalance_and_amp_accumulation_resume(tmp_path, device, precision, sampling, weighting):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    split, cache = training_fixture(tmp_path)
    settings = dict(
        device=device,
        precision=precision,
        sampling=sampling,
        class_weighting=weighting,
        accumulation_steps=2,
        batch_size=2,
    )
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train_binary(split, cache, full, steps=3, **settings)
    train_binary(split, cache, resumed, steps=1, **settings)
    train_binary(split, cache, resumed, steps=3, resume=resumed / "last.pt", **settings)
    _, a = load_binary_checkpoint(full / "last.pt")
    _, b = load_binary_checkpoint(resumed / "last.pt")
    assert a["history"] == b["history"]
    assert a["scaler_state"] == b["scaler_state"]
    assert all(torch.equal(a["model_state"][key], value) for key, value in b["model_state"].items())
    assert len(b["validation_predictions"]) == 2
    assert b["resources"]["loading_seconds"] > 0


def test_early_stopping_is_validation_based_and_resume_cannot_bypass_it(tmp_path, monkeypatch):
    split, cache = training_fixture(tmp_path)
    monkeypatch.setattr(
        "surveillance.training.indian_binary.evaluate_binary",
        lambda *args: {"pr_auc": 0.5, "f1": 0.5},
    )
    run = tmp_path / "stopped"
    train_binary(split, cache, run, steps=8, patience=2)
    _, saved = load_binary_checkpoint(run / "last.pt")
    assert saved["step"] == 3 and saved["early_stopped"]
    with pytest.raises(ValueError, match="Early-stopped"):
        train_binary(split, cache, run, steps=10, patience=2, resume=run / "last.pt")


def test_calibration_freezes_only_exact_validation_predictions(tmp_path):
    split, cache = training_fixture(tmp_path)
    run = tmp_path / "run"
    train_binary(split, cache, run, steps=2)
    result = calibrate_binary_checkpoint(
        split, run / "best.pt", tmp_path / "calibration", min_recall=1, candidates=[0, 0.5, 1]
    )
    rows, _ = verify_split(split)
    assert result["validation_clip_ids"] == sorted(r.video_id for r in rows if r.split == "val")
    assert not result["test_predictions_used"] and result["selected_metrics"]["recall"] == 1
    with pytest.raises(FileExistsError):
        calibrate_binary_checkpoint(split, run / "best.pt", tmp_path / "calibration")
    saved = torch.load(run / "best.pt", weights_only=True)
    saved["validation_predictions"][0]["clip_id"] = next(
        r.video_id for r in rows if r.split == "test"
    )
    torch.save(saved, run / "best.pt")  # Tamper only with this synthetic test's private checkpoint.
    with pytest.raises(ValueError, match="validation membership"):
        calibrate_binary_checkpoint(split, run / "best.pt", tmp_path / "bad-calibration")


def test_training_subset_and_relabel_are_bound_in_the_config(tmp_path):
    split_dir, cache = training_fixture(tmp_path)
    rows, _ = verify_split(split_dir)
    groups = sorted({r.source_video_id for r in rows if r.split == "train"})
    train_binary(split_dir, cache, tmp_path / "full", steps=2)
    train_binary(
        split_dir, cache, tmp_path / "half", steps=2, train_groups=groups[: len(groups) // 2 + 1]
    )
    full = json.loads((tmp_path / "full/registration.json").read_text())["settings"]
    half = json.loads((tmp_path / "half/registration.json").read_text())["settings"]
    assert half["training_clip_count"] < full["training_clip_count"]
    assert half["train_groups"] and full["train_groups"] is None
    with pytest.raises(ValueError, match="existing training groups"):
        train_binary(split_dir, cache, tmp_path / "bad", steps=1, train_groups=["no-such-group"])
    flipped = {r.video_id: 1 - r.label for r in rows if r.split != "test"}
    train_binary(
        split_dir, cache, tmp_path / "probe", steps=2, relabel=flipped, target_name="probe"
    )
    _, saved = load_binary_checkpoint(tmp_path / "probe/best.pt")
    assert all(p["label"] == flipped[p["clip_id"]] for p in saved["validation_predictions"])
    with pytest.raises(ValueError, match="Relabel"):
        train_binary(split_dir, cache, tmp_path / "bad2", steps=1, relabel={"x": 1})
