"""Frozen protocols guard splits, configuration, artifact identity and overrides."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from surveillance.experiments.protocol import (
    actor_config,
    content_hash,
    load_protocol,
    verify_receipt,
)
from surveillance.experiments.runner import freeze_experiment, receipt_path, run_experiment
from surveillance.experiments.synthetic import synthetic_protocol


@pytest.fixture
def protocol(tmp_path):
    return synthetic_protocol(tmp_path / "fixture")


def reload_changed(protocol, tmp_path, change):
    changed = copy.deepcopy(protocol)
    change(changed)
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    return load_protocol(path)


def test_default_real_protocol_is_explicit():
    protocol = load_protocol(Path("configs/experiments/collective_protocol.yaml"))
    assert protocol["seeds"] == [0, 1, 2]
    assert len(protocol["dataset"]["train_sequences"]) == 32
    assert len(protocol["dataset"]["test_sequences"]) == 12
    assert protocol["dataset"]["validation_sequences"] == [1, 2, 3]
    assert len(protocol["experiments"]) == 6
    assert protocol["features"]["pose"]["checkpoint"] is None


def test_resolution_and_independent_seed_configs(protocol):
    first = actor_config(protocol, "pose_gt", 0)
    second = actor_config(protocol, "pose_gt", 1)
    assert first["seed"] == 0 and second["seed"] == 1
    first["training"]["learning_rate"] = 999
    assert second["training"]["learning_rate"] == 0.0001
    assert Path(protocol["output_root"]).is_absolute()
    with pytest.raises(ValueError, match="declared"):
        actor_config(protocol, "pose_gt", 10)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda p: p["dataset"].update(validation_sequences=[5]), "training-only"),
        (lambda p: p["dataset"].update(test_sequences=[1]), "32/12"),
        (lambda p: p.update(seeds=[0, 0]), "unique"),
        (lambda p: p.update(unknown_setting=True), "exactly"),
        (lambda p: p["metrics"].update(checkpoint_selection="test_accuracy"), "Metric"),
        (lambda p: p["experiments"]["detected_pose"].update(mode="rgb_only"), "reuse"),
        (lambda p: p.update(evidence_kind="real"), "feature dimensions"),
    ],
)
def test_protocol_rejects_unsafe_or_ambiguous_settings(protocol, tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        reload_changed(protocol, tmp_path, change)


def test_freeze_is_idempotent_and_binds_feature_bytes(protocol):
    receipt = freeze_experiment(protocol, "pose_gt")
    assert freeze_experiment(protocol, "pose_gt") == receipt
    artifact = next(Path(path) for path in receipt["artifacts"] if path.endswith(".npy"))
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_receipt(protocol, "pose_gt", receipt_path(protocol, "pose_gt"))


def test_frozen_protocol_and_receipt_cannot_be_changed(protocol):
    freeze_experiment(protocol, "pose_gt")
    changed = copy.deepcopy(protocol)
    changed["actor"]["training"]["learning_rate"] = 0.01
    with pytest.raises(ValueError, match="mismatch"):
        verify_receipt(changed, "pose_gt", receipt_path(protocol, "pose_gt"))
    path = receipt_path(protocol, "pose_gt")
    receipt = json.loads(path.read_text())
    receipt["experiment"] = "detected_pose"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="mismatch"):
        verify_receipt(protocol, "pose_gt", path)


def test_unfrozen_test_evaluation_fails_before_run_creation(protocol):
    with pytest.raises(FileNotFoundError):
        run_experiment(protocol, "pose_gt", 0)
    assert not (Path(protocol["output_root"]) / "pose_gt").exists()


def test_bounds_require_non_benchmark_dry_run(protocol):
    with pytest.raises(ValueError, match="dry-run"):
        run_experiment(protocol, "pose_gt", 0, max_iterations=1)
    with pytest.raises(ValueError, match="positive"):
        run_experiment(protocol, "pose_gt", 0, dry_run=True)


def test_dry_run_is_validation_only_and_does_not_need_test_receipt(protocol):
    result = run_experiment(protocol, "pose_gt", 0, dry_run=True, max_scenes=1, max_iterations=1)
    assert result["dry_run"] and result["population"]["split"] == "val"
    assert result["population"]["scene_ids"] == ["seq01:0001"]
    assert result["result_scope"] == "NON-BENCHMARK PREFLIGHT"
    output = Path(protocol["output_root"]) / "preflight" / "pose_gt" / "seed_0"
    assert (output / "environment.json").exists()
    assert (output / "resolved_config.yaml").exists()
    assert (output / "history.jsonl").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        run_experiment(protocol, "pose_gt", 0, dry_run=True, max_scenes=1, max_iterations=1)


def test_hash_rejects_nonfinite_protocol_values():
    with pytest.raises(ValueError):
        content_hash({"bad": float("nan")})


def test_bounded_raw_feature_preflight_uses_no_test_scenes(protocol, tmp_path):
    from test_actor_backbones import archive

    from surveillance.experiments.preparation import preflight_experiment
    from surveillance.experiments.protocol import file_hash

    checkpoint = archive(tmp_path)
    protocol["features"]["pose"].update(
        checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint)
    )
    protocol["actor"]["model"]["pose_feature_dim"] = 98304
    result = preflight_experiment(protocol, "pose_gt", 0, max_scenes=1, max_iterations=1)
    assert result["population"]["scene_ids"] == ["seq01:0001"]
    assert result["dry_run"] and result["evidence_kind"] == "synthetic"
    destination = Path(protocol["output_root"]) / "preflight" / "artifacts" / "pose_gt"
    from surveillance.datasets.collective import read_actor_manifest

    rows = read_actor_manifest(destination / "features.jsonl")
    assert [row.split for row in rows] == ["train", "val"]
    assert len(rows) == 2
