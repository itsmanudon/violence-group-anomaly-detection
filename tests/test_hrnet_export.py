"""Export contracts using a tiny external fixture, never pretrained HRNet weights."""

import copy
import json
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from test_collective_validation import manifest, row, sequence

from surveillance.features.actor_backbones import METADATA_FILENAME
from surveillance.features.hrnet_export import export_hrnet_features
from surveillance.features.hrnet_pose import HRNetPoseExtractor
from surveillance.features.provenance import file_sha256


@pytest.fixture
def official_layout(tmp_path):
    """Mimic upstream's factory/layout; this network is explicitly NOT HRNet."""
    repo = tmp_path / "external-fixture"
    source = repo / "lib/models/pose_hrnet.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """import torch
from torch import nn
class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 1)
        self.final_layer = nn.Conv2d(32, 17, 1)
    def forward(self, x):
        y = self.stem(torch.nn.functional.avg_pool2d(x, 4))
        return self.final_layer(y)
def get_pose_net(cfg, is_train, **kwargs):
    assert not is_train, "must not call upstream pretrained initialization"
    assert cfg["MODEL"]["NUM_JOINTS"] == 17
    return Fixture()
""",
        encoding="utf-8",
    )
    config = {
        "MODEL": {
            "NAME": "pose_hrnet",
            "NUM_JOINTS": 17,
            "IMAGE_SIZE": [192, 256],
            "HEATMAP_SIZE": [48, 64],
            "EXTRA": {"FINAL_CONV_KERNEL": 1},
        }
    }
    for stage, modules, channels in (
        (2, 1, [32, 64]),
        (3, 4, [32, 64, 128]),
        (4, 3, [32, 64, 128, 256]),
    ):
        config["MODEL"]["EXTRA"][f"STAGE{stage}"] = {
            "NUM_MODULES": modules,
            "NUM_BRANCHES": len(channels),
            "NUM_CHANNELS": channels,
            "NUM_BLOCKS": [4] * len(channels),
            "BLOCK": "BASIC",
            "FUSE_METHOD": "SUM",
        }
    config_path = repo / "experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    state = {
        "stem.weight": torch.full((32, 3, 1, 1), 0.01),
        "stem.bias": torch.zeros(32),
        "final_layer.weight": torch.full((17, 32, 1, 1), 0.02),
        "final_layer.bias": torch.zeros(17),
    }
    checkpoint = tmp_path / "pose_hrnet_w32_256x192.pth"
    torch.save(state, checkpoint)
    return repo, checkpoint, tmp_path / "features.pt", config_path, state


def test_export_endpoint_metadata_and_adapter(official_layout):
    repo, checkpoint, output, _, _ = official_layout
    report = export_hrnet_features(repo, checkpoint, output)
    extractor = HRNetPoseExtractor(output).eval()
    metadata = extractor.metadata
    assert metadata["endpoint"] == "pre_final_layer"
    assert metadata["exported_feature_dim"] == 98304
    assert metadata["feature_shape"] == [32, 64, 48]
    assert metadata["source_checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert metadata["source_repository"]["model_source_sha256"] == file_sha256(
        repo / "lib/models/pose_hrnet.py"
    )
    assert metadata["preprocessing"]["input_size"] == [256, 192]
    assert metadata["inference_only"] is True
    assert report["valid"] and report["adapter_accepted"] and report["repeatable"]
    assert report["archive_sha256"] == file_sha256(output)
    frames = torch.full((2, 3, 256, 192), 0.5)
    boxes = torch.tensor(
        [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 0, 0]]], dtype=torch.float32
    )
    mask = torch.tensor([[True, True], [True, False]])
    features = extractor(frames, boxes, mask)
    assert features.shape == (2, 2, 98304)
    assert torch.isfinite(features).all()
    assert torch.equal(features, extractor(frames, boxes, mask))
    assert features[1, 1].count_nonzero() == 0
    normalized = (torch.tensor(0.5) - extractor.mean) / extractor.std
    expected = normalized.sum() * 0.01
    torch.testing.assert_close(features[0, 0], expected.expand(98304), atol=1e-6, rtol=1e-5)


def test_export_serializes_full_float32_convolution_policy(official_layout):
    repo, checkpoint, output, _, _ = official_layout
    report = export_hrnet_features(repo, checkpoint, output)
    exported = torch.jit.load(str(output))
    convolutions = [
        node for node in exported.inlined_graph.nodes() if node.kind() == "aten::_convolution"
    ]
    assert convolutions, "Fixture must exercise serialized convolution policy"
    assert all(list(node.inputs())[12].toIValue() is False for node in convolutions)
    assert report["metadata"]["exporter"]["cudnn_allow_tf32"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_export_preserves_cudnn_execution_backend(official_layout, enabled):
    """Disabling TF32 must not disable cuDNN through context-manager defaults."""
    repo, checkpoint, output, _, _ = official_layout
    with torch.backends.cudnn.flags(enabled=enabled):
        export_hrnet_features(repo, checkpoint, output)
        assert torch.backends.cudnn.enabled is enabled
    exported = torch.jit.load(str(output))
    convolutions = [
        node for node in exported.inlined_graph.nodes() if node.kind() == "aten::_convolution"
    ]
    assert convolutions
    assert all(list(node.inputs())[11].toIValue() is enabled for node in convolutions)


@pytest.mark.parametrize(
    "size", [(1, 3, 192, 256), (1, 3, 255, 192), (1, 1, 256, 192), (0, 3, 256, 192)]
)
def test_export_enforces_input_resolution(official_layout, size):
    repo, checkpoint, output, _, _ = official_layout
    export_hrnet_features(repo, checkpoint, output)
    model = torch.jit.load(str(output))
    with pytest.raises((RuntimeError, torch.jit.Error), match="HRNet"):
        model(torch.zeros(size))


def test_export_rejects_unfreezing_eval_trace(official_layout):
    repo, checkpoint, output, _, _ = official_layout
    export_hrnet_features(repo, checkpoint, output)
    with pytest.raises(ValueError, match="inference-only"):
        HRNetPoseExtractor(output, frozen=False)


@pytest.mark.parametrize("wrapper", ["bare", "state_dict", "module"])
def test_checkpoint_formats(official_layout, wrapper):
    repo, checkpoint, output, _, state = official_layout
    if wrapper == "module":
        state = {f"module.{key}": value for key, value in state.items()}
    torch.save({"state_dict": state} if wrapper == "state_dict" else state, checkpoint)
    assert export_hrnet_features(repo, checkpoint, output)["valid"]


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "nonfinite", "prefix"])
def test_incompatible_checkpoints_fail_without_archive(official_layout, corruption):
    repo, checkpoint, output, _, state = official_layout
    if corruption == "missing":
        del state["final_layer.weight"]
    elif corruption == "extra":
        state["unrelated"] = torch.ones(1)
    elif corruption == "shape":
        state["final_layer.weight"] = torch.zeros(16, 32, 1, 1)
    elif corruption == "nonfinite":
        state["stem.weight"][0, 0, 0, 0] = float("nan")
    else:
        state["module.stem.weight"] = state.pop("stem.weight")
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="checkpoint"):
        export_hrnet_features(repo, checkpoint, output)
    assert not output.exists()


@pytest.mark.parametrize("asset", ["repository", "checkpoint", "config", "model"])
def test_missing_external_assets(official_layout, asset):
    repo, checkpoint, output, config, _ = official_layout
    if asset == "repository":
        repo = repo / "absent"
    elif asset == "checkpoint":
        checkpoint = checkpoint.with_name("absent.pth")
    elif asset == "config":
        config.unlink()
    else:
        (repo / "lib/models/pose_hrnet.py").unlink()
    with pytest.raises(FileNotFoundError, match="HRNet"):
        export_hrnet_features(repo, checkpoint, output)


@pytest.mark.parametrize("property", ["IMAGE_SIZE", "HEATMAP_SIZE", "NUM_JOINTS", "stage"])
def test_wrong_architecture_config(official_layout, property):
    repo, checkpoint, output, config_path, _ = official_layout
    config = yaml.safe_load(config_path.read_text())
    if property == "stage":
        config["MODEL"]["EXTRA"]["STAGE4"]["NUM_CHANNELS"] = [48, 96, 192, 384]
    else:
        config["MODEL"][property] = 1
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="configuration"):
        export_hrnet_features(repo, checkpoint, output)


def test_checkpoint_fingerprint_and_no_overwrite(official_layout):
    repo, checkpoint, output, _, _ = official_layout
    with pytest.raises(ValueError, match="SHA256"):
        export_hrnet_features(repo, checkpoint, output, expected_checkpoint_sha256="0" * 64)
    assert not output.exists()
    export_hrnet_features(
        repo, checkpoint, output, expected_checkpoint_sha256=file_sha256(checkpoint)
    )
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_hrnet_features(repo, checkpoint, output)
    assert output.read_bytes() == before


def test_embedded_archive_contract(official_layout):
    repo, checkpoint, output, _, _ = official_layout
    export_hrnet_features(repo, checkpoint, output)
    extra = {METADATA_FILENAME: ""}
    loaded = torch.jit.load(str(output), _extra_files=extra)
    exporter = json.loads(extra[METADATA_FILENAME])["exporter"]
    assert exporter["version"] == 2
    assert exporter["cudnn_allow_tf32"] is False
    for count in (1, 2, 3):
        first = loaded(torch.zeros(count, 3, 256, 192))
        assert first.shape == (count, 32, 64, 48)
        assert torch.equal(first, loaded(torch.zeros(count, 3, 256, 192)))


def test_export_cli_then_inspection_and_strict_cache(official_layout):
    """Exercise the actual export/extraction CLIs on GT actors in synthetic images."""
    from surveillance.actor_config import DEFAULT_CONFIG
    from surveillance.features.provenance import validate_feature_caches

    repo, checkpoint, output, _, _ = official_layout
    process = subprocess.run(
        [
            sys.executable,
            "scripts/export_hrnet_features.py",
            "--hrnet-repo",
            str(repo),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(output.with_suffix(".json").read_text())["valid"]
    root = output.parent
    sequence(root)
    record = replace(
        row(root),
        actor_boxes=[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]],
        actor_labels=[0, 1],
    )
    source = manifest(root / "actors.jsonl", [record])
    inspection = runpy.run_path("scripts/inspect_actor_features.py")["inspect_features"]
    report = inspection(
        source, output, "pose", max_scenes=1, image_size=(12, 20), debug_dir=root / "debug"
    )
    assert report["valid"] and report["non_benchmark"]
    scene = report["scenes"][0]
    assert scene["actor_count"] == 2 and scene["crop_shape"] == [3, 256, 192]
    assert scene["actor_labels"] == [0, 1] and scene["actor_boxes"] == record.actor_boxes
    assert (root / "debug" / scene["debug_image"]).is_file()
    feature_manifest = root / "features.jsonl"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/extract_pose_features.py",
            "--manifest",
            str(source),
            "--checkpoint",
            str(output),
            "--output-manifest",
            str(feature_manifest),
            "--feature-dir",
            str(root / "cache"),
            "--image-size",
            "12",
            "20",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    config = {**DEFAULT_CONFIG, "data": {**DEFAULT_CONFIG["data"], "image_size": [12, 20]}}
    expected = {
        "pose": {
            "architecture": "pose_hrnet_w32",
            "endpoint": "pre_final_layer",
            "checkpoint_sha256": file_sha256(output),
        }
    }
    assert validate_feature_caches(feature_manifest, config, expected)["valid"]
    expected["pose"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        validate_feature_caches(feature_manifest, config, expected)


def test_external_sources_and_cpu_rng_are_unchanged(official_layout):
    repo, checkpoint, output, config, _ = official_layout
    before = {
        path: file_sha256(path) for path in (repo / "lib/models/pose_hrnet.py", config, checkpoint)
    }
    random_state = torch.random.get_rng_state().clone()
    export_hrnet_features(repo, checkpoint, output)
    assert torch.equal(random_state, torch.random.get_rng_state())
    assert before == {path: file_sha256(path) for path in before}
    assert not list(repo.rglob("__pycache__"))


def test_malformed_tensor_checkpoint_is_actionable(official_layout):
    repo, checkpoint, output, _, _ = official_layout
    checkpoint.write_text("not a tensor state dictionary")
    with pytest.raises(ValueError, match="Cannot read HRNet checkpoint"):
        export_hrnet_features(repo, checkpoint, output)
    assert not output.exists()


def test_cli_missing_assets_is_actionable(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            "scripts/export_hrnet_features.py",
            "--hrnet-repo",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path / "absent.pth"),
            "--output",
            str(tmp_path / "absent.pt"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1
    assert "docs/hrnet-export.md" in process.stdout
    assert "download" in process.stdout and "Traceback" not in process.stderr


def test_exported_pose_features_train_select_reload_and_evaluate(official_layout):
    """Five optimization steps on fixture images; never a real-data benchmark."""
    from surveillance.datasets.collective import read_actor_manifest
    from surveillance.experiments.preparation import extract_features, preflight_experiment
    from surveillance.experiments.protocol import load_protocol
    from surveillance.experiments.runner import freeze_experiment, run_experiment
    from surveillance.training.actor_transformer_trainer import load_checkpoint

    repo, checkpoint, output, _, _ = official_layout
    export_hrnet_features(repo, checkpoint, output)
    root = output.parent
    rows = []
    for sid, split, count in ((4, "train", 1), (12, "train", 2), (1, "val", 2), (5, "test", 1)):
        sequence(root, sid=sid)
        rows.append(
            replace(
                row(root, sid=sid, split=split),
                actor_boxes=[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]][:count],
                actor_labels=[0, 1][:count],
                group_label=0,
            )
        )
    authority = manifest(root / "authority.jsonl", rows)
    protocol = copy.deepcopy(
        yaml.safe_load(Path("configs/experiments/collective_protocol.yaml").read_text())
    )
    protocol.update(
        evidence_kind="synthetic",
        protocol_id="hrnet_export_fixture_only",
        seeds=[0],
        output_root=str(root / "runs"),
    )
    protocol["dataset"].update(root=str(root), manifest=str(authority))
    protocol["features"]["pose"].update(
        checkpoint=str(output), checkpoint_sha256=file_sha256(output)
    )
    protocol["actor"]["device"] = "cpu"
    protocol["actor"]["data"]["image_size"] = [12, 20]
    protocol["actor"]["model"]["embedding_dim"] = 8
    protocol["actor"]["model"]["transformer"].update(feedforward_dim=16, reference_size=[12, 20])
    protocol["actor"]["training"].update(
        max_iterations=5, batch_size=2, validation_interval=1, checkpoint_interval=5
    )
    protocol["experiments"] = {
        "pose_gt": {
            "mode": "pose_only",
            "box_source": "ground_truth",
            "manifest": str(root / "full_features.jsonl"),
        }
    }
    protocol_path = root / "fixture_protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    protocol = load_protocol(protocol_path)
    preflight = preflight_experiment(protocol, "pose_gt", 0, max_scenes=2, max_iterations=5)
    assert preflight["dry_run"] and preflight["population"]["split"] == "val"
    extract_features(protocol, "pose_gt")
    assert {
        len(record.actor_boxes) for record in read_actor_manifest(root / "full_features.jsonl")
    } == {1, 2}
    receipt = freeze_experiment(protocol, "pose_gt")
    assert receipt["protocol_hash"]
    result = run_experiment(protocol, "pose_gt", 0)
    assert result["result_scope"].startswith("NON-BENCHMARK")
    assert result["scene_count"] == 1 and result["gt_actor_count"] == 1
    assert result["population"]["split"] == "test"
    _, saved = load_checkpoint(root / "runs/pose_gt/seed_0/last.pt", "cpu")
    assert saved["iteration"] == 5
    assert (root / "runs/pose_gt/seed_0/errors.json").is_file()
    assert (root / "runs/pose_gt/seed_0/predictions.json").is_file()
    with pytest.raises(FileExistsError):
        run_experiment(protocol, "pose_gt", 0)
