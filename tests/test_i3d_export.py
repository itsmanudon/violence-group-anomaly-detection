"""I3D export-boundary tests using a tiny external fixture, never downloaded weights."""

import importlib
import importlib.util
import json
import math
import subprocess
import sys

import pytest
import torch

from surveillance.features.actor_backbones import METADATA_FILENAME
from surveillance.features.i3d import I3DActorExtractor
from surveillance.features.provenance import file_sha256
from surveillance.models.actor_transformer.actor_transformer import ActorTransformer


@pytest.fixture
def upstream(tmp_path):
    """Match upstream's factory/endpoint registration; this is NOT an I3D network."""
    repo = tmp_path / "upstream-fixture"
    repo.mkdir()
    source = repo / "pytorch_i3d.py"
    source.write_text(
        """import torch
from torch import nn
class InceptionI3d(nn.Module):
    VALID_ENDPOINTS = ("Conv3d_1a_7x7", "Mixed_4f", "Mixed_5c", "Logits")
    def __init__(self, num_classes=400, in_channels=3, final_endpoint="Logits"):
        super().__init__()
        assert num_classes == 400 and in_channels == 3
        self._num_classes = num_classes
        self.end_points = {
            "Conv3d_1a_7x7": nn.Sequential(
                nn.AvgPool3d((1,16,16), (1,16,16)),
                nn.Conv3d(3,832,1,bias=False), nn.BatchNorm3d(832,eps=.001)),
            "Mixed_4f": nn.AvgPool3d((4,1,1),(4,1,1),ceil_mode=True),
            "Mixed_5c": nn.Conv3d(832,1024,1),
        }
        # Reproduce upstream's intermediate-endpoint early return before registration.
        if final_endpoint != "Logits":
            return
        self.logits = nn.Conv3d(1024,400,1)
        for name, block in self.end_points.items():
            self.add_module(name,block)
    def forward(self,x):
        for block in self.end_points.values():
            x = block(x)
        return self.logits(x)
""",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("fixture_i3d", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.InceptionI3d()
    state = model.state_dict()
    state["Conv3d_1a_7x7.1.weight"].fill_(0.125)
    checkpoint = tmp_path / "rgb.pt"
    torch.save(state, checkpoint)
    return repo, checkpoint, tmp_path / "features.pt", state


def export(upstream, **kwargs):
    assert importlib.util.find_spec("surveillance.features.i3d_export") is not None, (
        "The upstream I3D export boundary has not been implemented"
    )
    function = importlib.import_module("surveillance.features.i3d_export").export_i3d_features
    repo, checkpoint, output, _ = upstream
    return function(
        repo,
        checkpoint,
        output,
        input_size=(32, 48),
        source_revision="0" * 40,
        expected_source_sha256=file_sha256(repo / "pytorch_i3d.py"),
        **kwargs,
    )


def test_export_returns_native_mixed4f_and_embeds_provenance(upstream):
    repo, checkpoint, output, _ = upstream
    report = export(upstream)
    metadata = I3DActorExtractor(output).metadata
    assert metadata["source_repository"]["revision"] == "0" * 40
    assert metadata["source_repository"]["model_source_sha256"] == file_sha256(
        repo / "pytorch_i3d.py"
    )
    assert metadata["source_checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert metadata["endpoint"] == "Mixed_4f"
    assert metadata["input_frames"] == 10
    assert metadata["preprocessing"]["mean"] == [0.5, 0.5, 0.5]
    assert metadata["preprocessing"]["std"] == [0.5, 0.5, 0.5]
    assert metadata["inference_only"] is True
    assert report["native_endpoint_verified"] and report["adapter_accepted"]
    assert report["archive_sha256"] == file_sha256(output)
    extras = {METADATA_FILENAME: ""}
    backbone = torch.jit.load(str(output), _extra_files=extras).eval()
    assert json.loads(extras[METADATA_FILENAME]) == metadata
    for count in (1, 2):
        inputs = torch.zeros(count, 3, 10, 32, 48)
        inputs[:, 0] = 1
        result = backbone(inputs)
        assert result.shape == (count, 832, 3, 2, 3)
        # Only the red channel is one: sum(channel inputs)*.125, then native BN.
        torch.testing.assert_close(result, torch.full_like(result, 0.125 / math.sqrt(1.001)))


def test_export_adapter_rgb_normalization_roi_padding_and_actor_transformer(upstream):
    export(upstream)
    adapter = I3DActorExtractor(upstream[2]).eval()
    frames = torch.zeros(2, 10, 3, 32, 48)
    frames[:, :, 0] = 1  # RGB [1,0,0] -> normalized [1,-1,-1].
    boxes = torch.tensor(
        [
            [[0.2, 0.2, 0.8, 0.8], [0.3, 0.3, 0.7, 0.7]],
            [[0.1, 0.1, 0.9, 0.9], [0, 0, 0, 0]],
        ]
    )
    mask = torch.tensor([[True, True], [True, False]])
    features = adapter(frames, boxes, mask)
    assert features.shape == (2, 2, 20800)
    torch.testing.assert_close(
        features[mask], torch.full_like(features[mask], -0.125 / math.sqrt(1.001))
    )
    assert features[1, 1].count_nonzero() == 0
    assert torch.equal(features, adapter(frames, boxes, mask))
    model = ActorTransformer(mode="rgb_only").eval()
    result = model(rgb_features=features, actor_boxes=boxes, actor_valid_mask=mask)
    assert result["group_logits"].shape == (2, 5)
    assert result["actor_logits"].shape == (2, 2, 5)
    assert torch.isfinite(result["group_logits"]).all()


@pytest.mark.parametrize(
    "size",
    [
        (1, 3, 9, 32, 48),
        (1, 10, 3, 32, 48),
        (1, 1, 10, 32, 48),
        (1, 3, 10, 31, 48),
        (1, 3, 10, 32, 49),
        (0, 3, 10, 32, 48),
    ],
)
def test_archive_rejects_wrong_temporal_channel_spatial_or_empty_input(upstream, size):
    export(upstream)
    model = torch.jit.load(str(upstream[2]))
    with pytest.raises((RuntimeError, torch.jit.Error), match="I3D"):
        model(torch.zeros(size))


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_endpoint",
        "missing_classifier",
        "extra",
        "shape",
        "dtype",
        "nonfinite",
    ],
)
def test_strict_checkpoint_rejects_corruption_without_creating_archive(upstream, corruption):
    _, checkpoint, output, state = upstream
    key = "Conv3d_1a_7x7.1.weight"
    if corruption == "missing_endpoint":
        del state[key]
    elif corruption == "missing_classifier":
        del state["logits.weight"]
    elif corruption == "extra":
        state["unexpected.weight"] = torch.zeros(1)
    elif corruption == "shape":
        state[key] = torch.zeros(1)
    elif corruption == "dtype":
        state[key] = state[key].double()
    else:
        state[key][0, 0, 0, 0, 0] = float("nan")
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="checkpoint"):
        export(upstream)
    assert not output.exists()


def test_legacy_bn_counter_is_explicitly_initialized_and_reported(upstream):
    _, checkpoint, _, state = upstream
    del state["Conv3d_1a_7x7.2.num_batches_tracked"]
    torch.save(state, checkpoint)
    report = export(upstream)
    assert report["metadata"]["source_checkpoint"]["legacy_bn_counters_initialized"] == [
        "Conv3d_1a_7x7.2.num_batches_tracked"
    ]


def test_export_refuses_to_overwrite_existing_archive(upstream):
    upstream[2].write_bytes(b"existing-user-artifact")
    with pytest.raises(FileExistsError):
        export(upstream)
    assert upstream[2].read_bytes() == b"existing-user-artifact"


def test_export_rejects_stale_checkpoint_fingerprint(upstream):
    with pytest.raises(ValueError, match="SHA256"):
        export(upstream, expected_checkpoint_sha256="0" * 64)
    assert not upstream[2].exists()


def test_inference_only_archive_cannot_be_unfrozen(upstream):
    export(upstream)
    with pytest.raises(ValueError, match="inference-only"):
        I3DActorExtractor(upstream[2], frozen=False)


@pytest.mark.parametrize("enabled", [False, True])
def test_full_precision_trace_preserves_cudnn_backend_and_caller_flags(upstream, enabled):
    with torch.backends.cudnn.flags(enabled=enabled, allow_tf32=True):
        export(upstream)
        assert torch.backends.cudnn.enabled is enabled
        assert torch.backends.cudnn.allow_tf32 is True
    backbone = torch.jit.load(str(upstream[2]))
    convolutions = [n for n in backbone.inlined_graph.nodes() if n.kind() == "aten::_convolution"]
    assert convolutions
    assert all(list(n.inputs())[11].toIValue() is enabled for n in convolutions)
    assert all(list(n.inputs())[12].toIValue() is False for n in convolutions)


def test_export_cli_creates_archive_and_structured_report(upstream):
    repo, checkpoint, output, _ = upstream
    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_i3d_features.py",
            "--i3d-repo",
            str(repo),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--source-revision",
            "0" * 40,
            "--source-sha256",
            file_sha256(repo / "pytorch_i3d.py"),
            "--input-size",
            "32",
            "48",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.exists()
    assert json.loads(output.with_suffix(".json").read_text())["adapter_accepted"]
