"""CPU adapter tests; synthetic archives here are NOT pretrained backbones."""

import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F


class SyntheticPose(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(1) == 3 and x.size(2) == 256 and x.size(3) == 192
        return (
            F.interpolate(x.mean(1, keepdim=True), size=(64, 48)).repeat(1, self.channels, 1, 1)
            * self.scale
        )


class SyntheticRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5 and x.size(1) == 3 and x.size(2) == 10
        assert x.size(3) == 8 and x.size(4) == 12
        return x.mean(1, keepdim=True).repeat(1, 832, 1, 1, 1) * self.scale


def archive(tmp_path, kind="pose", overrides=None, channels=32):
    metadata = {
        "schema_version": 1,
        "architecture": "pose_hrnet_w32" if kind == "pose" else "i3d",
        "endpoint": "pre_final_layer" if kind == "pose" else "Mixed_4f",
        "output_channels": 32 if kind == "pose" else 832,
        "feature_only": True,
        "preprocessing": {
            "color_order": "rgb",
            "input_range": [0, 1],
            "mean": [0, 0, 0],
            "std": [1, 1, 1],
            "input_size": [256, 192] if kind == "pose" else [8, 12],
        },
        "provenance": "synthetic test fixture; not a pretrained backbone",
    }
    if kind == "rgb":
        metadata["input_frames"] = 10
    metadata.update(overrides or {})
    path = tmp_path / f"{kind}.pt"
    module = SyntheticPose(channels) if kind == "pose" else SyntheticRGB()
    torch.jit.save(
        torch.jit.script(module),
        str(path),
        _extra_files={
            "actor_backbone.json": json.dumps(metadata),
        },
    )
    return path


def test_geometry_clipping_and_anisotropic_scaling():
    from surveillance.features.geometry import (
        clip_boxes,
        normalized_boxes_to_pixels,
        scale_boxes,
    )

    source = torch.tensor([[-10.0, 20.0, 300.0, 120.0]])
    torch.testing.assert_close(
        clip_boxes(source, 200, 100), torch.tensor([[0.0, 20.0, 200.0, 100.0]])
    )
    pixels = normalized_boxes_to_pixels(torch.tensor([[0.25, 0.1, 0.75, 0.9]]), 200, 100)
    torch.testing.assert_close(pixels, torch.tensor([[50.0, 10.0, 150.0, 90.0]]))
    scaled = scale_boxes(pixels, source_size=(100, 200), target_size=(90, 160))
    torch.testing.assert_close(scaled, torch.tensor([[40.0, 9.0, 120.0, 81.0]]))
    assert clip_boxes(torch.empty(0, 4), 200, 100).shape == (0, 4)


@pytest.mark.parametrize(
    "box", [[2, 1, 1, 2], [0, 0, 0, 1], [-5, 1, -1, 2], [0, 0, float("nan"), 2]]
)
def test_geometry_rejects_invalid_or_fully_clipped_boxes(box):
    from surveillance.features.geometry import clip_boxes

    with pytest.raises(ValueError):
        clip_boxes(torch.tensor([box], dtype=torch.float32), 100, 100)


def test_roi_coordinate_ramp_and_gradients():
    from surveillance.features.geometry import roi_align_actors

    # Pixel-center ramp f(x,y)=x+10*y; aligned ROI [2,2,6,6] center is (3.5,3.5).
    y, x = torch.meshgrid(torch.arange(8.0), torch.arange(8.0), indexing="ij")
    features = (x + 10 * y)[None, None].requires_grad_()
    boxes = torch.tensor([[[0.25, 0.25, 0.75, 0.75], [float("nan")] * 4]])
    output = roi_align_actors(features, boxes, torch.tensor([[True, False]]), (1, 1))
    torch.testing.assert_close(output, torch.tensor([[[[38.5]]]]))
    output.sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["pose", "rgb"])
def test_backbone_padding_shapes_frozen_and_unfrozen_gradients(tmp_path, kind):
    from surveillance.features.hrnet_pose import HRNetPoseExtractor
    from surveillance.features.i3d import I3DActorExtractor

    cls = HRNetPoseExtractor if kind == "pose" else I3DActorExtractor
    path = archive(tmp_path, kind)
    frozen = cls(path)
    frozen.train()
    assert not frozen.backbone.training
    assert all(not p.requires_grad for p in frozen.backbone.parameters())
    boxes = torch.tensor([[[0, 0, 1, 1], [float("nan")] * 4]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    shape = (1, 3, 12, 20) if kind == "pose" else (1, 10, 3, 12, 20)
    frames = torch.full(shape, 0.25, requires_grad=True)
    output = frozen(frames, boxes, mask)
    assert output.shape == (1, 2, 98304 if kind == "pose" else 20800)
    torch.testing.assert_close(output[:, 0], torch.full_like(output[:, 0], 0.5))
    assert torch.count_nonzero(output[:, 1]) == 0
    assert not output.requires_grad
    trainable = cls(path, frozen=False)
    trainable.train()
    assert trainable.backbone.training
    trainable(frames, boxes, mask).sum().backward()
    assert frames.grad is not None and frames.grad.abs().sum() > 0
    assert trainable.backbone.scale.grad is not None


@pytest.mark.parametrize("kind", ["pose", "rgb"])
def test_missing_checkpoint_has_actionable_error(kind):
    from surveillance.features.hrnet_pose import HRNetPoseExtractor
    from surveillance.features.i3d import I3DActorExtractor

    cls = HRNetPoseExtractor if kind == "pose" else I3DActorExtractor
    with pytest.raises(FileNotFoundError, match="local.*TorchScript"):
        cls(None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"architecture": "resnet"},
        {"endpoint": "heatmaps"},
        {"output_channels": 17},
        {"schema_version": 2},
        {"feature_only": False},
        {"provenance": ""},
        {"preprocessing": {}},
    ],
)
def test_checkpoint_contract_is_validated(tmp_path, overrides):
    from surveillance.features.hrnet_pose import HRNetPoseExtractor

    with pytest.raises(ValueError, match="checkpoint"):
        HRNetPoseExtractor(archive(tmp_path, overrides=overrides))


def test_checkpoint_wrong_runtime_shape_rejected(tmp_path):
    from surveillance.features.hrnet_pose import HRNetPoseExtractor

    model = HRNetPoseExtractor(archive(tmp_path, channels=17))
    with pytest.raises(ValueError, match="32,64,48"):
        model(
            torch.rand(1, 3, 8, 8),
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
            torch.ones(1, 1, dtype=torch.bool),
        )


def test_geometry_and_input_validation_at_model_boundary(tmp_path):
    from surveillance.features.hrnet_pose import HRNetPoseExtractor

    model = HRNetPoseExtractor(archive(tmp_path))
    frames = torch.rand(1, 3, 8, 8)
    mask = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="outside"):
        model(frames, torch.tensor([[[-0.1, 0.0, 1.0, 1.0]]]), mask)
    clipped = HRNetPoseExtractor(archive(tmp_path), clip_outside=True)
    assert clipped(frames, torch.tensor([[[-0.1, 0.0, 1.0, 1.0]]]), mask).isfinite().all()
    with pytest.raises(ValueError, match="RGB"):
        model(frames + 2, torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]), mask)


def test_preprocessing_uses_export_declared_mean_std(tmp_path):
    from surveillance.features.hrnet_pose import HRNetPoseExtractor

    preprocessing = {
        "color_order": "rgb",
        "input_range": [0, 1],
        "mean": [0.1, 0.1, 0.1],
        "std": [0.2, 0.2, 0.2],
        "input_size": [256, 192],
    }
    model = HRNetPoseExtractor(archive(tmp_path, overrides={"preprocessing": preprocessing}))
    result = model(
        torch.full((1, 3, 8, 8), 0.3),
        torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
        torch.ones(1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(result, torch.full_like(result, 2.0))


def test_empty_geometry_and_padded_batch_return_empty_or_zeros(tmp_path):
    from surveillance.features.geometry import roi_align_actors
    from surveillance.features.hrnet_pose import HRNetPoseExtractor

    frames = torch.rand(2, 3, 8, 8)
    boxes = torch.full((2, 2, 4), float("nan"))
    valid = torch.zeros(2, 2, dtype=torch.bool)
    assert roi_align_actors(frames, boxes, valid, (5, 5)).shape == (0, 3, 5, 5)
    assert torch.count_nonzero(HRNetPoseExtractor(archive(tmp_path))(frames, boxes, valid)) == 0


def test_i3d_temporal_mean_and_multi_scene_padding(tmp_path):
    from surveillance.features.i3d import I3DActorExtractor

    model = I3DActorExtractor(archive(tmp_path, "rgb"))
    frames = torch.zeros(2, 10, 3, 8, 12)
    frames[0, 0] = 1.0
    frames[1] = 0.75
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]] * 3] * 2)
    valid = torch.tensor([[True, False, True], [False, True, False]])
    output = model(frames, boxes, valid)
    torch.testing.assert_close(output[0, 0], torch.full((20800,), 0.2))
    torch.testing.assert_close(output[0, 2], torch.full((20800,), 0.2))
    torch.testing.assert_close(output[1, 1], torch.full((20800,), 1.5))
    assert torch.count_nonzero(output[~valid]) == 0
    with pytest.raises(ValueError, match="10"):
        model(frames[:, :9], boxes, valid)


def test_checkpoint_missing_embedded_metadata_and_non_archive_fail(tmp_path):
    from surveillance.features.hrnet_pose import HRNetPoseExtractor

    path = tmp_path / "missing.pt"
    torch.jit.save(torch.jit.script(SyntheticPose()), str(path))
    with pytest.raises(ValueError, match="actor_backbone.json"):
        HRNetPoseExtractor(path)
    path.write_text("not an archive")
    with pytest.raises(ValueError, match="TorchScript"):
        HRNetPoseExtractor(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mean", [0, 0]),
        ("std", [1, 0, 1]),
        ("std", [1, float("nan"), 1]),
        ("color_order", "bgr"),
        ("input_range", [-1, 1]),
        ("input_size", [64, 48]),
    ],
)
def test_invalid_preprocessing_fails_before_forward(tmp_path, field, value):
    from surveillance.features.hrnet_pose import HRNetPoseExtractor

    preprocessing = {
        "color_order": "rgb",
        "input_range": [0, 1],
        "mean": [0, 0, 0],
        "std": [1, 1, 1],
        "input_size": [256, 192],
    }
    preprocessing[field] = value
    with pytest.raises(ValueError, match="checkpoint"):
        HRNetPoseExtractor(archive(tmp_path, overrides={"preprocessing": preprocessing}))


def test_finetuning_rejects_export_without_parameters(tmp_path):
    from surveillance.features.hrnet_pose import HRNetPoseExtractor

    path = archive(tmp_path)
    extra = {"actor_backbone.json": ""}
    module = torch.jit.load(str(path), _extra_files=extra)
    optimized = torch.jit.freeze(module.eval())
    torch.jit.save(optimized, str(path), _extra_files=extra)
    with pytest.raises(ValueError, match="parameters"):
        HRNetPoseExtractor(path, frozen=False)


def test_raw_training_checkpoint_restores_updated_backbone_and_optimizer(tmp_path):
    import copy

    import cv2
    import numpy as np

    from surveillance.actor_config import DEFAULT_CONFIG
    from surveillance.training.actor_transformer_trainer import load_checkpoint, train

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["device"] = "cpu"
    config["data"].update(input_mode="raw", image_size=[12, 20])
    config["model"]["embedding_dim"] = 8
    config["model"]["transformer"].update(feedforward_dim=16, dropout=0.0)
    config["backbones"]["pose"].update(checkpoint=str(archive(tmp_path)), frozen=False)
    config["training"].update(
        max_iterations=1,
        batch_size=1,
        learning_rate=0.01,
        validation_interval=1,
        checkpoint_interval=1,
    )
    paths = []
    for index in range(10):
        path = tmp_path / f"frame_{index}.png"
        assert cv2.imwrite(str(path), np.full((12, 20, 3), 64 + index, dtype=np.uint8))
        paths.append(path.name)
    manifest = tmp_path / "raw.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "video_id": "one",
                "source_video_id": "one",
                "clip_id": "one",
                "split": "train",
                "frame_paths": paths,
                "frame_indices": list(range(10)),
                "actor_boxes": [[0.1, 0.1, 0.8, 0.9]],
                "actor_labels": [0],
                "group_label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = train(config, manifest, tmp_path / "trained")
    restored, saved = load_checkpoint(checkpoint)
    scale = restored.extractors["pose"].backbone.scale
    assert scale.requires_grad and scale.item() != 2.0
    torch.testing.assert_close(scale, saved["model_state"]["extractors.pose.backbone.scale"])
    # This scalar parameter is the synthetic backbone's only scalar; transformer
    # parameters are vectors/matrices. Adam's moments show it joined the optimizer.
    scalar_states = [
        state for state in saved["optimizer_state"]["state"].values() if state["exp_avg"].ndim == 0
    ]
    assert len(scalar_states) == 1 and scalar_states[0]["exp_avg"].abs() > 0
    assert saved["backbone_metadata"]["pose"] == restored.extractors["pose"].metadata
    changed_preprocessing = copy.deepcopy(restored.extractors["pose"].metadata["preprocessing"])
    changed_preprocessing["mean"] = [0.5, 0.5, 0.5]
    archive(tmp_path, overrides={"preprocessing": changed_preprocessing})
    with pytest.raises(ValueError, match="metadata|provenance"):
        load_checkpoint(checkpoint)
