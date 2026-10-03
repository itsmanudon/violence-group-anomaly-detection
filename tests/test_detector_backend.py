import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from surveillance.datasets.collective import ActorRecord, write_actor_manifest
from surveillance.detection import filter_detections, read_detections
from surveillance.detection import torchvision_detector as backend


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.BatchNorm2d(3))
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, images):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert images[0].shape == (3, 24, 40)
        return [
            {
                "boxes": torch.tensor([[1, 2, 11, 22], [20, 0, 30, 20]]),
                "scores": torch.tensor([0.9, 0.9]),
                "labels": torch.tensor([1, 2]),
            }
        ]


def test_real_architecture_without_download(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No pretrained download allowed")

    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    # Real factory, architecture and normalization on meta avoid allocating 160MB of weights.
    with torch.device("meta"):
        model = backend.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=91)
        backend._frozen_batch_norm(model.backbone)
    assert model.roi_heads.box_predictor.cls_score.out_features == 91
    norms = [m for m in model.backbone.modules() if isinstance(m, FrozenBatchNorm2d)]
    assert len(norms) == 53 and all(m.eps == 0 for m in norms)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.backbone.modules())


def test_missing_weights_fails_before_model_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "fasterrcnn_resnet50_fpn", lambda **kw: pytest.fail("built model"))
    with pytest.raises(ValueError, match="local pretrained"):
        backend.TorchvisionPersonDetector()
    with pytest.raises(FileNotFoundError):
        backend.TorchvisionPersonDetector(tmp_path / "absent.pth")


def test_adapter_local_strict_checkpoint_and_provenance(monkeypatch, tmp_path):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return FakeModel()

    monkeypatch.setattr(backend, "fasterrcnn_resnet50_fpn", factory)
    model = FakeModel()
    backend._frozen_batch_norm(model.backbone)
    checkpoint = tmp_path / "local.pth"
    torch.save(model.state_dict(), checkpoint)
    detector = backend.TorchvisionPersonDetector(checkpoint, device="cpu")
    result = detector.detect(torch.zeros(3, 24, 40))
    assert result.boxes.tolist() == [[1, 2, 11, 22]]
    assert result.metadata["untrained"] is False
    assert len(result.metadata["checkpoint_sha256"]) == 64
    assert calls == [
        {
            "weights": None,
            "weights_backbone": None,
            "num_classes": 91,
            "box_score_thresh": 0.0,
            "box_nms_thresh": 1.0,
            "box_detections_per_img": 1000,
        }
    ]
    assert result.metadata["backend_candidate_cap"] == 1000
    assert result.metadata["before_count_scope"] == "after_rpn_and_backend_candidate_cap"
    torch.save({"wrong": torch.ones(1)}, checkpoint)
    with pytest.raises(ValueError, match="Incompatible"):
        backend.TorchvisionPersonDetector(checkpoint, device="cpu")
    state = model.state_dict()
    state["weight"] = torch.zeros(2)
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="Incompatible"):
        backend.TorchvisionPersonDetector(checkpoint, device="cpu")


@pytest.mark.parametrize(
    "image",
    [
        torch.zeros(1, 24, 40),
        torch.zeros(3, 24, 40).long(),
        torch.full((3, 24, 40), float("nan")),
        torch.full((3, 24, 40), 2.0),
    ],
)
def test_adapter_input_contract(monkeypatch, image):
    monkeypatch.setattr(backend, "fasterrcnn_resnet50_fpn", lambda **kw: FakeModel())
    detector = backend.TorchvisionPersonDetector(device="cpu", allow_untrained=True)
    with pytest.raises(ValueError, match="RGB"):
        detector.detect(image)


def test_injected_cli_uses_native_rgb_center_and_source(tmp_path):
    image = np.zeros((24, 40, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    cv2.imwrite(str(tmp_path / "center.png"), image)
    manifest = tmp_path / "manifest.jsonl"
    row = ActorRecord(
        "collective",
        "v1",
        "collective:v1",
        "v1:5",
        "test",
        ["absent.png"] * 5 + ["center.png"] + ["absent.png"] * 4,
        list(range(10)),
        [[0.1, 0.1, 0.3, 0.4]],
        [0],
        0,
    )
    write_actor_manifest([row], manifest)

    class MockDetector:
        def detect(self, image):
            assert image.shape == (3, 24, 40)
            assert (image[0] == 1).all() and (image[2] == 0).all()
            return filter_detections(torch.empty(0, 4), [], [], (24, 40))

    script = Path(__file__).parents[1] / "scripts" / "detect_people.py"
    spec = importlib.util.spec_from_file_location("detect_people", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "detections.jsonl"
    module.run_detection(manifest, output, MockDetector())
    actual = read_detections(output)[0]
    assert actual.frame_index == 5
    assert actual.source_video_id == "collective:v1"
    assert actual.result.boxes.shape == (0, 4)
