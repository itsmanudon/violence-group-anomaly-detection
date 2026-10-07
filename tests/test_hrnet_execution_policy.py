"""Inference-only pose archives must not change execution graphs after a cold call."""

import pytest
import torch
from test_actor_backbones import archive
from torch import nn

from surveillance.features.hrnet_pose import HRNetPoseExtractor
from surveillance.features.provenance import extraction_config


class ProfilingSensitiveFeatures(nn.Module):
    """Model a profiling executor's cold/hot feature drift without needing CUDA."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, crops):
        self.calls += 1
        optimization_enabled = torch._C._get_graph_executor_optimize()
        value = 1.0 if not optimization_enabled or self.calls == 1 else 1.01
        return crops.new_full((len(crops), 32, 64, 48), value)


def test_inference_only_cold_and_hot_features_equal_and_scope_restored(tmp_path):
    extractor = HRNetPoseExtractor(archive(tmp_path, overrides={"inference_only": True}))
    extractor.backbone = ProfilingSensitiveFeatures()
    frames = torch.zeros(1, 3, 12, 20)
    boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    with torch.jit.optimized_execution(True):
        first, second = extractor(frames, boxes, valid), extractor(frames, boxes, valid)
        assert torch.equal(first, second)
        assert torch._C._get_graph_executor_optimize()


def test_execution_policy_is_in_cache_fingerprint(tmp_path):
    extractor = HRNetPoseExtractor(archive(tmp_path, overrides={"inference_only": True}))
    config = extraction_config("pose", (480, 720), "a" * 64, extractor.metadata, "ground_truth")
    assert config["jit_execution_policy"] == "unoptimized_inference_only"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Optional CUDA regression; CPU fixtures run everywhere"
)
def test_scripted_pose_cpu_cuda_correspondence(tmp_path):
    checkpoint = archive(tmp_path, overrides={"inference_only": True})
    frames = torch.linspace(0, 1, 3 * 12 * 20).reshape(1, 3, 12, 20)
    boxes = torch.tensor([[[0.1, 0.2, 0.8, 0.9]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    cpu = HRNetPoseExtractor(checkpoint)(frames, boxes, valid)
    gpu = HRNetPoseExtractor(checkpoint).cuda()
    first = gpu(frames.cuda(), boxes.cuda(), valid.cuda())
    second = gpu(frames.cuda(), boxes.cuda(), valid.cuda())
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(first.cpu(), cpu, rtol=1e-4, atol=1e-5)
