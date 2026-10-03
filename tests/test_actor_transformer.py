import pytest
import torch

from surveillance.models.actor_transformer.actor_transformer import ActorTransformer
from surveillance.models.actor_transformer.fusion import fuse_log_probabilities
from surveillance.models.actor_transformer.positional_encoding import SpatialPositionEncoding


def make_model(mode="pose_only"):
    return ActorTransformer(
        mode=mode,
        pose_feature_dim=6,
        rgb_feature_dim=4,
        embedding_dim=8,
        num_actor_classes=5,
        num_group_classes=5,
        transformer={"num_layers": 1, "num_heads": 1, "feedforward_dim": 16, "dropout": 0.0},
    )


@pytest.mark.parametrize(
    "mode", ["pose_only", "rgb_only", "pose_rgb_early_fusion", "pose_rgb_late_fusion"]
)
@pytest.mark.parametrize("actors", [1, 3])
def test_modes_and_gradients(mode, actors):
    model = make_model(mode)
    boxes = torch.tensor([0.1, 0.2, 0.5, 0.8]).expand(2, actors, 4)
    output = model(
        actor_boxes=boxes,
        actor_valid_mask=torch.ones(2, actors, dtype=torch.bool),
        pose_features=torch.randn(2, actors, 6),
        rgb_features=torch.randn(2, actors, 4),
        return_attention=True,
    )
    assert output["actor_logits"].shape == (2, actors, 5)
    assert output["group_logits"].shape == (2, 5)
    assert torch.isfinite(output["group_logits"]).all()
    output["group_logits"].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_position_axes_independent_and_deterministic():
    encoding = SpatialPositionEncoding(8, reference_size=(480, 720))
    xy = torch.tensor([[[0.1, 0.2], [0.5, 0.2], [0.1, 0.8]]])
    output = encoding(xy)
    assert output.shape == (1, 3, 8)
    torch.testing.assert_close(output, encoding(xy))
    torch.testing.assert_close(output[0, 0, 4:], output[0, 1, 4:])
    torch.testing.assert_close(output[0, 0, :4], output[0, 2, :4])
    assert not torch.equal(output[0, 0, :4], output[0, 1, :4])
    assert not torch.equal(output[0, 0, 4:], output[0, 2, 4:])
    with pytest.raises(ValueError):
        SpatialPositionEncoding(6)


def test_late_fusion_is_weighted_probability_average():
    pose = torch.tensor([[4.0, -1.0, 0.0]])
    rgb = torch.tensor([[-2.0, 1.0, 3.0]])
    fused = fuse_log_probabilities(pose, rgb, pose_weight=2.0)
    torch.testing.assert_close(fused.softmax(-1), (2 * pose.softmax(-1) + rgb.softmax(-1)) / 3)
    assert not torch.allclose(fused.softmax(-1), ((2 * pose + rgb) / 3).softmax(-1))


def test_model_late_fusion_matches_independent_branch_outputs():
    model = make_model("pose_rgb_late_fusion").eval()
    output = model(
        pose_features=torch.randn(1, 2, 6),
        rgb_features=torch.randn(1, 2, 4),
        actor_boxes=torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]).expand(1, 2, 4),
        actor_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    for key in ("actor_logits", "group_logits"):
        branches = output["branch_outputs"]
        expected = (2 * branches["pose"][key].softmax(-1) + branches["rgb"][key].softmax(-1)) / 3
        torch.testing.assert_close(output[key].softmax(-1), expected)
