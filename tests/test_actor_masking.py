import pytest
import torch

from surveillance.models.actor_transformer.actor_transformer import ActorTransformer, masked_max


@pytest.mark.parametrize(
    "mode", ["pose_only", "rgb_only", "pose_rgb_early_fusion", "pose_rgb_late_fusion"]
)
def test_padded_nan_actors_are_invisible(mode):
    torch.manual_seed(2)
    model = ActorTransformer(
        mode=mode,
        pose_feature_dim=6,
        rgb_feature_dim=4,
        embedding_dim=8,
        transformer={"dropout": 0.0},
    ).eval()
    pose, rgb = torch.randn(1, 2, 6), torch.randn(1, 2, 4)
    boxes = torch.tensor([[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]])
    clean = model(pose, rgb, boxes, torch.ones(1, 2, dtype=torch.bool), True)
    padded = model(
        torch.cat([pose, torch.full((1, 3, 6), float("nan"))], 1),
        torch.cat([rgb, torch.full((1, 3, 4), float("nan"))], 1),
        torch.cat([boxes, torch.full((1, 3, 4), float("nan"))], 1),
        torch.tensor([[True, True, False, False, False]]),
        True,
    )
    torch.testing.assert_close(clean["group_logits"], padded["group_logits"])
    torch.testing.assert_close(clean["actor_logits"], padded["actor_logits"][:, :2])
    assert torch.equal(padded["actor_logits"][:, 2:], torch.zeros(1, 3, 5))
    for attention in padded["attention"].values():
        assert torch.equal(attention[..., 2:, :], torch.zeros_like(attention[..., 2:, :]))
        assert torch.equal(attention[..., :, 2:], torch.zeros_like(attention[..., :, 2:]))
        torch.testing.assert_close(
            attention[..., :2, :].sum(-1), torch.ones_like(attention[..., :2, 0])
        )


def test_masked_pooling_and_empty_scene():
    features = torch.tensor([[[-5.0, -2.0], [100.0, 100.0]]])
    torch.testing.assert_close(masked_max(features, torch.tensor([[True, False]])), features[:, 0])
    with pytest.raises(ValueError, match="at least one"):
        masked_max(features, torch.zeros(1, 2, dtype=torch.bool))


def test_permutation_equivariance_and_one_valid_actor():
    model = ActorTransformer(
        pose_feature_dim=6, embedding_dim=8, transformer={"dropout": 0.0}
    ).eval()
    features = torch.randn(1, 3, 6)
    boxes = torch.tensor([[[0.0, 0.0, 0.2, 0.3], [0.4, 0.5, 0.7, 1.0], [0.0, 0.0, 1.0, 1.0]]])
    mask = torch.tensor([[True, False, False]])
    a = model(pose_features=features, actor_boxes=boxes, actor_valid_mask=mask)
    order = [2, 0, 1]
    b = model(
        pose_features=features[:, order],
        actor_boxes=boxes[:, order],
        actor_valid_mask=mask[:, order],
    )
    torch.testing.assert_close(a["group_logits"], b["group_logits"])
    torch.testing.assert_close(a["actor_logits"][:, order], b["actor_logits"])
