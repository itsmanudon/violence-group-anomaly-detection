import torch

from surveillance.models.actor_transformer.surveillance_transfer import (
    GroupOnlyLoss,
    transfer_rgb_model,
)
from surveillance.training.actor_transformer_trainer import ActorTransformerSystem


def test_group_only_loss_never_needs_actor_logits_or_labels():
    logits = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    loss = GroupOnlyLoss(torch.ones(6))(logits, torch.tensor([1]))
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, torch.tensor([1])))
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_transfer_preserves_rgb_representation_and_replaces_only_group_head(tmp_path):
    config = {
        "model": {
            "mode": "rgb_only",
            "rgb_feature_dim": 20800,
            "embedding_dim": 8,
            "num_actor_classes": 5,
            "num_group_classes": 5,
            "transformer": {"num_layers": 1, "num_heads": 1, "feedforward_dim": 16},
        }
    }
    source = ActorTransformerSystem(config)
    path = tmp_path / "source.pt"
    torch.save(
        dict(
            format_version=1,
            checkpoint_type="actor_transformer",
            config=config,
            model_state=source.state_dict(),
            optimizer_state={},
            scheduler_state={},
            iteration=1,
            best_value=0.0,
            selection_metric="validation_group_accuracy",
            manifest_fingerprint="fixture",
            backbone_metadata={},
        ),
        path,
    )
    adapted, receipt = transfer_rgb_model(path, seed=0)
    assert adapted.model.branches["rgb"].group_classifier.out_features == 6
    assert not adapted.model.branches["rgb"].actor_classifier.weight.requires_grad
    torch.testing.assert_close(
        adapted.model.rgb_projection.weight, source.model.rgb_projection.weight
    )
    torch.testing.assert_close(
        adapted.model.branches["rgb"].encoder.state_dict(),
        source.model.branches["rgb"].encoder.state_dict(),
    )
    assert receipt["supervision"] == "group_only"
    batch = {
        "rgb_features": torch.ones(1, 2, 20800),
        "actor_boxes": torch.tensor([[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]]),
        "actor_valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    output = adapted(batch)
    assert output["group_logits"].shape == (1, 6)
    GroupOnlyLoss()(output["group_logits"], torch.tensor([3])).backward()
    assert adapted.model.rgb_projection.weight.grad is not None
    assert adapted.model.branches["rgb"].actor_classifier.weight.grad is None
