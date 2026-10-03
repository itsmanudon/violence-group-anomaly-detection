import torch
import torch.nn.functional as F

from surveillance.models.actor_transformer.loss import ActorGroupLoss


def test_weighted_cross_entropy_ignores_padding_and_backward():
    actor = torch.randn(2, 3, 5, requires_grad=True)
    group = torch.randn(2, 5, requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    labels = torch.tensor([[2, -900, 99], [0, 1, 3]])
    target = torch.tensor([1, 2])
    loss = ActorGroupLoss(2.0, 3.0)(
        {"actor_logits": actor, "group_logits": group}, labels, target, mask
    )
    torch.testing.assert_close(loss["actor"], F.cross_entropy(actor[mask], labels[mask]))
    torch.testing.assert_close(loss["group"], F.cross_entropy(group, target))
    torch.testing.assert_close(loss["total"], 2 * loss["group"] + 3 * loss["actor"])
    loss["total"].backward()
    assert torch.isfinite(actor.grad).all() and torch.isfinite(group.grad).all()
    assert actor.grad[~mask].abs().sum() == 0
