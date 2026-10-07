"""Joint group and individual-action supervision."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from surveillance.models.actor_transformer.actor_transformer import validate_actor_mask


class ActorGroupLoss(nn.Module):
    """group_weight*CE(group) + actor_weight*CE(valid actors).

    Group CE averages over scenes. Actor CE averages over all real actors in
    the batch. Arbitrary padded targets never reach cross_entropy. In late
    fusion, this supervises the fused probability distribution jointly; separate
    single-modality pretraining is also available through the single modes.
    """

    def __init__(self, group_weight: float = 1.0, actor_weight: float = 1.0) -> None:
        super().__init__()
        if any(not math.isfinite(w) or w < 0 for w in (group_weight, actor_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")
        self.group_weight, self.actor_weight = group_weight, actor_weight

    def forward(
        self,
        output: dict,
        actor_labels: torch.Tensor,
        group_labels: torch.Tensor,
        actor_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return differentiable total and unweighted actor/group components."""
        actor_logits, group_logits = output["actor_logits"], output["group_logits"]
        validate_actor_mask(actor_logits, actor_valid_mask)
        if actor_labels.shape != actor_valid_mask.shape or group_labels.shape != (
            len(actor_labels),
        ):
            raise ValueError("Targets must be actor_labels [B,N] and group_labels [B]")
        actor = F.cross_entropy(actor_logits[actor_valid_mask], actor_labels[actor_valid_mask])
        group = F.cross_entropy(group_logits, group_labels)
        return {
            "total": self.group_weight * group + self.actor_weight * actor,
            "group": group,
            "actor": actor,
        }
