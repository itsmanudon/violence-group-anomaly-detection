"""Weak supervision through MIL ranking, sparsity, and temporal smoothness."""

import torch
from torch import nn


class MILLoss(nn.Module):
    """Mean paired hinge + lambda1*mean(sum(a)) + lambda2*mean(sum(diff(a)^2)).

    Hinge for each abnormal/normal pair is relu(margin-max(a)+max(n)).
    Regularization sums over segments and averages over bags, so coefficients
    do not grow with batch size. Only abnormal bags are regularized.
    """

    def __init__(
        self,
        ranking_margin: float = 1.0,
        sparsity_weight: float = 8e-5,
        smoothness_weight: float = 8e-5,
    ) -> None:
        super().__init__()
        if min(ranking_margin, sparsity_weight, smoothness_weight) < 0:
            raise ValueError("Loss parameters must be nonnegative")
        self.margin = ranking_margin
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, abnormal: torch.Tensor, normal: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return differentiable total and unweighted component losses."""
        if abnormal.ndim != 2 or abnormal.shape != normal.shape or abnormal.numel() == 0:
            raise ValueError("Expected nonempty paired [batch, segments] score tensors")
        if not torch.isfinite(abnormal).all() or not torch.isfinite(normal).all():
            raise ValueError("Scores must be finite")
        ranking = torch.relu(self.margin - abnormal.amax(1) + normal.amax(1)).mean()
        sparsity = abnormal.sum(1).mean()
        smoothness = abnormal.diff(dim=1).square().sum(1).mean()
        total = ranking + self.sparsity_weight * sparsity + self.smoothness_weight * smoothness
        return dict(total=total, ranking=ranking, sparsity=sparsity, smoothness=smoothness)
