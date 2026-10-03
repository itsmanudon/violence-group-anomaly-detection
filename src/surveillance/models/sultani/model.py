"""Paper-1 segment scoring network."""

from collections.abc import Sequence

import torch
from torch import nn


class SultaniScorer(nn.Module):
    """Score [..., segments, feature_dim] inputs as [..., segments] probabilities.

    Matches the authors' released network: ReLU on the first hidden layer,
    linear subsequent hidden layers, dropout after each, then sigmoid.
    """

    def __init__(
        self, feature_dim: int = 4096, hidden_dims: Sequence[int] = (512, 32), dropout: float = 0.6
    ) -> None:
        super().__init__()
        if feature_dim < 1 or not hidden_dims or any(d < 1 for d in hidden_dims):
            raise ValueError("feature_dim and hidden_dims must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.feature_dim = feature_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        layers: list[nn.Module] = []
        previous = feature_dim
        for index, dimension in enumerate(hidden_dims):
            layers.append(nn.Linear(previous, dimension))
            if index == 0:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous = dimension
        layers.extend([nn.Linear(previous, 1), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict one score for every feature row."""
        if features.ndim not in (2, 3) or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected [S,{self.feature_dim}] or [B,S,{self.feature_dim}]")
        if features.shape[-2] == 0 or not torch.isfinite(features).all():
            raise ValueError("Features must contain finite, nonempty segments")
        return self.network(features).squeeze(-1)
