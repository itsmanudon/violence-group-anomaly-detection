"""Explicit post-norm encoder with inspectable actor attention."""

import torch
from torch import nn


class ActorEncoderLayer(nn.Module):
    """Paper equations 4-6: attention/residual/norm then ReLU FFN/residual/norm.

    True in actor_valid_mask means a real actor. PyTorch key_padding_mask has
    the opposite meaning, so we pass ~valid. Key masking alone does NOT zero
    padded queries; these are explicitly cleared after every residual block.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 1,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )

    def forward(
        self, features: torch.Tensor, valid: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transform [B,N,D]; optional attention is [B,H,N,N]."""
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        attended, weights = self.attention(
            features,
            features,
            features,
            key_padding_mask=~valid,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        features = self.norm1(features + self.dropout(attended)).masked_fill(
            ~valid.unsqueeze(-1), 0
        )
        features = self.norm2(features + self.dropout(self.feedforward(features)))
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        if weights is not None:
            weights = weights.masked_fill(~valid[:, None, :, None], 0)
            weights = weights.masked_fill(~valid[:, None, None, :], 0)
        return features, weights


class ActorEncoder(nn.Module):
    """Stack independent encoder layers, preserving actor ordering."""

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int = 1,
        num_heads: int = 1,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(embedding_dim, num_layers, num_heads, feedforward_dim) < 1:
            raise ValueError("Encoder dimensions and layer/head counts must be positive")
        if embedding_dim % num_heads or not 0 <= dropout < 1:
            raise ValueError("embedding_dim must divide by num_heads; dropout must be in [0,1)")
        self.layers = nn.ModuleList(
            [
                ActorEncoderLayer(embedding_dim, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, features: torch.Tensor, valid: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return contextual actors and attention [B,L,H,N,N] when requested."""
        weights = []
        for layer in self.layers:
            features, attention = layer(features, valid, return_attention)
            if attention is not None:
                weights.append(attention)
        return features, torch.stack(weights, dim=1) if weights else None
