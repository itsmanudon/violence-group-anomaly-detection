"""Sinusoidal position features from actor bounding-box centers."""

import math
from collections.abc import Sequence

import torch
from torch import nn


class SpatialPositionEncoding(nn.Module):
    """Encode normalized xy centers [B,N,2] as [B,N,d].

    x occupies the first d/2 channels and y the remaining d/2. Each half
    alternates sin/cos with denominator 10000**(2*i/(d/2)). Normalized
    coordinates are scaled to reference (height,width) before the sinusoid.
    This coordinate scale is explicit: the paper gives center-based PE but
    does not specify its normalization. Set reference_size=(1,1) for unit scale.
    """

    def __init__(
        self, embedding_dim: int = 128, reference_size: Sequence[float] = (480, 720)
    ) -> None:
        super().__init__()
        if embedding_dim < 4 or embedding_dim % 4:
            raise ValueError("2D sinusoidal embedding_dim must be a positive multiple of 4")
        if len(reference_size) != 2 or any(not math.isfinite(v) or v <= 0 for v in reference_size):
            raise ValueError("reference_size must be finite positive [height,width]")
        self.embedding_dim = embedding_dim
        self.register_buffer(
            "scale", torch.tensor([reference_size[1], reference_size[0]], dtype=torch.float32)
        )
        frequency = torch.exp(
            -math.log(10000.0) * torch.arange(0, embedding_dim // 2, 2) / (embedding_dim // 2)
        )
        self.register_buffer("frequency", frequency)

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        """Encode finite normalized centers; callers clear padded coordinates first."""
        if centers.ndim != 3 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape [B,N,2]")
        if not torch.isfinite(centers).all() or ((centers < 0) | (centers > 1)).any():
            raise ValueError("Actor centers must be finite normalized xy in [0,1]")
        angles = centers.unsqueeze(-1) * self.scale[:, None] * self.frequency
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-3)
