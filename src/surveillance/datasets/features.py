"""Precomputed feature bags shared by training and evaluation."""

from pathlib import Path

import numpy as np
import torch

from surveillance.datasets.common import Record, resolve_path


def load_features(path: Path, num_segments: int = 32, feature_dim: int = 4096) -> torch.Tensor:
    """Load an explicit [S,D] .npy, tensor .pt, or whitespace .txt bag.

    No silent reshaping or aggregation of arbitrary input tensors is performed.
    Inputs should already be segment-aggregated and L2-normalized.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing features: {path}. Run scripts/extract_c3d.py first.")
    if path.suffix == ".npy":
        data = torch.from_numpy(np.load(path, allow_pickle=False)).float()
    elif path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"{path}: expected a tensor")
        data = data.float()
    elif path.suffix == ".txt":
        data = torch.from_numpy(np.loadtxt(path, dtype=np.float32)).reshape(-1, feature_dim)
    else:
        raise ValueError("Supported feature formats: .npy, .pt, .pth, .txt")
    if data.shape != (num_segments, feature_dim) or not torch.isfinite(data).all():
        raise ValueError(
            f"{path}: expected finite [{num_segments},{feature_dim}], got {data.shape}"
        )
    return data


def record_features(record: Record, manifest: Path, segments: int, dimension: int) -> torch.Tensor:
    """Load features referenced by a manifest record."""
    if record.feature_path is None:
        raise ValueError(
            f"{record.video_id}: feature_path missing; extract or attach features first"
        )
    return load_features(resolve_path(record.feature_path, manifest), segments, dimension)
