"""C3D FC6 architecture and strictly local checkpoint adapter; never downloads."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from surveillance.video.decode import iter_clips, probe_video
from surveillance.video.segmentation import aggregate_segments
from surveillance.video.transforms import c3d_transform


class FeatureExtractor(Protocol):
    """Extension point for future feature modalities."""

    feature_dim: int

    def extract_video(self, path: Path, num_segments: int = 32) -> np.ndarray:
        """Return normalized [segments, feature_dim] embeddings."""
        ...


class C3DFC6(nn.Module):
    """Canonical eight-convolution C3D trunk and post-ReLU FC6 embeddings.

    Checkpoints use conv1, conv2, conv3a/b, conv4a/b, conv5a/b, fc6 keys.
    Full classifier keys fc7/fc8 may be removed when converting to this schema.
    No random-weight fallback is exposed by the extraction adapter.
    """

    def __init__(self) -> None:
        super().__init__()
        previous = 3
        for name, channels in [
            ("conv1", 64),
            ("conv2", 128),
            ("conv3a", 256),
            ("conv3b", 256),
            ("conv4a", 512),
            ("conv4b", 512),
            ("conv5a", 512),
            ("conv5b", 512),
        ]:
            setattr(self, name, nn.Conv3d(previous, channels, 3, padding=1))
            previous = channels
        self.fc6 = nn.Linear(512 * 1 * 4 * 4, 4096)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Map [batch,3,16,112,112] to [batch,4096]."""
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (3, 16, 112, 112):
            raise ValueError("C3D expects [B,3,16,112,112]")
        x = torch.relu(self.conv1(clips))
        x = torch.nn.functional.max_pool3d(x, (1, 2, 2), (1, 2, 2))
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool3d(x, 2, 2)
        for block in (3, 4, 5):
            x = torch.relu(getattr(self, f"conv{block}a")(x))
            x = torch.relu(getattr(self, f"conv{block}b")(x))
            x = torch.nn.functional.max_pool3d(x, 2, 2, padding=(0, 1, 1) if block == 5 else 0)
        return torch.relu(self.fc6(x.flatten(1)))


class C3DExtractor:
    """Load compatible local C3D weights and aggregate 16-frame embeddings.

    mean/order must come from checkpoint provenance. Extraction streams clips;
    it stores embeddings only, rather than retaining every decoded video frame.
    """

    feature_dim = 4096

    def __init__(
        self,
        checkpoint: Path | None,
        device: str = "cpu",
        batch_size: int = 4,
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        channel_order: str = "bgr",
    ) -> None:
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                "C3D checkpoint missing. Supply --c3d-checkpoint with a local "
                "compatible pretrained FC6 state_dict; see data/README.md. "
                "Use --features for precomputed-feature inference."
            )
        if batch_size < 1:
            raise ValueError("C3D batch_size must be positive")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.mean = tuple(mean)
        self.channel_order = channel_order
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint_data, dict):
            raise ValueError("C3D checkpoint must contain a state_dict mapping")
        state = checkpoint_data.get("state_dict", checkpoint_data)
        if not isinstance(state, dict):
            raise ValueError("C3D checkpoint must contain a state_dict mapping")
        with torch.device("meta"):
            self.model = C3DFC6()
        try:
            self.model.load_state_dict(state, strict=True, assign=True)
        except RuntimeError as error:
            raise ValueError(
                "Incompatible C3D checkpoint: expected conv1..conv5b and fc6 "
                "PyTorch parameters. Convert names/layout explicitly; "
                "no torchvision backbone is substituted."
            ) from error
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def extract_video(self, path: Path, num_segments: int = 32) -> np.ndarray:
        """Extract FC6 clip units, average into segments, and L2 normalize."""
        metadata = probe_video(path)
        if metadata.fps <= 0:
            raise ValueError("Video needs positive FPS")
        outputs = []
        batch = []
        for clip in iter_clips(path):
            batch.append(c3d_transform(clip, self.mean, self.channel_order))
            if len(batch) == self.batch_size:
                outputs.append(self.model(torch.stack(batch).to(self.device)).cpu().numpy())
                batch = []
        if batch:
            outputs.append(self.model(torch.stack(batch).to(self.device)).cpu().numpy())
        return aggregate_segments(np.concatenate(outputs), num_segments)
