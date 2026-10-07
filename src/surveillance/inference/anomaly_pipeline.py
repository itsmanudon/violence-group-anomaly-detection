"""Reusable scorer inference from features or locally configured C3D."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from surveillance.features.c3d import FeatureExtractor
from surveillance.models.sultani.model import SultaniScorer
from surveillance.video.decode import probe_video
from surveillance.video.segmentation import c3d_segment_frame_ranges


@dataclass
class AnomalyResult:
    """Anomaly localization only; this scorer does not classify anomaly types."""

    overall_score: float
    segment_scores: list[float]
    timestamps: list[tuple[float, float]]
    suspicious_intervals: list[tuple[float, float]]
    metadata: dict

    def to_dict(self) -> dict:
        """Return a JSON-serializable result."""
        return asdict(self)


class AnomalyPipeline:
    """Own a trained scorer and optional feature extractor."""

    def __init__(
        self,
        model: SultaniScorer,
        extractor: FeatureExtractor | None = None,
        num_segments: int = 32,
    ) -> None:
        if num_segments < 1:
            raise ValueError("num_segments must be positive")
        self.model = model.eval()
        self.extractor = extractor
        self.num_segments = num_segments

    @torch.inference_mode()
    def predict_features(
        self,
        features: np.ndarray | torch.Tensor,
        duration_sec: float,
        threshold: float = 0.5,
        metadata: dict | None = None,
        timestamps: list[tuple[float, float]] | None = None,
    ) -> AnomalyResult:
        """Score one feature bag and merge consecutive thresholded intervals."""
        if not np.isfinite(duration_sec) or duration_sec <= 0 or not 0 <= threshold <= 1:
            raise ValueError("duration_sec must be positive/finite; threshold must be in [0,1]")
        tensor = torch.as_tensor(
            features, dtype=torch.float32, device=next(self.model.parameters()).device
        )
        if tensor.shape != (self.num_segments, self.model.feature_dim):
            raise ValueError(f"Expected [{self.num_segments},{self.model.feature_dim}] features")
        scores = self.model(tensor).cpu().tolist()
        timestamp_policy = "uniform-duration segments (approximate for padded C3D clips)"
        if timestamps is None:
            edges = np.linspace(0, duration_sec, self.num_segments + 1).tolist()
            timestamps = list(zip(edges[:-1], edges[1:]))
        else:
            values = np.asarray(timestamps, dtype=float)
            if (
                values.shape != (self.num_segments, 2)
                or not np.isfinite(values).all()
                or (values[:, 0] < 0).any()
                or (values[:, 0] >= values[:, 1]).any()
                or (values[:, 1] > duration_sec + 1e-8).any()
                or (np.diff(values[:, 0]) < 0).any()
            ):
                raise ValueError("Invalid explicit segment timestamps")
            timestamps = [tuple(pair) for pair in values.tolist()]
            timestamp_policy = "explicit extractor temporal-unit boundaries"
        intervals: list[tuple[float, float]] = []
        for score, (start, end) in zip(scores, timestamps):
            if score >= threshold:
                if intervals and start <= intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
                else:
                    intervals.append((start, end))
        info = dict(metadata or {})
        info.update(
            duration_sec=duration_sec,
            threshold=threshold,
            timestamp_policy=timestamp_policy,
        )
        return AnomalyResult(max(scores), scores, timestamps, intervals, info)

    def predict_video(self, path: Path, threshold: float = 0.5) -> AnomalyResult:
        """Decode/extract/score with a configured local extractor."""
        if self.extractor is None:
            raise ValueError(
                "Video inference requires C3DExtractor with local pretrained weights; "
                "or call predict_features with precomputed C3D features"
            )
        metadata = probe_video(path)
        features = self.extractor.extract_video(path, self.num_segments)
        unit_frames = getattr(self.extractor, "temporal_unit_frames", None)
        frame_ranges = (
            c3d_segment_frame_ranges(metadata.num_frames, self.num_segments, unit_frames)
            if unit_frames is not None
            else None
        )
        started = time.perf_counter()
        result = self.predict_features(
            features,
            metadata.duration_sec,
            threshold,
            dict(
                video_path=str(path),
                fps=metadata.fps,
                num_frames=metadata.num_frames,
                segment_frame_ranges=frame_ranges,
            ),
            timestamps=[(a / metadata.fps, b / metadata.fps) for a, b in frame_ranges]
            if frame_ranges is not None
            else None,
        )
        result.metadata["timings"] = {
            **getattr(self.extractor, "last_timings", {}),
            "sultani_seconds": time.perf_counter() - started,
        }
        return result
