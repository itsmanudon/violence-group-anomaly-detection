"""Manifest evaluation with explicit frame or reduced bag-level modes."""

from pathlib import Path

import numpy as np
import torch

from surveillance.datasets.common import check_leakage, read_manifest
from surveillance.datasets.features import record_features
from surveillance.evaluation.anomaly_metrics import binary_metrics, frame_labels, project_scores
from surveillance.training.sultani_trainer import load_checkpoint


@torch.inference_mode()
def evaluate(
    checkpoint: Path,
    manifest: Path,
    split: str = "test",
    mode: str = "bag",
    threshold: float = 0.5,
    projection: str = "repeat",
    include_predictions: bool = False,
) -> dict:
    """Evaluate held-out videos; frame mode refuses unknown abnormal annotations."""
    model, saved = load_checkpoint(checkpoint)
    config = saved["config"]
    records = read_manifest(manifest)
    check_leakage(records)
    records = [r for r in records if r.split == split]
    if not records:
        raise ValueError(f"No records in split {split}")
    if mode not in {"bag", "frame"}:
        raise ValueError("mode must be bag or frame")
    labels, scores, predictions = [], [], []
    for row in records:
        prediction = model(
            record_features(row, manifest, config["num_segments"], config["feature_dim"])
        ).numpy()
        if include_predictions:
            predictions.append(
                {
                    "video_id": row.video_id,
                    "source_video_id": row.source_video_id,
                    "path": row.path,
                    "bag_label": row.label,
                    "anomaly_type": row.anomaly_type,
                    "segment_scores": prediction.tolist(),
                    "overall_score": float(prediction.max()),
                    "num_frames": row.num_frames,
                    "fps": row.fps,
                }
            )
        if mode == "bag":
            labels.append(np.array([row.label]))
            scores.append(np.array([prediction.max()]))
        else:
            if row.num_frames is None or row.fps is None:
                raise ValueError(f"{row.video_id}: frame evaluation needs num_frames and fps")
            if row.label == 1 and row.temporal_annotations is None:
                raise ValueError(
                    f"{row.video_id}: temporal annotations unavailable; use --mode bag "
                    "for reduced evaluation, not frame-level benchmark results"
                )
            labels.append(frame_labels(row.num_frames, row.fps, row.temporal_annotations or []))
            scores.append(project_scores(prediction, row.num_frames, projection))
    result = binary_metrics(np.concatenate(labels), np.concatenate(scores), threshold)
    result.update(
        mode=mode,
        split=split,
        videos=len(records),
        projection=projection if mode == "frame" else None,
        explanation=(
            "Reduced video/bag evaluation using maximum segment score; not frame-level UCF ROC-AUC."
        )
        if mode == "bag"
        else "Frame-level evaluation using temporal ground truth and projected segment scores.",
    )
    if include_predictions:
        result["predictions"] = predictions
    return result
