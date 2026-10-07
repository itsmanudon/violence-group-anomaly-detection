"""Interpretable Sultani-to-observable-behavior cascade with explicit abstention."""

import math
import time
from pathlib import Path

from surveillance.datasets.dcsass_protocol import CLASSES


class SurveillancePipeline:
    """Route only top thresholded anomaly segments to a supplied behavior analyzer.

    Dependencies own their trained models. This orchestration does not download,
    train, calibrate or silently substitute a missing research checkpoint.
    """

    def __init__(
        self,
        anomaly,
        behavior,
        *,
        threshold: float = 0.5,
        max_windows: int = 3,
        provenance: dict | None = None,
    ):
        if not 0 <= threshold <= 1 or type(max_windows) is not int or max_windows < 1:
            raise ValueError("Invalid cascade threshold/window budget")
        self.anomaly, self.behavior = anomaly, behavior
        self.threshold, self.max_windows = threshold, max_windows
        self.provenance = provenance

    def predict_video(self, path: Path) -> dict:
        started = time.perf_counter()
        anomaly_started = time.perf_counter()
        anomaly = self.anomaly.predict_video(Path(path), threshold=self.threshold)
        anomaly_seconds = time.perf_counter() - anomaly_started
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in anomaly.segment_scores):
            raise ValueError("Invalid anomaly score in cascade")
        candidates = []
        used = set()
        ranked = sorted(
            range(len(anomaly.segment_scores)), key=lambda i: (-anomaly.segment_scores[i], i)
        )
        for index in ranked:
            score = anomaly.segment_scores[index]
            if score < self.threshold:
                continue
            interval = tuple(anomaly.timestamps[index])
            if interval in used:
                continue
            used.add(interval)
            candidates.append((index, interval, score))
            if len(candidates) == self.max_windows:
                break
        metadata = dict(anomaly.metadata)
        windows, alerts = [], []
        if candidates:
            fps, num_frames = metadata["fps"], metadata["num_frames"]
            for index, (start, end), score in candidates:
                ranges = metadata.get("segment_frame_ranges")
                frame_start, frame_end = (
                    ranges[index]
                    if ranges
                    else (
                        max(0, math.floor(start * fps + 1e-8)),
                        min(num_frames, math.ceil(end * fps - 1e-8)),
                    )
                )
                if frame_end <= frame_start:
                    raise ValueError("Suspicious interval has no real video frames")
                prediction = self.behavior.analyze_window(
                    Path(path), frame_start, frame_end, num_frames, fps
                )
                available = prediction["status"] == "available"
                label = prediction["prediction"] if available else None
                if available and (prediction["actor_count"] < 1 or label not in range(6)):
                    raise ValueError("Behavior prediction requires detected actors and valid class")
                predicted_behavior = CLASSES[label] if available else None
                disagreement = available and label == 0
                window = {
                    **prediction,
                    "segment_index": index,
                    "interval": [start, end],
                    "frame_range": [frame_start, frame_end],
                    "anomaly_score": score,
                    "predicted_behavior": predicted_behavior,
                    "model_disagreement": disagreement,
                }
                windows.append(window)
                state = "behavior_review" if available and label != 0 else "generic_anomaly"
                reason = (
                    f"Anomaly detected; behavior classifier predicts {predicted_behavior}. "
                    "Human review required."
                    if state == "behavior_review"
                    else "General anomaly detected; human-centric classifier predicts Normal. "
                    "Models disagree."
                    if disagreement
                    else "General anomaly detected; behavior unavailable because "
                    "no actors were detected."
                )
                alerts.append(
                    {
                        "state": state,
                        "severity": "review",
                        "reason": reason,
                        "behavior": predicted_behavior,
                        "timestamp_interval": [start, end],
                        "confidence": {
                            "anomaly_score": score,
                            "behavior_probability": prediction["probabilities"][label]
                            if available
                            else None,
                        },
                    }
                )
        final = (
            max(
                alerts,
                key=lambda a: (a["state"] == "behavior_review", a["confidence"]["anomaly_score"]),
            )
            if alerts
            else {
                "state": "no_anomaly",
                "severity": "none",
                "reason": "No segment reached the configured anomaly threshold.",
                "behavior": None,
                "timestamp_interval": None,
                "confidence": {
                    "anomaly_score": anomaly.overall_score,
                    "behavior_probability": None,
                },
            }
        )
        stages = {
            key: metadata.get("timings", {}).get(key, 0.0)
            for key in (
                "video_decode_seconds",
                "c3d_preprocess_seconds",
                "c3d_seconds",
                "sultani_seconds",
                "actor_preprocess_seconds",
                "person_detection_seconds",
                "i3d_seconds",
                "actor_transformer_seconds",
            )
        }
        for window in windows:
            for key, value in window.get("timings", {}).items():
                if key in stages:
                    stages[key] += value
        elapsed = time.perf_counter() - started
        return {
            "schema_version": 1,
            "provenance": self.provenance,
            "video_path": str(path),
            "video_metadata": metadata,
            "anomaly": anomaly.to_dict(),
            "behavior": {"classes": list(CLASSES), "windows": windows},
            "alerts": alerts,
            "final_alert": final,
            "timings": {
                **stages,
                "anomaly_total_seconds": anomaly_seconds,
                "total_seconds": elapsed,
                "other_overhead_seconds": max(0.0, elapsed - sum(stages.values())),
            },
            "policy": {
                "anomaly_threshold": self.threshold,
                "max_windows": self.max_windows,
                "confidence_calibrated": False,
                "no_actor_fallback": "preserve_generic_anomaly",
            },
        }
