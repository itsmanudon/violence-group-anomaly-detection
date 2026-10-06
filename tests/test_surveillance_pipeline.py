import json
from pathlib import Path

import pytest

from surveillance.inference.anomaly_pipeline import AnomalyResult
from surveillance.inference.surveillance_pipeline import SurveillancePipeline


class FixedAnomaly:
    def __init__(self, score):
        self.score = score

    def predict_video(self, path, threshold=0.5):
        return AnomalyResult(
            self.score,
            [self.score] * 32,
            [(0.0, 1.0)] * 32,
            [(0.0, 1.0)] if self.score >= threshold else [],
            {"fps": 30.0, "num_frames": 30, "duration_sec": 1.0},
        )


class FixedBehavior:
    def __init__(self, prediction=None, no_actors=False):
        self.prediction, self.no_actors = prediction, no_actors

    def analyze_window(self, *args):
        if self.no_actors:
            return {
                "status": "no_actors",
                "actor_count": 0,
                "probabilities": None,
                "prediction": None,
            }
        probabilities = [0.02] * 6
        probabilities[self.prediction] = 0.9
        return {
            "status": "available",
            "actor_count": 2,
            "prediction": self.prediction,
            "probabilities": probabilities,
        }


@pytest.mark.parametrize("budget", [0, 1.5, True])
def test_behavior_window_budget_requires_a_positive_integer(budget):
    with pytest.raises(ValueError, match="budget"):
        SurveillancePipeline(FixedAnomaly(0.9), FixedBehavior(3), max_windows=budget)


def test_invalid_low_rank_anomaly_scores_are_rejected_before_top_window_routing():
    class InvalidAnomaly(FixedAnomaly):
        def predict_video(self, *args, **kwargs):
            result = super().predict_video(*args, **kwargs)
            result.segment_scores[-1] = -0.1
            return result

    with pytest.raises(ValueError, match="anomaly score"):
        SurveillancePipeline(InvalidAnomaly(0.9), FixedBehavior(3), max_windows=1).predict_video(
            Path("fixture.mp4")
        )


def test_no_anomaly_bypasses_behavior():
    class MustNotRun:
        def analyze_window(self, *args):
            raise AssertionError("Normal video must bypass actor analysis")

    result = SurveillancePipeline(FixedAnomaly(0.1), MustNotRun()).predict_video(
        Path("fixture.mp4")
    )
    assert result["behavior"]["windows"] == []
    assert result["final_alert"]["state"] == "no_anomaly"


def test_anomaly_and_behavior_preserve_both_outputs_and_deduplicate_short_units():
    result = SurveillancePipeline(FixedAnomaly(0.9), FixedBehavior(3)).predict_video(
        Path("fixture.mp4")
    )
    assert result["final_alert"]["state"] == "behavior_review"
    assert result["final_alert"]["behavior"] == "Fighting"
    assert len(result["behavior"]["windows"]) == 1
    assert result["anomaly"]["overall_score"] == 0.9
    json.dumps(result, allow_nan=False)


def test_normal_behavior_disagreement_remains_generic_anomaly():
    result = SurveillancePipeline(FixedAnomaly(0.9), FixedBehavior(0)).predict_video(
        Path("fixture.mp4")
    )
    assert result["final_alert"]["state"] == "generic_anomaly"
    assert result["behavior"]["windows"][0]["predicted_behavior"] == "Normal"
    assert result["behavior"]["windows"][0]["model_disagreement"] is True


def test_no_actor_abstention_preserves_anomaly_and_has_no_normal_prediction():
    result = SurveillancePipeline(FixedAnomaly(0.9), FixedBehavior(no_actors=True)).predict_video(
        Path("fixture.mp4")
    )
    assert result["final_alert"]["state"] == "generic_anomaly"
    assert result["behavior"]["windows"][0]["predicted_behavior"] is None
    assert result["behavior"]["windows"][0]["status"] == "no_actors"
