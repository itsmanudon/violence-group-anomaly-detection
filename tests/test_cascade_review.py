import copy
import json

import pytest
from test_cascade_evaluation import record

from surveillance.evaluation.cascade_review import compare_review_modes
from surveillance.inference.cascade_review import ReviewAwarePipeline, ReviewPolicy, assess_review


def mixed_result():
    result = record("clip", 0, 0.9, "available", 4)["result"]
    other = copy.deepcopy(result["behavior"]["windows"][0])
    other.update(interval=[1, 2], prediction=0, probabilities=[1, 0, 0, 0, 0, 0])
    result["behavior"]["windows"].append(other)
    return result


def test_legacy_is_byte_equivalent_and_wrapper_is_opt_in():
    result = mixed_result()

    class Pipeline:
        def predict_video(self, path):
            return result

    assert ReviewAwarePipeline(Pipeline()).predict_video(None) is result
    assert json.dumps(assess_review(result)) == json.dumps(result)


def test_consistency_abstains_without_mutating_or_suppressing_original_alert():
    result = mixed_result()
    original = copy.deepcopy(result)
    a1 = assess_review(result, ReviewPolicy("coverage_consistency"))
    a2 = assess_review(result, ReviewPolicy("coverage"))
    assert a1["review_assessment"]["reason"] == "cross_window_class_disagreement"
    assert a1["review_assessment"]["suggested_behavior_class"] is None
    assert a2["review_assessment"]["suggested_behavior_class"] == 4
    assert a1["final_alert"] == original["final_alert"]
    assert result == original
    assert a1 == assess_review(result, ReviewPolicy("coverage_consistency"))


@pytest.mark.parametrize("status", [None, "no_actors", "available"])
def test_unexamined_no_actors_and_short_single_window_are_distinct(status):
    result = record(
        "x", 0, 0.1 if status is None else 0.9, status, 0 if status == "available" else None
    )["result"]
    assessment = assess_review(result, ReviewPolicy("coverage_consistency"))["review_assessment"]
    assert (
        assessment["reason"]
        == {
            None: "not_routed_actors_unexamined",
            "no_actors": "no_actors",
            "available": "covered_suggestion_requires_human_review",
        }[status]
    )
    assert assessment["binary_violence_decision"] is None


def test_missing_one_window_never_implies_normal_behavior():
    result = mixed_result()
    result["behavior"]["windows"][0].update(
        status="no_actors", actor_count=0, probabilities=None, prediction=None
    )
    assert (
        assess_review(result, ReviewPolicy("coverage"))["review_assessment"]["reason"]
        == "partial_actor_coverage"
    )


@pytest.mark.parametrize(
    "problem", ["nan", "shape", "sum", "negative", "interval", "count", "status", "prediction"]
)
def test_review_contract_rejects_invalid_evidence(problem):
    result = mixed_result()
    window = result["behavior"]["windows"][0]
    if problem == "nan":
        window["probabilities"][0] = float("nan")
    elif problem == "shape":
        window["probabilities"] = [1, 0]
    elif problem == "sum":
        window["probabilities"] = [0.1] * 6
    elif problem == "negative":
        window["probabilities"][0] = -1
    elif problem == "interval":
        window["interval"] = [1, 1]
    elif problem == "count":
        window["actor_count"] = -1
    elif problem == "status":
        window["status"] = "unknown"
    else:
        window["prediction"] = 1
    with pytest.raises(ValueError):
        assess_review(result, ReviewPolicy("coverage_consistency"))


def comparison_fixture():
    rows = [record("a", 0, 0.1), record("b", 4, 0.9, "available", 4)]
    for r in rows:
        r.update(split="val", video_sha256="a" * 64)
    return rows, [{k: v for k, v in r.items() if k != "result"} for r in rows]


def test_matched_comparison_includes_abstentions_in_full_denominator():
    rows, manifest = comparison_fixture()
    report = compare_review_modes(rows, manifest, {"fixture": True})
    assert report["variants"]["coverage_consistency"]["coverage"] == 0.5
    assert report["variants"]["legacy"]["correct_fraction_all_clips"] == 0.5
    assert report["raw_clip_alert_all_modes"]["confusion_matrix"] == [[1, 0], [0, 1]]


@pytest.mark.parametrize(
    "problem", ["test", "source", "hash", "labels", "identity", "membership", "leakage"]
)
def test_comparison_rejects_protected_or_incompatible_populations(problem):
    rows, manifest = comparison_fixture()
    provenance = {"fixture": True}
    if problem == "test":
        rows[0]["split"] = "test"
    elif problem == "source":
        rows[0]["source_video_id"] = "other"
    elif problem == "hash":
        rows[0]["video_sha256"] = "b" * 64
    elif problem == "labels":
        rows[0]["group_label"] = 1
    elif problem == "identity":
        provenance = {"different": True}
    elif problem == "membership":
        rows = rows[:1]
    else:
        manifest.append({**manifest[0], "clip_id": "test-clip", "split": "test"})
    with pytest.raises(ValueError):
        compare_review_modes(rows, manifest, provenance)
