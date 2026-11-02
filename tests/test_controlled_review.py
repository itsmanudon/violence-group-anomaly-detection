import copy

import pytest
from test_cascade_evaluation import record

from surveillance.evaluation.controlled_review import diagnose, eligible_validation


def eligibility_fixture():
    rows = [
        {
            "clip_id": "val",
            "source_video_id": "source-val",
            "split": "val",
            "selected": True,
            "sha256": "a" * 64,
        }
    ]
    receipt = {
        "source_assignments": {"source-val": "val"},
        "source_overlap": [],
        "duplicate_content_overlap": [],
    }
    protocol = {
        "source_assignments": {"source-val": "val"},
        "source_overlap": [],
        "cross_split_content_overlap": [],
    }
    return rows, receipt, protocol


def test_eligible_population_requires_validation_for_both_models():
    rows, receipt, protocol = eligibility_fixture()
    assert eligible_validation(rows, receipt, protocol) == rows


@pytest.mark.parametrize("problem", ["train", "test", "unknown", "alias", "hash", "human_source"])
def test_ineligible_or_leaking_validation_is_rejected(problem):
    rows, receipt, protocol = eligibility_fixture()
    if problem in {"train", "test", "unknown"}:
        protocol["source_assignments"]["source-val"] = problem
    elif problem == "alias":
        protocol["exact_content_duplicate_groups"] = [["source-val", "alias-train"]]
        protocol["source_assignments"]["alias-train"] = "train"
    elif problem == "hash":
        rows.append(
            {**rows[0], "clip_id": "test", "source_video_id": "source-test", "split": "test"}
        )
        receipt["source_assignments"]["source-test"] = "test"
    else:
        receipt["source_assignments"]["source-val"] = "train"
    with pytest.raises(ValueError):
        eligible_validation(rows, receipt, protocol)


def diagnostic_fixture():
    wrong = record("wrong", 0, 0.9, "available", 4)
    other = copy.deepcopy(wrong["result"]["behavior"]["windows"][0])
    other.update(prediction=0, probabilities=[1, 0, 0, 0, 0, 0], interval=[1, 2])
    wrong["result"]["behavior"]["windows"].append(other)
    right = record("right", 4, 0.9, "available", 4)
    other = copy.deepcopy(right["result"]["behavior"]["windows"][0])
    other.update(
        status="no_actors", actor_count=0, prediction=None, probabilities=None, interval=[1, 2]
    )
    right["result"]["behavior"]["windows"].append(other)
    rows = [wrong, right, record("miss", 3, 0.1), record("normal", 0, 0.1)]
    for i, r in enumerate(rows):
        r["source_video_id"] = f"source-{i}"
        r["result"]["alerts"] = []
    return rows


def test_diagnostic_abstentions_are_not_correct_negatives_or_resolved_alerts():
    rows = diagnostic_fixture()
    before = copy.deepcopy(rows)
    report, detail = diagnose(rows, bootstrap_repetitions=30, seed=7)
    a0 = report["modes"]["legacy"]
    a1 = report["modes"]["coverage_consistency"]
    assert a0["behavior_covered"] == 2 and a1["behavior_covered"] == 0
    assert a1["wrong_legacy_behavior_withheld"] == 1
    assert a1["correct_legacy_behavior_withheld"] == 1
    assert a1["binary_counts"] == dict(TN=1, FP=1, FN=1, TP=1)
    assert a1["binary_changed_decisions"] == a1["false_alerts_prevented"] == 0
    assert a1["binary_abstentions"] == 0
    assert a1["conditional_behavior_accuracy"] is None
    assert a1["true_violence_detections_lost"] is None
    assert len(detail) == 4 and rows == before
    assert report == diagnose(rows, bootstrap_repetitions=30, seed=7)[0]


def test_source_bootstrap_does_not_count_clip_as_independent_unit():
    rows = diagnostic_fixture()
    for row in rows:
        row["source_video_id"] = "same-original-recording"
    report, _ = diagnose(rows, bootstrap_repetitions=20)
    assert report["uncertainty"]["source_clusters"] == 1
    assert report["uncertainty"]["intervals"]["coverage_consistency_coverage_delta"][
        "percentile_95"
    ] == [-0.5, -0.5]


def test_coverage_only_ablation_removes_agreement_requirement():
    report, _ = diagnose(diagnostic_fixture(), bootstrap_repetitions=10)
    assert report["modes"]["coverage"]["behavior_covered"] == 1
    assert report["modes"]["coverage"]["wrong_legacy_behavior_withheld"] == 0
    assert report["modes"]["coverage"]["correct_legacy_behavior_withheld"] == 1
