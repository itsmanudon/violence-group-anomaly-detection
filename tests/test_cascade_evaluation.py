import pytest


def record(clip, label, score, status=None, prediction=None):
    probabilities = None if status != "available" else [float(i == prediction) for i in range(6)]
    return {
        "clip_id": clip,
        "source_video_id": "held-out-source",
        "group_label": label,
        "binary_label": int(label != 0),
        "result": {
            "schema_version": 1,
            "provenance": {"fixture": True},
            "policy": {"anomaly_threshold": 0.5, "max_windows": 3},
            "anomaly": {"overall_score": score},
            "final_alert": {
                "state": "no_anomaly" if score < 0.5 else "generic_anomaly",
                "timestamp_interval": None if score < 0.5 else [0, 1],
            },
            "behavior": {
                "windows": []
                if status is None
                else [
                    {
                        "status": status,
                        "actor_count": int(status == "available"),
                        "interval": [0, 1],
                        "probabilities": probabilities,
                        "prediction": prediction,
                        "model_disagreement": prediction == 0,
                    }
                ]
            },
            "timings": {"total_seconds": 1.0},
        },
    }


def test_cascade_accounts_for_skipped_and_no_actor_clips_without_normal_targets():
    from surveillance.evaluation.cascade import summarize_cascade

    rows = [
        record("normal", 0, 0.2),
        record("fighting", 3, 0.9, "available", 3),
        record("no-actors", 4, 0.8, "no_actors"),
        record("miss", 2, 0.1),
    ]
    summary = summarize_cascade(rows)
    assert summary["clip_alert"]["confusion_matrix"] == [[1, 0], [1, 2]]
    assert summary["clip_alert"]["accuracy"] == 0.75
    assert summary["coverage"] == {
        "total_clips": 4,
        "source_ids": 1,
        "routed_clips": 2,
        "bypassed_clips": 2,
        "covered_clips": 1,
        "no_actor_routed_clips": 1,
        "behavior_abstained_clips": 3,
        "coverage_overall": 0.25,
        "coverage_among_routed": 0.5,
        "analyzed_windows": 2,
        "no_actor_windows": 1,
    }
    assert summary["conditional_behavior"]["count"] == 1
    assert summary["conditional_behavior"]["accuracy"] == 1
    assert summary["temporal_localization_measured"] is False


def test_cascade_empty_actor_population_abstains_instead_of_fabricating_accuracy():
    from surveillance.evaluation.cascade import summarize_cascade

    summary = summarize_cascade([record("empty", 1, 0.9, "no_actors")])
    assert summary["conditional_behavior"] is None
    assert summary["coverage"]["behavior_abstained_clips"] == 1


@pytest.mark.parametrize("problem", ["duplicate", "policy", "provenance", "label"])
def test_cascade_summary_rejects_inconsistent_population_or_model_identity(problem):
    from surveillance.evaluation.cascade import summarize_cascade

    first, second = record("a", 0, 0.1), record("b", 1, 0.9, "no_actors")
    if problem == "duplicate":
        second["clip_id"] = "a"
    elif problem == "policy":
        second["result"]["policy"]["anomaly_threshold"] = 0.8
    elif problem == "provenance":
        second["result"]["provenance"] = {"different": True}
    else:
        second["binary_label"] = 0
    with pytest.raises(ValueError):
        summarize_cascade([first, second])


def test_cascade_population_requires_held_out_sources_in_both_branches():
    from surveillance.evaluation.cascade import held_out_cascade_rows

    rows = [
        {"clip_id": name, "source_video_id": name, "split": "test", "selected": True}
        for name in ["official", "excluded"]
    ]
    protocol = {
        "source_assignments": {"official": "test", "leaked": "train"},
        "excluded_author_train_held_out_dcsass": [{"source_video_id": "excluded"}],
    }
    assert [r["clip_id"] for r in held_out_cascade_rows(rows, protocol)] == ["excluded", "official"]
    with pytest.raises(ValueError, match="source"):
        held_out_cascade_rows(
            rows
            + [
                {
                    "clip_id": "leaked",
                    "source_video_id": "leaked",
                    "split": "test",
                    "selected": True,
                }
            ],
            protocol,
        )
    with pytest.raises(ValueError, match="source"):
        held_out_cascade_rows(
            rows
            + [
                {
                    "clip_id": "unknown",
                    "source_video_id": "unknown",
                    "split": "test",
                    "selected": True,
                }
            ],
            protocol,
        )
