import pytest

from surveillance.experiments.demo_examples import choose_demo_examples


def fixtures():
    rows, predictions = [], []
    for i in range(7):
        label = i if i < 6 else 0
        rows.append(
            {
                "clip_id": str(i),
                "source_video_id": str(i),
                "group_label": label,
                "split": "test",
                "num_frames": 60,
                "fps": 30.0,
            }
        )
        predictions.append(
            {
                "clip_id": str(i),
                "source_video_id": str(i),
                "target": label,
                "prediction": label if i < 6 else 4,
                "confidence": 0.99,
                "actor_count": 1,
            }
        )
    return rows, predictions


def test_examples_span_all_labels_and_include_a_disclosed_model_mistake():
    rows, predictions = fixtures()
    selected = choose_demo_examples(list(reversed(rows)), predictions)
    assert [r["group_label"] for r in selected[:6]] == list(range(6))
    assert selected[-1]["clip_id"] == "6"
    assert selected[-1]["demo_role"] == "known_behavior_baseline_mistake"
    assert all(r["split"] == "test" for r in selected)


@pytest.mark.parametrize("changed", ["split", "target", "source_video_id"])
def test_example_selection_rejects_nonheldout_or_inconsistent_prediction_identity(changed):
    rows, predictions = fixtures()
    if changed == "split":
        rows[0]["split"] = "train"
    else:
        predictions[0][changed] = "changed"
    with pytest.raises(ValueError, match="held-out"):
        choose_demo_examples(rows, predictions)
