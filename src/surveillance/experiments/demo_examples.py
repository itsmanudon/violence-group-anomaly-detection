"""Transparent held-out presentation examples; never used for model selection."""


def choose_demo_examples(rows: list[dict], predictions: list[dict]) -> list[dict]:
    held_out = {row["clip_id"]: row for row in rows if row["split"] == "test"}
    by_id = {}
    for prediction in predictions:
        row = held_out.get(prediction["clip_id"])
        if (
            row is None
            or prediction["clip_id"] in by_id
            or row["source_video_id"] != prediction["source_video_id"]
            or row["group_label"] != prediction["target"]
        ):
            raise ValueError("Demo predictions do not match the frozen held-out population")
        by_id[prediction["clip_id"]] = prediction
    eligible = sorted(
        [
            row
            for row in held_out.values()
            if row["clip_id"] in by_id
            and by_id[row["clip_id"]]["actor_count"] > 0
            and row["num_frames"] / row["fps"] >= 2
        ],
        key=lambda row: row["clip_id"],
    )
    selected = []
    for label in range(6):
        row = next((row for row in eligible if row["group_label"] == label), None)
        if row is None:
            raise ValueError(f"No readable-length covered held-out example for class {label}")
        selected.append(
            {
                **row,
                "demo_role": "first_covered_example_for_label",
                "baseline_prediction": by_id[row["clip_id"]],
            }
        )
    selected_ids = {row["clip_id"] for row in selected}
    mistakes = sorted(
        [
            p
            for p in predictions
            if p["target"] != p["prediction"] and p["clip_id"] not in selected_ids
        ],
        key=lambda p: (-p["confidence"], p["clip_id"]),
    )
    if not mistakes:
        raise ValueError("No separate held-out mistake is available for limitation discussion")
    mistake = mistakes[0]
    selected.append(
        {
            **held_out[mistake["clip_id"]],
            "demo_role": "known_behavior_baseline_mistake",
            "baseline_prediction": mistake,
        }
    )
    return selected
