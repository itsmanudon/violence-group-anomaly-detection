from surveillance.experiments.sultani_dcsass import generic_source_assignments


def test_generic_anomaly_protocol_preserves_actor_splits_and_reserves_ucf_sources():
    clips = [
        {"source_video_id": "fighting002", "category": "Fighting", "binary_label": 1},
        {"source_video_id": "fighting005", "category": "Fighting", "binary_label": 0},
        *[
            {"source_video_id": f"arson{i:03}", "category": "Arson", "binary_label": i % 2}
            for i in range(10)
        ],
        {"source_video_id": "unknown", "category": "Abuse", "binary_label": None},
    ]
    actor = {"fighting002": "val", "fighting005": "train"}
    assignments = generic_source_assignments(clips, actor, {"arson003"})
    assert assignments["fighting002"] == "val"
    assert assignments["fighting005"] == "train"
    assert assignments["arson003"] == "test"
    assert "unknown" not in assignments
    assert set(assignments.values()) == {"train", "val", "test"}
