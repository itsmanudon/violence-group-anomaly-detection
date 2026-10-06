from pathlib import Path

import pytest

from surveillance.datasets.dcsass_protocol import (
    behavior_label,
    read_raw_labels,
    split_sources,
)
from surveillance.datasets.preparation import source_identity


@pytest.mark.parametrize(
    "category,positive",
    [("Abuse", 1), ("Assault", 2), ("Fighting", 3), ("Robbery", 4), ("Vandalism", 5)],
)
def test_binary_normal_overrides_selected_category(category, positive):
    assert behavior_label(category, 0) == 0
    assert behavior_label(category, 1) == positive


@pytest.mark.parametrize(
    "category",
    [
        "Arrest",
        "Arson",
        "RoadAccidents",
        "Accident",
        "Burglary",
        "Explosion",
        "Shooting",
        "Stealing",
        "Shoplifting",
    ],
)
def test_excluded_categories_do_not_enter_actor_population_even_if_normal(category):
    assert behavior_label(category, 0) is None
    assert behavior_label(category, 1) is None


def test_unknown_category_and_nonbinary_label_fail():
    with pytest.raises(ValueError):
        behavior_label("invented", 0)
    with pytest.raises(ValueError):
        behavior_label("Fighting", 2)


def test_headerless_labels_report_duplicates_missing_values_without_relabeling(tmp_path):
    labels = tmp_path / "Labels"
    labels.mkdir()
    (labels / "Abuse.csv").write_text(
        "Abuse001_x264_0,Abuse,0\nAbuse001_x264_0,Abuse,0\n"
        "Abuse001_x264_1,Abuse,\nAbuse001_x264_2,Abuse,0\nAbuse001_x264_2,Abuse,1\n"
    )
    accepted, issues = read_raw_labels(labels)
    assert accepted == {"Abuse001_x264_0": {"category": "Abuse", "label": 0}}
    assert len(issues["duplicate_rows"]) == 2
    assert len(issues["invalid_rows"]) == 1
    assert issues["conflicting_clip_ids"] == ["Abuse001_x264_2"]


def test_group_split_is_order_independent_and_reserves_ucf_test_sources():
    sources = {f"fighting{i:03}": "Fighting" for i in range(20)}
    sources.update({f"abuse{i:03}": "Abuse" for i in range(20)})
    reserved = {"fighting003", "abuse004"}
    split = split_sources(sources, reserved, seed=0)
    assert split == split_sources(dict(reversed(list(sources.items()))), reserved, seed=0)
    assert split["fighting003"] == split["abuse004"] == "test"
    assert set(split.values()) == {"train", "val", "test"}
    assert len(split) == len(sources)
    assert sum(v == "train" for v in split.values()) == 28


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Fighting002_x264_31.mp4", "fighting002"),
        ("Abuse048_x264_0.mp4", "abuse048"),
        ("RoadAccidents001_x264_3.mp4", "roadaccidents001"),
    ],
)
def test_actual_dcsass_source_filename_examples(filename, expected):
    assert source_identity(Path(filename), {}) == expected


def test_split_builder_checks_labels_and_duplicate_content_leakage():
    from surveillance.datasets.dcsass_protocol import build_human_split

    clips = [
        {
            "clip_id": f"Fighting{i:03}_x264_0",
            "source_video_id": f"fighting{i:03}",
            "category": "Fighting",
            "binary_label": 1,
            "group_label": 3,
            "selected": True,
            "sha256": str(i),
        }
        for i in range(10)
    ]
    audit = {"bounded_preflight": False, "clips": clips}
    rows, report = build_human_split(audit, {"fighting001"})
    assert next(r for r in rows if r["source_video_id"] == "fighting001")["split"] == "test"
    assert report["source_overlap"] == []
    assert report["clip_counts"]["train"] == 7
    clips[0]["group_label"] = 0
    with pytest.raises(ValueError, match="label"):
        build_human_split(audit, set())
    clips[0]["group_label"] = 3
    for row in clips:
        row["sha256"] = "same-content"
    with pytest.raises(ValueError, match="content"):
        build_human_split(audit, set())


def test_bounded_audit_cannot_freeze_a_research_split():
    from surveillance.datasets.dcsass_protocol import build_human_split

    with pytest.raises(ValueError, match="bounded"):
        build_human_split({"bounded_preflight": True, "clips": []}, set())


def test_class_aware_split_repairs_normal_only_anomaly_source_in_validation():
    from surveillance.datasets.dcsass_protocol import build_human_split

    sources = {f"fighting{i:03}": "Fighting" for i in range(8)}
    assignment = split_sources(sources, {"fighting001"}, seed=0)
    val_source = next(s for s, split in assignment.items() if split == "val")
    clips = [
        {
            "clip_id": source,
            "source_video_id": source,
            "category": "Fighting",
            "binary_label": int(source != val_source),
            "group_label": 0 if source == val_source else 3,
            "selected": True,
            "sha256": source,
        }
        for source in sources
    ]
    rows, report = build_human_split({"bounded_preflight": False, "clips": clips}, {"fighting001"})
    assert report["class_counts"]["val"]["Fighting"] > 0
    assert report["class_counts"]["train"]["Fighting"] > 0
    assert next(r for r in rows if r["source_video_id"] == "fighting001")["split"] == "test"
