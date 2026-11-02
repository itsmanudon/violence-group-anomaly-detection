import copy

import pytest
from test_indian_readiness import row

from surveillance.datasets.indian_readiness import assess_annotation
from surveillance.datasets.indian_review import (
    build_review_queue,
    fingerprint_evidence,
    timing_observations,
)


def candidate_record(label="violence", readable=True):
    raw = row(label=label, violence_start_sec="", violence_end_sec="")
    return {
        "clip_id": "fixture",
        "raw_annotation": raw,
        "media": {"readable": readable, "resolved_local_path": "local.mp4"},
        "source_identity": {"source_video_id": None, "filename_family_hint": "candidate"},
        "catalog_record": {},
        "guard_group": "unresolved-collection",
        "blockers": [],
    }


def test_positive_with_missing_provenance_is_reviewed_never_approved():
    record = candidate_record()
    before = copy.deepcopy(record)
    q = build_review_queue([record], [])[0]
    assert q["priority"] == "first"
    assert "candidate_positive_second_look" in q["flags"]
    assert not q["training_approved"] and not q["human_recording_id"]
    assert record == before


def test_multiple_flags_are_one_record_and_raw_uncertain_values_survive():
    record = candidate_record("not_sure", False)
    record["raw_annotation"].update(violence_start_sec="00:00", violence_end_sec="00:06")
    q = build_review_queue([record], [])[0]
    assert "uncertain_label" in q["flags"] and "strict_loader_media_hold" in q["flags"]
    assert q["raw_violence_end_sec"] == "00:06"
    assert q["quality_status"] == "quarantine"
    assert q["original_label"] == "not_sure"
    assert q["human_disposition"] == ""


def test_catalog_source_conflict_is_actionable_not_silently_merged():
    record = candidate_record("non_violence")
    record["raw_annotation"]["source_video_id"] = "claimed-A"
    record["catalog_record"]["source_video_id"] = "claimed-B"
    assert "source_conflict" in build_review_queue([record], [])[0]["flags"]


def test_duplicate_queue_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate records"):
        build_review_queue([candidate_record(), candidate_record()], [])


def test_ordered_fingerprints_distinguish_ambiguous_static_matches():
    values = ["0000000000000000", "ffffffffffffffff", "5555555555555555"]
    strong = fingerprint_evidence(values, values)
    assert strong["ordered_sample_matches"] == 3
    assert strong["observation"] == "temporal_visual_candidate"
    assert not strong["source_verified"] and not strong["incident_verified"]
    ambiguous = fingerprint_evidence(["0"] * 8, ["0"] * 8)
    assert ambiguous["observation"] == "ambiguous_visual_similarity"
    assert fingerprint_evidence(["0"] * 8, ["ffffffffffffffff"] * 8)["ordered_sample_matches"] == 0


def test_mmss_duration_is_parsed_consistently_and_invalid_duration_quarantined():
    result = assess_annotation(row(duration_seconds="00:05.767"))
    assert result["rounding_proposal"]["proposed_end_seconds"] == 5.767
    assert result["end_excess_seconds"] == pytest.approx(0.233)
    invalid = assess_annotation(row(duration_seconds="invalid"))
    assert "invalid_duration" in invalid["annotation_issues"]
    assert invalid["rounding_proposal"] is None


def test_presentation_and_frame_clocks_are_observations_not_binary_quarantine():
    record = candidate_record("non_violence")
    record.update(dataset_name="fixture")
    record["media"].update(fps=30, num_frames=300)
    report, queue = timing_observations(
        [record], {"fixture": {"fps": "30", "duration_seconds": "12"}}
    )
    assert report["flagged"] == 1 and report["fps_disagreements_over_1e_minus_6"] == 0
    assert queue[0]["signed_difference_seconds"] == 2
    assert queue[0]["training_or_temporal_approval"] is False
    assert build_review_queue([record], [])[0]["quality_status"] == "candidate_only_not_approved"
    same_clock, queue = timing_observations(
        [record], {"fixture": {"fps": "30", "duration_seconds": "10.001"}}
    )
    assert same_clock["flagged"] == 0 and queue == []
