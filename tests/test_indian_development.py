import copy
import json

import pytest
from test_indian_review import candidate_record

from surveillance.datasets.indian_development import (
    ATTESTATION,
    propose_development,
    provisional_groups,
)


def fixture(key, label="violence", guard="same-negative-recording", collection="fixture"):
    r = candidate_record(label)
    r.update(clip_id=key, dataset_name=collection, guard_group=guard)
    r["raw_annotation"].update(clip_id=key, dataset_name=collection)
    r["media"]["sha256"] = key * 64
    r["source_identity"]["filename_family_hint"] = None
    r["annotation"] = {"normalized_label": label, "annotation_issues": []}
    return r


def test_distinct_violence_is_attested_not_independently_source_verified():
    rows = [fixture("a"), fixture("b")]
    before = copy.deepcopy(rows)
    retained, groups, excluded = provisional_groups(rows, [])
    assert groups == [["a"], ["b"]] and not excluded
    assert rows == before and len(retained) == 2
    assert not ATTESTATION["independently_verified_source_metadata"]
    assert not ATTESTATION["research_permission_confirmed"]


def test_connected_negative_recordings_and_visual_links_never_cross_groups():
    rows = [
        fixture("a", "non_violence"),
        fixture("b", "non_violence"),
        fixture("c", "non_violence", guard="other"),
    ]
    _, groups, _ = provisional_groups(rows, [{"pair": ["b", "c"], "kind": "visual_candidate"}])
    assert groups == [["a", "b", "c"]]


def test_exact_duplicate_link_applies_to_positive_and_cross_collection_clips():
    rows = [fixture("a"), fixture("b", collection="other")]
    rows[1]["media"]["sha256"] = rows[0]["media"]["sha256"]
    assert provisional_groups(rows, [])[1] == [["a", "b"]]


def test_holds_and_reservations_survive_without_relabeling():
    hold = fixture("a")
    hold["annotation"]["annotation_issues"] = ["interval_end_after_video"]
    municipal = fixture("b")
    municipal["blockers"] = ["municipal_reservation"]
    negative = fixture("c", "non_violence", collection="Snatch-1.0")
    before = copy.deepcopy([hold, municipal, negative])
    retained, groups, excluded = provisional_groups([hold, municipal, negative], [])
    assert retained == [] and groups == []
    assert len(excluded) == 2 and "a" not in excluded  # still a quality hold
    assert [hold, municipal, negative] == before


def test_unknown_overlap_id_and_duplicate_ids_fail():
    with pytest.raises(ValueError):
        provisional_groups([fixture("a")], [{"pair": ["a", "unknown"]}])
    with pytest.raises(ValueError):
        provisional_groups([fixture("a"), fixture("a")], [])


def test_proposal_never_creates_a_trainable_manifest_or_permission_approval(tmp_path, monkeypatch):
    audit = tmp_path / "audit"
    audit.mkdir()
    records = [fixture(f"p{i}") for i in range(12)] + [
        fixture(f"n{i}", "non_violence", guard=f"recording-{i // 2}") for i in range(12)
    ]
    (audit / "records.json").write_text(json.dumps(records))
    (audit / "readiness.json").write_text(json.dumps({"duplicate_candidates": []}))
    (audit / "registration.json").write_text(json.dumps({"input_sha256": {}}))

    def synthetic_recheck(audit, annotations, catalog, output):
        output.mkdir(parents=True)
        (output / "registration.json").write_text('{"synthetic_fixture": true}')
        return {
            "quality_candidates": 24,
            "quality_quarantine": 0,
            "candidate_label_counts": {"violence": 12, "non_violence": 12},
        }

    monkeypatch.setattr(
        "surveillance.datasets.indian_development.reuse_readiness", synthetic_recheck
    )
    output = tmp_path / "proposal"
    summary = propose_development(
        audit, tmp_path / "unused.csv", tmp_path / "unused-catalog.csv", output
    )
    assert summary["training_approved"] == summary["evaluation_approved"] == 0
    assert summary["proposed_records"] == 24 and not summary["test_frozen"]
    assert not (output / "manifest.jsonl").exists()
    assert not (output / "split-receipt.json").exists()
    for group in summary["groups"]:
        assert len({summary["assignments"][key] for key in group}) == 1
    assert "unverified" in (output / "permission-evidence-queue.csv").read_text()
