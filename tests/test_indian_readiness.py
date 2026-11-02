import csv
import json

import pytest
from test_indian_domain import make_video

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import import_annotations, parse_annotation_seconds
from surveillance.datasets.indian_readiness import (
    ACTUAL_HEADER,
    assess_annotation,
    audit_readiness,
    conservative_group,
    inspect_media,
    reuse_readiness,
)


@pytest.mark.parametrize(
    "raw,seconds",
    [("00:06", 6), ("01:02.5", 62.5), ("90:01", 5401), ("6.25", 6.25), (" 00:00 ", 0), (".5", 0.5)],
)
def test_decimal_or_mmss_even_when_header_says_sec(raw, seconds):
    assert parse_annotation_seconds(raw) == seconds


@pytest.mark.parametrize("raw", ["1:60", "1:2", "01:00:00", "-1", "NaN", "inf", "", "1,5"])
def test_reject_ambiguous_timestamps(raw):
    with pytest.raises(ValueError):
        parse_annotation_seconds(raw)


def row(**changes):
    return {
        **dict.fromkeys(ACTUAL_HEADER, ""),
        "clip_id": "fixture",
        "label": "violence",
        "duration_seconds": "5.767",
        "playback_status": "ok",
        "violence_start_sec": "00:00",
        "violence_end_sec": "00:06",
        "dataset_name": "HCI-User-Collection",
        **changes,
    }


def test_rounding_is_proposed_not_applied_and_uncertainty_preserved():
    raw = row()
    parsed = assess_annotation(raw)
    assert parsed["end_excess_seconds"] == pytest.approx(0.233)
    assert parsed["parsed_interval_seconds"] == [0, 6]
    assert parsed["rounding_proposal"]["applied"] is False
    assert raw["violence_end_sec"] == "00:06"
    uncertain = assess_annotation(row(label="not_sure", violence_end_sec="00:05"))
    assert uncertain["normalized_label"] == "not_sure"
    assert uncertain["parsed_interval_seconds"] == [0, 5]
    assert "uncertain_label" in uncertain["annotation_issues"]
    assert "interval_on_uncertain_or_nonviolent_label" in uncertain["annotation_issues"]


def test_filename_families_are_not_independent_sources():
    raw = row(relative_video_path="Snatch Theft/23_0.mp4", dataset_name="Snatch-1.0")
    source = conservative_group(raw, {}, {})
    assert source["filename_family_hint"] == "23"
    assert source["source_video_id"] is None and source["incident_id"] is None
    assert source["guard_group"] == "snatch-1.0-entire-collection-unresolved"


def test_importer_mmss_retains_original_sec_columns(tmp_path):
    make_video(tmp_path / "a.avi")
    csv_path = tmp_path / "a.csv"
    csv_path.write_text(
        "clip_id,file_path,label,violence_start_sec,violence_end_sec\na,a.avi,violence,00:00,00:00.5\n"
    )
    result = import_annotations(
        csv_path,
        tmp_path,
        tmp_path / "import",
        columns={
            "clip_id": "clip_id",
            "file_path": "file_path",
            "label": "label",
            "violence_start": "violence_start_sec",
            "violence_end": "violence_end_sec",
        },
    )
    assert result["accepted_binary_clips"] == 1
    saved = json.loads((tmp_path / "import/import.json").read_text())["records"][0]
    assert saved["temporal_annotations"] == [[0, 0.5]]
    assert saved["raw"]["violence_end_sec"] == "00:00.5"


def write_rows(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_inputs(tmp_path):
    make_video(tmp_path / "video.avi")
    annotation = row(
        relative_video_path="01_Videos/fixture.avi",
        duration_seconds="1",
        violence_end_sec="00:00.5",
        assigned_reviewer="reviewer_1",
    )
    csv_path = tmp_path / "combined.csv"
    write_rows(csv_path, [annotation])
    priority = tmp_path / "priority.csv"
    write_rows(priority, [], list(annotation) + ["issue_codes"])
    catalog = tmp_path / "catalog.csv"
    write_rows(
        catalog,
        [
            {
                "clip_id": "fixture",
                "dataset_name": "HCI-User-Collection",
                "planned_relative_video_path": "01_Videos/fixture.avi",
                "source_video_id": "",
                "parent_recording_id": "",
                "original_local_path": str(tmp_path / "video.avi"),
                "sha256": sha256(tmp_path / "video.avi"),
            }
        ],
    )
    sources = tmp_path / "sources.csv"
    write_rows(
        sources,
        [
            {
                "clip_id": "fixture",
                "source_identity_verified": "False",
                "municipal_reservation": "True",
            }
        ],
    )
    evidence, timing = tmp_path / "evidence.jsonl", tmp_path / "timing.jsonl"
    evidence.write_text("")
    timing.write_text("")
    return csv_path, priority, catalog, sources, evidence, timing, tmp_path / "dry-run"


def test_dry_run_reconciles_without_approving_or_making_manifest(tmp_path):
    args = fixture_inputs(tmp_path)
    before = {p: sha256(p) for p in args[:-1]}
    report = audit_readiness(*args, expected_inventory=2, workers=1)
    assert (
        report["annotations"]
        == report["media_readable"]
        == report["annotation_clean_candidates"]
        == 1
    )
    assert report["annotation_quarantined"] == 0
    assert report["training_eligible"] == 0 and not report["split_approved"]
    assert report["identifiable_missing_video_annotations"] == 0
    assert report["reference_count_difference"] == 1
    assert "unidentified_stated_inventory_records" not in report
    assert not (args[-1] / "manifest.jsonl").exists()
    assert before == {p: sha256(p) for p in args[:-1]}
    records = json.loads((args[-1] / "records.json").read_text())
    assert records[0]["raw_annotation"]["relative_video_path"] == "01_Videos/fixture.avi"
    assert "municipal_reservation" in records[0]["blockers"]
    with pytest.raises(FileExistsError):
        audit_readiness(*args)


def test_priority_conflict_blocks_without_outputs(tmp_path):
    args = fixture_inputs(tmp_path)
    write_rows(args[1], [row(clip_id="fixture", label="not_sure")])
    with pytest.raises(ValueError, match="conflicts"):
        audit_readiness(*args)
    assert not args[-1].exists()


def test_matching_decode_receipt_is_reused_only_for_unchanged_bytes(tmp_path):
    path = tmp_path / "video.avi"
    make_video(path)
    digest = sha256(path)
    catalog = {"original_local_path": str(path), "sha256": digest}
    prior = {
        "sha256": digest,
        "fps": 8,
        "strong_decode": {"status": "completed", "decoded_frames": 8},
    }
    result = inspect_media(catalog, prior, None)
    assert result["readable"] and result["decode_evidence"].startswith("reused_hash_bound")
    path.write_bytes(b"changed")
    result = inspect_media(catalog, prior, None)
    assert not result["readable"] and result["error"] == "catalog_content_changed"


def test_receipt_reuse_checks_media_and_durations_without_decoding(tmp_path, monkeypatch):
    args = fixture_inputs(tmp_path)
    report = audit_readiness(*args, workers=1)
    assert report["reference_video_count"] is None
    assert "stated_inventory_shortfall" not in report
    records = json.loads((args[-1] / "records.json").read_text())
    media = records[0]["media"]
    original = tmp_path / "original.csv"
    write_rows(
        original,
        [
            {
                "clip_id": "fixture",
                "file_path": media["resolved_local_path"],
                "sha256": media["sha256"],
                "duration_seconds": "1",
            }
        ],
    )
    monkeypatch.setattr(
        "surveillance.datasets.indian_readiness.inspect_media",
        lambda *a: pytest.fail("Receipt reuse must not re-decode"),
    )
    result = reuse_readiness(args[-1], args[0], original, tmp_path / "reused")
    assert result["quality_candidates"] == 1 and result["training_eligible"] == 0
    assert result["duration_seconds"]["annotation_and_media_candidates_presentation"] == "1"
    assert (tmp_path / "reused/source-provenance-queue.csv").is_file()
    assert not (tmp_path / "reused/manifest.jsonl").exists()
    from pathlib import Path

    Path(media["resolved_local_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Catalog/media changed"):
        reuse_readiness(args[-1], args[0], original, tmp_path / "reused2")
