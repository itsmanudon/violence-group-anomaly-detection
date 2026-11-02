"""Read-only annotation/catalog reconciliation; never approves or freezes a split."""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import cv2

from surveillance.datasets.dcsass_audit import audit_video, sha256
from surveillance.datasets.indian_domain import (
    duplicate_candidates,
    inspect_header,
    normalize_label,
    parse_annotation_seconds,
    visual_signature,
    write_new_json,
)

ACTUAL_HEADER = {
    "clip_id",
    "dataset_name",
    "relative_video_path",
    "video_url",
    "source_video_id",
    "duration_seconds",
    "media_status",
    "assigned_reviewer",
    "label",
    "playback_status",
    "violence_start_sec",
    "violence_end_sec",
    "reviewer_notes",
    "reviewed_at",
}


def read_csv(path: Path) -> list[dict]:
    inspect_header(path)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row width: {path}")
    return rows


def unique_index(rows: list[dict], name: str) -> dict:
    result = {}
    for row in rows:
        key = row.get("clip_id", "").strip()
        if not key or key in result:
            raise ValueError(f"Missing or duplicate clip_id in {name}: {key!r}")
        result[key] = row
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assess_annotation(raw: dict) -> dict:
    """Return derived values/issues without editing labels, statuses or original intervals."""
    issues, interval = [], None
    try:
        label = normalize_label(raw["label"])
        if label == "not_sure":
            issues.append("uncertain_label")
    except ValueError:
        label = None
        issues.append("missing_label" if not raw["label"].strip() else "unsupported_label")
    if raw["playback_status"].strip().casefold() != "ok":
        issues.append("playback_not_ok")
    try:
        duration = parse_annotation_seconds(raw["duration_seconds"])
        if duration <= 0:
            raise ValueError("nonpositive duration")
    except ValueError:
        duration = None
        issues.append("invalid_duration")
    start, end = raw["violence_start_sec"].strip(), raw["violence_end_sec"].strip()
    if start or end:
        try:
            if not start or not end:
                raise ValueError("partial interval")
            a, b = parse_annotation_seconds(start), parse_annotation_seconds(end)
            interval = [a, b]
            if a >= b:
                issues.append("invalid_interval_order")
            if duration is not None and b > duration:
                issues.append("interval_end_after_video")
            if duration is not None and a >= duration:
                issues.append("interval_start_after_video")
            if label != "violence":
                issues.append("interval_on_uncertain_or_nonviolent_label")
        except ValueError:
            issues.append("invalid_or_partial_time")
    excess = max(0.0, interval[1] - duration) if interval and duration is not None else None
    return {
        "normalized_label": label,
        "parsed_interval_seconds": interval,
        "annotation_issues": issues,
        "end_excess_seconds": excess,
        "rounding_proposal": {
            "original_end": raw["violence_end_sec"],
            "proposed_end_seconds": duration,
            "applied": False,
            "decision": "quarantine_pending_explicit_adjudication",
        }
        if excess and excess <= 0.5
        else None,
    }


def conservative_group(raw: dict, catalog: dict, source_row: dict) -> dict:
    """Retain real catalog identities; derive only explicitly provisional guard groups."""
    dataset = raw["dataset_name"]
    supplied_source = catalog.get("source_video_id") or catalog.get("parent_recording_id")
    source_verified = source_row.get("source_identity_verified", "").casefold() == "true"
    if source_verified and supplied_source:
        return {
            "source_video_id": supplied_source,
            "incident_id": None,
            "identity_status": "recording_verified_incident_unresolved",
            "guard_group": f"recording:{supplied_source}",
            "filename_family_hint": None,
        }
    # Collection-level grouping deliberately avoids invented independent clip sources.
    if dataset == "WalkIndia-50":
        group = source_row.get("conservative_leakage_group_id")
        if not group or "tour-" not in group:
            group = "walkindia-entire-release-unresolved"
    else:
        group = f"{dataset.casefold()}-entire-collection-unresolved"
    stem = Path(raw["relative_video_path"]).stem
    family = None
    if dataset == "Snatch-1.0" and re.fullmatch(r"(?:s_)?\d+(?:_\d+)?", stem):
        family = re.sub(r"_\d+$", "", stem) if not stem.startswith("s_") else stem
    elif dataset == "HCI-User-Collection" and re.fullmatch(r"krishna-clip-\d+", stem):
        family = "krishna-clip-series"
    return {
        "source_video_id": None,
        "incident_id": None,
        "identity_status": "unresolved_original_recording_and_incident",
        "guard_group": group,
        "catalog_guard_group": source_row.get("conservative_leakage_group_id"),
        "filename_family_hint": family,
    }


def inspect_media(catalog_row: dict, prior: dict | None, timing: dict | None) -> dict:
    """Hash current bytes, reuse matching complete-decode evidence, sample current frames.

    Full decode is performed only where no qualifying hash-bound prior count
    receipt exists. A sample check is never described as a fresh full decode.
    """
    path = Path(catalog_row["original_local_path"])
    result = {"resolved_local_path": str(path), "exists": path.is_file(), "readable": False}
    if not result["exists"]:
        return {**result, "error": "missing_local_video"}
    try:
        digest = sha256(path)
        result.update(sha256=digest, catalog_hash_matches=digest == catalog_row["sha256"])
        if not result["catalog_hash_matches"]:
            return {**result, "error": "catalog_content_changed"}
        reusable = None
        if prior and prior.get("sha256") == digest:
            strong = prior.get("strong_decode") or {}
            if strong.get("status") == "completed" and strong.get("decoded_frames", 0) > 0:
                reusable = {
                    "num_frames": strong["decoded_frames"],
                    "fps": prior["fps"],
                    "decode_evidence": "reused_hash_bound_catalog_complete_decode",
                }
        if timing and timing.get("sha256") == digest:
            if (
                timing.get("complete_original_sha256_matches") is True
                and timing.get("sequential_decoded_frames", 0) > 0
            ):
                reusable = {
                    "num_frames": timing["sequential_decoded_frames"],
                    "fps": prior.get("fps") if prior else None,
                    "decode_evidence": "reused_hash_bound_phase2a_sequential_decode",
                    "timing_classification": timing.get("classification"),
                    "frame_identity_mapping_verified": timing.get(
                        "decoded_frame_identity_mapping_verified"
                    ),
                }
        if reusable is None:
            decoded = audit_video(path)
            result.update(decoded)
            if decoded["sha256"] != digest or not decoded["readable"]:
                return {**result, "error": decoded.get("error", "decode_failed_or_changed")}
            result["decode_evidence"] = "fresh_full_decode"
        else:
            result.update(reusable)
        result["visual_signature"] = visual_signature(path, result["num_frames"])
        if sha256(path) != digest:
            return {**result, "readable": False, "error": "media_changed_during_check"}
        result["readable"] = True
    except (ValueError, OSError, cv2.error) as error:
        result.update(error=str(error), readable=False)
    return result


def summarize_sources(records: list[dict], key: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        grouped[row[key]].append(row)
    summary = []
    for source, group in sorted(grouped.items()):
        counts = Counter(row["annotation"]["normalized_label"] or "missing" for row in group)
        summary.append(
            {
                "source": source,
                "clips": len(group),
                **{
                    label: counts[label]
                    for label in ("violence", "non_violence", "not_sure", "missing")
                },
                "annotation_clean": sum(not r["annotation"]["annotation_issues"] for r in group),
                "readable": sum(r["media"]["readable"] for r in group),
                "positive_fraction_of_binary": counts["violence"]
                / (counts["violence"] + counts["non_violence"])
                if counts["violence"] + counts["non_violence"]
                else None,
            }
        )
    return summary


def audit_readiness(
    annotations: Path,
    priority: Path,
    catalog: Path,
    source_groups: Path,
    evidence_catalog: Path,
    timing_evidence: Path,
    output: Path,
    *,
    expected_inventory: int | None = None,
    workers: int = 2,
    historical_manifests: list[Path] | None = None,
    reviewer_files: list[Path] | None = None,
) -> dict:
    """Dry-run only: produce auditable receipts; never emits a training manifest."""
    if workers < 1 or workers > 4:
        raise ValueError("Use one to four bounded decode workers")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite readiness audit: {output}")
    inputs = [
        Path(p)
        for p in [annotations, priority, catalog, source_groups, evidence_catalog, timing_evidence]
    ]
    inputs += list(historical_manifests or []) + list(reviewer_files or [])
    input_hashes = {str(p.resolve()): sha256(p) for p in inputs}
    if not ACTUAL_HEADER <= set(inspect_header(annotations)):
        raise ValueError(
            "Unexpected combined annotation CSV header; inspect/mapping review required"
        )
    raw_rows, priority_rows = read_csv(annotations), read_csv(priority)
    raw_by_id, priority_by_id = (
        unique_index(raw_rows, "annotations"),
        unique_index(priority_rows, "priority"),
    )
    catalog_by_id = unique_index(read_csv(catalog), "catalog")
    guards = unique_index(read_csv(source_groups), "source groups")
    evidence = unique_index(
        [
            json.loads(line)
            for line in Path(evidence_catalog).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "prior evidence",
    )
    timing = unique_index(
        [
            json.loads(line)
            for line in Path(timing_evidence).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "prior timing evidence",
    )
    if set(raw_by_id) - catalog_by_id.keys():
        raise ValueError("Annotations reference clip IDs absent from actual catalog")
    if set(priority_by_id) - raw_by_id.keys():
        raise ValueError("Priority records reference IDs absent from combined annotations")
    for key, raw in raw_by_id.items():
        cat = catalog_by_id[key]
        if (
            raw["relative_video_path"] != cat["planned_relative_video_path"]
            or raw["dataset_name"] != cat["dataset_name"]
        ):
            raise ValueError(f"Catalog path/dataset identity conflict: {key}")
    for key, row in priority_by_id.items():
        if any(row[field] != raw_by_id[key][field] for field in ACTUAL_HEADER):
            raise ValueError(f"Priority CSV conflicts with combined annotations: {key}")
    provenance = defaultdict(list)
    for path in reviewer_files or []:
        for line, row in enumerate(read_csv(path), 2):
            key = row.get("clip_id")
            if key in raw_by_id:
                provenance[key].append(
                    {
                        "path": str(Path(path).resolve()),
                        "csv_row": line,
                        "matching_combined_fields": all(
                            row.get(k) == raw_by_id[key][k] for k in ACTUAL_HEADER
                        ),
                    }
                )
    local_rows = [catalog_by_id[raw["clip_id"]] for raw in raw_rows]
    print(
        f"Dry run: reconcile {len(raw_rows)} annotations; checking current media bytes/decoding",
        flush=True,
    )
    media = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            lambda r: inspect_media(r, evidence.get(r["clip_id"]), timing.get(r["clip_id"])),
            local_rows,
        )
        for index, row in enumerate(results, 1):
            media.append(row)
            if index % 50 == 0 or index == len(local_rows):
                print(f"Media checked {index}/{len(local_rows)}", flush=True)
    records = []
    for line, (raw, media_row) in enumerate(zip(raw_rows, media, strict=True), 2):
        key = raw["clip_id"]
        source = conservative_group(raw, catalog_by_id[key], guards.get(key, {}))
        annotation = assess_annotation(raw)
        issues = annotation["annotation_issues"]
        records.append(
            {
                "clip_id": key,
                "dataset_name": raw["dataset_name"],
                "guard_group": source["guard_group"],
                "csv_row": line,
                "raw_annotation": raw,
                "annotation": annotation,
                "media": media_row,
                "source_identity": source,
                "catalog_record": catalog_by_id[key],
                "literal_package_path_exists": (
                    Path(catalog).resolve().parent.parent / raw["relative_video_path"]
                ).is_file(),
                "catalog_source_group": guards.get(key, {}),
                "reviewer_csv_provenance": provenance.get(key, []),
                "priority_original": priority_by_id.get(key),
                "annotation_disposition": "quarantine" if issues else "binary_candidate_only",
                "training_eligible": False,
                "evaluation_eligible": False,
                "blockers": [
                    "source_incident_independence_unresolved",
                    "benchmark_overlap_not_cleared",
                    "no_split_approved",
                ]
                + (["annotation_priority_quarantine"] if issues else [])
                + (["human_review_timestamp_missing"] if not raw["reviewed_at"].strip() else [])
                + (["media_check_failed"] if not media_row["readable"] else [])
                + (
                    ["municipal_reservation"]
                    if guards.get(key, {}).get("municipal_reservation", "").casefold() == "true"
                    else []
                ),
            }
        )
    screen = [
        {
            "clip_id": r["clip_id"],
            "sha256": r["media"]["sha256"],
            "visual_signature": r["media"]["visual_signature"],
        }
        for r in records
        if r["media"]["readable"]
    ]
    candidates = duplicate_candidates(screen)
    historical_hashes, historical_sources, historical_inputs = set(), set(), []
    for path in historical_manifests or []:
        count = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if row.get("sha256"):
                historical_hashes.add(row["sha256"])
            if row.get("source_video_id"):
                historical_sources.add(row["source_video_id"].casefold())
        historical_inputs.append(
            {
                "path": str(Path(path).resolve()),
                "rows": count,
                "sha256": input_hashes[str(Path(path).resolve())],
            }
        )
    historical_hits = [
        {"clip_id": r["clip_id"], "sha256": r["media"].get("sha256")}
        for r in records
        if r["media"].get("sha256") in historical_hashes
    ]
    inventory_video_ids = {
        key for key, row in catalog_by_id.items() if row["planned_relative_video_path"]
    }
    missing_rows = [
        row for key, row in catalog_by_id.items() if key in inventory_video_ids - raw_by_id.keys()
    ]
    exclusions = [row for key, row in catalog_by_id.items() if key not in inventory_video_ids]
    clean = [r for r in records if not r["annotation"]["annotation_issues"]]
    quarantined = [r for r in records if r["annotation"]["annotation_issues"]]
    derived_priorities = {r["clip_id"] for r in quarantined}
    physical_inventory = []
    for dataset in sorted({r["dataset_name"] for r in catalog_by_id.values()}):
        group = [r for r in catalog_by_id.values() if r["dataset_name"] == dataset]
        paths = {Path(r["original_local_path"]).resolve() for r in group}
        root = Path(os.path.commonpath([str(p.parent) for p in paths]))
        found = {
            p.resolve()
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.casefold() in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
        }
        physical_inventory.append(
            {
                "dataset": dataset,
                "scope_root": str(root),
                "catalog_media": len(paths),
                "physical_media": len(found),
                "uncatalogued_media": sorted(str(p) for p in found - paths),
                "missing_catalogued_media": sorted(str(p) for p in paths - found),
            }
        )
    report = {
        "schema_version": 1,
        "mode": "dry_run_no_training_or_split_approval",
        "annotations": len(raw_rows),
        "catalog_media": len(catalog_by_id),
        "catalog_video_records": len(inventory_video_ids),
        "audio_or_nonvideo_exclusions": len(exclusions),
        "reference_video_count": expected_inventory,
        "reference_count_difference": expected_inventory - len(inventory_video_ids)
        if expected_inventory is not None
        else None,
        "identifiable_missing_video_annotations": len(missing_rows),
        "label_counts": dict(Counter(r["label"] or "missing" for r in raw_rows)),
        "reviewer_counts": dict(Counter(r["assigned_reviewer"] for r in raw_rows)),
        "original_media_status_counts": dict(Counter(r["media_status"] for r in raw_rows)),
        "playback_status_counts": dict(Counter(r["playback_status"] for r in raw_rows)),
        "blank_reviewed_at": sum(not r["reviewed_at"].strip() for r in raw_rows),
        "blank_source_video_id": sum(not r["source_video_id"].strip() for r in raw_rows),
        "blank_video_url": sum(not r["video_url"].strip() for r in raw_rows),
        "media_exists": sum(m["exists"] for m in media),
        "literal_package_paths_exist": sum(r["literal_package_path_exists"] for r in records),
        "physical_inventory_by_catalog_scope": physical_inventory,
        "media_readable": sum(m["readable"] for m in media),
        "decode_evidence_counts": dict(Counter(m.get("decode_evidence", "failed") for m in media)),
        "media_failures": [
            {"clip_id": r["clip_id"], "error": r["media"].get("error")}
            for r in records
            if not r["media"]["readable"]
        ],
        "annotation_clean_candidates": len(clean),
        "annotation_clean_label_counts": dict(
            Counter(r["annotation"]["normalized_label"] for r in clean)
        ),
        "annotation_quarantined": len(quarantined),
        "quarantine_issues": dict(
            Counter(issue for r in quarantined for issue in r["annotation"]["annotation_issues"])
        ),
        "priority_supplied": len(priority_rows),
        "priority_matches_derived_issues": set(priority_by_id) == derived_priorities,
        "priority_derived_not_supplied": sorted(derived_priorities - priority_by_id.keys()),
        "training_eligible": 0,
        "protected_evaluation_eligible": 0,
        "verified_independent_source_ids": len(
            {
                r["source_identity"]["source_video_id"]
                for r in records
                if r["source_identity"]["source_video_id"]
            }
        ),
        "split_approved": False,
        "dataset_source_counts": summarize_sources(records, "dataset_name"),
        "conservative_guard_counts": summarize_sources(records, "guard_group"),
        "exact_duplicate_pairs": sum(c["kind"] == "exact" for c in candidates),
        "near_duplicate_candidate_pairs": sum(c["kind"] != "exact" for c in candidates),
        "duplicate_candidates": candidates,
        "historical_screening": {
            "inputs": historical_inputs,
            "available_video_hashes": len(historical_hashes),
            "source_ids": len(historical_sources),
            "exact_content_hits": historical_hits,
            "source_and_visual_overlap": "unresolved_no_identity_clearance",
        },
        "originals_changed": False,
        "rounding_adjustments_applied": 0,
    }
    if any(sha256(Path(path)) != digest for path, digest in input_hashes.items()):
        raise ValueError("Readiness input changed while auditing")
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "readiness.json", report)
    write_new_json(output / "records.json", records)
    write_new_json(output / "priority-quarantine.json", quarantined)
    write_csv(
        output / "missing-annotations.csv", missing_rows, list(next(iter(catalog_by_id.values())))
    )
    write_csv(
        output / "nonvideo-exclusions.csv", exclusions, list(next(iter(catalog_by_id.values())))
    )
    write_csv(
        output / "source-label-counts.csv",
        report["dataset_source_counts"],
        [
            "source",
            "clips",
            "violence",
            "non_violence",
            "not_sure",
            "missing",
            "annotation_clean",
            "readable",
            "positive_fraction_of_binary",
        ],
    )
    write_new_json(
        output / "registration.json",
        {
            "input_sha256": input_hashes,
            "output_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
            "code_sha256": {
                Path(__file__).name: sha256(Path(__file__)),
                "indian_domain.py": sha256(Path(__file__).with_name("indian_domain.py")),
            },
        },
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "annotations",
                    "media_readable",
                    "annotation_clean_candidates",
                    "annotation_quarantined",
                    "training_eligible",
                    "near_duplicate_candidate_pairs",
                ]
            }
        ),
        flush=True,
    )
    return report


def reuse_readiness(audit: Path, annotations: Path, original_catalog: Path, output: Path) -> dict:
    """Revalidate bytes and reuse prior decoding; no footage rescan or split approval."""
    audit, output = Path(audit), Path(output)
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite readiness continuation: {output}")
    receipt = json.loads((audit / "registration.json").read_text(encoding="utf-8"))
    for name, digest in receipt["output_sha256"].items():
        if sha256(audit / name) != digest:
            raise ValueError(f"Prior readiness output changed: {name}")
    for path, digest in receipt["input_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Prior readiness input changed: {path}")
    records = json.loads((audit / "records.json").read_text(encoding="utf-8"))
    old_report = json.loads((audit / "readiness.json").read_text(encoding="utf-8"))
    current = unique_index(read_csv(annotations), "current workspace annotations")
    inventory = unique_index(read_csv(original_catalog), "original catalog")
    if set(current) != {r["clip_id"] for r in records}:
        raise ValueError("Annotation membership changed; run the affected ingestion checks")
    for row in records:
        key = row["clip_id"]
        if current[key] != row["raw_annotation"]:
            raise ValueError(f"Annotation changed since prior readiness: {key}")
        source = inventory.get(key, {})
        media = Path(row["media"]["resolved_local_path"])
        if (
            source.get("sha256") != row["media"].get("sha256")
            or Path(source.get("file_path", "")).resolve() != media.resolve()
            or sha256(media) != row["media"].get("sha256")
        ):
            raise ValueError(f"Catalog/media changed since prior readiness: {key}")
        if Decimal(source["duration_seconds"]) != Decimal(current[key]["duration_seconds"]):
            raise ValueError(f"Presentation duration conflicts with original catalog: {key}")
    quality = [
        r for r in records if r["annotation"]["annotation_issues"] or not r["media"]["readable"]
    ]
    candidates = [r for r in records if r not in quality]

    def seconds(rows):
        return str(
            sum((Decimal(r["raw_annotation"]["duration_seconds"]) for r in rows), Decimal(0))
        )

    exclusions = [r for key, r in inventory.items() if key not in current]
    prior_exclusions = unique_index(read_csv(audit / "nonvideo-exclusions.csv"), "prior exclusions")
    if {r["clip_id"] for r in exclusions} != set(prior_exclusions):
        raise ValueError(
            "Original inventory exclusions changed; do not infer new audio-only records"
        )
    by_reviewer = defaultdict(Counter)
    for raw in current.values():
        by_reviewer[raw["assigned_reviewer"]][raw["label"] or "missing"] += 1
    summary = {
        "mode": "receipt_reuse_current_hashes_no_new_decoding",
        "catalog_path": str(Path(original_catalog).resolve()),
        "catalog_media": len(inventory),
        "catalog_video_records": old_report["catalog_video_records"],
        "audio_only_exclusions": len(exclusions),
        "annotated_videos": len(current),
        "missing_video_annotations": old_report["identifiable_missing_video_annotations"],
        "current_media_hash_matches": len(records),
        "strict_loader_qualified": sum(r["media"]["readable"] for r in records),
        "quality_candidates": len(candidates),
        "quality_quarantine": len(quality),
        "candidate_label_counts": dict(
            Counter(r["annotation"]["normalized_label"] for r in candidates)
        ),
        "label_counts": old_report["label_counts"],
        "reviewer_label_counts": dict(by_reviewer),
        "duration_seconds": {
            "catalog_presentation_all_videos": seconds(records),
            "strict_media_qualified_presentation": seconds(
                [r for r in records if r["media"]["readable"]]
            ),
            "annotation_and_media_candidates_presentation": seconds(candidates),
            "definition": "sum of catalog presentation durations, not continuous camera-hours",
        },
        "training_eligible": 0,
        "protected_evaluation_eligible": 0,
        "split_approved": False,
        "unassigned": len(current),
        "source_guards": old_report["conservative_guard_counts"],
        "near_duplicate_candidates": old_report["duplicate_candidates"],
        "blockers": [
            "original_recording_and_incident_ids_unresolved",
            "historical_and_visual_overlap_review_pending",
            "annotation_adjudication_pending",
        ],
        "prior_registration_sha256": sha256(audit / "registration.json"),
        "workspace_annotations_sha256": sha256(annotations),
        "original_catalog_sha256": sha256(original_catalog),
    }
    queue = []
    for row in quality:
        queue.append(
            {
                **row["raw_annotation"],
                "resolved_local_path": row["media"]["resolved_local_path"],
                "issues": ";".join(
                    row["annotation"]["annotation_issues"]
                    + (
                        []
                        if row["media"]["readable"]
                        else [row["media"].get("error", "media_hold")]
                    )
                ),
                "proposed_adjustment_applied": False,
                "adjudicator": "",
                "adjudicated_at": "",
                "decision_evidence": "",
            }
        )
    source_queue = [
        {
            "clip_id": r["clip_id"],
            "relative_video_path": r["raw_annotation"]["relative_video_path"],
            "conservative_guard": r["guard_group"],
            "original_recording_id": "",
            "incident_id": "",
            "source_url": "",
            "evidence": "",
            "training_permission": inventory[r["clip_id"]].get("train_permission", "unverified"),
            "municipal_reservation": r["catalog_source_group"].get(
                "municipal_reservation", "False"
            ),
        }
        for r in records
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "readiness.json", summary)
    write_csv(
        output / "priority-review-queue.csv",
        queue,
        list(queue[0]) if queue else ["clip_id", "issues"],
    )
    write_csv(output / "source-provenance-queue.csv", source_queue, list(source_queue[0]))
    write_new_json(
        output / "registration.json",
        {
            "parent_registration_sha256": summary["prior_registration_sha256"],
            "annotations_sha256": sha256(annotations),
            "original_catalog_sha256": sha256(original_catalog),
            "output_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
            "code_sha256": sha256(Path(__file__)),
        },
    )
    return summary
