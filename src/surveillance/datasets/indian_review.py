"""Automated review candidates from verified receipts; never approve or relabel."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import duplicate_candidates, write_new_json
from surveillance.datasets.indian_readiness import (
    assess_annotation,
    read_csv,
    reuse_readiness,
    unique_index,
    write_csv,
)


def fingerprint_evidence(a: list[str], b: list[str], distance: int = 4) -> dict:
    """Ordered sampled dHash alignment; crop/incident equivalence is NOT established."""
    av, bv = [int(v, 16) for v in a], [int(v, 16) for v in b]
    lengths = [[0] * (len(bv) + 1) for _ in range(len(av) + 1)]
    edges = []
    for i, x in enumerate(av):
        for j, y in enumerate(bv):
            d = (x ^ y).bit_count()
            if d <= distance:
                edges.append([i, j, d])
            lengths[i + 1][j + 1] = max(
                lengths[i][j + 1],
                lengths[i + 1][j],
                lengths[i][j] + int(d <= distance),
            )
    ordered = lengths[-1][-1]
    distinct = min(len(set(a)), len(set(b)))
    return {
        "method": "existing_8_sample_64bit_dhash_ordered_alignment",
        "max_hamming_distance": distance,
        "matching_sample_edges_i_j_distance": edges,
        "ordered_sample_matches": ordered,
        "minimum_distinct_hashes": distinct,
        "observation": "temporal_visual_candidate"
        if ordered >= 3 and distinct >= 3
        else "ambiguous_visual_similarity",
        "source_verified": False,
        "incident_verified": False,
        "crop_or_trim_equivalence": "NOT ESTABLISHED",
    }


def build_review_queue(records: list[dict], candidates: list[dict]) -> list[dict]:
    if len({r["clip_id"] for r in records}) != len(records):
        raise ValueError("Duplicate records in review queue")
    candidate_by_id = defaultdict(list)
    ids = {r["clip_id"] for r in records}
    for candidate in candidates:
        if len(candidate["pair"]) != 2 or not set(candidate["pair"]) <= ids:
            raise ValueError("Duplicate candidate references unknown IDs")
        for key in candidate["pair"]:
            candidate_by_id[key].append(candidate)
    queue = []
    for row in records:
        raw, media, source = row["raw_annotation"], row["media"], row["source_identity"]
        annotation = assess_annotation(raw)
        issues = annotation["annotation_issues"]
        quality_hold = bool(issues) or not media["readable"]
        flags = ["unverified_source_incident_and_overlap"]
        if quality_hold:
            flags.extend(issues)
            if not media["readable"]:
                flags.append("strict_loader_media_hold")
        if not quality_hold and annotation["normalized_label"] == "violence":
            flags.append("candidate_positive_second_look")
        if candidate_by_id[row["clip_id"]]:
            flags.append("duplicate_or_visual_candidate_review")
        conflict = any(
            raw.get(k) and row["catalog_record"].get(k) and raw[k] != row["catalog_record"][k]
            for k in ("source_video_id", "media_status")
        )
        if conflict:
            flags.append("annotation_catalog_status_or_source_conflict")
        if "municipal_reservation" in row["blockers"]:
            flags.append("municipal_reservation_keep_out_of_optimization")
        if not raw["reviewed_at"].strip():
            flags.append("review_timestamp_missing")
        priority = (
            quality_hold
            or bool(candidate_by_id[row["clip_id"]])
            or "candidate_positive_second_look" in flags
            or conflict
        )
        queue.append(
            {
                "clip_id": row["clip_id"],
                "file_path": raw["relative_video_path"],
                "resolved_local_path": media["resolved_local_path"],
                "collection": raw["dataset_name"],
                "original_label": raw["label"],
                "original_reviewer": raw["assigned_reviewer"],
                "original_media_status": raw["media_status"],
                "original_playback_status": raw["playback_status"],
                "quality_status": "quarantine" if quality_hold else "candidate_only_not_approved",
                "priority": "first" if priority else "source_evidence",
                "flags": ";".join(sorted(set(flags))),
                "conservative_guard_not_verified_source": row["guard_group"],
                "filename_family_hint_not_verified": source.get("filename_family_hint") or "",
                "raw_violence_start_sec": raw["violence_start_sec"],
                "raw_violence_end_sec": raw["violence_end_sec"],
                "parsed_interval_seconds": json.dumps(annotation["parsed_interval_seconds"]),
                "rounding_proposal_not_applied": json.dumps(annotation["rounding_proposal"]),
                "duplicate_observations": json.dumps(
                    candidate_by_id[row["clip_id"]], sort_keys=True
                ),
                "review_action": (
                    "Adjudicate flagged content/media and supply recording/incident/rights/"
                    "overlap evidence; retain reservation"
                ),
                "human_disposition": "",
                "human_recording_id": "",
                "human_incident_id": "",
                "human_evidence": "",
                "human_reviewer": "",
                "human_reviewed_at": "",
                "training_approved": False,
            }
        )
    return sorted(queue, key=lambda r: (r["priority"] != "first", r["collection"], r["clip_id"]))


def timing_observations(records: list[dict], inventory: dict) -> tuple[dict, list[dict]]:
    """Keep container/presentation and decoded-frame clocks explicitly separate."""
    observations, queue = [], []
    for row in sorted(records, key=lambda r: r["clip_id"]):
        media, cat = row["media"], inventory[row["clip_id"]]
        if not media.get("fps") or not media.get("num_frames"):
            continue
        fps = float(media["fps"])
        presentation = float(cat["duration_seconds"])
        frame_clock = media["num_frames"] / fps
        delta = presentation - frame_clock
        observed = {
            "clip_id": row["clip_id"],
            "collection": row["dataset_name"],
            "original_label": row["raw_annotation"]["label"],
            "file_path": row["raw_annotation"]["relative_video_path"],
            "presentation_seconds": presentation,
            "decoded_count_over_fps_seconds": frame_clock,
            "signed_difference_seconds": delta,
            "catalog_fps": float(cat["fps"]),
            "decoder_fps": fps,
            "beyond_one_frame_plus_1ms": abs(delta) > 1 / fps + 0.001,
            "raw_violence_end_sec": row["raw_annotation"]["violence_end_sec"],
            "training_or_temporal_approval": False,
            "review_action": "Resolve presentation/sample-to-frame timeline before temporal use",
            "human_disposition": "",
            "human_evidence": "",
        }
        observations.append(observed)
        if observed["beyond_one_frame_plus_1ms"]:
            queue.append(observed)
    return {
        "scope": "prior_decode_metadata_on_current_hash_verified_bytes_no_new_decoding",
        "definition": "presentation minus decoded count/FPS; one-frame+1ms flag is diagnostic only",
        "observations": observations,
        "available": len(observations),
        "flagged": len(queue),
        "flagged_collections": dict(Counter(r["collection"] for r in queue)),
        "fps_disagreements_over_1e_minus_6": sum(
            abs(r["catalog_fps"] - r["decoder_fps"]) > 1e-6 for r in observations
        ),
        "interpretation": (
            "does not change binary labels, source approval or existing quality holds"
        ),
    }, queue


def screen_review(audit: Path, annotations: Path, catalog: Path, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refuse to overwrite review screening")
    # This reuses existing complete-decode/fingerprint evidence only after checking
    # every registered input and current media hash. No resampling is needed.
    summary = reuse_readiness(audit, annotations, catalog, output / "receipt-recheck")
    records = json.loads((Path(audit) / "records.json").read_text())
    by_id = {r["clip_id"]: r for r in records}
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
    for candidate in candidates:
        a, b = [by_id[k] for k in candidate["pair"]]
        candidate["evidence"] = fingerprint_evidence(
            a["media"]["visual_signature"], b["media"]["visual_signature"]
        )
        candidate["different_collections"] = a["dataset_name"] != b["dataset_name"]
        candidate["conflicting_binary_annotations"] = {
            a["raw_annotation"]["label"],
            b["raw_annotation"]["label"],
        } == {"violence", "non_violence"}
    queue = build_review_queue(records, candidates)
    catalog_rows = unique_index(read_csv(catalog), "current original inventory")
    timing_report, timing_queue = timing_observations(records, catalog_rows)
    audio = [r for k, r in catalog_rows.items() if k not in by_id]
    audio_checks = []
    for r in audio:
        p = Path(r["file_path"])
        audio_checks.append(
            {
                "clip_id": r["clip_id"],
                "exists": p.is_file(),
                "hash_matches": p.is_file() and sha256(p) == r["sha256"],
            }
        )
    families = defaultdict(list)
    for r in records:
        hint = r["source_identity"].get("filename_family_hint")
        if hint:
            families[(r["dataset_name"], hint)].append(r["clip_id"])
    groups = [
        {"collection": k[0], "filename_family_hint": k[1], "clip_ids": sorted(v), "verified": False}
        for k, v in sorted(families.items())
        if len(v) > 1
    ]
    collections = defaultdict(Counter)
    geography = defaultdict(Counter)
    flags = Counter()
    for r, q in zip(
        sorted(records, key=lambda r: r["clip_id"]),
        sorted(queue, key=lambda r: r["clip_id"]),
        strict=True,
    ):
        collections[r["dataset_name"]][r["raw_annotation"]["label"] or "missing"] += 1
        geography[r["catalog_record"].get("geographic_origin", "unknown")][
            r["raw_annotation"]["label"] or "missing"
        ] += 1
        flags.update(q["flags"].split(";"))
    report = {
        "scope": "receipt_reuse_current_media_hashes_no_new_full_decode_or_frames",
        "readiness": summary,
        "audio_current_checks": audio_checks,
        "deduplicated_queue_records": len(queue),
        "first_priority_records": sum(r["priority"] == "first" for r in queue),
        "flag_counts_not_disjoint": dict(flags),
        "collection_label_counts": {k: dict(v) for k, v in sorted(collections.items())},
        "catalog_geography_label_counts": {k: dict(v) for k, v in sorted(geography.items())},
        "unknown_original_source_records": sum(
            not r["source_identity"].get("source_video_id") for r in records
        ),
        "verified_incidents": 0,
        "near_duplicate_pairs": len(candidates),
        "filename_families_not_proven_incidents": groups,
        "candidate_hard_negatives": "NOT AVAILABLE: no approved Indian prediction scores",
        "crop_trim_cross_collection_clearance": (
            "NOT ESTABLISHED: sampled dHash can miss transformations"
        ),
        "split_approved": False,
    }
    write_csv(output / "review-queue.csv", queue, list(queue[0]))
    first = [r for r in queue if r["priority"] == "first"]
    write_csv(output / "first-priority.csv", first, list(queue[0]))
    write_new_json(output / "duplicate-candidates.json", candidates)
    write_new_json(output / "quality.json", report)
    write_new_json(output / "timing-observations.json", timing_report)
    write_csv(
        output / "timing-review-queue.csv",
        timing_queue,
        list(timing_queue[0]) if timing_queue else ["clip_id", "human_disposition"],
    )
    write_new_json(
        output / "registration.json",
        {
            "parent_recheck_sha256": sha256(output / "receipt-recheck/registration.json"),
            "code_sha256": {
                "indian_review.py": sha256(Path(__file__)),
                "indian_readiness.py": sha256(Path(__file__).with_name("indian_readiness.py")),
            },
            "outputs": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        },
    )
    return report
