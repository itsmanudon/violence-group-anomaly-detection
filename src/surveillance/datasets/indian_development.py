"""Provisional attestation-aware grouping; produces no training-approved manifest."""

import json
from collections import defaultdict
from pathlib import Path

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import write_new_json
from surveillance.datasets.indian_readiness import reuse_readiness, write_csv
from surveillance.datasets.indian_split import assign_groups

ATTESTATION = {
    "id": "human-message-20261026-distinct-violence-incidents",
    "reported_by": "user reporting human annotators",
    "date_local": "2026-10-26",
    "timezone": "Asia/Calcutta",
    "statement": "All labeled violence clips represent distinct incidents.",
    "evidence_type": "annotator_attestation",
    "independently_verified_source_metadata": False,
    "original_source_urls_required_for_exploratory_grouping": False,
    "research_permission_confirmed": False,
}


def provisional_groups(
    records: list[dict], candidates: list[dict]
) -> tuple[list[dict], list[list[str]], dict]:
    """Keep negative continuity cohorts intact and all global duplicate links intact.

    Negative collection/tour cohorts deliberately overgroup unresolved recordings.
    The two mixed-label collections' unknown negative contexts are excluded from
    this initial proposal, avoiding an invented claim of independence from positives.
    Existing HCI municipal reservations remain outside optimization.
    """
    ids = [r["clip_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate development IDs")
    quality = [
        r for r in records if not r["annotation"]["annotation_issues"] and r["media"]["readable"]
    ]
    excluded = {}
    retained = []
    for r in quality:
        key, label = r["clip_id"], r["annotation"]["normalized_label"]
        if "municipal_reservation" in r["blockers"]:
            excluded[key] = "existing_municipal_reservation"
        elif label == "non_violence" and r["dataset_name"] in {
            "Snatch-1.0",
            "Unidentified-Refined",
        }:
            excluded[key] = "unresolved_negative_recording_context_in_positive_collection"
        elif label not in {"violence", "non_violence"}:
            raise ValueError("Uncertain labels must remain quality quarantine")
        else:
            retained.append(r)
    parent = {r["clip_id"]: r["clip_id"] for r in retained}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def join(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    seen = {}
    for r in retained:
        key = r["clip_id"]
        tokens = [("content_sha256", r["media"]["sha256"])]
        if r["annotation"]["normalized_label"] == "non_violence":
            tokens.append(("negative_recording_continuity_guard", r["guard_group"]))
        hint = r["source_identity"].get("filename_family_hint")
        if hint:
            tokens.append(("unresolved_filename_recording_family", (r["dataset_name"], hint)))
        for token in tokens:
            if token in seen:
                join(key, seen[token])
            seen[token] = key
    for c in candidates:
        if len(c["pair"]) != 2 or not set(c["pair"]) <= set(ids):
            raise ValueError("Unknown duplicate link")
        a, b = c["pair"]
        if a in parent and b in parent:
            join(a, b)  # unresolved near matches remain together, never declared independent
    components = defaultdict(list)
    for key in sorted(parent):
        components[root(key)].append(key)
    return retained, sorted(components.values()), excluded


def propose_development(audit: Path, annotations: Path, catalog: Path, output: Path, *, seed=7):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing development proposal")
    recheck = reuse_readiness(audit, annotations, catalog, output / "current-hash-recheck")
    records = json.loads((Path(audit) / "records.json").read_text())
    report = json.loads((Path(audit) / "readiness.json").read_text())
    retained, groups, excluded = provisional_groups(records, report["duplicate_candidates"])
    rows = [
        {"clip_id": r["clip_id"], "label": int(r["annotation"]["normalized_label"] == "violence")}
        for r in retained
    ]
    assignment = assign_groups(rows, groups, seed, (0.7, 0.15, 0.15))
    group_by_id = {key: f"provisional-cohort-{i:03d}" for i, g in enumerate(groups) for key in g}
    assignments = []
    for r in sorted(retained, key=lambda r: r["clip_id"]):
        raw = r["raw_annotation"]
        assignments.append(
            {
                "clip_id": r["clip_id"],
                "file_path": r["media"]["resolved_local_path"],
                "original_relative_path": raw["relative_video_path"],
                "collection": r["dataset_name"],
                "original_label": raw["label"],
                "label": int(r["annotation"]["normalized_label"] == "violence"),
                "proposed_partition": assignment[r["clip_id"]],
                "provisional_group": group_by_id[r["clip_id"]],
                "incident_evidence": ATTESTATION["id"]
                if raw["label"] == "violence"
                else "unresolved_negative_recording_continuity_guard",
                "original_source_video_id": raw["source_video_id"],
                "original_video_url": raw["video_url"],
                "independently_verified_source": False,
                "research_permission": "unverified",
                "training_approved": False,
                "test_is_final_independent": False,
            }
        )
    # Retain automatic all-clip/all-dataset exact checks; never restrict them to negatives.
    historical = set()
    registration = json.loads((Path(audit) / "registration.json").read_text())
    for p in registration["input_sha256"]:
        if (
            Path(p).suffix == ".jsonl"
            and "manifests" in Path(p).parts
            and Path(p).name != "master-clip-inventory.jsonl"
        ):
            for line in Path(p).read_text().splitlines():
                if line.strip():
                    item = json.loads(line)
                    if item.get("sha256"):
                        historical.add(item["sha256"])
    hits = [r["clip_id"] for r in records if r["media"].get("sha256") in historical]
    if hits:
        raise ValueError("Known cross-dataset duplicate hits require explicit proposal exclusions")
    counts = {
        split: {
            "clips": sum(r["proposed_partition"] == split for r in assignments),
            "violence": sum(
                r["proposed_partition"] == split and r["label"] == 1 for r in assignments
            ),
            "non_violence": sum(
                r["proposed_partition"] == split and r["label"] == 0 for r in assignments
            ),
            "provisional_groups": len(
                {r["provisional_group"] for r in assignments if r["proposed_partition"] == split}
            ),
        }
        for split in ("train", "val", "test")
    }
    collections = []
    for collection in sorted({r["dataset_name"] for r in records}):
        for split in ("train", "val", "test"):
            subset = [
                r
                for r in assignments
                if r["collection"] == collection and r["proposed_partition"] == split
            ]
            collections.append(
                {
                    "collection": collection,
                    "partition": split,
                    "violence": sum(r["label"] == 1 for r in subset),
                    "non_violence": sum(r["label"] == 0 for r in subset),
                }
            )
    summary = {
        "scope": "provisional_exploratory_development_proposal_not_approved_training_split",
        "annotator_attestation": ATTESTATION,
        "quality_candidates": recheck["quality_candidates"],
        "quality_holds_unchanged": recheck["quality_quarantine"],
        "attested_original_violence_records": sum(
            r["raw_annotation"]["label"] == "violence" for r in records
        ),
        "candidate_positive_records": recheck["candidate_label_counts"]["violence"],
        "proposed_records": len(assignments),
        "separate_development_exclusions": excluded,
        "requested_group_ratios": [0.7, 0.15, 0.15],
        "seed": seed,
        "counts": counts,
        "collection_by_partition": collections,
        "groups": groups,
        "assignments": assignment,
        "research_permissions": "all_six_collections_unverified_per_user_instruction",
        "training_approved": 0,
        "evaluation_approved": 0,
        "test_frozen": False,
        "test_semantics": "proposed development reserve only; not a final independent-domain test",
        "historical_hashes_screened": len(historical),
        "global_exact_historical_hits": hits,
        "global_near_candidates_kept_connected": report["duplicate_candidates"],
        "limitations": [
            "attestation does not verify recording URLs or usage rights",
            "negative guards overgroup unresolved recordings",
            "distinct incidents may share cameras/collection style",
            "sampled near matches cannot exhaust transformations",
            "all48 original positives have unknown catalog geography",
        ],
    }
    write_new_json(output / "proposal.json", summary)
    write_csv(output / "proposed-membership.csv", assignments, list(assignments[0]))
    write_csv(output / "collection-partition-counts.csv", collections, list(collections[0]))
    permission_queue = [
        {
            "collection": c,
            "research_training_permission": "unverified",
            "evidence": "",
            "authorized_attestor": "",
            "checked_at": "",
            "training_approved": False,
        }
        for c in sorted({r["dataset_name"] for r in records})
    ]
    write_csv(output / "permission-evidence-queue.csv", permission_queue, list(permission_queue[0]))
    write_new_json(
        output / "registration.json",
        {
            "attestation": ATTESTATION,
            "recheck_sha256": sha256(output / "current-hash-recheck/registration.json"),
            "code_sha256": sha256(Path(__file__)),
            "output_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        },
    )
    return summary
