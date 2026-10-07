"""Freeze the completed UCF audit with explicit content-alias leakage protection."""

import json
from collections import Counter, defaultdict
from dataclasses import asdict, fields
from pathlib import Path

from _common import run_cli

from surveillance.datasets.common import Record, check_leakage, read_manifest, write_manifest
from surveillance.datasets.dcsass_audit import sha256
from surveillance.experiments.dcsass_cache import write_json
from surveillance.experiments.sultani_ucf import protect_shared_sources, remove_training_duplicates

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / "runs/ucf-crime/sultani_shared_safe_v1"
    manifest = ROOT / "data/manifests/ucf_sultani_shared_safe_v1.jsonl"
    if manifest.exists() or (run / "protocol.json").exists():
        raise FileExistsError("UCF source protocol is already frozen")
    audit = json.loads((run / "audit.json").read_text())
    if audit["failures"] or audit["audited"] != audit["videos"] or audit["videos"] != 1900:
        raise ValueError("Complete original-video audit required")
    root = Path(audit["root"])
    train = root / "Anomaly_Train.txt"
    test = root / "UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Test.txt"
    annotations = root / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    for supplied, pinned in [
        (train, ROOT / "data/splits/Anomaly_Train.txt"),
        (annotations, ROOT / "data/splits/Temporal_Anomaly_Annotation.txt"),
    ]:
        if supplied.read_bytes() != pinned.read_bytes():
            raise ValueError("Author provenance changed after audit")
    accepted = {field.name for field in fields(Record)}
    records = [
        Record(**{k: v for k, v in row.items() if k in accepted}) for row in audit["records"]
    ]
    author_test_names = {
        Path(line.split()[0]).name.casefold() for line in test.read_text().splitlines()
    }
    if {Path(r.path).name.casefold() for r in records if r.split == "test"} != author_test_names:
        raise ValueError("Author test population changed after audit")
    identities = {row["video_id"]: row["sha256"] for row in audit["records"]}
    deduplicated, duplicate_exclusions = remove_training_duplicates(records, identities)
    shared_manifest = ROOT / "data/manifests/dcsass_sultani_generic_v1.jsonl"
    shared_protocol = ROOT / "runs/dcsass/sultani_generic_v1/protocol.json"
    if sha256(shared_manifest) != json.loads(shared_protocol.read_text())["manifest_sha256"]:
        raise ValueError("Frozen shared DCSASS source manifest changed")
    shared_records = read_manifest(shared_manifest)
    shared = {r.source_video_id: r.split for r in shared_records}
    retained, shared_exclusions = protect_shared_sources(deduplicated, shared, seed=0)
    check_leakage(retained + shared_records)
    hash_splits = defaultdict(set)
    for row in retained:
        hash_splits[identities[row.video_id]].add(row.split)
    if any(len(splits) > 1 for splits in hash_splits.values()):
        raise ValueError("Content leakage remains after duplicate quarantine")
    if sum(r.split == "test" for r in retained) != 290:
        raise ValueError("Original 290-video test set was not preserved")
    write_manifest(retained, manifest)
    receipt = {
        "protocol_id": "ucf_sultani_shared_safe_v1",
        "seed": 0,
        "root": str(root),
        "manifest_sha256": sha256(manifest),
        "audit_sha256": sha256(run / "audit.json"),
        "shared_manifest_sha256": sha256(shared_manifest),
        "author_train_sha256": sha256(train),
        "author_test_sha256": sha256(test),
        "temporal_annotations_sha256": sha256(annotations),
        "source_counts": dict(Counter(r.split for r in retained)),
        "unique_content_counts": {
            s: len({identities[r.video_id] for r in retained if r.split == s})
            for s in ("train", "val", "test")
        },
        "binary_counts": {
            s: dict(Counter(str(r.label) for r in retained if r.split == s))
            for s in ("train", "val", "test")
        },
        "source_assignments": {r.source_video_id: r.split for r in retained},
        "excluded_author_train_held_out_dcsass": [asdict(r) for r in shared_exclusions],
        "excluded_duplicate_training_copies": [asdict(r) for r in duplicate_exclusions],
        "exact_content_duplicate_groups": audit["exact_content_duplicate_groups"],
        "selection_metric": "validation_bag_roc_auc",
        "evaluation_mode": "frame",
        "anomaly_threshold": 0.5,
        "frame_projection": "c3d_units",
        "paper_reproduction": False,
        "annotation_boundary_adjustments": audit["annotation_boundary_adjustments"],
        "annotation_boundary_policy": "Verified-original 1-2 frame endpoint intersections",
        "source_overlap": [],
        "cross_split_content_overlap": [],
        "deviation": "Preserve DCSASS membership; exclude its test sources and exact duplicate "
        "training copies; retain all author-test entries including one duplicate pair; "
        "remaining category-aware validation 15%; modern C3D preprocessing",
        "held_out_used_for_selection": False,
    }
    write_json(run / "protocol.json", receipt)
    print(
        json.dumps(
            {k: receipt[k] for k in ("source_counts", "unique_content_counts", "binary_counts")},
            indent=2,
        )
    )


if __name__ == "__main__":
    run_cli(main)
