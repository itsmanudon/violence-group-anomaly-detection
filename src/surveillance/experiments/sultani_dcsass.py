"""Provisional real DCSASS clip-level Sultani baseline; never a UCF frame benchmark."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from surveillance.datasets.common import Record, check_leakage, read_manifest, write_manifest
from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.dcsass_protocol import split_sources
from surveillance.datasets.preparation import source_identity
from surveillance.experiments.dcsass_cache import write_json
from surveillance.features.c3d import C3DExtractor
from surveillance.training.sultani_trainer import seed_everything, select_device
from surveillance.video.segmentation import c3d_segment_frame_ranges

EXPERIMENT = Path("runs/dcsass/sultani_generic_v1")
MANIFEST = Path("data/manifests/dcsass_sultani_generic_v1.jsonl")
FEATURE_MANIFEST = Path("data/manifests/dcsass_sultani_generic_features_v1.jsonl")
C3D_CHECKPOINT = Path("checkpoints/c3d_fc6_openmmlab_v1.pt")
C3D_SHA256 = "beac4ff065de3663bf5c6bab36aedca4fe56ec59b231c66bf5374c23eda14831"


def generic_source_assignments(
    clips: list[dict], actor_assignments: dict, reserved: set[str]
) -> dict:
    """All valid binary categories; shared actor sources retain identical membership."""
    sources = {r["source_video_id"]: r["category"] for r in clips if r["binary_label"] in (0, 1)}
    remaining = {s: c for s, c in sources.items() if s not in actor_assignments}
    assignments = split_sources(remaining, reserved, seed=0)
    assignments.update({s: split for s, split in actor_assignments.items() if s in sources})
    if any(assignments[s] != "test" for s in reserved & set(assignments)):
        raise ValueError("Sultani protocol would expose an official UCF test source")
    return assignments


def prepare_generic() -> dict:
    if MANIFEST.exists() or (EXPERIMENT / "protocol.json").exists():
        raise FileExistsError("Generic Sultani protocol already frozen")
    audit = json.loads(Path("runs/dcsass/audit_v1.json").read_text())
    if audit["bounded_preflight"]:
        raise ValueError("Sultani preparation requires a full real audit")
    actor = json.loads(Path("runs/dcsass/human_centric_v1/final_split_receipt.json").read_text())
    test_file = Path("data/splits/Temporal_Anomaly_Annotation.txt")
    reserved = {
        source_identity(Path(line.split()[0]), {})
        for line in test_file.read_text().splitlines()
        if line.strip()
    }
    assignments = generic_source_assignments(audit["clips"], actor["source_assignments"], reserved)
    records = [
        Record(
            dataset="dcsass",
            video_id=f"{r['category']}/{r['clip_id']}",
            source_video_id=r["source_video_id"],
            path=r["path"],
            split=assignments[r["source_video_id"]],
            label=r["binary_label"],
            anomaly_type=r["category"],
            duration_sec=r["duration_sec"],
            fps=r["fps"],
            num_frames=r["num_frames"],
            temporal_annotations=None,
        )
        for r in audit["clips"]
        if r["readable"] and r["binary_label"] in (0, 1)
    ]
    check_leakage(records)
    write_manifest(records, MANIFEST)
    report = {
        "protocol_id": "dcsass_sultani_generic_clip_v1",
        "dataset": "DCSASS",
        "evidence_kind": "real",
        "evaluation_scope": "binary clip/bag labels only",
        "frame_annotations_available": False,
        "paper_reproduction": False,
        "seed": 0,
        "source_assignments": assignments,
        "audit_sha256": sha256(Path("runs/dcsass/audit_v1.json")),
        "actor_split_receipt_sha256": sha256(
            Path("runs/dcsass/human_centric_v1/final_split_receipt.json")
        ),
        "manifest_sha256": sha256(MANIFEST),
        "c3d_sha256": C3D_SHA256,
        "source_counts": dict(Counter(assignments.values())),
        "clip_counts": dict(Counter(r.split for r in records)),
        "binary_counts": {
            split: {
                str(label): sum(r.split == split and r.label == label for r in records)
                for label in (0, 1)
            }
            for split in ("train", "val", "test")
        },
        "shared_actor_sources": len(actor["source_assignments"]),
        "unknown_annotations_excluded": audit["binary_label_counts"]["None"],
        "source_overlap": [],
        "ucf_frame_benchmark": "pending external dataset download",
    }
    write_json(EXPERIMENT / "protocol.json", report)
    return report


def extract_generic(max_clips: int | None = None) -> dict:
    if sha256(C3D_CHECKPOINT) != C3D_SHA256:
        raise ValueError("Approved C3D export identity changed")
    protocol = json.loads((EXPERIMENT / "protocol.json").read_text())
    if sha256(MANIFEST) != protocol["manifest_sha256"]:
        raise ValueError("Generic Sultani manifest changed")
    records = read_manifest(MANIFEST)
    if max_clips is not None:
        chosen = []
        for split in ("train", "val"):
            for label in (0, 1):
                chosen.extend(
                    [r for r in records if r.split == split and r.label == label][: max_clips // 4]
                )
        records = chosen
    prefix = EXPERIMENT / ("preflight_cache" if max_clips is not None else "cache")
    output_manifest = prefix / "features.jsonl" if max_clips is not None else FEATURE_MANIFEST
    seed_everything(0)
    extractor = C3DExtractor(
        C3D_CHECKPOINT,
        str(select_device()),
        batch_size=4,
        mean=(104, 117, 128),
        channel_order="rgb",
    )
    audit = json.loads(Path("runs/dcsass/audit_v1.json").read_text())
    by_path = {r["path"]: r for r in audit["clips"]}
    result = []
    for i, row in enumerate(records, 1):
        path = prefix / "features" / f"{row.video_id}.npy"
        sidecar = path.with_suffix(".json")
        source = by_path[row.path]
        if sha256(Path(row.path)) != source["sha256"]:
            raise ValueError(f"Video changed: {row.video_id}")
        expected = {
            "video_id": row.video_id,
            "source_video_id": row.source_video_id,
            "video_sha256": source["sha256"],
            "c3d_sha256": C3D_SHA256,
            "shape": [32, 4096],
            "mean": [104, 117, 128],
            "channel_order": "rgb",
            "unit_frames": 16,
            "segment_frame_ranges": [
                list(pair) for pair in c3d_segment_frame_ranges(row.num_frames)
            ],
        }
        if path.exists() and sidecar.exists():
            saved = json.loads(sidecar.read_text())
            if any(saved.get(k) != v for k, v in expected.items()) or saved[
                "feature_sha256"
            ] != sha256(path):
                raise ValueError(f"Stale C3D cache: {row.video_id}")
            values = np.load(path, allow_pickle=False)
        else:
            values = extractor.extract_video(Path(row.path), 32)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, values)
            write_json(
                sidecar,
                {**expected, "feature_sha256": sha256(path), "preprocessing": extractor.metadata},
            )
        if values.shape != (32, 4096) or not np.isfinite(values).all():
            raise ValueError("Invalid C3D feature bag")
        from dataclasses import replace

        result.append(replace(row, feature_path=str(path.resolve())))
        if i % 200 == 0 or i == len(records):
            print(f"C3D: {i}/{len(records)} clips", flush=True)
    write_manifest(result, output_manifest)
    report = {
        "feature_manifest": str(output_manifest),
        "clip_count": len(result),
        "bounded_preflight": max_clips is not None,
        "manifest_sha256": sha256(output_manifest),
        "c3d_sha256": C3D_SHA256,
    }
    write_json(prefix / "extraction.json", report)
    return report
