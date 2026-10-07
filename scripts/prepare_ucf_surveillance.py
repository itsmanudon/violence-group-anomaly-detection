"""Audit installed original UCF videos and freeze a shared-source-safe protocol."""

import argparse
import json
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
from _common import run_cli

from surveillance.datasets.common import check_leakage, read_manifest, write_manifest
from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.ucf_crime import (
    apply_frame_annotations,
    apply_official_splits,
    prepare_ucf_crime,
)
from surveillance.experiments.dcsass_cache import write_json
from surveillance.experiments.sultani_ucf import protect_shared_sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run = ROOT / "runs/ucf-crime/sultani_shared_safe_v1"
    manifest = ROOT / "data/manifests/ucf_sultani_shared_safe_v1.jsonl"
    if manifest.exists() or (run / "protocol.json").exists():
        raise FileExistsError("UCF protocol is already frozen; preserve existing evidence")
    train = args.root / "Anomaly_Train.txt"
    test = args.root / "UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Test.txt"
    annotations = args.root / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    for supplied, pinned in [
        (train, ROOT / "data/splits/Anomaly_Train.txt"),
        (annotations, ROOT / "data/splits/Temporal_Anomaly_Annotation.txt"),
    ]:
        if supplied.read_bytes() != pinned.read_bytes():
            raise ValueError("Supplied split/annotation differs from pinned author provenance")
    started = time.perf_counter()
    records = apply_official_splits(prepare_ucf_crime(args.root, manifest), train, test)
    by_name = {Path(row.path).name.casefold(): row for row in records}
    integrity = []
    for line in annotations.read_text().splitlines():
        fields = line.split()
        row = by_name[fields[0].casefold()]
        pairs = list(zip(map(int, fields[2::2]), map(int, fields[3::2])))
        if not any(end > row.num_frames for _, end in pairs):
            continue
        if any(start > row.num_frames or end > row.num_frames + 2 for start, end in pairs):
            raise ValueError("Unexpected annotation overrun requires scientific diagnosis")
        video = Path(row.path)
        relative = video.relative_to(args.root)
        with zipfile.ZipFile(args.root / (relative.parts[0] + ".zip")) as archive:
            original = archive.getinfo(relative.as_posix())
        crc = 0
        with video.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                crc = zlib.crc32(chunk, crc)
        if video.stat().st_size != original.file_size or crc != original.CRC:
            raise ValueError("Annotation overrun video does not match original ZIP bytes")
        integrity.append(
            {
                "filename": video.name,
                "archive_size": original.file_size,
                "archive_crc32": original.CRC,
                "crc_matches": True,
            }
        )
    adjustments = []
    records = apply_frame_annotations(
        records, annotations, end_overrun_tolerance=2, boundary_adjustments=adjustments
    )
    shared_manifest = ROOT / "data/manifests/dcsass_sultani_generic_v1.jsonl"
    shared_records = read_manifest(shared_manifest)
    check_leakage(shared_records)
    shared = {r.source_video_id: r.split for r in shared_records}
    retained, excluded = protect_shared_sources(records, shared, seed=0)
    check_leakage(retained + shared_records)
    audit, failures = [], []
    for index, row in enumerate(records, 1):
        video = Path(row.path)
        capture = cv2.VideoCapture(str(video))
        try:
            first_ok, first = capture.read()
            capture.set(cv2.CAP_PROP_POS_FRAMES, row.num_frames - 1)
            last_ok, _ = capture.read()
            if not first_ok or not last_ok:
                raise ValueError("Cannot decode first/last reported frame")
            height, width = first.shape[:2]
            audit.append(
                {
                    **asdict(row),
                    "bytes": video.stat().st_size,
                    "sha256": sha256(video),
                    "width": width,
                    "height": height,
                    "decode_check": "first and last reported frames; full decode at extraction",
                }
            )
        except (ValueError, OSError) as error:
            failures.append({"video_id": row.video_id, "error": str(error)})
        finally:
            capture.release()
        if index % 50 == 0 or index == len(records):
            print(f"UCF audit: {index}/{len(records)}, failures {len(failures)}", flush=True)
    by_hash = defaultdict(list)
    for row in audit:
        by_hash[row["sha256"]].append(row["source_video_id"])
    duplicates = [sources for sources in by_hash.values() if len(sources) > 1]
    report = {
        "root": str(args.root.resolve()),
        "videos": len(records),
        "audited": len(audit),
        "failures": failures,
        "exact_content_duplicate_groups": duplicates,
        "records": audit,
        "seconds": time.perf_counter() - started,
        "event_recognition_files_excluded": True,
        "compressed_archives_used": False,
        "annotation_boundary_adjustments": adjustments,
        "annotation_overrun_video_integrity": integrity,
    }
    write_json(run / "audit.json", report)
    if failures or duplicates:
        raise ValueError("UCF audit requires diagnosis before protocol freeze; see audit.json")
    write_manifest(retained, manifest)
    receipt = {
        "protocol_id": "ucf_sultani_shared_safe_v1",
        "seed": 0,
        "root": str(args.root.resolve()),
        "manifest_sha256": sha256(manifest),
        "audit_sha256": sha256(run / "audit.json"),
        "shared_manifest_sha256": sha256(shared_manifest),
        "author_train_sha256": sha256(train),
        "author_test_sha256": sha256(test),
        "temporal_annotations_sha256": sha256(annotations),
        "source_counts": dict(Counter(r.split for r in retained)),
        "binary_counts": {
            s: dict(Counter(str(r.label) for r in retained if r.split == s))
            for s in ("train", "val", "test")
        },
        "source_assignments": {r.source_video_id: r.split for r in retained},
        "excluded_author_train_held_out_dcsass": [asdict(r) for r in excluded],
        "selection_metric": "validation_bag_roc_auc",
        "anomaly_threshold": 0.5,
        "frame_projection": "c3d_units",
        "paper_reproduction": False,
        "deviation": "Preserve shared DCSASS membership; exclude its test sources from "
        "authors' train list; remaining category-aware validation split 15%",
        "held_out_used_for_selection": False,
        "annotation_boundary_policy": "Verified-original end overruns of at most two frames "
        "intersect available frames; no existing-frame label changed",
        "annotation_boundary_adjustments": adjustments,
    }
    write_json(run / "protocol.json", receipt)
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k in {"source_counts", "binary_counts"}}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    run_cli(main)
