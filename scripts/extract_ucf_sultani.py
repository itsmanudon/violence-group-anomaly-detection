"""Resumable identity-checked full C3D extraction, or bounded train/val preflight."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from _common import run_cli

from surveillance.datasets.common import read_manifest, write_manifest
from surveillance.datasets.dcsass_audit import sha256
from surveillance.experiments.dcsass_cache import write_json
from surveillance.experiments.sultani_ucf import C3D_SHA256, extract_cached_bag
from surveillance.features.c3d import C3DExtractor
from surveillance.training.sultani_trainer import seed_everything, select_device

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    run = ROOT / "runs/ucf-crime/sultani_shared_safe_v1"
    manifest = ROOT / "data/manifests/ucf_sultani_shared_safe_v1.jsonl"
    checkpoint = ROOT / "checkpoints/c3d_fc6_openmmlab_v1.pt"
    receipt = json.loads((run / "protocol.json").read_text())
    if sha256(manifest) != receipt["manifest_sha256"] or sha256(checkpoint) != C3D_SHA256:
        raise ValueError("Frozen UCF manifest or approved C3D export changed")
    if sha256(run / "audit.json") != receipt["audit_sha256"]:
        raise ValueError("Frozen UCF audit changed")
    records = read_manifest(manifest)
    if args.preflight:
        records = [
            row
            for split in ("train", "val")
            for label in (0, 1)
            for row in sorted(
                [r for r in records if r.split == split and r.label == label],
                key=lambda r: (r.num_frames, r.video_id),
            )[:2]
        ]
    cache = run / ("preflight_cache" if args.preflight else "cache")
    output = (
        cache / "features.jsonl"
        if args.preflight
        else ROOT / "data/manifests/ucf_sultani_shared_safe_features_v1.jsonl"
    )
    audit = json.loads((run / "audit.json").read_text())
    identities = {row["video_id"]: row["sha256"] for row in audit["records"]}
    seed_everything(0)
    extractor = C3DExtractor(
        checkpoint, str(select_device()), batch_size=4, mean=(104, 117, 128), channel_order="rgb"
    )
    result, timings = [], []
    started = time.perf_counter()
    reused_count = 0
    for index, row in enumerate(records, 1):
        before = time.perf_counter()
        feature, reused = extract_cached_bag(row, identities[row.video_id], extractor, cache)
        reused_count += int(reused)
        result.append(
            replace(row, feature_path=feature.resolve().as_posix(), feature_sha256=sha256(feature))
        )
        timings.append(
            {
                "video_id": row.video_id,
                "frames": row.num_frames,
                "seconds": time.perf_counter() - before,
                "reused": reused,
            }
        )
        print(
            f"UCF C3D {index}/{len(records)}: {row.video_id} "
            f"({timings[-1]['seconds']:.2f}s, reused={reused})",
            flush=True,
        )
    write_manifest(result, output)
    write_json(
        cache / "extraction.json",
        {
            "feature_manifest": str(output),
            "manifest_sha256": sha256(output),
            "video_count": len(result),
            "reused": reused_count,
            "preflight": args.preflight,
            "full_sequential_decode_and_frame_count_verified": True,
            "c3d_sha256": C3D_SHA256,
            "seconds": time.perf_counter() - started,
            "timings": timings,
        },
    )


if __name__ == "__main__":
    run_cli(main)
