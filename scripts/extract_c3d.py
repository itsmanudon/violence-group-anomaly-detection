"""Extract normalized C3D feature bags from an installed video manifest."""

import argparse
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from _common import run_cli

from surveillance.datasets.common import check_leakage, read_manifest, resolve_path, write_manifest
from surveillance.features.c3d import C3DExtractor
from surveillance.training.sultani_trainer import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--c3d-checkpoint", type=Path, required=True)
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--mean",
        nargs=3,
        type=float,
        required=True,
        help="Channel means from chosen checkpoint preprocessing",
    )
    parser.add_argument("--channel-order", choices=["rgb", "bgr"], required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    check_leakage(rows)
    extractor = C3DExtractor(
        args.c3d_checkpoint,
        str(select_device(args.device)),
        args.batch_size,
        args.mean,
        args.channel_order,
    )
    result = []
    for i, row in enumerate(rows, 1):
        video_path = resolve_path(row.path, args.manifest)
        feature_path = args.feature_dir / row.dataset / f"{row.video_id}.npy"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        features = extractor.extract_video(video_path, args.num_segments)
        np.save(feature_path, features)
        result.append(
            replace(
                row,
                path=Path(
                    os.path.relpath(video_path, args.output_manifest.resolve().parent)
                ).as_posix(),
                feature_path=Path(
                    os.path.relpath(feature_path.resolve(), args.output_manifest.resolve().parent)
                ).as_posix(),
            )
        )
        print(f"[{i}/{len(rows)}] {row.video_id} -> {feature_path}")
    write_manifest(result, args.output_manifest)


if __name__ == "__main__":
    run_cli(main)
