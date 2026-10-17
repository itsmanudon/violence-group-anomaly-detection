"""Generate an explicit synthetic-only dry-run adapter fixture, never a benchmark."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/municipal-fixtures"))
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.count <= 10000:
        parser.error("count must be between 1 and 10000")
    if args.output.exists():
        parser.error("output exists; choose a new fixture directory")
    args.output.mkdir(parents=True)
    clips = []
    for index in range(args.count):
        filename = f"synthetic-{index:04}.avi"
        writer = cv2.VideoWriter(
            str(args.output / filename), cv2.VideoWriter_fourcc(*"MJPG"), 8, (32, 24)
        )
        if not writer.isOpened():
            raise RuntimeError("Local MJPG codec unavailable")
        for offset in range(8):
            writer.write(
                np.random.default_rng(index * 8 + offset).integers(
                    0, 255, (24, 32, 3), dtype=np.uint8
                )
            )
        writer.release()
        clips.append(
            {
                "clip_id": f"synthetic-{index:04}",
                "source_video_id": f"generator-{index:04}",
                "file_path": filename,
                "source": "deterministic synthetic generator",
                "license": "synthetic software fixture",
                "external_handover_permission": "no",
                "training_permission": "no",
                "assignment": "test",
                "cohort": "synthetic_fixture",
                "selection_method": "generated software fixture; no benchmark interpretation",
                "geography": {"location": None, "confidence": "unknown"},
            }
        )
    path = args.output / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "clips": clips}, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
