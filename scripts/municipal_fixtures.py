"""Generate explicitly synthetic local integration media; never benchmark data."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from surveillance.municipal.catalog import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.count <= 10000:
        parser.error("Choose 1–10000 fixtures")
    args.output.mkdir(parents=True, exist_ok=True)
    clips = []
    for index in range(args.count):
        path = args.output / f"synthetic-{index:05d}.avi"
        if path.exists():
            parser.error("Fixture output exists; choose a fresh directory")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        for step in range(3):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            # Binary visual identity remains unique beyond 255 fixtures.
            for bit in range(14):
                if index & (1 << bit):
                    x, y = (bit % 7) * 8, (bit // 7) * 16
                    frame[y : y + 8, x : x + 8] = 180 + step
            writer.write(frame)
        writer.release()
        clips.append(
            dict(
                clip_id=f"synthetic-{index:05d}",
                source_video_id=f"synthetic-source-{index:05d}",
                file_path=path.name,
                sha256=digest(path),
                source="Generated synthetic software fixture",
                license="Generated fixture; no real people",
                population="synthetic",
                assignment="demo",
                selection_method="Deterministic synthetic media for integration checks only",
            )
        )
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(dict(schema_version=1, clips=clips), indent=2))
    print(manifest)


if __name__ == "__main__":
    main()
