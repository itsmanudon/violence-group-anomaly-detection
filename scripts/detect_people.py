"""Detect native-resolution middle frames; precomputed mode is an explicit offline fixture."""

import argparse
import os
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import torch

from surveillance.datasets.collective import read_actor_manifest
from surveillance.datasets.common import resolve_path
from surveillance.detection import (
    DetectionConfig,
    DetectionRecord,
    PersonDetector,
    load_detection_config,
    read_detections,
    write_detections,
)
from surveillance.detection.torchvision_detector import TorchvisionPersonDetector


def run_detection(manifest: Path, output: Path, detector: PersonDetector) -> list[DetectionRecord]:
    """Read only frame_paths[5], detect at native size, and preserve manifest identities."""
    records = []
    for row in read_actor_manifest(manifest):
        path = resolve_path(row.frame_paths[5], Path(manifest))
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing or undecodable middle frame: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        result = detector.detect(tensor)
        if result.image_size != tuple(image.shape[:2]):
            raise ValueError("Detector result image_size differs from native middle frame")
        records.append(
            DetectionRecord(
                row.dataset,
                row.video_id,
                row.source_video_id,
                row.clip_id,
                row.frame_indices[5],
                result,
            )
        )
    write_detections(records, output)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/person_detector.yaml"))
    parser.add_argument(
        "--precomputed",
        type=Path,
        help="Validate and copy existing detection JSONL (no detector inference)",
    )
    args = parser.parse_args()
    config = load_detection_config(args.config)
    if args.precomputed:
        expected = {(r.dataset, r.clip_id): r for r in read_actor_manifest(args.manifest)}
        records = read_detections(args.precomputed)
        if set(expected) != {r.key for r in records}:
            raise ValueError(
                "Precomputed detection clip keys must exactly match the actor manifest"
            )
        relocated = []
        for record in records:
            row = expected[record.key]
            if (record.video_id, record.source_video_id, record.frame_index) != (
                row.video_id,
                row.source_video_id,
                row.frame_indices[5],
            ):
                raise ValueError(f"Detection source/center mismatch: {record.key}")
            changes = {}
            for field in ("pose_feature_path", "rgb_feature_path"):
                value = getattr(record, field)
                if value is not None:
                    resolved = resolve_path(value, args.precomputed).resolve()
                    try:
                        changes[field] = Path(
                            os.path.relpath(resolved, args.output.resolve().parent)
                        ).as_posix()
                    except ValueError:
                        changes[field] = resolved.as_posix()
            relocated.append(replace(record, **changes))
        write_detections(relocated, args.output)
        print(
            f"Validated {len(records)} precomputed scenes -> {args.output}; no detector inference"
        )
    else:
        settings = config["detector"]
        filters = DetectionConfig(**{key: settings[key] for key in asdict(DetectionConfig())})
        detector = TorchvisionPersonDetector(settings["checkpoint"], filters, config["device"])
        records = run_detection(args.manifest, args.output, detector)
        print(f"Detected persons in {len(records)} scenes -> {args.output}")


if __name__ == "__main__":
    from _common import run_cli

    run_cli(main)
