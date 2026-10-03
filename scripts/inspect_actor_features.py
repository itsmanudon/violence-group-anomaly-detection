"""Inspect local actor extraction contracts. This produces no benchmark accuracy."""

import argparse
import json
from pathlib import Path

import torch

from surveillance.datasets.actor_batch import collate_actors
from surveillance.datasets.collective import ActorFeatureDataset
from surveillance.features.geometry import normalized_boxes_to_pixels, scale_boxes
from surveillance.features.hrnet_pose import HRNetPoseExtractor
from surveillance.features.i3d import I3DActorExtractor
from surveillance.features.provenance import (
    config_fingerprint,
    extraction_config,
    file_sha256,
)
from surveillance.training.sultani_trainer import select_device


def inspect_features(
    manifest: Path,
    checkpoint: Path,
    modality: str,
    max_scenes: int = 10,
    image_size: tuple[int, int] = (480, 720),
    device: str = "cpu",
) -> dict:
    """Run real adapters twice; check shapes, finiteness, order and coordinate mapping."""
    if max_scenes < 1:
        raise ValueError("max_scenes must be positive")
    if modality not in {"pose", "rgb"}:
        raise ValueError("modality must be pose or rgb")
    cls = HRNetPoseExtractor if modality == "pose" else I3DActorExtractor
    selected_device = select_device(device)
    extractor = cls(checkpoint, frozen=True).to(selected_device).eval()
    dataset = ActorFeatureDataset(
        manifest, split=None, mode=f"{modality}_only", input_mode="raw", image_size=image_size
    )
    config = extraction_config(
        modality, image_size, file_sha256(checkpoint), extractor.metadata, "ground_truth"
    )
    report = {
        "schema_version": 1,
        "valid": True,
        "non_benchmark": True,
        "description": "Local extraction contract checks; no accuracy measured",
        "extraction_config": config,
        "extraction_config_hash": config_fingerprint(config),
        "scenes": [],
        "errors": [],
    }
    with torch.inference_mode():
        for index in range(min(max_scenes, len(dataset))):
            row = dataset.records[index]
            try:
                batch = collate_actors([dataset[index]])
                frames = batch["frames"].to(selected_device)
                if modality == "pose":
                    frames = frames[:, 5]
                boxes = batch["actor_boxes"].to(selected_device)
                mask = batch["actor_valid_mask"].to(selected_device)
                first = extractor(frames, boxes, mask)
                second = extractor(frames, boxes, mask)
                permutation = torch.arange(boxes.shape[1] - 1, -1, -1, device=selected_device)
                reordered = extractor(frames, boxes[:, permutation], mask[:, permutation])
                shape_ok = tuple(first.shape) == (1, len(row.actor_boxes), extractor.feature_dim)
                finite = bool(torch.isfinite(first).all())
                repeatable = bool(torch.allclose(first, second, rtol=1e-5, atol=1e-6))
                order_ok = bool(
                    torch.allclose(first[:, permutation], reordered, rtol=1e-5, atol=1e-6)
                )
                target = (90, 160) if modality == "rgb" else (64, 48)
                pixels = normalized_boxes_to_pixels(boxes, image_size[1], image_size[0])
                mapped = scale_boxes(pixels, image_size, target)
                direct = normalized_boxes_to_pixels(boxes, target[1], target[0])
                coordinates_ok = bool(torch.allclose(mapped, direct, rtol=1e-5, atol=1e-6))
                checks = {
                    "clip_id": row.clip_id,
                    "shape": list(first.shape),
                    "finite": finite,
                    "shape_valid": shape_ok,
                    "repeatable": repeatable,
                    "actor_order_verified": order_ok,
                    "image_map_coordinates_verified": coordinates_ok,
                }
                report["scenes"].append(checks)
                if not all((shape_ok, finite, repeatable, order_ok, coordinates_ok)):
                    report["errors"].append(f"{row.clip_id}: extraction contract check failed")
            except (OSError, ValueError, RuntimeError) as error:
                report["errors"].append(f"{row.clip_id}: {error}")
    report["valid"] = not report["errors"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--modality", choices=["pose", "rgb"], required=True)
    parser.add_argument("--max-scenes", type=int, default=10)
    parser.add_argument("--image-size", nargs=2, type=int, default=[480, 720])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = inspect_features(
            args.manifest,
            args.checkpoint,
            args.modality,
            args.max_scenes,
            tuple(args.image_size),
            args.device,
        )
    except (OSError, ValueError, RuntimeError) as error:
        report = {
            "schema_version": 1,
            "valid": False,
            "non_benchmark": True,
            "scenes": [],
            "errors": [str(error)],
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Inspection {'passed' if report['valid'] else 'failed'}; report: {args.report}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
