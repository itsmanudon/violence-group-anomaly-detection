"""Inspect local actor extraction contracts. This produces no benchmark accuracy."""

import argparse
import json
from pathlib import Path

import cv2
import torch

from surveillance.datasets.actor_batch import collate_actors
from surveillance.datasets.collective import ActorFeatureDataset
from surveillance.datasets.common import resolve_path
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
    debug_dir: Path | None = None,
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
                    "source_video_id": row.source_video_id,
                    "reference_frame_index": row.frame_indices[5],
                    "reference_frame_path": str(resolve_path(row.frame_paths[5], manifest)),
                    "image_size": list(image_size),
                    "actor_count": len(row.actor_boxes),
                    "actor_indices": list(range(len(row.actor_boxes))),
                    "actor_boxes": row.actor_boxes,
                    "box_convention": "normalized_xyxy_image_edges",
                    "actor_labels": row.actor_labels,
                    "group_label": row.group_label,
                    "crop_shape": [3, 256, 192] if modality == "pose" else None,
                    "shape": list(first.shape),
                    "finite": finite,
                    "shape_valid": shape_ok,
                    "repeatable": repeatable,
                    "actor_order_verified": order_ok,
                    "image_map_coordinates_verified": coordinates_ok,
                }
                if debug_dir is not None:
                    # A processed reference frame, not a copy used as a new dataset source.
                    reference = frames[0] if modality == "pose" else frames[0, 5]
                    image = reference.detach().cpu().permute(1, 2, 0).numpy()
                    image = cv2.cvtColor((image * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
                    for actor, box in enumerate(pixels[0].detach().cpu().tolist()):
                        x1, y1, x2, y2 = [round(value) for value in box]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(
                            image,
                            str(actor),
                            (x1, max(y1, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                        )
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    destination = debug_dir / f"scene_{index:04d}.png"
                    if destination.exists():
                        raise FileExistsError(f"Debug image already exists: {destination}")
                    if not cv2.imwrite(str(destination), image):
                        raise OSError(f"Cannot write debug image: {destination}")
                    checks["debug_image"] = destination.name
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
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Optional local frame/GT-box overlays; use an ignored runs/ path",
    )
    args = parser.parse_args()
    try:
        report = inspect_features(
            args.manifest,
            args.checkpoint,
            args.modality,
            args.max_scenes,
            tuple(args.image_size),
            args.device,
            args.debug_dir,
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
