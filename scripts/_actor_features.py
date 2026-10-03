"""Shared extraction CLI for local feature-only actor backbones."""

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from surveillance.datasets.actor_batch import collate_actors
from surveillance.datasets.collective import ActorFeatureDataset, write_actor_manifest
from surveillance.datasets.common import resolve_path
from surveillance.datasets.detected_actors import (
    DetectedActorDataset,
    collate_actor_inputs,
    source_fingerprint,
)
from surveillance.detection.records import detection_fingerprint, write_detections
from surveillance.features.hrnet_pose import HRNetPoseExtractor
from surveillance.features.i3d import I3DActorExtractor
from surveillance.training.sultani_trainer import select_device


def manifest_reference(path: Path, output_manifest: Path) -> str:
    """Use portable relative references, or absolute paths across Windows drives."""
    path = path.resolve()
    try:
        return Path(os.path.relpath(path, output_manifest.resolve().parent)).as_posix()
    except ValueError:
        return path.as_posix()


def extract_features(modality: str) -> None:
    """Extract [N,D] arrays for every manifest scene while preserving actor order."""
    parser = argparse.ArgumentParser(description=f"Extract {modality} actor features locally")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--box-source", choices=["ground_truth", "detections"], default="ground_truth"
    )
    parser.add_argument(
        "--detections", type=Path, help="Detection JSONL for --box-source detections"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Vetted feature-only TorchScript archive; see docs/actor-backbones.md",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--image-size", nargs=2, type=int, default=[480, 720], metavar=("HEIGHT", "WIDTH")
    )
    args = parser.parse_args()
    detected = args.box_source == "detections"
    if detected != (args.detections is not None):
        raise ValueError("--detections is required exactly when --box-source detections")
    device = select_device(args.device)
    extractor_type = HRNetPoseExtractor if modality == "pose" else I3DActorExtractor
    extractor = extractor_type(args.checkpoint, frozen=True).to(device).eval()
    settings = dict(
        split=None, mode=f"{modality}_only", input_mode="raw", image_size=tuple(args.image_size)
    )
    dataset = (
        DetectedActorDataset(args.manifest, args.detections, **settings)
        if detected
        else ActorFeatureDataset(args.manifest, **settings)
    )
    destination = args.feature_dir.resolve() / ("detections" if detected else "") / modality
    destination.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    records = []

    def relocated(value: str | None) -> str | None:
        if value is None:
            return None
        source = args.detections if detected else args.manifest
        return manifest_reference(resolve_path(value, source), args.output_manifest)

    with torch.inference_mode():
        for index, record in enumerate(dataset.records):
            sample = dataset[index]
            if detected and len(sample["actor_boxes"]) == 0:
                features = np.empty((0, extractor.feature_dim), dtype=np.float32)
            else:
                batch = (collate_actor_inputs if detected else collate_actors)([sample])
                frames = batch["frames"].to(device)
                if modality == "pose":
                    frames = frames[:, 5]
                features = (
                    extractor(
                        frames,
                        batch["actor_boxes"].to(device),
                        batch["actor_valid_mask"].to(device),
                    )[0]
                    .cpu()
                    .numpy()
                )
            identity = f"{record.dataset}/{record.video_id}/{record.clip_id}"
            if detected:
                det = dataset.detections[index]
                identity += detection_fingerprint(det.result) + checkpoint_hash
                identity += source_fingerprint(record, args.manifest) + str(args.image_size)
            filename = hashlib.sha256(identity.encode()).hexdigest()[:24]
            path = destination / f"{filename}.npy"
            np.save(path, features)
            provenance = {
                "dataset": record.dataset,
                "clip_id": record.clip_id,
                "source_video_id": record.source_video_id,
                "actor_boxes": sample["actor_boxes"].tolist() if detected else record.actor_boxes,
                "box_source": args.box_source,
                "checkpoint_sha256": checkpoint_hash,
                "backbone": extractor.metadata,
                "image_size": args.image_size,
                "shape": list(features.shape),
            }
            if detected:
                provenance["detection_fingerprint"] = detection_fingerprint(det.result)
                provenance["source_fingerprint"] = source_fingerprint(record, args.manifest)
                with path.open("rb") as stream:
                    provenance["feature_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            path.with_suffix(".json").write_text(
                json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8"
            )
            output_record = det if detected else record
            changes = {
                f"{name}_feature_path": relocated(getattr(output_record, f"{name}_feature_path"))
                for name in ("pose", "rgb")
            }
            if detected:
                changes["feature_metadata"] = {**det.feature_metadata, modality: provenance}
            else:
                changes["frame_paths"] = [relocated(value) for value in record.frame_paths]
            changes[f"{modality}_feature_path"] = manifest_reference(path, args.output_manifest)
            records.append(replace(output_record, **changes))
            print(f"[{index + 1}/{len(dataset)}] {record.clip_id}: {features.shape} -> {path}")
    (write_detections if detected else write_actor_manifest)(records, args.output_manifest)
    print(f"Wrote {args.box_source} feature manifest {args.output_manifest}")
