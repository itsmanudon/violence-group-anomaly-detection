"""Resumable DCSASS detected-actor and frozen I3D caches bound to clip bytes."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from surveillance.datasets.dcsass_actor_inputs import coverage_report
from surveillance.datasets.dcsass_audit import sha256
from surveillance.detection.config import DetectionConfig
from surveillance.detection.person_detector import DetectionResult
from surveillance.detection.records import detection_fingerprint
from surveillance.detection.torchvision_detector import TorchvisionPersonDetector
from surveillance.features.i3d import I3DActorExtractor
from surveillance.training.sultani_trainer import seed_everything, select_device
from surveillance.video.actor_window import centered_frame_indices, read_actor_window

DETECTOR_SHA256 = "258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384"
I3D_SHA256 = "fe7fc30ca6f26430232e2e0f6bdffd8514478a071db065e452bff46f60f4e0c4"


def write_json(path: Path, value: dict) -> None:
    """Commit atomically, retrying brief Windows reader/scanner replacement locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def detection_payload(row: dict, window: dict, result: DetectionResult) -> dict:
    """Serialize actual clip-local reference geometry and empty detections."""
    return {
        "schema_version": 1,
        **{key: row[key] for key in ("clip_id", "source_video_id", "group_label", "split")},
        "video_sha256": row["sha256"],
        "reference_frame_coordinate_system": "zero_based_clip_local",
        "reference_frame": window["reference_frame"],
        "frame_indices": window["frame_indices"],
        "image_size": window["image_size"],
        "boxes": result.boxes.tolist(),
        "scores": result.scores.tolist(),
        "person_labels": result.class_ids.tolist(),
        "metadata": result.metadata,
        "detection_fingerprint": detection_fingerprint(result),
    }


def validate_detection_cache(payload: dict, row: dict, detector_hash: str) -> DetectionResult:
    """Reject stale identity, sampling, detector, filter, box or score artifacts."""
    for key in ("clip_id", "source_video_id", "group_label", "split"):
        if payload.get(key) != row[key]:
            raise ValueError(f"DCSASS cache identity mismatch: {key}")
    if (
        payload.get("schema_version") != 1
        or payload.get("video_sha256") != row["sha256"]
        or payload.get("reference_frame_coordinate_system") != "zero_based_clip_local"
        or payload.get("frame_indices") != centered_frame_indices(row["num_frames"])
        or payload.get("reference_frame") != payload["frame_indices"][5]
        or payload["metadata"].get("checkpoint_sha256") != detector_hash
        or payload["metadata"].get("filter_config")
        != asdict(DetectionConfig(confidence_threshold=0.7))
    ):
        raise ValueError("Stale DCSASS detection provenance")
    result = DetectionResult(
        torch.tensor(payload["boxes"]).reshape(-1, 4),
        torch.tensor(payload["scores"]),
        torch.tensor(payload["person_labels"], dtype=torch.long),
        tuple(payload["image_size"]),
        payload["metadata"],
    )
    if (result.class_ids != 1).any() or detection_fingerprint(result) != payload[
        "detection_fingerprint"
    ]:
        raise ValueError("Changed DCSASS detected actors")
    return result


def choose_rows(rows: list[dict], max_clips: int | None) -> list[dict]:
    """Bounded preflight uses training/validation only, spanning class targets."""
    if max_clips is None:
        return rows
    eligible = [r for r in rows if r["split"] != "test"]
    chosen = {}
    for split in ("train", "val"):
        for label in range(6):
            row = next(
                (r for r in eligible if r["split"] == split and r["group_label"] == label), None
            )
            if row:
                chosen[row["clip_id"]] = row
    for row in eligible:
        if len(chosen) >= max_clips:
            break
        chosen[row["clip_id"]] = row
    return list(chosen.values())[:max_clips]


def run_cache(
    manifest: Path,
    output: Path,
    stage: str,
    detector_checkpoint: Path,
    i3d_checkpoint: Path,
    device: str = "auto",
    max_clips: int | None = None,
) -> dict:
    """Precompute fixed deployment boxes or features; resume only verified cache rows."""
    if stage not in ("detect", "extract"):
        raise ValueError("Cache stage must be detect or extract")
    manifest, output = Path(manifest), Path(output)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len({r["clip_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate DCSASS clip IDs")
    seen = {}
    for row in rows:
        source = row["source_video_id"]
        if source in seen and seen[source] != row["split"]:
            raise ValueError("DCSASS source leakage")
        seen[source] = row["split"]
    rows = choose_rows(rows, max_clips)
    if sha256(detector_checkpoint) != DETECTOR_SHA256 or sha256(i3d_checkpoint) != I3D_SHA256:
        raise ValueError("DCSASS cache backbone asset identity changed")
    registration = {
        "manifest_sha256": sha256(manifest),
        "detector_sha256": DETECTOR_SHA256,
        "i3d_sha256": I3D_SHA256,
        "confidence_threshold": 0.7,
        "nms_iou": 0.5,
        "max_actors": 20,
        "max_clips": max_clips,
        "clip_ids": [r["clip_id"] for r in rows],
        "frame_policy": "clip-local center[-5,+4], edge replicate, reference index5",
        "rgb_policy": "opencv resize480x720 rgb0..1, frozen I3D Mixed_4f, "
        "mean-time, resize90x160, roi5x5",
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "registration.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != registration:
            raise ValueError("Cache registration changed; use a separately registered directory")
    else:
        write_json(receipt, registration)
    seed_everything(0)
    detector = (
        TorchvisionPersonDetector(
            detector_checkpoint, DetectionConfig(confidence_threshold=0.7), device
        )
        if stage == "detect"
        else None
    )
    extractor = (
        I3DActorExtractor(i3d_checkpoint).to(select_device(device)).eval()
        if stage == "extract"
        else None
    )
    started = time.perf_counter()
    payloads, timings = [], []
    for i, row in enumerate(rows, 1):
        detection_path = output / "detections" / f"{row['clip_id']}.json"
        window = None
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"Video bytes changed: {row['clip_id']}")
        before = time.perf_counter()
        if detection_path.exists():
            payload = json.loads(detection_path.read_text())
            result = validate_detection_cache(payload, row, DETECTOR_SHA256)
        elif stage == "detect":
            window = read_actor_window(Path(row["path"]), row["num_frames"])
            result = detector.detect(window["reference_rgb"])
            payload = detection_payload(row, window, result)
            write_json(detection_path, payload)
        else:
            raise FileNotFoundError(f"Missing DCSASS detection cache: {row['clip_id']}")
        if stage == "extract" and len(result.boxes):
            feature_path = output / "rgb" / f"{row['clip_id']}.npy"
            feature_sidecar = feature_path.with_suffix(".json")
            expected = {
                "clip_id": row["clip_id"],
                "source_video_id": row["source_video_id"],
                "video_sha256": row["sha256"],
                "reference_frame": payload["reference_frame"],
                "frame_indices": payload["frame_indices"],
                "detection_fingerprint": payload["detection_fingerprint"],
                "detector_sha256": DETECTOR_SHA256,
                "i3d_sha256": I3D_SHA256,
                "shape": [len(result.boxes), 20800],
            }
            if feature_path.exists() and feature_sidecar.exists():
                saved = json.loads(feature_sidecar.read_text())
                if any(saved.get(k) != v for k, v in expected.items()) or saved.get(
                    "feature_sha256"
                ) != sha256(feature_path):
                    raise ValueError(f"Stale RGB cache: {row['clip_id']}")
                values = np.load(feature_path, allow_pickle=False)
            else:
                window = window or read_actor_window(Path(row["path"]), row["num_frames"])
                height, width = result.image_size
                boxes = result.boxes / torch.tensor([width, height, width, height])
                target_device = next(extractor.buffers()).device
                with torch.inference_mode():
                    values = (
                        extractor(
                            window["frames"][None].to(target_device),
                            boxes[None].to(target_device),
                            torch.ones(1, len(boxes), dtype=torch.bool, device=target_device),
                        )[0]
                        .cpu()
                        .numpy()
                    )
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(feature_path, values)
                write_json(
                    feature_sidecar,
                    {
                        **expected,
                        "feature_sha256": sha256(feature_path),
                        "backbone_metadata": extractor.metadata,
                    },
                )
            if values.shape != (len(result.boxes), 20800) or not np.isfinite(values).all():
                raise ValueError("Invalid DCSASS I3D tensor")
        payloads.append(payload)
        timings.append(time.perf_counter() - before)
        if i % 100 == 0 or i == len(rows):
            print(
                f"{stage}: {i}/{len(rows)} clips, {time.perf_counter() - started:.1f}s", flush=True
            )
    report = coverage_report(payloads)
    report.update(
        stage=stage,
        seconds=time.perf_counter() - started,
        bounded_preflight=max_clips is not None,
        latency_quantiles_sec=np.quantile(timings, [0, 0.5, 0.95, 1]).tolist(),
    )
    report["by_split"] = {
        split: coverage_report([p for p in payloads if p["split"] == split])
        for split in ("train", "val", "test")
        if any(p["split"] == split for p in payloads)
    }
    write_json(output / f"{stage}_report.json", report)
    return report
