"""Offline detector/feature/robustness smoke; synthetic fixtures, not research weights."""

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from _common import run_cli, write_json
from torch import nn

from surveillance.actor_config import validate_actor_config
from surveillance.datasets.collective import ActorRecord, write_actor_manifest
from surveillance.datasets.detected_actors import normalized_detection_boxes, source_fingerprint
from surveillance.detection.config import DetectionConfig
from surveillance.detection.person_detector import filter_detections
from surveillance.detection.records import DetectionRecord, detection_fingerprint, write_detections
from surveillance.evaluation.box_robustness import evaluate_box_robustness
from surveillance.features.geometry import roi_align_actors
from surveillance.inference.detected_group_activity import DetectedGroupActivityPipeline
from surveillance.training.actor_transformer_trainer import load_checkpoint, train


class SyntheticColorFeatures(nn.Module):
    """Three RGB crop means for software checks; NOT an HRNet or pretrained substitute."""

    def forward(self, frames, boxes, valid):
        values = roi_align_actors(frames, boxes, valid, (1, 1)).flatten(1)
        result = frames.new_zeros((*valid.shape, 3))
        result[valid] = values
        return result


class FixtureDetector:
    """Return predetermined person detections to exercise the live detector protocol."""

    def __init__(self, result):
        self.result = result

    def detect(self, image):
        if tuple(image.shape[-2:]) != self.result.image_size:
            raise ValueError("Synthetic detector image size mismatch")
        return self.result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/detection_smoke"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest, detection_path = root / "gt.jsonl", root / "detections.jsonl"
    torch.manual_seed(42)
    extractor = SyntheticColorFeatures()
    records, detections, test_frames = [], [], []
    height, width = 16, 24
    for index, split in enumerate(["train", "train", "val", "test", "test", "test"]):
        image = np.random.default_rng(index).integers(0, 256, (height, width, 3), dtype=np.uint8)
        frame_path = root / f"scene{index}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frames = torch.from_numpy(image).permute(2, 0, 1).float()[None].repeat(10, 1, 1, 1) / 255
        gt_boxes = torch.tensor([[0.05, 0.1, 0.4, 0.9], [0.6, 0.1, 0.95, 0.9]])
        if index == 0:
            gt_boxes = gt_boxes[:1]
        row = ActorRecord(
            "synthetic_detection_smoke",
            f"scene{index}",
            f"scene{index}",
            f"scene{index}:5",
            split,
            [str(frame_path)] * 10,
            list(range(10)),
            gt_boxes.tolist(),
            [index % 5] * len(gt_boxes),
            index % 5,
        )
        gt_values = extractor(
            frames[5:6], gt_boxes[None], torch.ones(1, len(gt_boxes), dtype=torch.bool)
        )[0]
        gt_feature = root / f"gt_{index}.npy"
        np.save(gt_feature, gt_values.numpy())
        records.append(replace(row, pose_feature_path=str(gt_feature)))
        boxes = gt_boxes * torch.tensor([width, height, width, height])
        if index == 4:
            boxes = torch.stack(
                [
                    boxes[0] + torch.tensor([0.1, 0.0, 0.1, 0.0]),
                    torch.tensor([10.0, 1.0, 13.0, 5.0]),
                ]
            )
        if index == 5:
            boxes = torch.empty(0, 4)
        result = filter_detections(
            boxes,
            torch.full((len(boxes),), 0.9),
            torch.ones(len(boxes), dtype=torch.long),
            (height, width),
            DetectionConfig(min_box_width=1, min_box_height=1),
        )
        result.metadata["synthetic_fixture"] = True
        det = DetectionRecord(
            row.dataset, row.video_id, row.source_video_id, row.clip_id, 5, result
        )
        if len(boxes):
            values = extractor(
                frames[5:6],
                normalized_detection_boxes(result)[None],
                torch.ones(1, len(boxes), dtype=torch.bool),
            )[0]
            feature_path = root / f"det_{index}.npy"
            np.save(feature_path, values.numpy())
            with feature_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            det = replace(
                det,
                pose_feature_path=str(feature_path),
                feature_metadata={
                    "pose": {
                        "detection_fingerprint": detection_fingerprint(result),
                        "source_fingerprint": source_fingerprint(row, manifest),
                        "feature_sha256": digest,
                        "synthetic_fixture": "RGB crop means, not HRNet",
                    }
                },
            )
        detections.append(det)
        if split == "test":
            test_frames.append(frames)
    write_actor_manifest(records, manifest)
    write_detections(detections, detection_path)
    config = validate_actor_config(
        {
            "device": "cpu",
            "data": {"image_size": [height, width]},
            "model": {"pose_feature_dim": 3, "embedding_dim": 8},
            "training": {
                "max_iterations": 2,
                "batch_size": 2,
                "validation_interval": 1,
                "checkpoint_interval": 1,
            },
        }
    )
    checkpoint = train(config, manifest, root / "run")
    comparison = evaluate_box_robustness(
        checkpoint, manifest, detection_path, device="cpu", return_attention=True
    )
    comparison["synthetic_smoke_only"] = True
    write_json(comparison, root / "comparison.json")
    system, _ = load_checkpoint(checkpoint)
    pipeline = DetectedGroupActivityPipeline(system)
    live = [
        pipeline.predict_clip(
            frames, FixtureDetector(det.result), pose_extractor=extractor, return_attention=True
        )
        for frames, det in zip(test_frames, detections[-3:], strict=True)
    ]
    assert [s["status"] for s in live] == ["ok", "ok", "no_actors_detected"]
    assert [len(s["actors"]) for s in live] == [2, 2, 0]
    assert comparison["detected_boxes"]["detection"]["missed_gt_actors"] == 3
    assert comparison["detected_boxes"]["detection"]["unmatched_detections"] == 1
    write_json({"synthetic_smoke_only": True, "scenes": live}, root / "live.json")
    print(f"Synthetic A/B/C workflows passed: {root}; no real accuracy claim.")


if __name__ == "__main__":
    run_cli(main)
