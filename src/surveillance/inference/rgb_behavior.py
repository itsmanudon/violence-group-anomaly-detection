"""Real local detected-actor RGB behavior analysis inside suspicious intervals."""

import time
from pathlib import Path

import torch

from surveillance.datasets.dcsass_audit import sha256
from surveillance.detection.config import DetectionConfig
from surveillance.detection.torchvision_detector import TorchvisionPersonDetector
from surveillance.features.i3d import I3DActorExtractor
from surveillance.training.dcsass_behavior import load_behavior_checkpoint
from surveillance.training.sultani_trainer import select_device
from surveillance.video.actor_window import read_actor_window


class RGBBehaviorAnalyzer:
    def __init__(
        self,
        behavior_checkpoint: Path,
        detector_checkpoint: Path,
        i3d_checkpoint: Path,
        device: str = "auto",
    ):
        self.device = select_device(device)
        self.model, self.saved = load_behavior_checkpoint(behavior_checkpoint, str(self.device))
        if self.saved["dry_run"]:
            raise ValueError("A preflight checkpoint cannot be used as a trained behavior model")
        for name, path in (("detector", detector_checkpoint), ("i3d", i3d_checkpoint)):
            if sha256(path) != self.saved["config"][name]["checkpoint_sha256"]:
                raise ValueError(f"Live {name} backbone differs from behavior training provenance")
        self.detector = TorchvisionPersonDetector(
            detector_checkpoint, DetectionConfig(confidence_threshold=0.7), str(self.device)
        )
        self.extractor = I3DActorExtractor(i3d_checkpoint).to(self.device).eval()

    @torch.inference_mode()
    def analyze_window(self, path: Path, start: int, end: int, num_frames: int, fps: float) -> dict:
        window = read_actor_window(path, num_frames, start=start, end=end)
        timings = dict(window["timings"])
        before = time.perf_counter()
        result = self.detector.detect(window["reference_rgb"])
        timings["person_detection_seconds"] = time.perf_counter() - before
        base = {
            "reference_frame": window["reference_frame"],
            "reference_timestamp": window["reference_frame"] / fps,
            "sampled_frame_indices": window["frame_indices"],
            "image_size": window["image_size"],
            "actor_count": len(result.boxes),
            "boxes": result.boxes.tolist(),
            "detector_scores": result.scores.tolist(),
        }
        if not len(result.boxes):
            return {
                **base,
                "status": "no_actors",
                "prediction": None,
                "probabilities": None,
                "timings": timings,
            }
        height, width = result.image_size
        boxes = (result.boxes / torch.tensor([width, height, width, height]))[None].to(self.device)
        valid = torch.ones(1, len(result.boxes), dtype=torch.bool, device=self.device)
        before = time.perf_counter()
        features = self.extractor(window["frames"][None].to(self.device), boxes, valid)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings["i3d_seconds"] = time.perf_counter() - before
        before = time.perf_counter()
        output = self.model(
            {"rgb_features": features, "actor_boxes": boxes, "actor_valid_mask": valid}
        )
        probabilities = output["group_logits"].softmax(-1)[0].cpu().tolist()
        timings["actor_transformer_seconds"] = time.perf_counter() - before
        return {
            **base,
            "status": "available",
            "prediction": probabilities.index(max(probabilities)),
            "probabilities": probabilities,
            "timings": timings,
        }
