"""Source adaptation around the unchanged Actor-Transformer inference pipeline."""

from pathlib import Path

import torch

from surveillance.datasets.detected_actors import (
    DetectedActorDataset,
    collate_actor_inputs,
    normalized_detection_boxes,
)
from surveillance.inference.group_activity_pipeline import GroupActivityPipeline


class DetectedGroupActivityPipeline:
    """Predict all retained detections; report empty scenes without self-attention.

    Confidence and absolute boxes are mapped to the same actor indices as every
    branch's attention axes. Matching labels are only attached after inference.
    """

    def __init__(self, system):
        self.base = GroupActivityPipeline(system)
        self.system, self.config = system, system.config

    def predict_sample(self, sample: dict, return_attention: bool = False) -> dict:
        """Predict one raw/cached scene, including a valid zero-detection result."""
        result = sample["detection_result"]
        if sample["actor_boxes"].shape != (len(result.boxes), 4) or not torch.allclose(
            sample["actor_boxes"].cpu(), normalized_detection_boxes(result).cpu()
        ):
            raise ValueError("Actor boxes must match ordered detection boxes")
        metadata = {
            **sample.get("metadata", {}),
            "box_source": "detections",
            "detector": result.metadata,
            "image_size": list(result.image_size),
        }
        if not len(result.boxes):
            scene = {
                "status": "no_actors_detected",
                "box_source": "detections",
                "group_prediction": None,
                "group_name": None,
                "group_probabilities": None,
                "actors": [],
                "metadata": metadata,
            }
            if return_attention:
                scene["attention"] = {}
        else:
            batch = collate_actor_inputs([{**sample, "metadata": metadata}])
            scene = self.base.predict_batch(batch, return_attention)[0]
            scene.update(status="ok", box_source="detections")
            for actor in scene["actors"]:
                index = actor["index"]
                actor["box_pixels"] = result.boxes[index].tolist()
                actor["detection_score"] = float(result.scores[index])
                actor["detector_class_id"] = int(result.class_ids[index])
                if "actor_labels" in sample:
                    label = int(sample["actor_labels"][index])
                    actor["label"] = label if label >= 0 else None
        if "group_label" in sample:
            scene["group_label"] = sample["group_label"]
        if "matching" in sample:
            matched = sample["matching"]
            mapping = dict(zip(matched.detection_indices, matched.gt_indices, strict=True))
            for actor in scene["actors"]:
                actor["matched_gt_index"] = mapping.get(actor["index"])
            scene["matching"] = matched.to_dict()
        return scene

    def predict_detections(
        self,
        result,
        *,
        pose_features=None,
        rgb_features=None,
        frames=None,
        metadata=None,
        return_attention: bool = False,
    ) -> dict:
        """Predict explicit absolute-pixel detector results with aligned features.

        Features are [N,D]; frames, when the checkpoint uses raw mode, are
        [10,3,H,W]. This API requires no annotations or actor class targets.
        """
        sample = {
            "actor_boxes": normalized_detection_boxes(result),
            "detection_result": result,
            "metadata": metadata or {},
        }
        for name, value in (
            ("pose_features", pose_features),
            ("rgb_features", rgb_features),
            ("frames", frames),
        ):
            if value is not None:
                sample[name] = value
        return self.predict_sample(sample, return_attention)

    @torch.inference_mode()
    def predict_clip(
        self,
        frames: torch.Tensor,
        detector,
        *,
        pose_extractor=None,
        rgb_extractor=None,
        return_attention: bool = False,
    ) -> dict:
        """Detect the native center frame, then reuse raw or provided feature adapters.

        The detector receives RGB [3,H,W] in [0,1]. No temporal identity tracking
        is needed: all RoIs refer to frame index 5. Precomputed-mode checkpoints
        require supplied local extractors for their enabled modalities.
        """
        if (
            frames.ndim != 4
            or frames.shape[:2] != (10, 3)
            or min(frames.shape[2:]) < 1
            or not frames.is_floating_point()
            or not torch.isfinite(frames).all()
            or (frames < 0).any()
            or (frames > 1).any()
        ):
            raise ValueError("frames must be finite RGB [10,3,H,W] in [0,1]")
        result = detector.detect(frames[5])
        if tuple(result.image_size) != tuple(frames.shape[-2:]):
            raise ValueError("Detector image_size does not match the input center frame")
        if not len(result.boxes):
            return self.predict_detections(result, return_attention=return_attention)
        device = self.base.device
        frames = frames.to(device)
        target_size = tuple(self.config["data"]["image_size"])
        frames = torch.nn.functional.interpolate(
            frames, size=target_size, mode="bilinear", align_corners=False
        )
        if self.config["data"]["input_mode"] == "raw":
            return self.predict_detections(result, frames=frames, return_attention=return_attention)
        boxes = normalized_detection_boxes(result).to(device)[None]
        valid = torch.ones(boxes.shape[:2], dtype=torch.bool, device=device)
        features = {}
        mode = self.config["model"]["mode"]
        for name, extractor in (("pose", pose_extractor), ("rgb", rgb_extractor)):
            if mode == ("rgb_only" if name == "pose" else "pose_only"):
                continue
            if extractor is None:
                raise ValueError(f"Live {name} features require a configured local extractor")
            inputs = frames[5:6] if name == "pose" else frames[None]
            features[f"{name}_features"] = extractor.to(device).eval()(inputs, boxes, valid)[0]
        return self.predict_detections(result, **features, return_attention=return_attention)

    def predict_manifest(
        self,
        manifest: Path,
        detections: Path,
        split: str | None = "test",
        iou_threshold: float = 0.5,
        return_attention: bool = False,
    ) -> list[dict]:
        """Evaluate cached detections/features or raw features with the saved config."""
        model, data = self.config["model"], self.config["data"]
        dataset = DetectedActorDataset(
            manifest,
            detections,
            split,
            model["mode"],
            model["pose_feature_dim"],
            model["rgb_feature_dim"],
            data["input_mode"],
            tuple(data["image_size"]),
            iou_threshold,
        )
        scenes = []
        for sample in dataset:
            if (
                sample["group_label"] >= model["num_group_classes"]
                or (sample["actor_labels"] >= model["num_actor_classes"]).any()
            ):
                raise ValueError("Scene labels outside configured class vocabulary")
            scenes.append(self.predict_sample(sample, return_attention))
        return scenes
