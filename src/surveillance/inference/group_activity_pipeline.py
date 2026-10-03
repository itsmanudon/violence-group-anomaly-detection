"""Serializable scene/actor predictions with optional unpadded attention."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from surveillance.datasets.actor_batch import collate_actors
from surveillance.training.actor_transformer_trainer import make_dataset, to_device


class GroupActivityPipeline:
    """Inference with the same feature extraction and masks used during training."""

    def __init__(self, system, config: dict | None = None):
        self.system = system.eval()
        self.config = config if config is not None else system.config
        self.device = next(system.parameters()).device

    def _vocabulary(self, kind: str, count: int) -> list[str]:
        labels = self.config.get(f"{kind}_classes")
        if labels is None:
            defaults = ["crossing", "waiting", "queueing", "walking", "talking"]
            return defaults if count == 5 else [str(index) for index in range(count)]
        if len(labels) != count:
            raise ValueError(f"Configured {kind} vocabulary must contain {count} names")
        return list(labels)

    @torch.inference_mode()
    def predict_batch(self, batch: dict, return_attention: bool = False) -> list[dict]:
        self.system.eval()
        batch = to_device(batch, self.device)
        output = self.system(batch, return_attention=return_attention)
        group_probabilities = output["group_logits"].softmax(-1)
        actor_probabilities = output["actor_logits"].softmax(-1)
        group_names = self._vocabulary("group", group_probabilities.shape[-1])
        actor_names = self._vocabulary("actor", actor_probabilities.shape[-1])
        results = []
        for index, probabilities in enumerate(group_probabilities):
            valid = batch["actor_valid_mask"][index]
            group_prediction = int(probabilities.argmax())
            scene = {
                "metadata": batch.get("metadata", [{}] * len(group_probabilities))[index],
                "group_prediction": group_prediction,
                "group_name": group_names[group_prediction],
                "group_probabilities": probabilities.cpu().tolist(),
                "group_vocabulary": group_names,
                "actor_vocabulary": actor_names,
                "actors": [],
            }
            if "group_labels" in batch:
                scene["group_label"] = int(batch["group_labels"][index])
            for actor_index in valid.nonzero(as_tuple=False).flatten().tolist():
                values = actor_probabilities[index, actor_index]
                prediction = int(values.argmax())
                actor = {
                    "index": actor_index,
                    "prediction": prediction,
                    "name": actor_names[prediction],
                    "probabilities": values.cpu().tolist(),
                    "box": batch["actor_boxes"][index, actor_index].cpu().tolist(),
                }
                if "actor_labels" in batch:
                    actor["label"] = int(batch["actor_labels"][index, actor_index])
                scene["actors"].append(actor)
            if return_attention:
                attention = output.get("attention", {})
                if isinstance(attention, torch.Tensor):
                    attention = {"fused": attention}
                scene["attention"] = {
                    branch: weights[index][..., valid, :][..., valid].cpu().tolist()
                    for branch, weights in attention.items()
                }
            results.append(scene)
        return results

    def predict_manifest(
        self,
        manifest: Path,
        split: str | None = "test",
        batch_size: int | None = None,
        return_attention: bool = False,
    ) -> list[dict]:
        dataset = make_dataset(self.config, manifest, split)
        if not len(dataset):
            raise ValueError(f"No scenes available for split {split!r}")
        batch_size = batch_size or int(self.config["training"]["batch_size"])
        results = []
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_actors):
            results.extend(self.predict_batch(batch, return_attention))
        return results
