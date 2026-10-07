"""Strict variable-actor batching; True denotes valid actors throughout the pipeline."""

import torch


def collate_actors(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("Cannot collate an empty actor batch")
    optional = {key for key in ("pose_features", "rgb_features", "frames") if key in samples[0]}
    counts = []
    for sample in samples:
        present = {key for key in ("pose_features", "rgb_features", "frames") if key in sample}
        if present != optional:
            raise ValueError("All samples in a batch must have matching modalities")
        boxes, labels = sample["actor_boxes"], sample["actor_labels"]
        if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] == 0:
            raise ValueError("Each scene requires nonempty [N,4] actor boxes")
        n = boxes.shape[0]
        if (
            not torch.isfinite(boxes).all()
            or (boxes < 0).any()
            or (boxes > 1).any()
            or (boxes[:, 2:] <= boxes[:, :2]).any()
        ):
            raise ValueError("Boxes must be finite normalized xyxy with positive area")
        if labels.shape != (n,) or labels.dtype != torch.long or (labels < 0).any():
            raise ValueError("Actor labels must be matching [N] nonnegative long integers")
        group = sample["group_label"]
        if type(group) is not int or group < 0:
            raise ValueError("Group label must be a nonnegative integer")
        for key in optional:
            data = sample[key]
            if not data.is_floating_point() or not torch.isfinite(data).all():
                raise ValueError(f"{key} must contain finite floating point values")
            if key == "frames":
                if data.ndim != 4 or data.shape[0:2] != (10, 3) or min(data.shape[2:]) == 0:
                    raise ValueError("frames must have shape [10,3,H,W]")
                if (data < 0).any() or (data > 1).any():
                    raise ValueError("frames must contain RGB values in [0,1]")
            elif data.ndim != 2 or data.shape[0] != n or data.shape[1] == 0:
                raise ValueError(f"{key} must have shape [N,D]")
            if data.shape[1:] != samples[0][key].shape[1:]:
                raise ValueError(f"{key} dimensions differ between samples")
        counts.append(n)
    batch_size, actors = len(samples), max(counts)
    batch = {
        "actor_boxes": torch.zeros(batch_size, actors, 4),
        "actor_labels": torch.full((batch_size, actors), -100, dtype=torch.long),
        "actor_valid_mask": torch.zeros(batch_size, actors, dtype=torch.bool),
        "group_labels": torch.tensor([s["group_label"] for s in samples], dtype=torch.long),
        "metadata": [s.get("metadata", {}) for s in samples],
    }
    for key in optional - {"frames"}:
        batch[key] = torch.zeros(batch_size, actors, samples[0][key].shape[1])
    for i, (sample, n) in enumerate(zip(samples, counts, strict=True)):
        batch["actor_boxes"][i, :n] = sample["actor_boxes"]
        batch["actor_labels"][i, :n] = sample["actor_labels"]
        batch["actor_valid_mask"][i, :n] = True
        for key in optional - {"frames"}:
            batch[key][i, :n] = sample[key]
    if "frames" in optional:
        batch["frames"] = torch.stack([s["frames"] for s in samples])
    return batch
