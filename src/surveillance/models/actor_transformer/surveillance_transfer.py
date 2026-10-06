"""Collective RGB representation transfer with genuine group-only supervision."""

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from surveillance.datasets.dcsass_audit import sha256
from surveillance.training.actor_transformer_trainer import ActorTransformerSystem, load_checkpoint
from surveillance.training.sultani_trainer import seed_everything


class GroupOnlyLoss(nn.Module):
    """Six-class cross entropy; actor predictions/targets are never consumed."""

    def __init__(self, weights: torch.Tensor | None = None):
        super().__init__()
        if weights is not None and (
            weights.shape != (6,) or not torch.isfinite(weights).all() or (weights <= 0).any()
        ):
            raise ValueError("Group class weights require six positive finite values")
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape != (len(labels), 6) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite six-class group logits")
        return F.cross_entropy(logits, labels, weight=self.weights)


def transfer_rgb_model(checkpoint: Path, seed: int = 0) -> tuple[ActorTransformerSystem, dict]:
    """Transfer all RGB representation weights, then initialize a fresh group head."""
    source, saved = load_checkpoint(checkpoint, "cpu")
    if source.model.mode != "rgb_only" or source.model.rgb_feature_dim != 20800:
        raise ValueError("Transfer requires the validated RGB-only 20,800-feature architecture")
    seed_everything(seed)
    config = copy.deepcopy(saved["config"])
    config["model"]["num_group_classes"] = 6
    model = ActorTransformerSystem(config)
    prefix = "model.branches.rgb.group_classifier."
    transferable = {
        key: value for key, value in source.state_dict().items() if not key.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    if set(missing) != {prefix + "weight", prefix + "bias"} or unexpected:
        raise ValueError("Unexpected transfer parameter mismatch")
    for parameter in model.model.branches["rgb"].actor_classifier.parameters():
        parameter.requires_grad_(False)
    receipt = {
        "source_checkpoint_sha256": sha256(checkpoint),
        "source_selected_iteration": saved["iteration"],
        "source_classes": config["model"]["num_actor_classes"],
        "target_group_classes": 6,
        "supervision": "group_only",
        "transferred_keys": sorted(transferable),
        "reset_keys": sorted(missing),
        "unused_actor_head": "structurally retained, frozen, not evaluated",
    }
    return model, receipt
