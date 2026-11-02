"""Separate binary heads; existing six-class checkpoints and outputs stay intact."""

import copy

import torch
from torch import nn

from surveillance.models.actor_transformer.actor_transformer import masked_max


class FrozenBagBinaryHead(nn.Module):
    """Mean aggregate frozen C3D segments then classify a weakly labeled clip.

    This lightweight whole-clip baseline is not a replacement MIL localizer or
    an Actor-Transformer result. Temporal labels are not manufactured.
    """

    def __init__(self, feature_dim: int = 4096):
        super().__init__()
        if type(feature_dim) is not int or feature_dim < 1:
            raise ValueError("Positive feature dimension required")
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 3
            or features.shape[-1] != self.feature_dim
            or min(features.shape[:2]) < 1
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Expected finite frozen features [B,S,D]")
        return self.classifier(features.mean(1))


class BinaryActorAdapter(nn.Module):
    """New group-only head on a copied pretrained RGB actor representation.

    By default every representation weight is frozen and kept in eval mode.
    classifier_tuning permits only RGB projection/encoder changes, not I3D or
    person-detector changes. No actor targets are implied by binary clip labels.
    """

    def __init__(self, system: nn.Module, classifier_tuning: bool = False):
        super().__init__()
        if system.model.mode != "rgb_only" or system.extractors:
            raise ValueError("Binary actor adapter requires cached RGB-only actor features")
        self.representation = copy.deepcopy(system)
        self.classifier_tuning = classifier_tuning
        for parameter in self.representation.parameters():
            parameter.requires_grad_(False)
        branch = self.representation.model.branches["rgb"]
        self.classifier = nn.Linear(branch.group_classifier.in_features, 2)
        if classifier_tuning:
            for module in (self.representation.model.rgb_projection, branch.encoder):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.classifier_tuning:
            self.representation.eval()
        return self

    def forward(self, batch: dict) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.classifier_tuning):
            output = self.representation(batch)
            pooled = masked_max(output["actor_features"], batch["actor_valid_mask"])
        return self.classifier(pooled)
