"""Prediction fusion that preserves probability semantics."""

import math

import torch
import torch.nn.functional as F


def fuse_log_probabilities(
    pose_logits: torch.Tensor, rgb_logits: torch.Tensor, pose_weight: float = 2.0
) -> torch.Tensor:
    """Return log((w*softmax(pose)+softmax(rgb))/(w+1)) stably.

    The paper weights static predictions twice dynamic predictions. The choice
    of normalized probabilities as scores is explicit; these log probabilities
    can be passed directly to cross_entropy or softmax without double softmax.
    """
    if not math.isfinite(pose_weight) or pose_weight <= 0:
        raise ValueError("late_fusion_pose_weight must be finite and positive")
    if pose_logits.shape != rgb_logits.shape:
        raise ValueError("Late fusion branches must predict identical class shapes")
    return torch.logaddexp(
        F.log_softmax(pose_logits, -1) + math.log(pose_weight), F.log_softmax(rgb_logits, -1)
    ) - math.log(pose_weight + 1)
