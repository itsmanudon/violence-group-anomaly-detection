"""Actor representations, contextual attention, and dual classification heads."""

import torch
from torch import nn

from surveillance.models.actor_transformer.encoder import ActorEncoder
from surveillance.models.actor_transformer.fusion import fuse_log_probabilities
from surveillance.models.actor_transformer.positional_encoding import SpatialPositionEncoding

MODES = {"pose_only", "rgb_only", "pose_rgb_early_fusion", "pose_rgb_late_fusion"}


def validate_actor_mask(features: torch.Tensor, valid: torch.Tensor) -> None:
    """Require Boolean [B,N] masking with at least one real actor per scene."""
    if valid.dtype != torch.bool or features.shape[:2] != valid.shape or valid.ndim != 2:
        raise ValueError("actor_valid_mask must be Boolean [B,N] matching features")
    if valid.shape[0] == 0 or not valid.any(dim=1).all():
        raise ValueError("Every scene must contain at least one valid actor")


def masked_max(features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Max pool [B,N,D] to [B,D], excluding padded tokens even if negative."""
    validate_actor_mask(features, valid)
    return features.masked_fill(~valid.unsqueeze(-1), float("-inf")).amax(dim=1)


class ActorBranch(nn.Module):
    """One independent transformer and its individual/group classifiers."""

    def __init__(
        self, embedding_dim: int, num_actor_classes: int, num_group_classes: int, transformer: dict
    ) -> None:
        super().__init__()
        settings = dict(transformer)
        position_enabled = settings.pop("positional_encoding", True)
        reference = settings.pop("reference_size", (480, 720))
        self.position = (
            SpatialPositionEncoding(embedding_dim, reference) if position_enabled else None
        )
        self.encoder = ActorEncoder(embedding_dim, **settings)
        self.actor_classifier = nn.Linear(embedding_dim, num_actor_classes)
        self.group_classifier = nn.Linear(embedding_dim, num_group_classes)

    def forward(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        return_attention: bool,
    ) -> dict:
        """Contextualize actor embeddings and produce masked predictions."""
        if self.position is not None:
            centers = (boxes[..., :2] + boxes[..., 2:]) / 2
            features = features + self.position(centers)
        features, attention = self.encoder(features, valid, return_attention)
        return {
            "actor_logits": self.actor_classifier(features).masked_fill(~valid[..., None], 0),
            "group_logits": self.group_classifier(masked_max(features, valid)),
            "actor_features": features,
            "attention": attention,
        }


class ActorTransformer(nn.Module):
    """Independent actor-scene model with four configurable feature modes.

    Inputs: pose [B,N,Dp], RGB [B,N,Dr], normalized xyxy boxes [B,N,4], valid
    mask [B,N]. Outputs: actor_logits [B,N,Ca], group_logits [B,Cg]. For late
    fusion logits are normalized log probabilities; each modality owns its
    encoder and heads. Optional attention maps are keyed by branch name.
    """

    def __init__(
        self,
        mode: str = "pose_only",
        pose_feature_dim: int = 98304,
        rgb_feature_dim: int = 20800,
        embedding_dim: int = 128,
        num_actor_classes: int = 5,
        num_group_classes: int = 5,
        transformer: dict | None = None,
        late_fusion_pose_weight: float = 2.0,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown feature mode {mode}; expected one of {sorted(MODES)}")
        if (
            min(
                pose_feature_dim,
                rgb_feature_dim,
                embedding_dim,
                num_actor_classes,
                num_group_classes,
            )
            < 1
        ):
            raise ValueError("Feature dimensions and class counts must be positive")
        self.mode = mode
        self.pose_feature_dim = pose_feature_dim
        self.rgb_feature_dim = rgb_feature_dim
        self.late_fusion_pose_weight = late_fusion_pose_weight
        self.pose_projection = (
            nn.Linear(pose_feature_dim, embedding_dim) if mode != "rgb_only" else None
        )
        self.rgb_projection = (
            nn.Linear(rgb_feature_dim, embedding_dim) if mode != "pose_only" else None
        )
        self.early_projection = (
            nn.Linear(2 * embedding_dim, embedding_dim) if mode == "pose_rgb_early_fusion" else None
        )
        names = (
            ["pose", "rgb"]
            if mode == "pose_rgb_late_fusion"
            else ["fused" if self.early_projection else mode.split("_")[0]]
        )
        self.branches = nn.ModuleDict(
            {
                name: ActorBranch(
                    embedding_dim, num_actor_classes, num_group_classes, transformer or {}
                )
                for name in names
            }
        )

    @staticmethod
    def _project(
        features: torch.Tensor | None, projection: nn.Linear, valid: torch.Tensor, name: str
    ) -> torch.Tensor:
        if features is None or features.shape != (*valid.shape, projection.in_features):
            raise ValueError(f"{name}_features must have shape [B,N,{projection.in_features}]")
        if not torch.isfinite(features[valid]).all():
            raise ValueError(f"Valid {name} features must be finite")
        # masked_fill, rather than multiplication, safely clears even NaN padding.
        return projection(features.masked_fill(~valid[..., None], 0)).masked_fill(
            ~valid[..., None], 0
        )

    def forward(
        self,
        pose_features: torch.Tensor | None = None,
        rgb_features: torch.Tensor | None = None,
        actor_boxes: torch.Tensor | None = None,
        actor_valid_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict:
        """Predict observable benchmark activity using supplied annotated actors."""
        if actor_boxes is None or actor_valid_mask is None:
            raise ValueError("actor_boxes and actor_valid_mask are required")
        valid = actor_valid_mask
        validate_actor_mask(actor_boxes, valid)
        if actor_boxes.shape != (*valid.shape, 4):
            raise ValueError("actor_boxes must be normalized xyxy [B,N,4]")
        boxes = actor_boxes.masked_fill(~valid[..., None], 0)
        selected = boxes[valid]
        if (
            not torch.isfinite(selected).all()
            or ((selected < 0) | (selected > 1)).any()
            or (selected[:, 2:] <= selected[:, :2]).any()
        ):
            raise ValueError("Valid actor boxes require finite normalized xyxy with positive area")
        embedded = {}
        if self.pose_projection is not None:
            embedded["pose"] = self._project(pose_features, self.pose_projection, valid, "pose")
        if self.rgb_projection is not None:
            embedded["rgb"] = self._project(rgb_features, self.rgb_projection, valid, "rgb")
        if self.early_projection is not None:
            embedded = {
                "fused": self.early_projection(torch.cat([embedded["pose"], embedded["rgb"]], -1))
            }
        outputs = {
            name: self.branches[name](features, boxes, valid, return_attention)
            for name, features in embedded.items()
        }
        if self.mode == "pose_rgb_late_fusion":
            result = {
                key: fuse_log_probabilities(
                    outputs["pose"][key], outputs["rgb"][key], self.late_fusion_pose_weight
                )
                for key in ("actor_logits", "group_logits")
            }
            result["actor_logits"] = result["actor_logits"].masked_fill(~valid[..., None], 0)
            result["branch_outputs"] = outputs
        else:
            result = dict(next(iter(outputs.values())))
        result["attention"] = (
            {name: output["attention"] for name, output in outputs.items()}
            if return_attention
            else {}
        )
        return result
