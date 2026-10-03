"""Iteration-based Actor Transformer training with resumable local checkpoints."""

import copy
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from surveillance.actor_config import validate_actor_config
from surveillance.datasets.actor_batch import collate_actors
from surveillance.datasets.collective import ActorFeatureDataset, read_actor_manifest
from surveillance.evaluation.group_activity_metrics import group_activity_metrics
from surveillance.models.actor_transformer import ActorTransformer
from surveillance.models.actor_transformer.loss import ActorGroupLoss
from surveillance.training.sultani_trainer import seed_everything, select_device

LOGGER = logging.getLogger(__name__)


def make_dataset(config: dict, manifest: Path, split: str | None) -> ActorFeatureDataset:
    """Build a leakage-checked split and verify its labels against model class counts."""
    model, data = config["model"], config.get("data", {})
    dataset = ActorFeatureDataset(
        manifest,
        split=split,
        mode=model["mode"],
        pose_feature_dim=model["pose_feature_dim"],
        rgb_feature_dim=model["rgb_feature_dim"],
        input_mode=data.get("input_mode", "precomputed"),
        image_size=tuple(data.get("image_size", [480, 720])),
    )
    for row in dataset.records:
        if any(label >= model["num_actor_classes"] for label in row.actor_labels):
            raise ValueError(f"{row.clip_id}: actor label outside configured actor classes")
        if row.group_label >= model["num_group_classes"]:
            raise ValueError(f"{row.clip_id}: group label outside configured group classes")
    return dataset


def to_device(batch: dict, device: torch.device) -> dict:
    """Move collated tensors while preserving JSON-compatible scene metadata."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class ActorTransformerSystem(nn.Module):
    """One serializable module containing transformer and optional raw backbones."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.model = ActorTransformer(**config["model"])
        self.extractors = nn.ModuleDict()
        if config.get("data", {}).get("input_mode", "precomputed") == "raw":
            mode = config["model"]["mode"]
            if mode != "rgb_only":
                from surveillance.features.hrnet_pose import HRNetPoseExtractor

                settings = config.get("backbones", {}).get("pose", {})
                self.extractors["pose"] = HRNetPoseExtractor(
                    settings.get("checkpoint"), frozen=settings.get("frozen", True)
                )
            if mode != "pose_only":
                from surveillance.features.i3d import I3DActorExtractor

                settings = config.get("backbones", {}).get("rgb", {})
                self.extractors["rgb"] = I3DActorExtractor(
                    settings.get("checkpoint"), frozen=settings.get("frozen", True)
                )

    def forward(self, batch: dict, return_attention: bool = False) -> dict:
        """Extract optional raw features and return actor/group logits with shared masks."""
        pose, rgb = batch.get("pose_features"), batch.get("rgb_features")
        boxes, valid = batch["actor_boxes"], batch["actor_valid_mask"]
        if self.extractors:
            frames = batch["frames"]
            if "pose" in self.extractors:
                pose = self.extractors["pose"](frames[:, frames.shape[1] // 2], boxes, valid)
            if "rgb" in self.extractors:
                rgb = self.extractors["rgb"](frames, boxes, valid)
        return self.model(
            pose_features=pose,
            rgb_features=rgb,
            actor_boxes=boxes,
            actor_valid_mask=valid,
            return_attention=return_attention,
        )


def _read_checkpoint(path: Path, device: str | torch.device) -> dict:
    saved = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(saved, dict) or saved.get("checkpoint_type") != "actor_transformer":
        raise ValueError("Expected Actor Transformer checkpoint")
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported Actor Transformer checkpoint version")
    required = {
        "config",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "iteration",
        "best_value",
        "selection_metric",
        "manifest_fingerprint",
        "backbone_metadata",
    }
    if not required.issubset(saved):
        raise ValueError("Incomplete Actor Transformer checkpoint")
    return saved


def load_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[ActorTransformerSystem, dict]:
    """Restore all weights and verify local raw-backbone preprocessing provenance."""
    saved = _read_checkpoint(path, device)
    system = ActorTransformerSystem(saved["config"])
    metadata = {name: extractor.metadata for name, extractor in system.extractors.items()}
    if metadata != saved["backbone_metadata"]:
        raise ValueError("Local backbone metadata differs from checkpoint preprocessing provenance")
    system.load_state_dict(saved["model_state"], strict=True)
    return system.to(device).eval(), saved


def _batch_metrics(predictions, batch, model_config):
    metrics = group_activity_metrics(
        batch["group_labels"],
        predictions["group_logits"].argmax(-1),
        batch["actor_labels"],
        predictions["actor_logits"].argmax(-1),
        batch["actor_valid_mask"],
        model_config["num_group_classes"],
        model_config["num_actor_classes"],
    )
    return {
        f"{level}_{metric}": metrics[level][metric]
        for level in ("group", "actor")
        for metric in ("accuracy", "macro_f1", "count")
    }


def _validation_metrics(system, dataset, batch_size, device):
    """Aggregate all validation scenes; exclude padded and ignored actor labels."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_actors)
    groups, group_predictions, actors, actor_predictions, masks = [], [], [], [], []
    system.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            predictions = system(batch)
            groups.append(batch["group_labels"].cpu())
            group_predictions.append(predictions["group_logits"].argmax(-1).cpu())
            actors.append(batch["actor_labels"].reshape(-1).cpu())
            actor_predictions.append(predictions["actor_logits"].argmax(-1).reshape(-1).cpu())
            masks.append(batch["actor_valid_mask"].reshape(-1).cpu())
    model = system.config["model"]
    metrics = group_activity_metrics(
        torch.cat(groups),
        torch.cat(group_predictions),
        torch.cat(actors),
        torch.cat(actor_predictions),
        torch.cat(masks),
        model["num_group_classes"],
        model["num_actor_classes"],
    )
    return {
        f"{level}_{metric}": metrics[level][metric]
        for level in ("group", "actor")
        for metric in ("accuracy", "macro_f1", "count")
    }


def _validation_accuracy(system, dataset, batch_size, device):
    return _validation_metrics(system, dataset, batch_size, device)["group_accuracy"]


def train(config: dict, manifest: Path, output: Path, resume: Path | None = None) -> Path:
    """Train exactly max_iterations with deterministic iteration sampling.

    Best uses validation group accuracy; without validation it explicitly uses
    negative training loss. Test records never participate in model selection.
    Resume permits extending max_iterations only and verifies manifest bytes.
    """
    config = validate_actor_config(config)
    seed = int(config.get("seed", 7))
    seed_everything(seed)
    device = select_device(config.get("device", "auto"))
    settings = config["training"]
    horizon, batch_size = int(settings["max_iterations"]), int(settings["batch_size"])
    validation_interval = int(settings.get("validation_interval", 100))
    checkpoint_interval = int(settings.get("checkpoint_interval", 100))
    if min(horizon, batch_size, validation_interval, checkpoint_interval) < 1:
        raise ValueError("Iteration, batch and interval settings must be positive")
    fingerprint = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    training = make_dataset(config, manifest, "train")
    records = read_actor_manifest(manifest)
    validation = (
        make_dataset(config, manifest, "val") if any(r.split == "val" for r in records) else []
    )
    if not len(training):
        raise ValueError("Actor training requires a nonempty train split")
    selection = "validation_group_accuracy" if len(validation) else "negative_train_loss"
    if not len(validation):
        LOGGER.warning("No validation scenes: best checkpoint uses negative training loss")
    system = ActorTransformerSystem(config).to(device)
    start, best, saved = 0, float("-inf"), None
    if resume:
        saved = _read_checkpoint(resume, device)
        previous = copy.deepcopy(saved["config"])
        previous["training"]["max_iterations"] = horizon
        if previous != config:
            raise ValueError("Resume config must match except max_iterations")
        if saved.get("manifest_fingerprint") != fingerprint:
            raise ValueError("Resume manifest fingerprint does not match checkpoint")
        if horizon <= saved["iteration"]:
            raise ValueError("Resume max_iterations must exceed completed iteration")
        if saved.get("selection_metric") != selection:
            raise ValueError("Resume checkpoint selection metric does not match dataset")
        system, saved = load_checkpoint(resume, device)
        system.config = copy.deepcopy(config)
        start, best = saved["iteration"], saved["best_value"]
    parameters = [p for p in system.parameters() if p.requires_grad]
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[
        settings["optimizer"]
    ]
    optimizer_options = {"lr": float(settings["learning_rate"])}
    if settings["optimizer"] != "sgd":
        optimizer_options.update(betas=tuple(settings["betas"]), eps=float(settings["eps"]))
    optimizer = optimizer_type(parameters, **optimizer_options)
    scheduler = MultiStepLR(
        optimizer,
        milestones=settings.get("lr_milestones", [5000, 10000]),
        gamma=float(settings.get("lr_gamma", 0.1)),
    )
    if saved:
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
    loss_fn = ActorGroupLoss(**config.get("loss", {}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    history_path = output / "history.jsonl"
    prefix = []
    if resume:
        previous_history = Path(resume).parent / "history.jsonl"
        if previous_history.is_file():
            prefix = [
                json.loads(line)
                for line in previous_history.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            prefix = [row for row in prefix if row["iteration"] <= start]
            if [row["iteration"] for row in prefix] != sorted({row["iteration"] for row in prefix}):
                raise ValueError("Resume history contains duplicate or unordered iterations")
    history_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in prefix),
        encoding="utf-8",
    )
    writer = SummaryWriter(str(output / "tensorboard"), purge_step=start + 1 if resume else None)
    history = history_path.open("a", encoding="utf-8")
    try:
        writer.add_text("selection/metric", selection, start)
        for iteration in range(start + 1, horizon + 1):
            seed_everything(seed + iteration)
            rng = np.random.default_rng(seed + iteration)
            indices = rng.choice(len(training), size=batch_size, replace=len(training) < batch_size)
            batch = to_device(collate_actors([training[int(i)] for i in indices]), device)
            system.train()
            optimizer.zero_grad(set_to_none=True)
            predictions = system(batch)
            components = loss_fn(
                predictions,
                batch["actor_labels"],
                batch["group_labels"],
                batch["actor_valid_mask"],
            )
            if not torch.isfinite(components["total"]):
                raise ValueError("Nonfinite Actor Transformer loss")
            components["total"].backward()
            clipping = settings.get("gradient_clip", 1.0)
            if clipping is not None:
                nn.utils.clip_grad_norm_(parameters, clipping)
            optimizer.step()
            scheduler.step()
            train_metrics = {name: value.detach().item() for name, value in components.items()}
            train_metrics.update(_batch_metrics(predictions, batch, config["model"]))
            train_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            for name, value in components.items():
                writer.add_scalar(f"train/{name}", value.detach().item(), iteration)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar("train/group_accuracy", train_metrics["group_accuracy"], iteration)
            writer.add_scalar("train/actor_accuracy", train_metrics["actor_accuracy"], iteration)
            writer.add_scalar("iteration", iteration, iteration)
            value = None
            validation_metrics = None
            if len(validation):
                if iteration % validation_interval == 0 or iteration == horizon:
                    validation_metrics = _validation_metrics(system, validation, batch_size, device)
                    value = validation_metrics["group_accuracy"]
                    for name, metric in validation_metrics.items():
                        writer.add_scalar(f"validation/{name}", metric, iteration)
            else:
                value = -components["total"].detach().item()
            improved = value is not None and value > best
            if improved:
                best = value
            if value is not None:
                writer.add_scalar("selection/value", value, iteration)
            history.write(
                json.dumps(
                    {
                        "iteration": iteration,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "selection_metric": selection,
                        "selection_value": value,
                        "best_value": best if np.isfinite(best) else None,
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            history.flush()
            archive = dict(
                format_version=1,
                checkpoint_type="actor_transformer",
                config=config,
                model_state=system.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                iteration=iteration,
                best_value=best,
                selection_metric=selection,
                manifest_fingerprint=fingerprint,
                manifest_path=str(Path(manifest).resolve()),
                backbone_metadata={
                    name: extractor.metadata for name, extractor in system.extractors.items()
                },
            )
            if improved:
                torch.save(archive, output / "best.pt")
            if iteration % checkpoint_interval == 0 or iteration == horizon:
                torch.save(archive, output / "last.pt")
            if iteration % validation_interval == 0 or iteration == horizon:
                LOGGER.info(
                    "iteration %d/%d loss=%.6f selection=%s device=%s",
                    iteration,
                    horizon,
                    components["total"].item(),
                    value,
                    device,
                )
    finally:
        history.close()
        writer.close()
    return output / "last.pt"
