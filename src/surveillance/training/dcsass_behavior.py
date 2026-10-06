"""Validation-selected RGB group-only training; explicit abstention accounting."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from surveillance.datasets.dcsass_actor_inputs import collate_behavior
from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.dcsass_behavior import BehaviorFeatureDataset
from surveillance.datasets.dcsass_protocol import CLASSES
from surveillance.evaluation.anomaly_metrics import binary_metrics
from surveillance.evaluation.group_activity_metrics import classification_metrics
from surveillance.experiments.dcsass_cache import write_json
from surveillance.models.actor_transformer.surveillance_transfer import (
    GroupOnlyLoss,
    transfer_rgb_model,
)
from surveillance.training.actor_transformer_trainer import ActorTransformerSystem, to_device
from surveillance.training.sultani_trainer import seed_everything, select_device


def training_class_weights(targets: list[int]) -> tuple[np.ndarray, list[int]]:
    """Balanced N/(6*n_c) weights from covered training examples alone."""
    counts = np.bincount(targets, minlength=6)
    if len(counts) != 6 or not counts.all():
        raise ValueError("Every behavior class requires covered training support")
    return (len(targets) / (6 * counts)).astype(np.float32), counts.tolist()


def load_behavior_checkpoint(
    path: Path, device: str = "cpu"
) -> tuple[ActorTransformerSystem, dict]:
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("checkpoint_type") != "dcsass_group_behavior" or saved.get("classes") != list(
        CLASSES
    ):
        raise ValueError("Expected six-class DCSASS group-only checkpoint")
    model = ActorTransformerSystem(saved["actor_config"])
    model.load_state_dict(saved["model_state"], strict=True)
    for parameter in model.model.branches["rgb"].actor_classifier.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval(), saved


@torch.inference_mode()
def predict_dataset(
    model: ActorTransformerSystem,
    dataset: BehaviorFeatureDataset,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    model.eval()
    records = []
    for batch in DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_behavior, shuffle=False
    ):
        probabilities = model(to_device(batch, device))["group_logits"].softmax(-1).cpu().tolist()
        for meta, probs in zip(batch["metadata"], probabilities, strict=True):
            records.append(
                {
                    "clip_id": meta["clip_id"],
                    "source_video_id": meta["source_video_id"],
                    "path": meta["path"],
                    "target": meta["group_label"],
                    "probabilities": probs,
                    "prediction": int(np.argmax(probs)),
                    "actor_count": meta["actor_count"],
                    "confidence": max(probs),
                    "detector_scores": meta["detection_scores"],
                    "covered": True,
                }
            )
    return records


def behavior_metrics(records: list[dict]) -> dict:
    targets = [r["target"] for r in records]
    predicted = [r["prediction"] for r in records]
    report = classification_metrics(targets, predicted, 6)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predicted, labels=list(range(6)), zero_division=0
    )
    report["balanced_accuracy"] = float(np.mean(recall[np.asarray(support) > 0]))
    report["per_class"] = {
        name: dict(
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i, name in enumerate(CLASSES)
    }
    scores = [1 - r["probabilities"][0] for r in records]
    truth = np.asarray(targets) != 0
    binary = binary_metrics(truth.astype(int), scores, threshold=0.5)
    binary["accuracy"] = float(np.mean(truth == (np.asarray(scores) >= 0.5)))
    report["binary_anomaly"] = binary
    return report


def train_behavior(
    config: dict,
    output: Path,
    *,
    cache: Path | None = None,
    preflight_iterations: int | None = None,
) -> Path:
    """Run seed 0 using validation macro F1; never load test data during training."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite a behavior run: {output}")
    if (
        config["classes"] != list(CLASSES)
        or config["checkpoint_selection"] != "validation_macro_f1"
    ):
        raise ValueError("Behavior vocabulary/checkpoint selection changed")
    checkpoint = Path(config["transfer_checkpoint"])
    if sha256(checkpoint) != config["transfer_checkpoint_sha256"]:
        raise ValueError("Collective transfer checkpoint identity changed")
    manifest = Path(config["manifest"])
    manifest_hash = sha256(manifest)
    split = json.loads(Path(config["split_receipt"]).read_text())
    if split["manifest_sha256"] != manifest_hash:
        raise ValueError("Frozen behavior split changed")
    cache = Path(cache or config["cache"])
    preflight = preflight_iterations is not None
    training = BehaviorFeatureDataset(manifest, cache, "train", preflight=preflight)
    validation = BehaviorFeatureDataset(manifest, cache, "val", preflight=preflight)
    targets = [s["group_label"] for s in training.samples]
    weights, counts = (
        (np.ones(6, dtype=np.float32), np.bincount(targets, minlength=6).tolist())
        if preflight
        else training_class_weights(targets)
    )
    seed = int(config["seed"])
    seed_everything(seed)
    device = select_device(config["device"])
    model, transfer = transfer_rgb_model(checkpoint, seed)
    model = model.to(device)
    loss_fn = GroupOnlyLoss(torch.tensor(weights, device=device))
    settings = config["training"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        parameters,
        lr=settings["learning_rate"],
        betas=tuple(settings["betas"]),
        eps=settings["eps"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings["lr_milestones"], gamma=settings["lr_gamma"]
    )
    horizon = preflight_iterations or settings["max_iterations"]
    output.mkdir(parents=True)
    resolved = {
        **config,
        "resolved_class_weights": weights.tolist(),
        "covered_training_class_counts": counts,
        "preflight_iterations": preflight_iterations,
    }
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    write_json(
        output / "training_registration.json",
        {
            "manifest_sha256": sha256(manifest),
            "split_receipt_sha256": sha256(Path(config["split_receipt"])),
            "cache_registration_sha256": sha256(cache / "registration.json"),
            "resolved_config_sha256": sha256(output / "resolved_config.yaml"),
            "transfer": transfer,
            "dry_run": preflight,
            "covered_train": len(training),
            "covered_val": len(validation),
            "train_no_actors": len(training.uncovered),
            "val_no_actors": len(validation.uncovered),
        },
    )
    writer = SummaryWriter(str(output / "tensorboard"))
    best = -1.0
    started = time.perf_counter()
    try:
        with (output / "history.jsonl").open("w", encoding="utf-8") as history:
            for iteration in range(1, horizon + 1):
                seed_everything(seed + iteration)
                rng = np.random.default_rng(seed + iteration)
                indices = rng.choice(
                    len(training),
                    size=settings["batch_size"],
                    replace=len(training) < settings["batch_size"],
                )
                batch = to_device(collate_behavior([training[int(i)] for i in indices]), device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)["group_logits"]
                loss = loss_fn(logits, batch["group_labels"])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite behavior training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"])
                optimizer.step()
                scheduler.step()
                metrics = None
                evaluate = (
                    preflight
                    or iteration % settings["validation_interval"] == 0
                    or iteration == horizon
                )
                if evaluate:
                    records = predict_dataset(model, validation, settings["batch_size"], device)
                    metrics = behavior_metrics(records)
                value = metrics["macro_f1"] if metrics else None
                improved = value is not None and value > best
                if improved:
                    best = value
                row = dict(
                    iteration=iteration,
                    loss=float(loss.detach()),
                    learning_rate=optimizer.param_groups[0]["lr"],
                    validation=metrics,
                    best_macro_f1=best,
                )
                history.write(json.dumps(row, allow_nan=False) + "\n")
                history.flush()
                writer.add_scalar("train/group_loss", row["loss"], iteration)
                writer.add_scalar("train/learning_rate", row["learning_rate"], iteration)
                if metrics:
                    writer.add_scalar("validation/macro_f1", value, iteration)
                    writer.add_scalar("validation/accuracy", metrics["accuracy"], iteration)
                    print(
                        f"iteration {iteration}/{horizon} loss={row['loss']:.4f} "
                        f"val_macro_f1={value:.4f} val_accuracy={metrics['accuracy']:.4f} "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
                archive = {
                    "format_version": 1,
                    "checkpoint_type": "dcsass_group_behavior",
                    "classes": list(CLASSES),
                    "actor_config": model.config,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "iteration": iteration,
                    "best_macro_f1": best,
                    "selection_metric": "validation_macro_f1",
                    "config": resolved,
                    "dry_run": preflight,
                    "transfer": transfer,
                    "manifest_sha256": manifest_hash,
                }
                if improved:
                    torch.save(archive, output / "best.pt")
                if iteration % settings["checkpoint_interval"] == 0 or iteration == horizon:
                    torch.save(archive, output / "last.pt")
    finally:
        writer.close()
    loaded, saved = load_behavior_checkpoint(output / "best.pt", str(device))
    reloaded = behavior_metrics(predict_dataset(loaded, validation, settings["batch_size"], device))
    if reloaded["macro_f1"] != saved["best_macro_f1"]:
        raise ValueError("Behavior checkpoint reload changed validation predictions")
    write_json(
        output / "selection.json",
        {
            "selected_iteration": saved["iteration"],
            "checkpoint_sha256": sha256(output / "best.pt"),
            "validation": reloaded,
            "dry_run": preflight,
            "selection_frozen": True,
        },
    )
    return output / "best.pt"
