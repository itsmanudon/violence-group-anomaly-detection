"""Reproducible precomputed-feature MIL training and portable checkpoints."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from surveillance.datasets.common import check_leakage, read_manifest
from surveillance.datasets.features import record_features
from surveillance.evaluation.anomaly_metrics import binary_metrics
from surveillance.models.sultani.loss import MILLoss
from surveillance.models.sultani.model import SultaniScorer

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch; require deterministic torch operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA when available; fall back to CPU for unavailable CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[SultaniScorer, dict]:
    """Load tensor/primitive-only scorer checkpoint with architecture metadata."""
    saved = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(saved, dict) or saved.get("format_version") != 1:
        raise ValueError("Expected surveillance format_version=1 scorer checkpoint")
    config = saved["config"]
    model = SultaniScorer(feature_dim=config["feature_dim"], **config["model"])
    model.load_state_dict(saved["model_state"], strict=True)
    return model.to(device).eval(), saved


def train(config: dict, manifest: Path, output: Path, resume: Path | None = None) -> Path:
    """Train abnormal/normal pairs; save last and best after each epoch.

    batch_size counts PAIRS. Each epoch visits every bag in the larger class;
    the smaller class is cycled through independently shuffled indices.
    Best = maximum validation bag ROC-AUC when both classes are available;
    otherwise minimum training objective (explicitly logged, not test selection).
    Resumption restores optimizer and best value. Per-epoch seeding makes an
    uninterrupted and resumed run deterministic on the same device/software.
    """
    seed = int(config.get("seed", 7))
    seed_everything(seed)
    device = select_device(config.get("device", "auto"))
    records = read_manifest(manifest)
    check_leakage(records)
    settings = config["training"]
    batch_size = int(settings.get("batch_size", 30))
    epochs = int(settings.get("epochs", 20))
    if min(batch_size, epochs) < 1:
        raise ValueError("batch_size and epochs must be positive")
    segments, dimension = config["num_segments"], config["feature_dim"]
    classes = [[r for r in records if r.split == "train" and r.label == label] for label in (1, 0)]
    if not all(classes):
        raise ValueError("Training split requires both normal (0) and abnormal (1) bags")
    validation = [r for r in records if r.split == "val"]
    use_auc = {r.label for r in validation} == {0, 1}
    if not use_auc:
        LOGGER.warning("No two-class validation split: best checkpoint uses training loss")
    model = SultaniScorer(feature_dim=dimension, **config["model"]).to(device)
    loss_fn = MILLoss(**config.get("loss", {}))
    optimizers = {"adagrad": torch.optim.Adagrad, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    name = settings.get("optimizer", "adagrad").lower()
    if name not in optimizers:
        raise ValueError(f"Unsupported optimizer {name}; choose adagrad, adam, or sgd")
    optimizer = optimizers[name](model.parameters(), lr=float(settings.get("learning_rate", 0.001)))
    start, best = 0, float("-inf")
    if resume:
        model, saved = load_checkpoint(resume, device)
        previous_config = {**saved["config"], "training": dict(saved["config"]["training"])}
        previous_config["training"]["epochs"] = epochs
        if previous_config != config:
            raise ValueError("Resume config must match checkpoint config except total epochs")
        if epochs <= saved["epoch"]:
            raise ValueError("Resume target epochs must exceed completed checkpoint epoch")
        optimizer = optimizers[name](
            model.parameters(), lr=float(settings.get("learning_rate", 0.001))
        )
        optimizer.load_state_dict(saved["optimizer_state"])
        start, best = saved["epoch"], saved["best_value"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output / "tensorboard"))
    weight_l2 = float(settings.get("weight_l2", 0.001))
    if weight_l2 < 0:
        raise ValueError("weight_l2 must be nonnegative")
    try:
        for epoch in range(start + 1, epochs + 1):
            seed_everything(seed + epoch)
            rng = np.random.default_rng(seed + epoch)
            indices = [rng.permutation(len(rows)) for rows in classes]
            count = max(map(len, classes))
            model.train()
            sums = dict(total=0.0, ranking=0.0, sparsity=0.0, smoothness=0.0, weight_l2=0.0)
            for offset in range(0, count, batch_size):
                size = min(batch_size, count - offset)
                bags = []
                for rows, order in zip(classes, indices):
                    bags.append(
                        torch.stack(
                            [
                                record_features(
                                    rows[int(order[i % len(order)])], manifest, segments, dimension
                                )
                                for i in range(offset, offset + size)
                            ]
                        ).to(device)
                    )
                optimizer.zero_grad(set_to_none=True)
                components = loss_fn(model(bags[0]), model(bags[1]))
                regularizer = weight_l2 * sum(
                    p.square().sum() for n, p in model.named_parameters() if n.endswith("weight")
                )
                components["weight_l2"] = regularizer
                components["total"] = components["total"] + regularizer
                components["total"].backward()
                optimizer.step()
                for key, value in components.items():
                    sums[key] += value.detach().item() * size
            means = {key: value / count for key, value in sums.items()}
            for key, value in means.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            model.eval()
            value = -means["total"]
            if use_auc:
                with torch.inference_mode():
                    predictions = [
                        model(record_features(r, manifest, segments, dimension).to(device))
                        .max()
                        .item()
                        for r in validation
                    ]
                metrics = binary_metrics([r.label for r in validation], predictions)
                value = metrics["roc_auc"]
                writer.add_scalar("validation/bag_roc_auc", value, epoch)
            improved = value > best
            best = max(best, value)
            saved = dict(
                format_version=1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                config=config,
                epoch=epoch,
                best_value=best,
                selection_metric="validation_bag_roc_auc" if use_auc else "negative_train_loss",
            )
            torch.save(saved, output / "last.pt")
            if improved:
                torch.save(saved, output / "best.pt")
            LOGGER.info(
                "epoch %d/%d loss=%.6f selection=%.6f device=%s",
                epoch,
                epochs,
                means["total"],
                value,
                device,
            )
    finally:
        writer.close()
    return output / "last.pt"
