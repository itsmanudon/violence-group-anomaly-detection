"""Five real cached-feature MIL iterations and checkpoint reload; no held-out evaluation."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from _common import run_cli, write_json

from surveillance.datasets.common import check_leakage, read_manifest
from surveillance.datasets.features import record_features
from surveillance.models.sultani.loss import MILLoss
from surveillance.models.sultani.model import SultaniScorer
from surveillance.training.sultani_trainer import load_checkpoint, seed_everything, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preflight output already exists")
    config = yaml.safe_load(args.config.read_text())
    rows = read_manifest(args.manifest)
    check_leakage(rows)
    if any(r.split == "test" for r in rows):
        raise ValueError("Sultani preflight must not contain held-out test clips")
    seed_everything(config["seed"])
    device = select_device(config["device"])
    model = SultaniScorer(feature_dim=config["feature_dim"], **config["model"]).to(device)
    loss_fn = MILLoss(**config["loss"])
    optimizer = torch.optim.Adagrad(model.parameters(), lr=config["training"]["learning_rate"])
    groups = [[r for r in rows if r.split == "train" and r.label == label] for label in (1, 0)]
    if not all(groups):
        raise ValueError("Preflight needs positive and normal training clips")
    bags = [
        torch.stack([record_features(r, args.manifest, 32, 4096) for r in group]).to(device)
        for group in groups
    ]
    losses = []
    for iteration in range(5):
        seed_everything(config["seed"] + iteration)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        parts = loss_fn(model(bags[0]), model(bags[1]))
        loss = parts["total"] + config["training"]["weight_l2"] * sum(
            p.square().sum() for n, p in model.named_parameters() if n.endswith("weight")
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite real Sultani preflight loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite Sultani gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
    args.output.mkdir(parents=True)
    checkpoint = args.output / "preflight.pt"
    torch.save(
        dict(
            format_version=1,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config=config,
            epoch=1,
            best_value=-losses[-1],
            selection_metric="preflight_only",
            dry_run=True,
        ),
        checkpoint,
    )
    reloaded, _ = load_checkpoint(checkpoint, device)
    model.eval()
    with torch.inference_mode():
        original = model(bags[0])
        restored = reloaded(bags[0])
    torch.testing.assert_close(original, restored, rtol=0, atol=0)
    write_json(
        {
            "iterations": 5,
            "losses": losses,
            "finite": bool(np.isfinite(losses).all()),
            "checkpoint_reload_equal": True,
            "dry_run": True,
            "held_out_evaluated": False,
        },
        args.output / "report.json",
    )
    print(losses, "checkpoint reload passed")


if __name__ == "__main__":
    run_cli(main)
