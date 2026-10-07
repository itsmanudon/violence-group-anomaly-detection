"""Offline synthetic train/evaluate/infer smoke; this is not a benchmark."""

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml
from _common import run_cli, write_json

from surveillance.actor_config import DEFAULT_CONFIG, load_actor_config
from surveillance.datasets.collective import ActorRecord, write_actor_manifest
from surveillance.evaluation.group_activity_metrics import evaluate
from surveillance.inference.group_activity_pipeline import GroupActivityPipeline
from surveillance.models.actor_transformer.actor_transformer import MODES
from surveillance.training.actor_transformer_trainer import load_checkpoint, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/actor_smoke"))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--mode", choices=sorted(MODES), default="pose_only")
    args = parser.parse_args()
    root = args.output.resolve()
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["device"] = "cpu"
    config["model"].update(mode=args.mode, pose_feature_dim=8, rgb_feature_dim=6, embedding_dim=8)
    config["model"]["transformer"].update(num_heads=2, feedforward_dim=16)
    config["training"].update(
        max_iterations=args.iterations,
        batch_size=2,
        validation_interval=1,
        checkpoint_interval=1,
        lr_milestones=[2],
    )
    records = []
    for index, split in enumerate(["train", "train", "train", "val", "test", "test"]):
        count = (index % 3) + 1
        rng = np.random.default_rng(42 + index)
        pose_path, rgb_path = f"pose_{index}.npy", f"rgb_{index}.npy"
        np.save(data / pose_path, rng.normal(size=(count, 8)).astype("float32"))
        np.save(data / rgb_path, rng.normal(size=(count, 6)).astype("float32"))
        records.append(
            ActorRecord(
                dataset="synthetic_smoke",
                video_id=f"scene{index}",
                source_video_id=f"scene{index}",
                clip_id=f"clip{index}",
                split=split,
                frame_paths=[f"unused_scene{index}/frame{i}.jpg" for i in range(10)],
                frame_indices=list(range(10)),
                actor_boxes=[[0.1, 0.1, 0.6, 0.9]] * count,
                actor_labels=[index % 5] * count,
                group_label=index % 5,
                pose_feature_path=pose_path,
                rgb_feature_path=rgb_path,
            )
        )
    manifest = data / "scenes.jsonl"
    write_actor_manifest(records, manifest)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint = train(load_actor_config(config_path), manifest, root / "run")
    metrics = evaluate(checkpoint, manifest, "test", device="cpu")
    metrics["synthetic_smoke_only"] = True
    write_json(metrics, root / "metrics.json")
    system, _ = load_checkpoint(checkpoint)
    predictions = GroupActivityPipeline(system).predict_manifest(
        manifest, "test", return_attention=True
    )
    write_json({"synthetic_smoke_only": True, "scenes": predictions}, root / "predictions.json")
    print(f"Synthetic {args.mode} smoke complete: {root}. No benchmark accuracy claim.")


if __name__ == "__main__":
    run_cli(main)
