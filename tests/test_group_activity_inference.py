"""Workflow contracts: padding, held-out metrics, and reproducible resumption."""

import copy
import json

import numpy as np
import pytest
import torch


def tiny_config(iterations=3):
    return {
        "seed": 13,
        "device": "cpu",
        "actor_classes": ["a", "b"],
        "group_classes": ["a", "b"],
        "data": {"input_mode": "precomputed", "image_size": [480, 720], "num_frames": 10},
        "model": {
            "mode": "pose_only",
            "pose_feature_dim": 8,
            "rgb_feature_dim": 6,
            "embedding_dim": 8,
            "num_actor_classes": 2,
            "num_group_classes": 2,
            "transformer": {
                "num_layers": 1,
                "num_heads": 2,
                "feedforward_dim": 16,
                "dropout": 0.1,
                "positional_encoding": True,
                "reference_size": [480, 720],
            },
        },
        "loss": {"group_weight": 1.0, "actor_weight": 1.0},
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "max_iterations": iterations,
            "lr_milestones": [2],
            "lr_gamma": 0.1,
            "batch_size": 2,
            "gradient_clip": 1.0,
            "validation_interval": 1,
            "checkpoint_interval": 1,
        },
    }


def tiny_manifest(tmp_path):
    rows = []
    for index, (split, count) in enumerate(
        [("train", 1), ("train", 3), ("val", 1), ("val", 3), ("test", 3)]
    ):
        path = tmp_path / f"features_{index}.npy"
        np.save(path, np.random.default_rng(index).normal(size=(count, 8)).astype("float32"))
        rows.append(
            {
                "dataset": "synthetic",
                "video_id": f"video{index}",
                "source_video_id": f"video{index}",
                "clip_id": f"clip{index}",
                "split": split,
                "frame_paths": [f"scene{index}/unused_{i}.jpg" for i in range(10)],
                "frame_indices": list(range(10)),
                "actor_boxes": [[0.1, 0.1, 0.5, 0.9]] * count,
                "actor_labels": [index % 2] * count,
                "group_label": index % 2,
                "pose_feature_path": path.name,
            }
        )
    manifest = tmp_path / "scenes.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


def test_metrics_exclude_padded_actors_and_keep_missing_classes():
    from surveillance.evaluation.group_activity_metrics import group_activity_metrics

    result = group_activity_metrics(
        [0, 1],
        [0, 0],
        [[0, -100], [1, 1]],
        [[0, 1], [1, 0]],
        [[True, False], [True, True]],
        num_group_classes=3,
        num_actor_classes=2,
    )
    assert result["group"]["count"] == 2
    assert result["group"]["accuracy"] == 0.5
    assert result["group"]["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 0]]
    assert result["group"]["per_class_accuracy"] == [1.0, 0.0, None]
    assert result["actor"]["count"] == 3
    assert result["actor"]["accuracy"] == pytest.approx(2 / 3)


def test_resume_matches_uninterrupted_and_inference_omits_padding(tmp_path):
    from surveillance.evaluation.group_activity_metrics import evaluate
    from surveillance.inference.group_activity_pipeline import GroupActivityPipeline
    from surveillance.training.actor_transformer_trainer import load_checkpoint, train

    manifest = tiny_manifest(tmp_path)
    config = tiny_config()
    complete = train(config, manifest, tmp_path / "complete")
    partial_config = copy.deepcopy(config)
    partial_config["training"]["max_iterations"] = 1
    partial = train(partial_config, manifest, tmp_path / "resumed")
    resumed = train(config, manifest, tmp_path / "resumed", partial)
    model, saved = load_checkpoint(resumed)
    expected, _ = load_checkpoint(complete)
    assert saved["iteration"] == 3
    assert saved["selection_metric"] == "validation_group_accuracy"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
    scenes = GroupActivityPipeline(model).predict_manifest(manifest, "val", return_attention=True)
    assert [len(scene["actors"]) for scene in scenes] == [1, 3]
    assert sum(scenes[0]["group_probabilities"]) == pytest.approx(1.0)
    assert "attention" in scenes[0]
    json.dumps(scenes, allow_nan=False)
    metrics = evaluate(resumed, manifest, "test", device="cpu")
    assert metrics["group"]["count"] == 1
    assert metrics["actor"]["count"] == 3
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    extended = copy.deepcopy(config)
    extended["training"]["max_iterations"] = 4
    with pytest.raises(ValueError, match="manifest"):
        train(extended, manifest, tmp_path / "resumed", resumed)


def test_training_rejects_labels_outside_configured_vocabulary(tmp_path):
    from surveillance.training.actor_transformer_trainer import train

    manifest = tiny_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["actor_labels"] = [2]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="actor.*class"):
        train(tiny_config(1), manifest, tmp_path / "invalid")


def test_resume_rejects_wrong_checkpoint_type_before_accessing_config(tmp_path):
    from surveillance.training.actor_transformer_trainer import train

    manifest = tiny_manifest(tmp_path)
    checkpoint = tmp_path / "unrelated.pt"
    torch.save({"model": {}}, checkpoint)
    with pytest.raises(ValueError, match="Actor Transformer checkpoint"):
        train(tiny_config(1), manifest, tmp_path / "invalid", checkpoint)


def test_no_validation_uses_explicit_training_loss_and_ignores_test_labels(tmp_path):
    from surveillance.evaluation.group_activity_metrics import evaluate
    from surveillance.training.actor_transformer_trainer import load_checkpoint, train

    manifest = tiny_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["split"] != "val"]
    rows[-1]["group_label"] = 8
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    checkpoint = train(tiny_config(1), manifest, tmp_path / "no_validation")
    _, saved = load_checkpoint(checkpoint)
    assert saved["selection_metric"] == "negative_train_loss"
    assert saved["best_value"] < 0
    assert (checkpoint.parent / "best.pt").is_file()
    with pytest.raises(ValueError, match="group.*class"):
        evaluate(checkpoint, manifest, device="cpu")
