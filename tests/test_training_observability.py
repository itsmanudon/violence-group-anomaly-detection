"""Training logs expose masked metrics without changing optimization or resume."""

import copy
import json

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from test_group_activity_inference import tiny_config, tiny_manifest


def test_validation_metrics_aggregate_classes_and_exclude_padding():
    from surveillance.training.actor_transformer_trainer import _validation_metrics

    class ConstantSystem(torch.nn.Module):
        config = {"model": {"num_group_classes": 3, "num_actor_classes": 2}}

        def forward(self, batch):
            size, actors = batch["actor_labels"].shape
            return {
                "group_logits": torch.tensor([2.0, 0.0, 0.0]).repeat(size, 1),
                "actor_logits": torch.tensor([2.0, 0.0]).repeat(size, actors, 1),
            }

    samples = [
        {
            "actor_boxes": torch.tensor([[0.1, 0.1, 0.5, 0.9]] * count),
            "actor_labels": torch.tensor(labels),
            "group_label": group,
        }
        for count, labels, group in [(1, [0], 0), (3, [1, 1, 0], 1)]
    ]
    result = _validation_metrics(ConstantSystem(), samples, 2, torch.device("cpu"))
    assert result["group_accuracy"] == 0.5
    assert result["group_macro_f1"] == pytest.approx(2 / 9)
    assert result["actor_accuracy"] == 0.5
    assert result["actor_macro_f1"] == pytest.approx(1 / 3)
    assert result["group_count"] == 2
    assert result["actor_count"] == 4


def test_training_emits_metrics_and_history_resumes_without_duplicate_iterations(tmp_path):
    from surveillance.training.actor_transformer_trainer import load_checkpoint, train

    manifest = tiny_manifest(tmp_path)
    config = tiny_config(2)
    output = tmp_path / "run"
    first_config = copy.deepcopy(config)
    first_config["training"]["max_iterations"] = 1
    first = train(first_config, manifest, output)
    train(config, manifest, output, first)
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in history] == [1, 2]
    assert history[1]["selection_metric"] == "validation_group_accuracy"
    assert history[1]["selection_value"] == history[1]["validation"]["group_accuracy"]
    assert history[1]["validation"]["actor_count"] == 4
    for name in ("total", "group", "actor", "group_accuracy", "actor_accuracy", "learning_rate"):
        assert isinstance(history[1]["train"][name], (int, float))
    events = EventAccumulator(str(output / "tensorboard")).Reload()
    metric_text = events.Tensors("selection/metric/text_summary")[-1].tensor_proto.string_val[0]
    assert metric_text.decode() == "validation_group_accuracy"
    for tag in (
        "train/total",
        "train/group",
        "train/actor",
        "train/group_accuracy",
        "train/actor_accuracy",
        "train/learning_rate",
        "iteration",
        "selection/value",
        "validation/group_accuracy",
        "validation/group_macro_f1",
        "validation/actor_accuracy",
        "validation/actor_macro_f1",
    ):
        assert [item.step for item in events.Scalars(tag)] == [1, 2]
    _, saved = load_checkpoint(output / "last.pt")
    assert saved["selection_metric"] == "validation_group_accuracy"
    assert saved["format_version"] == 1


def test_observability_uses_one_training_forward_per_iteration(tmp_path, monkeypatch):
    from surveillance.training import actor_transformer_trainer as trainer

    manifest = tiny_manifest(tmp_path)
    actual_forward = trainer.ActorTransformerSystem.forward
    training_calls = []

    def counting_forward(system, batch, return_attention=False):
        if system.training:
            training_calls.append(True)
        return actual_forward(system, batch, return_attention=return_attention)

    monkeypatch.setattr(trainer.ActorTransformerSystem, "forward", counting_forward)
    trainer.train(tiny_config(2), manifest, tmp_path / "run")
    assert len(training_calls) == 2
