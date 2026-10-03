from pathlib import Path

import pytest
import yaml

from surveillance.actor_config import load_actor_config


@pytest.mark.parametrize("filename", ["actor_transformer.yaml", "actor_transformer_pose_only.yaml"])
def test_paper_config_defaults(filename):
    config = load_actor_config(Path("configs") / filename)
    assert config["data"]["num_frames"] == 10
    assert config["model"]["embedding_dim"] == 128
    assert config["training"]["lr_milestones"] == [5000, 10000]
    assert config["training"]["max_iterations"] == 20000


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "mode", "bad"),
        ("model", "embedding_dim", 7),
        ("data", "num_frames", 0),
        ("training", "learning_rate", float("nan")),
        ("training", "lr_milestones", [10, 5]),
        ("training", "batch_size", 0),
    ],
)
def test_bad_configuration_has_useful_error(tmp_path, section, key, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({section: {key: value}}))
    with pytest.raises(ValueError):
        load_actor_config(path)


@pytest.mark.parametrize(
    "update",
    [
        {"model": {"transformer": {"reference_size": None}}},
        {"training": {"betas": None}},
        {"model": {"transformer": {"dropout": None}}},
    ],
)
def test_null_config_values_fail_cleanly(tmp_path, update):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(update))
    with pytest.raises(ValueError):
        load_actor_config(path)
