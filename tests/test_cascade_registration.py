import pytest

from surveillance.inference.provenance import ASSET_KEYS


def context(root):
    config = {key: key + ".pt" for key in ASSET_KEYS}
    config.update(
        model_sha256={key: "a" * 64 for key in ASSET_KEYS},
        training_population={"anomaly": "fixture", "behavior": "fixture"},
        c3d_mean=[104, 117, 128],
        c3d_channel_order="rgb",
        anomaly_threshold=0.5,
        max_behavior_windows=3,
    )
    anomaly = {
        "selection_frozen": True,
        "held_out_used_for_selection": False,
        "checkpoint_sha256": "a" * 64,
        "protocol_sha256": "p" * 64,
        "population": "ucf_sultani_shared_safe_v1",
    }
    behavior = {
        "selection_frozen": True,
        "held_out_used_for_selection": False,
        "checkpoint_sha256": "a" * 64,
    }
    return config, anomaly, behavior


def test_cascade_experiment_has_canonical_one_pass_identity(tmp_path):
    from surveillance.experiments.cascade_registration import register_cascade_experiment

    config, anomaly, behavior = context(tmp_path)
    output = tmp_path / "runs/integration/ucf_sultani_human_v1"
    registered = register_cascade_experiment(
        tmp_path, config, anomaly, behavior, "m" * 64, "p" * 64, output
    )
    assert registered["complete"] is False
    assert (output / "registration.json").is_file()
    with pytest.raises(FileExistsError):
        register_cascade_experiment(tmp_path, config, anomaly, behavior, "m" * 64, "p" * 64, output)
    with pytest.raises(ValueError, match="canonical"):
        register_cascade_experiment(
            tmp_path, config, anomaly, behavior, "m" * 64, "p" * 64, tmp_path / "other"
        )


@pytest.mark.parametrize("drift", ["threshold", "windows", "behavior", "protocol"])
def test_cascade_registration_rejects_drift_before_any_test_scoring(tmp_path, drift):
    from surveillance.experiments.cascade_registration import register_cascade_experiment

    config, anomaly, behavior = context(tmp_path)
    if drift == "threshold":
        config["anomaly_threshold"] = 0.8
    elif drift == "windows":
        config["max_behavior_windows"] = 1
    elif drift == "behavior":
        config["model_sha256"]["behavior_checkpoint"] = "b" * 64
    else:
        anomaly["protocol_sha256"] = "changed"
    output = tmp_path / "runs/integration/ucf_sultani_human_v1"
    with pytest.raises(ValueError):
        register_cascade_experiment(tmp_path, config, anomaly, behavior, "m" * 64, "p" * 64, output)
    assert not output.exists()
