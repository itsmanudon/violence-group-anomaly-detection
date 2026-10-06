"""Canonical, immutable model-pair registration for the prospective cascade test."""

import json
from pathlib import Path

from surveillance.experiments.dcsass_cache import write_json
from surveillance.inference.provenance import pipeline_identity

EXPERIMENT = Path("runs/integration/ucf_sultani_human_v1")


def register_cascade_experiment(
    root, config, anomaly, behavior, manifest_sha, protocol_sha, output, *, resume=False
):
    canonical = (Path(root) / EXPERIMENT).resolve()
    if Path(output).resolve() != canonical:
        raise ValueError(
            "Use the canonical cascade experiment output; a new path cannot rerun test"
        )
    if (
        config["anomaly_threshold"] != 0.5
        or config["max_behavior_windows"] != 3
        or anomaly.get("anomaly_threshold", 0.5) != 0.5
        or anomaly.get("population") != "ucf_sultani_shared_safe_v1"
        or anomaly.get("protocol_sha256") != protocol_sha
        or config["model_sha256"]["sultani_checkpoint"] != anomaly["checkpoint_sha256"]
        or config["model_sha256"]["behavior_checkpoint"] != behavior["checkpoint_sha256"]
        or any(
            r.get("selection_frozen") is not True
            or r.get("held_out_used_for_selection") is not False
            for r in (anomaly, behavior)
        )
    ):
        raise ValueError(
            "Cascade models or 0.5/three-window policy differ from prospective selection"
        )
    identity = {
        "schema_version": 1,
        "experiment_id": "ucf_sultani_human_v1",
        "provenance": pipeline_identity(config),
        "manifest_sha256": manifest_sha,
        "anomaly_protocol_sha256": protocol_sha,
        "population": "All frozen human-centric test clips, weak clip labels only",
        "behavior_choice": "Final-alert covered window, else first covered window, matching UI",
        "temporal_localization_claim": False,
        "actor_accuracy_claim": False,
        "held_out_used_for_selection": False,
        "complete": False,
    }
    registration = canonical / "registration.json"
    if registration.exists():
        if not resume or (canonical / "completion.json").exists():
            raise FileExistsError(
                "Canonical cascade pass already registered; inspect saved results"
            )
        if json.loads(registration.read_text()) != identity:
            raise ValueError("Cannot resume a cascade experiment with changed identity")
        return identity
    if canonical.exists():
        raise FileExistsError("Preserve the unregistered canonical cascade directory")
    write_json(registration, identity)
    return identity
