"""Future validation-only frozen baseline, gated by the existing source-safe receipt."""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import write_new_json
from surveillance.datasets.indian_split import verify_split
from surveillance.evaluation.indian_binary import evaluate_binary
from surveillance.experiments.environment import environment_metadata
from surveillance.inference.cascade_review import ReviewPolicy, assess_review
from surveillance.inference.loading import load_surveillance_pipeline
from surveillance.inference.provenance import ASSET_KEYS, pipeline_identity


def run_frozen_validation(split_dir: Path, config_path: Path, output: Path, *, root: Path) -> dict:
    """No test inference or selection: anomaly score is a disclosed binary proxy."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refuse to overwrite baseline")
    rows, receipt = verify_split(split_dir)
    validation = sorted([r for r in rows if r.split == "val"], key=lambda r: r.video_id)
    if not validation or {r.label for r in validation} != {0, 1}:
        raise ValueError("Frozen baseline requires both validation classes")
    config = yaml.safe_load(Path(config_path).read_text())
    registration = {
        "scope": "source_approved_validation_only_not_final_test",
        "split_freeze_hash": receipt["freeze_hash"],
        "split_dir": str(Path(split_dir).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256(config_path),
        "provenance": pipeline_identity(config),
        "code_sha256": {
            name: sha256(Path(root) / "src/surveillance" / name)
            for name in (
                "evaluation/indian_frozen.py",
                "inference/surveillance_pipeline.py",
                "inference/cascade_review.py",
            )
        },
        "validation_ids": [r.video_id for r in validation],
        "binary_mapping": (
            "fixed anomaly score vs binary violence clip truth; proxy, not six-class conversion"
        ),
        "anomaly_threshold": config["anomaly_threshold"],
        "protected_test_inference": False,
        "optimization_performed": False,
    }
    before_loading = time.perf_counter()
    pipeline = load_surveillance_pipeline(config, Path(root))
    device = pipeline.behavior.device
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    registration["model_loading_seconds"] = time.perf_counter() - before_loading
    registration["environment"] = environment_metadata(
        registration["config_sha256"], seed=0, device=str(device)
    )
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "registration.json", registration)
    (output / "clips").mkdir()
    predictions, result_hashes = [], {}
    for index, record in enumerate(validation):
        result = pipeline.predict_video(Path(record.path))
        if result["provenance"] != registration["provenance"]:
            raise ValueError("Baseline inference provenance mismatch")
        if sha256(Path(record.path)) != receipt["video_sha256"][record.video_id]:
            raise ValueError("Validation media changed during inference")
        prediction = {
            "clip_id": record.video_id,
            "source_video_id": record.source_video_id,
            "binary_label": record.label,
            "video_sha256": receipt["video_sha256"][record.video_id],
            "split": "val",
            "result": result,
        }
        path = output / "clips" / f"{index:06d}.json"
        write_new_json(path, prediction)
        result_hashes[str(path.relative_to(output))] = sha256(path)
        predictions.append(prediction)
    # Reverify the complete frozen receipt/inputs/media before publishing completion.
    _, after = verify_split(split_dir)
    if after != receipt or sha256(config_path) != registration["config_sha256"]:
        raise ValueError("Baseline frozen inputs changed during inference")
    for key in ASSET_KEYS:
        if sha256(Path(root) / config[key]) != registration["provenance"]["models"][key]:
            raise ValueError("Baseline model changed during inference")
    labels = [p["binary_label"] for p in predictions]
    scores = [p["result"]["anomaly"]["overall_score"] for p in predictions]
    report = {
        "scope": registration["scope"],
        "binary_mapping": registration["binary_mapping"],
        "count": len(predictions),
        "source_count": len({r.source_video_id for r in validation}),
        "stage_latency_seconds": {
            key: {
                "median": float(np.median([p["result"]["timings"][key] for p in predictions])),
                "p95": float(np.quantile([p["result"]["timings"][key] for p in predictions], 0.95)),
            }
            for key in predictions[0]["result"]["timings"]
        },
        "metrics": evaluate_binary(labels, scores, config["anomaly_threshold"]),
        "source_metrics": {
            source: evaluate_binary(
                [p["binary_label"] for p in predictions if p["source_video_id"] == source],
                [
                    p["result"]["anomaly"]["overall_score"]
                    for p in predictions
                    if p["source_video_id"] == source
                ],
                config["anomaly_threshold"],
            )
            for source in sorted({r.source_video_id for r in validation})
        },
        "six_class_metrics": "NOT AVAILABLE: binary annotations do not provide six-class truth",
    }
    write_new_json(output / "metrics.json", report)
    write_new_json(
        output / "completion.json",
        {
            "complete": True,
            "registration_sha256": sha256(output / "registration.json"),
            "metrics_sha256": sha256(output / "metrics.json"),
            "result_sha256": result_hashes,
        },
    )
    return report


def replay_frozen_review(baseline: Path, split_dir: Path, output: Path) -> dict:
    """Paired A0/A1/A2 from the same verified validation outputs; no new inference."""
    baseline, output = Path(baseline), Path(output)
    if output.exists():
        raise FileExistsError("Refuse to overwrite review replay")
    rows, receipt = verify_split(split_dir)
    registration_path = baseline / "registration.json"
    completion = json.loads((baseline / "completion.json").read_text())
    registration = json.loads(registration_path.read_text())
    if (
        not completion["complete"]
        or completion["registration_sha256"] != sha256(registration_path)
        or completion["metrics_sha256"] != sha256(baseline / "metrics.json")
        or registration["split_freeze_hash"] != receipt["freeze_hash"]
        or registration["scope"] != "source_approved_validation_only_not_final_test"
    ):
        raise ValueError("Baseline receipt changed or population is not approved validation")
    expected = {r.video_id: r for r in rows if r.split == "val"}
    predictions = []
    for name, digest in completion["result_sha256"].items():
        path = (baseline / name).resolve()
        if not path.is_relative_to(baseline.resolve()) or sha256(path) != digest:
            raise ValueError("Saved baseline prediction hash/path mismatch")
        p = json.loads(path.read_text())
        r = expected.get(p["clip_id"])
        if (
            r is None
            or p.get("split") != "val"
            or p["source_video_id"] != r.source_video_id
            or p["binary_label"] != r.label
            or p["video_sha256"] != receipt["video_sha256"][r.video_id]
            or p["result"]["provenance"] != registration["provenance"]
        ):
            raise ValueError("Replay validation membership/identity mismatch")
        predictions.append(p)
    if len(predictions) != len(expected) or {p["clip_id"] for p in predictions} != set(expected):
        raise ValueError("Replay requires exact validation membership")
    modes = {}
    for mode in ("legacy", "coverage_consistency", "coverage"):
        reasons, latencies = Counter(), []
        for p in predictions:
            before = time.perf_counter()
            assessed = assess_review(p["result"], ReviewPolicy(mode))
            latencies.append(time.perf_counter() - before)
            assert assessed["final_alert"] == p["result"]["final_alert"]
            if mode == "legacy":
                covered = any(w["status"] == "available" for w in assessed["behavior"]["windows"])
                reasons["legacy_covered" if covered else "legacy_abstained"] += 1
            else:
                reasons[assessed["review_assessment"]["reason"]] += 1
        modes[mode] = {
            "review_states": dict(reasons),
            "count": len(predictions),
            "cpu_assessment_median_seconds": float(np.median(latencies)),
            "cpu_assessment_p95_seconds": float(np.quantile(latencies, 0.95)),
        }
    report = {
        "scope": "paired_validation_only_review_evidence_not_binary_alert_suppression",
        "split_freeze_hash": receipt["freeze_hash"],
        "baseline_completion_sha256": sha256(baseline / "completion.json"),
        "code_sha256": sha256(Path(__file__)),
        "review_code_sha256": sha256(Path(__file__).parents[1] / "inference/cascade_review.py"),
        "identical_raw_binary_metrics_all_modes": json.loads(
            (baseline / "metrics.json").read_text()
        )["metrics"],
        "modes": modes,
        "behavior_accuracy": "NOT AVAILABLE: no six-class truth",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "comparison.json", report)
    return report
