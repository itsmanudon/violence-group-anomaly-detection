"""Run explicit stages of one resolved Collective protocol; never download artifacts."""

import argparse
import subprocess
from pathlib import Path

from surveillance.datasets.collective_validation import DatasetValidationError
from surveillance.experiments.preparation import (
    detect_people,
    extract_features,
    inspect_features,
    preflight_experiment,
    prepare_data,
    tune_detector,
)
from surveillance.experiments.protocol import load_protocol
from surveillance.experiments.runner import freeze_experiment, run_experiment, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/experiments/collective_protocol.yaml")
    )
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "preflight",
            "inspect",
            "extract",
            "tune",
            "collect-and-tune",
            "detect",
            "freeze",
            "run",
        ],
        required=True,
    )
    parser.add_argument("--experiment", default="pose_gt")
    parser.add_argument(
        "--seed", type=int, help="One declared seed; otherwise run all declared seeds"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Bounded validation-only NON-BENCHMARK training"
    )
    parser.add_argument(
        "--max-scenes", type=int, help="Inspection subset or dry-run scenes per split"
    )
    parser.add_argument(
        "--max-iterations", type=int, help="Bound for preflight or --dry-run training"
    )
    args = parser.parse_args()
    try:
        protocol = load_protocol(args.protocol)
        if any(value is not None and value < 1 for value in (args.max_scenes, args.max_iterations)):
            raise ValueError("Scene/iteration bounds must be positive")
        if args.max_scenes is not None and args.stage not in {"inspect", "preflight", "run"}:
            raise ValueError("Scene bounds require inspection, preflight, or --dry-run training")
        if args.experiment not in protocol["experiments"]:
            raise ValueError("Experiment must be declared by the protocol")
        if args.dry_run and args.stage != "run":
            raise ValueError(
                "--dry-run applies to the training run stage; inspection is always non-benchmark"
            )
        if args.max_iterations is not None and not args.dry_run and args.stage != "preflight":
            raise ValueError("Iteration overrides require --dry-run")
        if args.stage == "prepare":
            prepare_data(protocol)
        elif args.stage == "preflight":
            result = preflight_experiment(
                protocol,
                args.experiment,
                args.seed if args.seed is not None else protocol["seeds"][0],
                args.max_scenes or 10,
                args.max_iterations or 5,
            )
            print(result["result_scope"])
        elif args.stage == "inspect":
            inspect_features(protocol, args.experiment, args.max_scenes or 10)
        elif args.stage == "extract":
            extract_features(protocol, args.experiment)
        elif args.stage in {"tune", "collect-and-tune"}:
            tune_detector(protocol, collect=args.stage == "collect-and-tune")
        elif args.stage == "detect":
            detect_people(protocol)
        elif args.stage == "freeze":
            freeze_experiment(protocol, args.experiment)
        else:
            for seed in [args.seed] if args.seed is not None else protocol["seeds"]:
                result = run_experiment(
                    protocol,
                    args.experiment,
                    seed,
                    dry_run=args.dry_run,
                    max_scenes=args.max_scenes,
                    max_iterations=args.max_iterations,
                )
                print(f"{result['result_scope']}: {args.experiment}, seed={seed}")
        return 0
    except DatasetValidationError as error:
        report = args.protocol.parent / "validation_failure.json"
        # Place generated reports under the configured ignored run root when available.
        if "protocol" in locals():
            report = Path(protocol["output_root"]) / "validation_failure.json"
        write_json(report, error.report)
        print(f"Validation failed; report: {report}")
        return 1
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Experiment failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
