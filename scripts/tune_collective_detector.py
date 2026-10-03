"""Collect broad local validation candidates and freeze a confidence-only detector sweep."""

import argparse
from dataclasses import asdict
from pathlib import Path

from surveillance.detection.config import DetectionConfig, load_detection_config
from surveillance.experiments.thresholds import (
    collect_validation_candidates,
    tune_detector_thresholds,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--confidences", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--min-recall", type=float, default=0.5)
    parser.add_argument("--detector-config", type=Path)
    parser.add_argument("--collect-candidates", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Compatible local COCO detector weights")
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_detection_config(args.detector_config) if args.detector_config else None
    filters = (
        DetectionConfig(**{key: config["detector"][key] for key in asdict(DetectionConfig())})
        if config
        else DetectionConfig()
    )
    iou_threshold = config["matching"]["iou_threshold"] if config else 0.5
    checkpoint = args.checkpoint or (config["detector"]["checkpoint"] if config else None)
    if args.collect_candidates:
        if args.candidates or not args.candidate_output or not checkpoint:
            raise ValueError(
                "--collect-candidates requires --candidate-output and local --checkpoint"
            )
        if args.candidate_output.resolve() in {args.manifest.resolve(), args.output.resolve()}:
            raise ValueError("Candidate output must differ from manifest and receipt output")
        if Path(checkpoint).resolve() in {args.candidate_output.resolve(), args.output.resolve()}:
            raise ValueError("Candidate and receipt outputs must differ from checkpoint input")
        collect_validation_candidates(args.manifest, args.candidate_output, checkpoint, args.device)
        candidates = args.candidate_output
    elif args.candidates:
        candidates = args.candidates
    else:
        raise ValueError("Provide --candidates or --collect-candidates with local detector weights")
    receipt = tune_detector_thresholds(
        args.manifest,
        candidates,
        args.output,
        args.confidences,
        filters,
        iou_threshold=iou_threshold,
        min_recall=args.min_recall,
    )
    print(f"Validation-only confidence selected: {receipt['selected_confidence']} -> {args.output}")


if __name__ == "__main__":
    from _common import run_cli

    run_cli(main)
