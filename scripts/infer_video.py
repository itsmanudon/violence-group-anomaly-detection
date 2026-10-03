"""Local video or precomputed feature inference with JSON and timeline outputs."""

import argparse
from pathlib import Path

import yaml
from _common import run_cli, write_json

from surveillance.datasets.features import load_features
from surveillance.features.c3d import C3DExtractor
from surveillance.inference.anomaly_pipeline import AnomalyPipeline
from surveillance.training.sultani_trainer import load_checkpoint, select_device
from surveillance.visualization.timeline import save_timeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--features", type=Path)
    parser.add_argument(
        "--duration-sec", type=float, help="Required for precomputed feature timestamps"
    )
    parser.add_argument("--c3d-checkpoint", type=Path)
    parser.add_argument("--mean", nargs=3, type=float)
    parser.add_argument("--channel-order", choices=["rgb", "bgr"])
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/anomaly.json"))
    parser.add_argument("--timeline", type=Path, default=Path("outputs/anomaly_timeline.png"))
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    threshold = args.threshold if args.threshold is not None else settings["threshold"]
    device = select_device(args.device)
    model, saved = load_checkpoint(args.checkpoint, device)
    segments = saved["config"]["num_segments"]
    extractor = None
    if args.video:
        if args.c3d_checkpoint is None or not args.c3d_checkpoint.is_file():
            raise FileNotFoundError(
                "Video inference needs --c3d-checkpoint with compatible local pretrained C3D "
                "weights. See data/README.md; --features works without C3D weights."
            )
        if args.mean is None or args.channel_order is None:
            raise ValueError(
                "Video inference requires --mean and --channel-order from C3D checkpoint provenance"
            )
        extractor = C3DExtractor(
            args.c3d_checkpoint, str(device), mean=args.mean, channel_order=args.channel_order
        )
    pipeline = AnomalyPipeline(model, extractor, segments)
    if args.features:
        if args.duration_sec is None:
            raise ValueError("--features requires --duration-sec for meaningful timestamps")
        result = pipeline.predict_features(
            load_features(args.features, segments, model.feature_dim), args.duration_sec, threshold
        )
    else:
        result = pipeline.predict_video(args.video, threshold)
    write_json(result.to_dict(), args.output)
    save_timeline(result.timestamps, result.segment_scores, args.timeline, threshold)
    print(
        f"Overall anomaly score={result.overall_score:.4f}; "
        f"intervals={result.suspicious_intervals}; output={args.output}"
    )


if __name__ == "__main__":
    run_cli(main)
