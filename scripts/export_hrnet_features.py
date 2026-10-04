"""Export local official COCO HRNet-W32 assets to the existing frozen pose contract."""

import argparse
import json
from pathlib import Path

from surveillance.features.hrnet_export import export_hrnet_features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrnet-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", help="Optional expected hash from a trusted source")
    parser.add_argument(
        "--report", type=Path, help="Default: output archive path with .json suffix"
    )
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix(".json")
    try:
        if report_path.exists():
            raise FileExistsError(f"Export report already exists: {report_path}")
        if report_path.resolve() == args.output.resolve():
            raise ValueError("Report and archive paths must differ")
        report = export_hrnet_features(
            args.hrnet_repo,
            args.checkpoint,
            args.output,
            expected_checkpoint_sha256=args.checkpoint_sha256,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"HRNet export failed: {error}")
        return 1
    print(
        f"HRNet export validated: {args.output}; 32x64x48 = 98304 features; report: {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
