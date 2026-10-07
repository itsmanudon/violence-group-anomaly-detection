"""Export supplied converted DeepMind RGB I3D; no automatic downloads."""

import argparse
import json
from pathlib import Path

from surveillance.features.i3d_export import (
    PINNED_REVISION,
    PINNED_SOURCE_SHA256,
    export_i3d_features,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i3d-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", default=PINNED_REVISION)
    parser.add_argument("--source-sha256", default=PINNED_SOURCE_SHA256)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--input-size", type=int, nargs=2, default=[480, 720])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix(".json")
    try:
        if report_path.exists():
            raise FileExistsError(f"I3D export report already exists: {report_path}")
        if report_path.resolve() == args.output.resolve():
            raise ValueError("Report and archive paths must differ")
        report = export_i3d_features(
            args.i3d_repo,
            args.checkpoint,
            args.output,
            source_revision=args.source_revision,
            expected_source_sha256=args.source_sha256,
            expected_checkpoint_sha256=args.checkpoint_sha256,
            input_size=tuple(args.input_size),
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"I3D export failed: {error}")
        return 1
    print(f"I3D Mixed_4f export validated: {args.output}; report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
