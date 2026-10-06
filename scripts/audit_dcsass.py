"""Audit the observed headerless DCSASS export, fully decoding every video."""

import argparse
import json
from pathlib import Path

from _common import run_cli, write_json

from surveillance.datasets.dcsass_audit import audit_dcsass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-clips", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refuse to overwrite audit: {args.output}")
    report = audit_dcsass(args.root, args.workers, args.max_clips)
    write_json(report, args.output)
    summary = {k: v for k, v in report.items() if k != "clips"}
    args.output.with_suffix(".md").write_text(
        "# DCSASS decoded audit\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("annotation_issues", "duplicates", "missing_video_annotations")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_cli(main)
