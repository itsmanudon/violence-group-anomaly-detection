"""Aggregate compatible seed metrics into explicit ablation rows."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from surveillance.experiments.reporting import (
    aggregate_experiments,
    aggregate_table_markdown,
    read_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", type=Path, required=True, help="Explicit per-seed metrics.json paths"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path or directory for aggregate.json and aggregate.md",
    )
    args = parser.parse_args(argv)
    groups = defaultdict(list)
    for path in args.runs:
        run = read_json(path)
        try:
            population = run["population"]
            identity = (
                run["experiment"],
                run["protocol_hash"],
                run["evidence_kind"],
                run["dry_run"],
                population["box_source"],
                population["actor_population"],
            )
        except (KeyError, TypeError) as error:
            parser.error(f"Invalid run schema in {path}: {error}")
        groups[identity].append(path)
    try:
        aggregates = [aggregate_experiments(paths) for _, paths in sorted(groups.items())]
    except ValueError as error:
        parser.error(str(error))
    output = (
        args.output if args.output.suffix.lower() == ".json" else (args.output / "aggregate.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"schema_version": 1, "aggregates": aggregates}, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(aggregate_table_markdown(aggregates), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
