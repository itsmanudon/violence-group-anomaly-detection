"""Inspect CSV headers or import local Indian-domain annotations into a new version."""

import argparse
import json
from pathlib import Path

from _common import run_cli

from surveillance.datasets.indian_domain import import_annotations, inspect_header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--inspect-header", action="store_true")
    parser.add_argument(
        "--reuse-audit", type=Path, help="Reuse verified decoding receipts; recheck hashes"
    )
    parser.add_argument("--original-catalog", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read-only catalog reconciliation; no trainable manifest",
    )
    parser.add_argument("--priority", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--source-groups", type=Path)
    parser.add_argument("--evidence-catalog", type=Path)
    parser.add_argument("--timing-evidence", type=Path)
    parser.add_argument("--historical-manifests", type=Path, nargs="*", default=[])
    parser.add_argument("--reviewer-files", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--expected-inventory",
        type=int,
        help="Optional reference count; never invents missing records",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--columns", type=Path, help="JSON canonical -> actual header")
    parser.add_argument("--provenance", type=Path, help="JSON clip_id -> provenance object")
    args = parser.parse_args()
    print(json.dumps({"observed_header": inspect_header(args.csv)}, indent=2))
    if args.inspect_header:
        return
    if args.reuse_audit:
        from surveillance.datasets.indian_readiness import reuse_readiness

        if args.original_catalog is None or args.output is None:
            parser.error("Receipt reuse requires --original-catalog and --output")
        print(
            json.dumps(
                reuse_readiness(args.reuse_audit, args.csv, args.original_catalog, args.output),
                indent=2,
            )
        )
        return
    if args.dry_run:
        from surveillance.datasets.indian_readiness import audit_readiness

        required = [
            args.priority,
            args.catalog,
            args.source_groups,
            args.evidence_catalog,
            args.timing_evidence,
            args.output,
        ]
        if any(value is None for value in required):
            parser.error(
                "Dry run requires priority, catalog, source-groups, "
                "evidence-catalog, timing-evidence and output"
            )
        audit_readiness(
            args.csv,
            args.priority,
            args.catalog,
            args.source_groups,
            args.evidence_catalog,
            args.timing_evidence,
            args.output,
            expected_inventory=args.expected_inventory,
            workers=args.workers,
            historical_manifests=args.historical_manifests,
            reviewer_files=args.reviewer_files,
        )
        return
    if args.media_root is None or args.output is None:
        parser.error("Import requires --media-root and --output")
    report = import_annotations(
        args.csv,
        args.media_root,
        args.output,
        columns=json.loads(args.columns.read_text()) if args.columns else None,
        provenance=json.loads(args.provenance.read_text()) if args.provenance else None,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run_cli(main)
