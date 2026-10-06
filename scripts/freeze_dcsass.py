"""Freeze Human-Centric v1 from the real audit and published UCF test reservations."""

import argparse
import hashlib
import json
from pathlib import Path

from _common import run_cli, write_json

from surveillance.datasets.dcsass_audit import inventory, sha256
from surveillance.datasets.dcsass_protocol import build_human_split
from surveillance.datasets.preparation import source_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--ucf-test-annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.manifest.exists() or args.receipt.exists():
        raise FileExistsError("Refuse to overwrite a frozen DCSASS protocol")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    current = inventory(Path(audit["dataset_root"]))
    current_hash = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    if current_hash != audit["inventory_sha256"]:
        raise ValueError("DCSASS changed since the audited snapshot")
    reserved = {
        source_identity(Path(line.split()[0]), {})
        for line in args.ucf_test_annotations.read_text().splitlines()
        if line.strip()
    }
    rows, receipt = build_human_split(audit, reserved, args.seed)
    if any(not count for count in receipt["class_counts"]["train"].values()):
        raise ValueError("At least one behavior class has no training examples")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    args.manifest.write_text(content, encoding="utf-8")
    receipt.update(
        audit_sha256=sha256(args.audit),
        manifest_sha256=sha256(args.manifest),
        ucf_test_annotations_sha256=sha256(args.ucf_test_annotations),
        dataset_inventory_sha256=current_hash,
        label_sha256=audit["label_sha256"],
    )
    write_json(receipt, args.receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "source_assignments"}, indent=2))


if __name__ == "__main__":
    run_cli(main)
