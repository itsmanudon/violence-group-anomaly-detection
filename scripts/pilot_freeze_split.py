"""Indian pilot Phase C: freeze the group-wise, label x collection stratified split once."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_pilot import (
    DISPOSITIONS,
    INVENTORY,
    PILOT_OVERRIDE,
    pilot_candidates,
)
from surveillance.datasets.indian_split import freeze_pilot_split

ROOT = Path("/Volumes/Seagate_2TB/violence-datasets")
ANNOTATIONS = (
    "violence-group-anomaly-detection/data/indian-domain/annotations-20261025-v1/"
    "indian_annotations_combined_unverified.csv"
)
# Fixed before any assignment was viewed; see runs/indian_pilot/DECISIONS.md.
REQUIREMENTS = [
    ("violence", {"label": 1}, 6),
    ("unknown_geography non_violence", {"label": 0, "geography": "unknown_geography"}, 10),
    ("documented_indian non_violence", {"label": 0, "geography": "documented_indian"}, 10),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--groups", type=Path, default=Path("manifests/v2-pilot/source_groups.csv"))
    parser.add_argument(
        "--review", type=Path, default=Path("manifests/v2-pilot/review_decisions.json")
    )
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=Path("runs/indian_pilot/features/fingerprints/fingerprints.json"),
    )
    parser.add_argument("--forced-train-groups", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("manifests/v2-pilot/split_v1"))
    args = parser.parse_args()

    candidates, excluded = pilot_candidates(args.root)
    with args.groups.open(encoding="utf-8") as stream:
        groups = {row["clip"]: row for row in csv.DictReader(stream)}
    if set(groups) != {c["clip_id"] for c in candidates}:
        raise ValueError("Source groups must cover exactly the eligible candidates")
    prints = json.loads(args.fingerprints.read_text())
    clips = {}
    for c in candidates:
        item = prints[c["clip_id"]]
        clips[c["clip_id"]] = {
            **c,
            "fps": item["fps"],
            "num_frames": item["decoded_frames"],
        }
    group_of = {key: row["group_id"] for key, row in groups.items()}
    forced = {key for key, group in group_of.items() if group in set(args.forced_train_groups)}
    receipt = freeze_pilot_split(
        clips,
        group_of,
        args.output,
        pilot_override=PILOT_OVERRIDE,
        seed=args.seed,
        requirements=REQUIREMENTS,
        forced_train=forced,
        artifacts=[
            args.groups,
            args.review,
            args.root / DISPOSITIONS,
            args.root / INVENTORY,
            args.root / ANNOTATIONS,
        ],
        excluded=excluded,
    )
    table = args.output.parent / f"{args.output.name}.csv"
    fields = ["clip", "split", "group_id", "collection", "geography", "label", "relative_path"]
    with table.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in sorted(clips, key=lambda k: (receipt["assignments"][k], group_of[k], k)):
            row = groups[key]
            writer.writerow(
                {
                    "clip": key,
                    "split": receipt["assignments"][key],
                    "group_id": row["group_id"],
                    "collection": row["collection"],
                    "geography": row["geography"],
                    "label": row["label"],
                    "relative_path": row["relative_path"],
                }
            )
    digest = sha256(table)
    (table.parent / f"{table.name}.sha256").write_text(f"{digest}  {table.name}\n")
    split = receipt["assignments"]
    counts = Counter(
        (split[k], c["geography"], c["collection"], c["label"]) for k, c in clips.items()
    )
    print("freeze_hash", receipt["freeze_hash"])
    print("split_v1.csv sha256", digest)
    print("clips", receipt["clip_counts"], "groups", receipt["group_counts"])
    print("class", receipt["class_counts"])
    print("requirements", receipt["balance"]["requirements_met"])
    for key in sorted(counts):
        print("  ", key, counts[key])


if __name__ == "__main__":
    run_cli(main)
