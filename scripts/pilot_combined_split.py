"""Indian pilot v2: merge Indian split v1 (train/val) with the frozen DCSASS split.

Assignments are copied from both frozen splits; see runs/indian_pilot/DCSASS_PROTOCOL.md.
"""

import json
from collections import Counter
from pathlib import Path

from _common import run_cli

from surveillance.datasets.indian_split import merge_frozen_splits, verify_split

INDIAN = Path("manifests/v2-pilot/split_v1")
BUNDLE = Path("/Volumes/Seagate_2TB/professor_demo_bundle")
DCSASS = BUNDLE / "data/manifests/dcsass_human_centric_v1_final.jsonl"
DCSASS_RECEIPT = BUNDLE / "runs/dcsass/human_centric_v1/final_split_receipt.json"
DCSASS_ROOT = BUNDLE / "data/raw/dcsass/DCSASS Dataset/DCSASS Dataset"
DCSASS_SHA256 = "f3e7fc11159fce191f61b8ecf1d0d667cb68c9f949afa3ab84758c0532ee7258"
VIOLENT = {"Abuse", "Assault", "Fighting", "Robbery"}
OUTPUT = Path("manifests/v2-pilot/split_combined_v1")


def main():
    from surveillance.datasets.dcsass_audit import sha256

    if sha256(DCSASS) != DCSASS_SHA256:
        raise ValueError("DCSASS manifest differs from its pinned SHA256")
    indian_rows, indian = verify_split(INDIAN)
    records, excluded = [], {}
    for r in indian_rows:
        if r.split == "test":
            continue  # the Indian v1 test split is spent
        records.append(
            {
                "dataset": "indian",
                "video_id": r.video_id,
                "source_video_id": r.source_video_id,
                "path": r.path,
                "split": r.split,
                "label": r.label,
                "sha256": indian["video_sha256"][r.video_id],
                "fps": r.fps,
                "num_frames": r.num_frames,
            }
        )
    for line in DCSASS.read_text().splitlines():
        r = json.loads(line)
        if r["category"] not in VIOLENT:
            excluded[r["clip_id"]] = f"DCSASS category {r['category']} excluded (not violence)"
            continue
        relative = r["path"].replace("\\", "/").split("DCSASS Dataset/DCSASS Dataset/", 1)[1]
        records.append(
            {
                "dataset": "dcsass",
                "video_id": r["clip_id"],
                "source_video_id": r["source_video_id"],
                "path": DCSASS_ROOT / relative,
                "split": r["split"],
                "label": int(r["binary_label"]),
                "sha256": r["sha256"],
                "fps": r["fps"],
                "num_frames": r["num_frames"],
            }
        )
    receipt = merge_frozen_splits(
        records,
        OUTPUT,
        artifacts=[INDIAN / "split-receipt.json", DCSASS, DCSASS_RECEIPT],
        scope="indian v1 train/val + dcsass human-centric v1 (violent categories)",
        excluded=excluded,
    )
    print("freeze_hash", receipt["freeze_hash"])
    print(
        json.dumps(
            {k: receipt[k] for k in ("clip_counts", "class_counts", "dataset_counts")}, indent=1
        )
    )
    print(
        "groups",
        len(receipt["groups"]),
        "excluded",
        Counter(v.split(" ")[2] for v in excluded.values()),
    )


if __name__ == "__main__":
    run_cli(main)
