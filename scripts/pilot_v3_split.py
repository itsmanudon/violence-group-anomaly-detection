"""Indian pilot v3 (V3_PROTOCOL.md): freeze the v3 split.

Indian and DCSASS keep their v2 assignments (the spent DCSASS test joins train); UBI-Fights
follows its official test list (-> val); XD-Violence sources split 80/20 (seed 7, stratified);
IITB-Corridor clips are the test. Fingerprint links override: a new source linked to DCSASS
or Indian takes that source's split, one linked to IITB test is dropped.
"""

import json
from collections import Counter
from pathlib import Path

from _common import run_cli

from surveillance.datasets.indian_split import merge_frozen_splits, verify_split
from surveillance.datasets.v3_clips import split_sources

V2 = Path("manifests/v2-pilot/split_combined_v1")
V3 = Path("/Volumes/Seagate_2TB/violence-training/v3")
PROTOCOL = Path("runs/indian_pilot/V3_PROTOCOL.md")
OUTPUT = Path("manifests/v3-pilot/split_v3")
XD_VAL_FRACTION, XD_SEED = 0.2, 7


def main():
    v2_rows, v2 = verify_split(V2)
    records, excluded = [], {}
    split_of = {}
    for r in v2_rows:
        split = "train" if r.dataset == "dcsass" and r.split == "test" else r.split
        split_of[r.source_video_id] = split
        if r.dataset == "indian":  # Indian fingerprints are keyed by clip, not source
            split_of[f"indian:{r.video_id}"] = split
        records.append(
            {
                "dataset": r.dataset,
                "video_id": r.video_id,
                "source_video_id": r.source_video_id.split(":", 1)[1],
                "path": r.path,
                "split": split,
                "label": r.label,
                "sha256": v2["video_sha256"][r.video_id],
                "fps": r.fps,
                "num_frames": r.num_frames,
            }
        )
    clips = json.loads((V3 / "clips.json").read_text())
    links = json.loads((V3 / "links.json").read_text())
    xd = {}
    for c in clips:
        if c["dataset"] == "xd":
            xd[c["source"]] = xd.get(c["source"], False) or c["label"] == 1
    xd_split = split_sources(xd, XD_VAL_FRACTION, XD_SEED)
    overrides = Counter()
    for c in clips:
        source = f"{c['dataset']}:{c['source']}"
        split = {"ubi": c["pool"], "iitb": "test", "xd": xd_split.get(c["source"])}[c["dataset"]]
        linked = links.get(source, {})
        if any(ref.startswith("iitb:") for ref in linked):
            excluded[c["clip_id"]] = f"source {source} shares frames with IITB test"
            overrides["dropped_iitb_link"] += 1
            continue
        inherited = {split_of[ref] for ref in linked if ref in split_of}
        if len(inherited) > 1:  # D27: either split would leak, so the source is dropped
            excluded[c["clip_id"]] = f"source {source} shares frames with train and val sources"
            overrides["dropped_conflicting_links"] += 1
            continue
        if inherited and inherited != {split}:
            overrides[f"{split}->{next(iter(inherited))}"] += 1
            split = next(iter(inherited))
        records.append(
            {
                "dataset": c["dataset"],
                "video_id": c["clip_id"],
                "source_video_id": c["source"],
                "path": c["path"],
                "split": split,
                "label": c["label"],
                "sha256": c["sha256"],
                "fps": c["fps"],
                "num_frames": c["num_frames"],
            }
        )
    receipt = merge_frozen_splits(
        records,
        OUTPUT,
        artifacts=[
            V2 / "split-receipt.json",
            V3 / "plan.json",
            V3 / "clips.json",
            V3 / "links.json",
            PROTOCOL,
        ],
        scope="v3: indian v1 + dcsass (v2, test->train) + ubi-fights + xd-violence; iitb test",
        excluded=excluded,
    )
    print("freeze_hash", receipt["freeze_hash"])
    print("link overrides", dict(overrides))
    table = Counter(
        (r["dataset"], r["split"], r["label"]) for r in records if r["video_id"] not in excluded
    )
    for key in sorted(table):
        print(key, table[key])


if __name__ == "__main__":
    run_cli(main)
