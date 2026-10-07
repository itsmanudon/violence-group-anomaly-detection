"""Add a disclosed post-hoc normal-bypass case; retain every original failure example."""

import json
import shutil
from pathlib import Path

import yaml
from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.experiments.dcsass_cache import write_json
from surveillance.experiments.demo_cache import cache_examples
from surveillance.inference.provenance import verify_selected_checkpoint


def main():
    root = Path.cwd()
    path = Path("configs/demo_examples.json")
    entries = json.loads(path.read_text())
    if any(entry["name"] == "Normal: no-alert example" for entry in entries):
        raise FileExistsError("Supplemental normal example is already registered")
    config = yaml.safe_load(Path("configs/surveillance_demo.yaml").read_text())
    run = Path(config["sultani_checkpoint"]).parent
    selected = verify_selected_checkpoint(run / "best.pt")
    registration = json.loads((run / "held_out/evaluation_registration.json").read_text())
    if registration["checkpoint_sha256"] != selected["checkpoint_sha256"]:
        raise ValueError("Saved normal-case predictions differ from selected Sultani model")
    scores = {
        r["video_id"].rsplit("/", 1)[-1]: r
        for r in json.loads((run / "held_out/predictions.json").read_text())["records"]
    }
    rows = [
        json.loads(line)
        for line in Path("data/manifests/dcsass_human_centric_v1_final.jsonl")
        .read_text()
        .splitlines()
    ]
    eligible = sorted(
        [
            row
            for row in rows
            if row["split"] == "test"
            and row["group_label"] == 0
            and row["num_frames"] / row["fps"] >= 2
            and row["clip_id"] in scores
            and scores[row["clip_id"]]["bag_label"] == 0
            and scores[row["clip_id"]]["overall_score"] < config["anomaly_threshold"]
        ],
        key=lambda row: row["clip_id"],
    )
    if not eligible:
        raise ValueError("No saved held-out normal-bypass candidate exists")
    row = eligible[0]
    source = Path(row["path"])
    if sha256(source) != row["sha256"]:
        raise ValueError("Normal example source bytes changed")
    video = Path("data/examples/normal_bypass.mp4")
    if video.exists():
        raise FileExistsError("Preserve the existing supplemental video")
    shutil.copy2(source, video)
    entry = {
        "name": "Normal: no-alert example",
        "video": str(video.as_posix()),
        "result_cache": "outputs/demo/examples/normal_bypass.json",
        "video_sha256": row["sha256"],
        "clip_id": row["clip_id"],
        "source_video_id": row["source_video_id"],
        "split": "test",
        "expected_behavior": "Normal",
        "demo_role": "supplemental_correct_normal_bypass",
        "selection_policy": "Post-hoc presentation case: first test Normal clip by ID with "
        "saved Sultani bag score below the already frozen threshold; "
        "all original failures retained",
        "sultani_checkpoint_sha256": selected["checkpoint_sha256"],
        "cascade_result_status": "pending genuine supplemental inference",
    }
    subset = Path("runs/dcsass/normal-bypass-example.json")
    subset.write_text(json.dumps([entry]))
    configured = {**config, "examples": str(subset)}
    report = cache_examples(configured, root)
    write_json(Path("runs/dcsass/normal-bypass-receipt.json"), report)
    entries.append(entry)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(row["clip_id"], report["records"][0]["final_alert"]["state"])


if __name__ == "__main__":
    run_cli(main)
