"""Copy seven transparent held-out demo examples without deleting or changing raw data."""

import argparse
import json
import shutil
from pathlib import Path

from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.dcsass_protocol import CLASSES
from surveillance.experiments.demo_examples import choose_demo_examples
from surveillance.inference.provenance import verify_selected_checkpoint
from surveillance.video.decode import probe_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/dcsass_human_centric_v1_final.jsonl")
    )
    parser.add_argument(
        "--run", type=Path, default=Path("runs/dcsass/human_rgb_detected_v1/random_init_seed_0")
    )
    parser.add_argument("--output", type=Path, default=Path("configs/demo_examples.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Demo example selection already exists; preserve its provenance")
    selected = verify_selected_checkpoint(args.run / "best.pt")
    registration = json.loads((args.run / "held_out/evaluation_registration.json").read_text())
    if registration["checkpoint_sha256"] != selected["checkpoint_sha256"] or (
        registration["manifest_sha256"] != sha256(args.manifest)
    ):
        raise ValueError(
            "Demo examples require the frozen evaluated behavior checkpoint/population"
        )
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    predictions = json.loads((args.run / "held_out/predictions.json").read_text())["records"]
    chosen = choose_demo_examples(rows, predictions)
    output = []
    for row in chosen:
        label = CLASSES[row["group_label"]]
        failure = row["demo_role"] == "known_behavior_baseline_mistake"
        stem = "limitation" if failure else label.lower()
        path = Path("data/examples") / (stem + ".mp4")
        source = Path(row["path"])
        if sha256(source) != row["sha256"]:
            raise ValueError("Selected example source bytes changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if sha256(path) != row["sha256"]:
                raise FileExistsError(f"Conflicting local example exists: {path}")
        else:
            shutil.copy2(source, path)
        probe_video(path)
        baseline = row["baseline_prediction"]
        output.append(
            {
                "name": "Limitation: high-confidence mistake" if failure else label + " example",
                "video": str(path.as_posix()),
                "result_cache": "outputs/demo/examples/" + stem + ".json",
                "video_sha256": row["sha256"],
                "clip_id": row["clip_id"],
                "source_video_id": row["source_video_id"],
                "split": "test",
                "expected_behavior": label,
                "demo_role": row["demo_role"],
                "baseline_prediction": CLASSES[baseline["prediction"]],
                "baseline_confidence": baseline["confidence"],
                "baseline_checkpoint_sha256": selected["checkpoint_sha256"],
                "selection_policy": "First covered test clip by clip ID per label, at least 2s; "
                "separate post-hoc high-confidence error for discussion; no test tuning",
                "cascade_result_status": "not yet measured; cache pending trained Sultani",
            }
        )
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Prepared", len(output), "real held-out examples; cascade results remain pending")


if __name__ == "__main__":
    run_cli(main)
