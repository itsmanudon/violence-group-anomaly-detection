"""Score a folder of video clips with the frozen demo model and write one CSV row per clip.

Runs the same code path as the demo's live upload (professor profile + Indian pilot head), so
each score equals what the UI would show for that file. It never trains, tunes, or caches
anything, and it does not change the threshold.

    python scripts/score_pilot_clips.py --clips path/to/clips --output scores.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from surveillance.demo_profile import load_professor_profile
from surveillance.demo_service import DemoService
from surveillance.inference.indian_pilot import load_pilot_spec

ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
COLUMNS = [
    "file",
    "video_sha256",
    "duration_sec",
    "pilot_violence_score",
    "pilot_flagged",
    "pilot_threshold",
    "cascade_alert_state",
    "cascade_behavior",
    "sultani_max_score",
    "seconds",
    "error",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, required=True, help="folder of video files")
    parser.add_argument("--output", type=Path, required=True, help="CSV to write")
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "runs/indian_pilot/demo/indian_pilot_spec_v3.json",
        help="pilot head spec (default: the frozen v3 demo model)",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu")
    parser.add_argument("--local-assets", type=Path)
    parser.add_argument("--json-dir", type=Path, help="optional: save each full result as JSON")
    args = parser.parse_args()

    clips = sorted(
        p for p in args.clips.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if not clips:
        parser.error(f"No video files found under {args.clips}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}; choose a new file name")

    config = load_professor_profile(ROOT, args.local_assets)
    config["device"] = args.device
    config["indian_pilot"] = load_pilot_spec(args.spec)
    service = DemoService(config, ROOT)
    if args.json_dir:
        args.json_dir.mkdir(parents=True, exist_ok=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    flagged = failed = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for index, clip in enumerate(clips, 1):
            started = time.perf_counter()
            row: dict[str, object] = {"file": str(clip.relative_to(args.clips))}
            try:
                result = service.analyze(str(clip), example="", use_cache=False)
                pilot = result["indian_pilot"]
                final = result["final_alert"]
                row.update(
                    video_sha256=result["video_sha256"],
                    duration_sec=round(result["video_metadata"]["duration_sec"], 3),
                    pilot_violence_score=round(pilot["violence_score"], 6),
                    pilot_flagged=int(pilot["flagged"]),
                    pilot_threshold=pilot["threshold"],
                    cascade_alert_state=final["state"],
                    cascade_behavior=final.get("behavior") or "",
                    sultani_max_score=round(result["anomaly"]["overall_score"], 6),
                )
                flagged += int(pilot["flagged"])
                if args.json_dir:
                    name = clip.relative_to(args.clips).as_posix().replace("/", "__")
                    (args.json_dir / f"{name}.json").write_text(
                        json.dumps(result, indent=2, default=str), encoding="utf-8"
                    )
            except Exception as error:  # one bad file must not stop the batch
                failed += 1
                row["error"] = f"{type(error).__name__}: {error}"
            row["seconds"] = round(time.perf_counter() - started, 2)
            writer.writerow(row)
            handle.flush()
            score = row.get("pilot_violence_score", "ERROR")
            print(f"[{index}/{len(clips)}] {row['file']}: {score} ({row['seconds']}s)")
    print(
        f"Done: {len(clips)} clips, {flagged} flagged as violence, {failed} failed. "
        f"CSV: {args.output}"
    )
    sys.exit(1 if failed == len(clips) else 0)


if __name__ == "__main__":
    main()
