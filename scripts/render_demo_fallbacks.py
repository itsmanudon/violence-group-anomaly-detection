"""Render trusted cached cases to offline timelines, reference frames and review videos."""

import argparse
import html
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from _common import run_cli

from surveillance.demo_profile import load_professor_profile
from surveillance.demo_rendering import reference_gallery, timeline_figure
from surveillance.demo_service import DemoService
from surveillance.video.decode import probe_video

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = [
    "Normal activity (previous bypass example)",
    "Robbery: correct cascade example (curated)",
    "Limitation: high-confidence mistake",
]


def presentation_frame(image, title, caption):
    canvas = np.full((540, 640, 3), 245, dtype=np.uint8)
    height, width = image.shape[:2]
    scale = min(640 / width, 480 / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    h, w = resized.shape[:2]
    canvas[(480 - h) // 2 : (480 + h) // 2, (640 - w) // 2 : (640 + w) // 2] = resized
    cv2.putText(canvas, title[:80], (14, 503), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (50, 65, 80), 1)
    cv2.putText(canvas, caption[:92], (14, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 65, 80), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo/fallbacks")
    parser.add_argument("--example", action="append")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    service = DemoService(load_professor_profile(ROOT), ROOT)
    catalog = [
        "<!doctype html><meta charset='utf-8'><title>Offline demo fallbacks</title>",
        "<h1>Cached research demonstration</h1>",
        "<p>Original clips, analyzed reference frames and model timelines. "
        "Actor boxes apply only to the displayed reference frames.</p>",
    ]
    for index, name in enumerate(args.example or DEFAULTS, 1):
        entry = service.examples[name]
        if not (ROOT / entry["result_cache"]).is_file():
            raise FileNotFoundError("A genuine cache is required; run preflight first")
        result = service.analyze(None, name, True)
        if result["execution_mode"] != "cached_example":
            raise ValueError("Fallback rendering must use a validated genuine cache")
        stem = f"case_{index:02}"
        timeline = args.output / (stem + "_timeline.png")
        timeline_figure(result).savefig(timeline)
        intermediate = args.output / (stem + ".avi")
        source_fps = probe_video(Path(result["video_path"])).fps
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"MJPG"), source_fps, (640, 540)
        )
        if not writer.isOpened():
            raise OSError("Cannot create the local fallback video")
        cap = cv2.VideoCapture(result["video_path"])
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(presentation_frame(frame, "Original clip - cached model output", name))
            for actor_index, (rgb, caption) in enumerate(reference_gallery(result), 1):
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(args.output / f"{stem}_actors_{actor_index}.png"), image)
                frame = presentation_frame(
                    image, "Analyzed reference frame - no tracking claim", caption
                )
                for _ in range(round(2 * source_fps)):
                    writer.write(frame)
            frame = presentation_frame(
                cv2.imread(str(timeline)), "Frozen Sultani timeline", result["final_alert"]["state"]
            )
            for _ in range(round(3 * source_fps)):
                writer.write(frame)
        finally:
            cap.release()
            writer.release()
        destination = args.output / (stem + ".mp4")
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(intermediate),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            check=True,
        )
        intermediate.unlink()
        (args.output / (stem + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False))
        catalog.append(
            f"<h2>{html.escape(name)}</h2><video controls width='640' src='{stem}.mp4'></video>"
            f"<p><a href='{stem}_timeline.png'>Timeline</a> · <a href='{stem}.json'>Result</a></p>"
        )
        print(destination)
    (args.output / "index.html").write_text("\n".join(catalog))


if __name__ == "__main__":
    run_cli(main)
