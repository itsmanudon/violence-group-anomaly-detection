"""Exercise the installed local demo over its public Gradio API."""

import argparse
import json
from pathlib import Path

from gradio_client import Client, handle_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/dcsass/demo-api-validation.json"
    )
    args = parser.parse_args()
    client = Client(args.url, verbose=False)
    checks = []
    preview = client.predict("Normal: no-alert example", api_name="/preview_example")
    assert Path(preview).is_file()
    for source, cached, video, expected in [
        ("Normal: no-alert example", True, None, "no_anomaly"),
        ("Robbery example", True, None, "behavior_review"),
        ("Limitation: high-confidence mistake", True, None, "generic_anomaly"),
        ("Robbery example", False, None, "behavior_review"),
        (
            "Upload a video",
            True,
            handle_file(str(ROOT / "data/examples/normal_bypass.mp4")),
            "no_anomaly",
        ),
    ]:
        outputs = client.predict(video, source, cached, api_name="/analyze_video")
        result = outputs[9]
        assert "error" not in result, result
        assert result["final_alert"]["state"] == expected
        assert len(result["anomaly"]["segment_scores"]) == 32
        assert outputs[2]["type"] == "matplotlib"
        assert Path(outputs[8]).is_file()
        assert result["execution_mode"] == (
            "cached_example" if cached and source != "Upload a video" else "on_demand"
        )
        checks.append(
            {
                "source": source,
                "mode": result["execution_mode"],
                "alert": result["final_alert"],
                "timings": result["timings"],
                "gallery_frames": len(outputs[3]),
                "json_download": outputs[8],
            }
        )
    failure = client.predict(None, "Upload a video", True, api_name="/analyze_video")
    assert "error" in failure[9]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "verified": True,
                "preview": preview,
                "checks": checks,
                "empty_input_error": failure[9],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Verified {len(checks)} real cached/live/upload requests: {args.output}")


if __name__ == "__main__":
    main()
