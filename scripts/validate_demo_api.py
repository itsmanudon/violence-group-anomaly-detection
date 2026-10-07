"""Exercise the installed local demo over its public Gradio API."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml
from gradio_client import Client, handle_file

from surveillance.inference.provenance import pipeline_identity

ROOT = Path(__file__).resolve().parents[1]


def validation_receipt_path(config: dict, requested: Path | None) -> Path:
    path = requested or ROOT / config["outputs"] / "api-validation" / f"{uuid4().hex}.json"
    if path.exists():
        raise FileExistsError(f"Preserve existing validation receipt: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surveillance_demo.yaml")
    parser.add_argument(
        "--output", type=Path, help="New receipt path; existing evidence is preserved"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_path = validation_receipt_path(config, args.output)
    provenance = pipeline_identity(config)
    entries = json.loads((ROOT / config["examples"]).read_text(encoding="utf-8"))
    normal = next(entry for entry in reversed(entries) if entry["expected_behavior"] == "Normal")
    robbery = next(entry for entry in reversed(entries) if entry["expected_behavior"] == "Robbery")
    limitation = next(
        entry for entry in entries if entry["demo_role"] == "known_behavior_baseline_mistake"
    )
    expected_results = {}
    for entry in (normal, robbery, limitation):
        payload = json.loads((ROOT / entry["result_cache"]).read_text(encoding="utf-8"))
        assert payload["result"]["provenance"] == provenance
        expected_results[entry["name"]] = payload["result"]
    client = Client(args.url, verbose=False)
    checks = []
    preview = client.predict(normal["name"], api_name="/preview_example")
    if isinstance(preview, tuple):
        preview = preview[0]
    assert Path(preview).is_file()
    for source, cached, video, expected_name in [
        (normal["name"], True, None, normal["name"]),
        (robbery["name"], True, None, robbery["name"]),
        (limitation["name"], True, None, limitation["name"]),
        (robbery["name"], False, None, robbery["name"]),
        (
            "Upload a video",
            True,
            handle_file(str(ROOT / normal["video"])),
            normal["name"],
        ),
        (
            "Upload a video",
            True,
            handle_file(str(ROOT / robbery["video"])),
            robbery["name"],
        ),
    ]:
        outputs = client.predict(video, source, cached, api_name="/analyze_video")
        result = outputs[9]
        assert "error" not in result, result
        expected = expected_results[expected_name]
        assert result["provenance"] == provenance
        assert result["final_alert"]["state"] == expected["final_alert"]["state"]
        np.testing.assert_allclose(
            result["anomaly"]["segment_scores"],
            expected["anomaly"]["segment_scores"],
            rtol=1e-5,
            atol=1e-6,
        )
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as receipt:
        receipt.write(
            json.dumps(
                {
                    "verified": True,
                    "config": str(args.config),
                    "provenance": provenance,
                    "scope": "API consistency against genuine frozen example caches; "
                    "not model accuracy",
                    "preview": preview,
                    "checks": checks,
                    "empty_input_error": failure[9],
                },
                indent=2,
            ),
        )
    print(f"Verified {len(checks)} real cached/live/upload requests: {output_path}")


if __name__ == "__main__":
    main()
