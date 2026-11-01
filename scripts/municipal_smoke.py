"""Real frozen-model offline smoke; qualitative evidence, never benchmark metrics."""

import argparse
import json
import socket
from pathlib import Path
from unittest.mock import patch

from surveillance.demo_profile import load_professor_profile
from surveillance.demo_service import DemoService

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example", required=True, help="Exact installed, byte-pinned example name"
    )
    parser.add_argument("--local-assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; historical results are never overwritten")
    config = load_professor_profile(ROOT, args.local_assets)
    service = DemoService(config, ROOT)
    if args.example not in service.examples:
        parser.error("Example is not in the frozen local index")
    # This guards Python network operations during real model loading and inference.
    with patch.object(
        socket.socket, "connect", side_effect=RuntimeError("Offline smoke blocks network")
    ):
        result = service.analyze(None, args.example, use_cache=False)
    assert result["execution_mode"] == "on_demand"
    assert len(result["anomaly"]["segment_scores"]) == 32
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            dict(
                execution_mode=result["execution_mode"],
                segments=len(result["anomaly"]["segment_scores"]),
                windows=len(result["behavior"]["windows"]),
                timings=result["timings"],
                profile_sha256=result["profile_sha256"],
                network_connect_blocked=True,
                benchmark=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
