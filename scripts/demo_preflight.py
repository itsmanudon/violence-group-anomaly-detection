"""Check the frozen professor demo before presentation; exits nonzero on FAIL."""

import argparse
from pathlib import Path

from surveillance.demo_preflight import preflight, print_report
from surveillance.demo_profile import load_professor_profile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--local-assets", type=Path)
    parser.add_argument(
        "--skip-model-load", action="store_true", help="Inventory only, not readiness"
    )
    args = parser.parse_args()
    try:
        report = preflight(
            load_professor_profile(ROOT, args.local_assets),
            ROOT,
            load_models=not args.skip_model_load,
        )
    except (OSError, ValueError, TypeError) as error:
        report = {
            "status": "FAIL",
            "checks": [
                {
                    "name": "profile",
                    "status": "FAIL",
                    "detail": str(error),
                    "fix": "Use the frozen professor profile and valid local path mapping",
                }
            ],
        }
    print_report(report, args.json)
    raise SystemExit(1 if report["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
