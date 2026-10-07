"""Qualitative external rehearsal: predeclare expectations, never tune the pipeline."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.dcsass_protocol import CLASSES
from surveillance.demo_profile import load_professor_profile, profile_hash
from surveillance.demo_service import DemoService

ROOT = Path(__file__).resolve().parents[1]


def validate_manifest(manifest: dict, base: Path) -> list[dict]:
    entries = manifest.get("clips", [])
    if not entries:
        raise ValueError("Manifest requires clips")
    ids = set()
    for entry in entries:
        if entry["id"] in ids or not isinstance(entry["id"], str):
            raise ValueError("Clip IDs must be unique strings")
        ids.add(entry["id"])
        expected = entry["expected"]
        if expected["behavior"] not in CLASSES or type(expected["anomaly_expected"]) is not bool:
            raise ValueError("Predeclare a known behavior and boolean anomaly expectation")
        for key in ("used_for_training", "used_for_validation", "used_for_test", "used_for_tuning"):
            if entry["provenance"].get(key) is not False:
                raise ValueError("External qualitative clips must be unused in all research stages")
        for key in ("description", "license", "url"):
            value = entry["source"].get(key)
            if not value or "REPLACE" in value:
                raise ValueError("Document source and permission before inference")
        if not (base / entry["file"]).is_file():
            raise FileNotFoundError(base / entry["file"])
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "demo_external/manifest.yaml")
    parser.add_argument("--local-assets", type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    entries = validate_manifest(yaml.safe_load(raw), args.manifest.resolve().parent)
    config = load_professor_profile(ROOT, args.local_assets)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "runs/demo_rehearsal" / stamp
    output.mkdir(parents=True, exist_ok=False)
    # Persist expectations and exact input identities BEFORE the first model call.
    declaration = {
        "manifest_sha256": sha256(args.manifest),
        "manifest": yaml.safe_load(raw),
        "profile_sha256": profile_hash(config),
        "clip_hashes": {e["id"]: sha256(args.manifest.parent / e["file"]) for e in entries},
    }
    (output / "predeclared.json").write_text(json.dumps(declaration, indent=2))
    ledger_path = ROOT / "runs/demo_rehearsal/expectations.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
    for entry in entries:
        clip_hash = declaration["clip_hashes"][entry["id"]]
        if clip_hash in ledger and ledger[clip_hash] != entry["expected"]:
            raise ValueError("Expectations changed after first rehearsal; preserve original labels")
        ledger[clip_hash] = entry["expected"]
    ledger_path.write_text(json.dumps(ledger, indent=2))
    service = DemoService(config, ROOT)
    records = []
    for entry in entries:
        path = args.manifest.parent / entry["file"]
        if sha256(path) != declaration["clip_hashes"][entry["id"]]:
            raise ValueError("Clip changed after predeclaration")
        record = {
            "id": entry["id"],
            "expected": entry["expected"],
            "video_sha256": declaration["clip_hashes"][entry["id"]],
        }
        try:
            result = service.analyze(str(path), "Upload a video", False)
            alert = result["final_alert"]
            record.update(
                result=result,
                anomaly_matches=(alert["state"] != "no_anomaly")
                == entry["expected"]["anomaly_expected"],
                behavior_matches=alert.get("behavior") == entry["expected"]["behavior"],
            )
        except (OSError, ValueError, RuntimeError) as error:
            record["error"] = str(error)
        records.append(record)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "scope": "qualitative unseen examples; not benchmark data",
                    "declaration": declaration,
                    "models_or_thresholds_changed": False,
                    "records": records,
                },
                indent=2,
                allow_nan=False,
            )
        )
    lines = [
        "# External qualitative rehearsal",
        "",
        "Not benchmark data. No model or threshold tuning.",
        "",
    ]
    for r in records:
        lines.append(
            f"- {r['id']}: "
            + (
                r["error"]
                if "error" in r
                else f"anomaly expectation matches={r['anomaly_matches']}; "
                f"behavior matches={r['behavior_matches']}; "
                f"latency={r['result']['timings']['request_total_seconds']:.3f}s"
            )
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(output)
    if any("error" in r for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    run_cli(main)
