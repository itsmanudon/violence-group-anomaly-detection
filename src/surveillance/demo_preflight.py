"""Read-only model/example verification and output-directory readiness."""

import json
import tempfile
from pathlib import Path

import torch

from surveillance.datasets.dcsass_audit import sha256
from surveillance.demo_profile import profile_hash
from surveillance.demo_service import DemoService
from surveillance.inference.loading import load_surveillance_pipeline
from surveillance.inference.provenance import ASSET_KEYS, verify_selected_checkpoint


def preflight(config: dict, root: Path, *, load_models: bool = True) -> dict:
    checks = []

    def check(name, action, fix, severity="FAIL"):
        try:
            detail = action()
            checks.append(dict(name=name, status="PASS", detail=detail))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            checks.append(dict(name=name, status=severity, detail=str(error), fix=fix))

    def verify_asset(key):
        path = root / config[key]
        actual = sha256(path)
        if actual != config["model_sha256"][key]:
            raise ValueError(f"Hash mismatch: {path}")
        if key in ("sultani_checkpoint", "behavior_checkpoint"):
            verify_selected_checkpoint(path)
        return str(path)

    for key in ASSET_KEYS:
        check(
            key, lambda key=key: verify_asset(key), "Copy exact frozen asset and selection receipt"
        )
    models_ready = all(item["status"] == "PASS" for item in checks)
    if load_models and models_ready:

        def load():
            pipeline = load_surveillance_pipeline(config, root)
            return pipeline.component_devices

        check(
            "model_archives", load, "Inspect archive/software compatibility; never replace weights"
        )
    else:
        checks.append(dict(name="model_archives", status="WARN", detail="Not loaded"))

    service = DemoService(config, root)
    if not service.examples:
        checks.append(dict(name="examples", status="FAIL", detail="Example index missing/empty"))
    for name, entry in service.examples.items():

        def example(entry=entry, name=name):
            if not (root / entry["result_cache"]).is_file():
                raise FileNotFoundError(entry["result_cache"])
            result = service.analyze(None, name, True)
            return result["execution_mode"]

        check("example: " + name, example, "Copy byte-pinned video and matching genuine cache")

    def writable():
        destination = root / config["outputs"]
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=destination) as handle:
            handle.write(b"preflight")
        return str(destination)

    check("output_directory", writable, "Choose a writable local outputs path")

    def frontend_bundle():
        path = root / "frontend/dist/index.html"
        if not path.is_file():
            raise FileNotFoundError(path)
        path.read_bytes()
        return str(path)

    check(
        "frontend_bundle",
        frontend_bundle,
        "Run npm ci --prefix frontend, then npm run build --prefix frontend",
    )
    checks.append(
        dict(
            name="MPS",
            status="PASS" if torch.backends.mps.is_available() else "WARN",
            detail={
                "built": torch.backends.mps.is_built(),
                "available": torch.backends.mps.is_available(),
                "policy": "auto selects CPU on Mac; no unvalidated MPS component is used",
            },
        )
    )
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS"
    return dict(
        status=status,
        profile_sha256=profile_hash(config),
        checks=checks,
        model_loading_requested=load_models,
        model_readiness_verified=load_models
        and models_ready
        and any(c["name"] == "model_archives" and c["status"] == "PASS" for c in checks),
    )


def print_report(report: dict, as_json: bool):
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        for item in report["checks"]:
            print(f"{item['status']}: {item['name']}: {item['detail']}")
            if item.get("fix"):
                print(f"  Fix: {item['fix']}")
        print(report["status"])
