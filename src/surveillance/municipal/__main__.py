"""Offline pilot launcher and explicit catalog administration (no downloads)."""

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from surveillance.demo_profile import load_professor_profile
from surveillance.inference.provenance import ASSET_KEYS
from surveillance.municipal.catalog import Catalog, digest

ROOT = Path(__file__).resolve().parents[3]


def settings(args):
    config = load_professor_profile(ROOT, args.local_assets)
    config["device"] = args.device
    config["municipal_directory"] = str(args.data_dir.resolve())
    config["municipal_import_root"] = str(args.import_root.resolve()) if args.import_root else None
    config["municipal_concurrency"] = args.concurrency
    config["municipal_max_upload_bytes"] = args.max_upload_mb * 1024 * 1024
    return config


def preflight(config, *, load_models=True):
    import torch

    from surveillance.inference.loading import load_surveillance_pipeline
    from surveillance.inference.provenance import verify_selected_checkpoint

    checks = []
    for key in ASSET_KEYS:
        try:
            path = Path(config[key])
            if not path.is_file():
                raise ValueError("Missing frozen asset; copy the exact hash-pinned file")
            if digest(path) != config["model_sha256"][key]:
                raise ValueError(
                    "Corrupt/different frozen asset; checkpoint substitution forbidden"
                )
            if key in {"sultani_checkpoint", "behavior_checkpoint"}:
                verify_selected_checkpoint(path)
            checks.append(dict(name=key, status="PASS"))
        except (OSError, ValueError) as exc:
            checks.append(dict(name=key, status="FAIL", detail=str(exc)))
    assets_ready = all(c["status"] == "PASS" for c in checks)
    if load_models and assets_ready:
        try:
            pipeline = load_surveillance_pipeline(config, ROOT)
            checks.append(
                dict(name="model_loading", status="PASS", devices=pipeline.component_devices)
            )
            del pipeline
        except Exception as exc:
            checks.append(dict(name="model_loading", status="FAIL", detail=str(exc)))
    else:
        checks.append(dict(name="model_loading", status="WARN", detail="Not loaded"))
    bundle = ROOT / "frontend/dist/index.html"
    checks.append(dict(name="frontend_static_build", status="PASS" if bundle.is_file() else "FAIL"))
    catalog = Catalog(config["municipal_directory"], config["municipal_import_root"])
    with catalog.connect() as db:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    checks.append(dict(name="sqlite", status="PASS" if integrity == "ok" else "FAIL"))
    checks.append(
        dict(
            name="device",
            status="FAIL"
            if config["device"] == "cuda" and not torch.cuda.is_available()
            else "PASS",
            cuda=torch.cuda.is_available(),
        )
    )
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torchvision",
            "numpy",
            "opencv-python",
            "fastapi",
            "uvicorn",
            "python-multipart",
        )
    }
    return dict(
        schema_version=1,
        app_version="0.2.0",
        platform=platform.platform(),
        python=sys.version.split()[0],
        packages=packages,
        checks=checks,
        model_readiness_verified=load_models
        and assets_ready
        and any(c["name"] == "model_loading" and c["status"] == "PASS" for c in checks),
        status="FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS",
        storage=str(catalog.directory),
        network_required=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/municipal")
    parser.add_argument("--import-root", type=Path)
    parser.add_argument("--local-assets", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--max-upload-mb", type=int, default=500)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=7860)
    check = sub.add_parser("preflight")
    check.add_argument("--skip-model-load", action="store_true")
    ingest = sub.add_parser("import")
    ingest.add_argument("manifest", type=Path)
    ingest.add_argument("--commit", action="store_true", help="Default is dry run")
    reference = sub.add_parser("reference")
    reference.add_argument("clip_id")
    reference.add_argument("payload", type=Path)
    reference.add_argument("--author", required=True)
    reference.add_argument("--reason", required=True)
    reference.add_argument("--expected-version", type=int, required=True)
    sub.add_parser("backup")
    retention = sub.add_parser("retention")
    retention.add_argument("--days", type=int, required=True)
    retention.add_argument(
        "--apply", action="store_true", help="Default lists eligible managed uploads"
    )
    args = parser.parse_args()
    if args.max_upload_mb < 1:
        parser.error("Upload limit must be positive")
    try:
        config = settings(args)
        catalog = Catalog(
            config["municipal_directory"],
            config["municipal_import_root"],
            config["municipal_max_upload_bytes"],
        )
        if args.command == "preflight":
            report = preflight(config, load_models=not args.skip_model_load)
            print(json.dumps(report, indent=2))
            raise SystemExit(1 if report["status"] == "FAIL" else 0)
        if args.command == "import":
            print(
                json.dumps(
                    catalog.import_manifest(
                        json.loads(args.manifest.read_text()), dry_run=not args.commit
                    ),
                    indent=2,
                )
            )
        elif args.command == "reference":
            version = catalog.reference(
                args.clip_id,
                json.loads(args.payload.read_text()),
                args.author,
                args.reason,
                args.expected_version,
            )
            print(json.dumps(dict(clip_id=args.clip_id, version=version)))
        elif args.command == "backup":
            print(catalog.backup())
        elif args.command == "retention":
            if args.days < 1:
                parser.error("Retention days must be at least one")
            cutoff = datetime.now(UTC) - timedelta(days=args.days)
            clips = [
                c
                for c in catalog.clips()
                if c["owned_upload"]
                and c["status"] != "deleted"
                and datetime.fromisoformat(c["imported_at"]) < cutoff
            ]
            deleted, skipped = [], []
            for clip in clips:
                if args.apply:
                    try:
                        catalog.delete_upload(clip["clip_id"])
                        deleted.append(clip["clip_id"])
                    except ValueError:
                        skipped.append(clip["clip_id"])
            print(
                json.dumps(
                    dict(
                        eligible=[c["clip_id"] for c in clips],
                        deleted=deleted,
                        skipped_active=skipped,
                        dry_run=not args.apply,
                    )
                )
            )
        elif args.command == "serve":
            import uvicorn

            from surveillance.demo_api import create_api

            if not (ROOT / "frontend/dist/index.html").is_file():
                parser.error("Build frontend offline bundle first: npm run build --prefix frontend")
            uvicorn.run(
                create_api(config, ROOT), host="127.0.0.1", port=args.port, access_log=False
            )
    except (ValueError, OSError) as error:
        parser.exit(1, f"Local platform unavailable: {error}\n")


if __name__ == "__main__":
    main()
