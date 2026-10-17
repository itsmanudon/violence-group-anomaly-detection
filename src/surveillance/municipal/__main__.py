"""One-command offline launch, preflight, dry-run manifest import and backup."""

import argparse
import json
import os
from pathlib import Path

import yaml

from surveillance.municipal.service import MunicipalService

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surveillance_demo.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/municipal")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-upload-mib", type=int, default=512)
    parser.add_argument("--retention-hours", type=float, default=24)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--commit-import", action="store_true", help="Default importer is dry-run")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--purge-temporary", action="store_true")
    args = parser.parse_args()
    if args.max_upload_mib < 1:
        parser.error("--max-upload-mib must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["GRADIO_TEMP_DIR"] = str(args.data_dir.resolve() / "temporary/gradio")
    service = MunicipalService(
        args.data_dir,
        config,
        ROOT,
        concurrency=args.concurrency,
        max_bytes=args.max_upload_mib * 1024 * 1024,
        retention_hours=args.retention_hours,
    )
    removed = service.catalog.purge_temporary(args.retention_hours)
    if args.purge_temporary:
        print(json.dumps({"removed_temporary_files": removed}))
        return
    if args.preflight:
        report = service.check_assets()
        import gradio
        import torch
        import torchvision

        report["runtime"] = {
            "gradio": gradio.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.cuda.is_available(),
        }
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ready"] else 1)
    if args.manifest:
        if not args.media_root:
            parser.error("--manifest requires an explicit --media-root")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        for record in manifest.get("clips", []):
            record["file_path"] = str(args.media_root / record["file_path"])
        print(
            json.dumps(
                service.import_manifest(manifest, args.media_root, dry_run=not args.commit_import),
                indent=2,
            )
        )
        return
    if args.backup:
        import uuid

        print(
            service.catalog.backup(
                service.catalog.directory / "exports" / (uuid.uuid4().hex + ".sqlite3")
            )
        )
        return
    import gradio as gr
    import uvicorn

    from surveillance.municipal.api import create_api
    from surveillance.municipal.console import create_console

    ui = create_console(service)
    ui.queue(default_concurrency_limit=1)
    api = create_api(service, allowed_host=args.host)
    app = gr.mount_gradio_app(
        api,
        ui,
        path="/",
        theme=ui.municipal_theme,
        css=ui.municipal_css,
        allowed_paths=[str(service.catalog.media), str(service.catalog.directory / "exports")],
        max_file_size=args.max_upload_mib * 1024 * 1024,
        footer_links=[],
    )
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
