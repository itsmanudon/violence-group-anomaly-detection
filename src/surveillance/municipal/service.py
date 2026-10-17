"""Shared application operations for HTTP and the React-backed Gradio console."""

import copy
import json
import uuid
from collections import Counter
from pathlib import Path

from surveillance.datasets.dcsass_audit import sha256
from surveillance.inference.provenance import (
    ASSET_KEYS,
    pipeline_identity,
    verify_selected_checkpoint,
)
from surveillance.municipal.catalog import Catalog, fingerprint, now
from surveillance.municipal.evaluation import (
    export_csv,
    export_json,
    metrics,
    printable_report,
    records,
)
from surveillance.municipal.queue import FrozenRunner, JobQueue


def preflight(config, root):
    identity = pipeline_identity(config)
    assets = {}
    for key in ASSET_KEYS:
        path = Path(root) / config[key]
        state = "missing"
        if path.is_file():
            state = "verified" if sha256(path) == identity["models"][key] else "hash_mismatch"
            if state == "verified" and key in ("sultani_checkpoint", "behavior_checkpoint"):
                try:
                    verify_selected_checkpoint(path)
                except (ValueError, OSError, KeyError):
                    state = "invalid_selection_receipt"
        assets[key] = state
    return {
        "ready": all(v == "verified" for v in assets.values()),
        "assets": assets,
        "profile_hash": fingerprint([identity, fingerprint(config)]),
        "config_sha256": fingerprint(config),
        "guidance": "Install the exact five frozen assets and selection receipts; "
        "see docs/municipal-demo-runbook.md",
    }


class MunicipalService:
    def __init__(
        self,
        directory,
        config,
        root,
        *,
        runner=None,
        concurrency=1,
        max_bytes=512 * 1024 * 1024,
        retention_hours=24,
    ):
        self.config, self.root = copy.deepcopy(config), Path(root)
        self.catalog = Catalog(directory, max_bytes=max_bytes)
        self.retention_hours = retention_hours
        identity = pipeline_identity(config)
        self.queue = JobQueue(
            self.catalog,
            identity,
            runner or FrozenRunner(config, root),
            concurrency=concurrency,
            config_sha256=fingerprint(config),
        )
        self.asset_report = {
            "ready": False,
            "state": "not_checked",
            "guidance": "Run asset preflight",
        }

    def health(self):
        jobs = self.queue.jobs()
        clips = self.catalog.clips()
        with self.catalog.connect() as db:
            schema = db.execute("PRAGMA user_version").fetchone()[0]
            pending = db.execute(
                "SELECT COUNT(*) FROM items i WHERE state='completed' "
                "AND NOT EXISTS(SELECT 1 FROM reviews r WHERE r.item_id=i.item_id)"
            ).fetchone()[0]
        return {
            "status": "ok",
            "schema_version": schema,
            "local_only_default": True,
            "library_count": len(clips),
            "library_composition": {
                "sources": len({c["source_video_id"] for c in clips}),
                "cohorts": dict(Counter(c["cohort"] for c in clips)),
                "assignments": dict(Counter(c["assignment"] for c in clips)),
                "selection_methods": dict(Counter(c["selection_method"] for c in clips)),
                "geography_confidence": dict(Counter(c["geography"]["confidence"] for c in clips)),
            },
            "reviews_pending": pending,
            "jobs": {
                state: sum(j["state"] == state for j in jobs)
                for state in ("queued", "running", "completed", "failed", "cancelled")
            },
            "worker_active": self.queue.executor is not None,
            "assets": self.asset_report,
            "profile_hash": self.queue.profile_hash,
        }

    def check_assets(self):
        self.asset_report = preflight(self.config, self.root)
        return self.asset_report

    def import_upload(self, path, *, source="unknown; operator upload", license="unknown"):
        return self.catalog.import_clip(
            {
                "file_path": str(path),
                "source_video_id": "upload-" + sha256(Path(path)),
                "source": source,
                "license": license,
                "external_handover_permission": "unknown",
                "training_permission": "unknown",
                "assignment": "upload",
                "cohort": "arbitrary_upload",
                "selection_method": "operator supplied arbitrary footage",
            },
            self.catalog.temporary,
        )

    def item(self, item_id):
        with self.catalog.connect() as db:
            row = db.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
        if not row:
            raise ValueError("Unknown processing item")
        saved = dict(row)
        saved["result"] = json.loads(saved["result"]) if saved["result"] else None
        return saved

    def reveal(self, item_id):
        item = self.item(item_id)
        reference = self.catalog.reference(item["clip_id"])
        with self.catalog.connect() as db:
            self.catalog.audit(
                db,
                "reference_reveal",
                item_id,
                {"version": reference["version"] if reference else None},
            )
        return reference

    def review(self, item_id, reviewer, decision, notes="", event_intervals=None):
        with self.catalog.connect() as db:
            revealed = db.execute(
                "SELECT 1 FROM audit WHERE action='reference_reveal' AND subject=?", (item_id,)
            ).fetchone()
        return self.catalog.review(
            item_id, reviewer, decision, notes, event_intervals, blinded=not bool(revealed)
        )

    def export(self, format, *, include_references=False, item_ids=None):
        if format not in ("json", "csv", "html"):
            raise ValueError("Export format must be json, csv or html")
        if include_references:
            for row in records(self.catalog, item_ids=item_ids):
                self.reveal(row["item_id"])
        if format == "json":
            data = export_json(
                self.catalog,
                self.queue.profile_hash,
                include_references=include_references,
                item_ids=item_ids,
            )
        elif format == "csv":
            data = export_csv(
                self.catalog, include_references=include_references, item_ids=item_ids
            )
        else:
            data = printable_report(self.catalog, self.queue.profile_hash)
        directory = self.catalog.directory / "exports"
        directory.mkdir(exist_ok=True)
        path = directory / (uuid.uuid4().hex + "." + format)
        path.write_text(data, encoding="utf-8")
        with self.catalog.connect() as db:
            self.catalog.audit(
                db,
                "export",
                path.name,
                {
                    "include_references": include_references,
                    "item_ids": item_ids,
                    "created_at": now(),
                },
            )
        return path

    def evaluation(self, cohort="untouched_evaluation", mode="on_demand"):
        return metrics(self.catalog, self.queue.profile_hash, cohort=cohort, mode=mode)

    def import_manifest(self, manifest, allowed_root, *, dry_run=True):
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("clips"), list):
            raise ValueError("Expected manifest schema_version=1 and clips array")
        if len(manifest["clips"]) > 10000:
            raise ValueError("Import at most 10000 clips per manifest")
        # Dry run uses a disposable database to enforce within-manifest duplicates and source rules.
        if dry_run:
            import tempfile

            with tempfile.TemporaryDirectory(dir=self.catalog.temporary) as directory:
                trial = Catalog(directory, max_bytes=self.catalog.max_bytes)
                with self.catalog.connect() as source, trial.connect() as target:
                    source.backup(target)
                outcomes = [trial.import_clip(record, allowed_root) for record in manifest["clips"]]
                return {"dry_run": True, "outcomes": outcomes, "count": len(outcomes)}
        outcomes = [self.catalog.import_clip(record, allowed_root) for record in manifest["clips"]]
        return {
            "dry_run": False,
            "outcomes": outcomes,
            "count": len(outcomes),
            "transaction_scope": "per_clip; reruns deduplicate committed clips",
        }
