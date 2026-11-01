# Municipal offline pilot runbook

Use a dedicated local OS account and authorized footage. This prototype requires
human review and is not reliable crime detection. No network service is required.
Windows + NVIDIA is the primary pilot target; Mac CPU is the secondary rehearsal
path. Windows hardware execution still requires an on-device rehearsal.

## Prepare once, before going offline

Python 3.11+ and Node 20.19+ (or supported Node 22) are needed for setup. Keep the
existing validated Windows CUDA torch/torchvision installation; do not replace it
with the Mac environment lock. `pyproject.toml`/`uv.lock` retain the research
requirements. `requirements/municipal-macos-py312.txt` pins the exact installed
Mac Python 3.12 rehearsal environment. It is not a Windows wheel lock.

```powershell
# Windows, repository root, after installing the matched CUDA runtime/wheels
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,demo]"
npm ci --prefix frontend
npm run build --prefix frontend
.venv\Scripts\python.exe -m pip list --format=freeze --exclude surveillance-research > outputs\municipal-windows-requirements.txt
.venv\Scripts\python.exe -m pip download -r outputs\municipal-windows-requirements.txt -d outputs\windows-wheelhouse
```

On Windows, use the CUDA wheel source/version already validated by this project's
research environment; freeze/download on that target. Remove the editable local
project line from the wheel requirements before `pip download`; the project is
installed separately from this checkout. Copy the matching wheelhouse, target
requirements lock, repository source/configs and `frontend/dist` to the pilot.
For installation without internet:

```powershell
.venv\Scripts\python.exe -m pip install --no-index --find-links outputs\windows-wheelhouse -r outputs\municipal-windows-requirements.txt
.venv\Scripts\python.exe -m pip install --no-index --no-deps --no-build-isolation -e .
```

The lock must exclude its editable project line for both commands above. Prepare
pip/setuptools in the environment/wheelhouse as well. Keep the source checkout
because frozen configs and the static bundle are repository resources. Do not run
setup downloads on demo day. Wheel compatibility, GPU drivers and codecs are
target-specific.

Mac, with the already validated environment or matching prepared wheelhouse:

```bash
source .venv/bin/activate
# Existing verified snapshot: requirements/municipal-macos-py312.txt
# Clean install online during preparation only:
python -m pip install -r requirements/municipal-macos-py312.txt
python -m pip install --no-deps --no-build-isolation -e .
npm ci --prefix frontend
npm run build --prefix frontend
```

No pretrained weights or videos are bundled in Git. Copy the exact five assets,
selection receipts and (if desired) installed example videos/caches according to
[mac-demo-assets.md](mac-demo-assets.md). Relocate paths only in ignored
`configs/demo/local_assets.yaml`. Never edit model hashes, thresholds or profiles.
The municipal library works empty; no Indian data is needed. Cached professor
examples are optional, require their own pinned media/cache, and remain separate.

## Preflight and one-command launch

```powershell
.venv\Scripts\python.exe -m surveillance.municipal --device cuda preflight
scripts\start_municipal.cmd --device cuda
```

```bash
.venv/bin/python -m surveillance.municipal --device cpu preflight
./scripts/start_municipal.sh --device cpu
```

Open `http://127.0.0.1:7860`. Node is not needed after the build. Direct equivalent:
`python -m surveillance.municipal serve`. Global options precede `serve`, e.g.
`python -m surveillance.municipal --data-dir outputs/pilot --import-root /local/authorized
--device cpu serve --port 7861`. Launcher scripts accept global options and use
port 7860. They always bind loopback; no remote bind option is provided.
`--concurrency 1` is the default; more dispatch threads never increase GPU model
concurrency beyond one. `--max-upload-mb 500` controls the size ceiling.

`preflight` verifies frozen hashes, learned-model receipts, all model loading,
SQLite integrity, runtime versions, frontend build and device availability.
`preflight --skip-model-load` is an inventory only and reports
model_readiness_verified=false even when inventory passes. Missing/corrupt assets
fail explicitly; the library/review UI can still launch to inspect existing data.
Health: `GET /api/platform/health`; worker, schema, frozen identity and current
state are local only. Model readiness is established by full preflight, not health.

## Operate independently

1. Open Dashboard; check pending work and library composition. No data is imported automatically.
2. Read [dataset-import-contract.md](dataset-import-contract.md). Put authorized
   videos beneath the configured import root, prepare provenance/rights/source
   groups, and run a manifest dry run. The UI can read a manifest and commit it
   after inspection. Arbitrary uploads are unverified/unlabeled by default.
3. Search/filter library, select multiple clips and Batch submit. Progress reflects
   completed clips. Inspect failures; fix assets/codec/paths then explicitly retry.
4. Open a completed item. Inspect original playback, 32-segment timeline,
   suspicious windows, actor reference boxes, behavior scores, latency and hashes.
   Browser codec support varies; if original media cannot play, retain its bytes
   and acquire a separately documented authorized playable derivative.
5. Assess correct/incorrect/uncertain/not_assessable, notes and optional event
   intervals. Save before revealing a reference. A saved review is not a verified label.
6. Results includes feedback/error examples. Download per-clip or complete CSV/JSON;
   JSON includes aggregate evaluation. Select references deliberately. Use external
   handover filtering and inspect permissions/privacy before sharing. No raw media
   or imagery is embedded. Print report / save PDF produces the browser summary.
7. Use Professor console for explicit installed caches; never describe them as new
   inference. Cache timing belongs to the original run. Fresh uploads always infer.

No actors preserves the generic anomaly, with unavailable behavior. No routed
windows means behavior was unexamined. Neither is an invented Normal label.
Metrics need verified references and protected eligibility; a curated 100-clip
library is not automatically an unbiased benchmark.

## Restart, backup and retention

Ctrl-C gracefully finishes the active clip. Relaunch the same command/data directory.
Queued clips persist. A crashed running clip requeues once by default; completed
clips remain complete. Repeated interruption exhausts retries and exposes failure.
Do not start two processes for the same catalog. A stuck model call is not forcibly
killed by cancellation; restart the process if necessary, preserving the database.

Use Dashboard's Download SQLite backup or `python -m surveillance.municipal backup`.
The online backup includes references, reviews, private paths and results, but
not video/weights. Keep it local/private; use filtered result exports for handover.
For recovery: stop the app, preserve current catalog.sqlite3 plus WAL/SHM, restore
a backup to catalog.sqlite3 in a separate data directory and launch the matching
app version. Keep the same import root and upload files/paths. Verify preflight
and counts before returning to normal operation. Do not hot-replace an open DB.

```bash
# List managed uploads older than 7 days (does not delete):
python -m surveillance.municipal retention --days 7
# Explicitly delete eligible inactive upload bytes, retaining audit/results:
python -m surveillance.municipal retention --days 7 --apply
```

Individual managed uploads can be deleted in the library. Active queued/running
clips are protected; cancel and finish first. Imported source footage is never
deleted by this command. Deletion is filesystem unlink, not forensic secure erase.
Backups and manually saved exports have separate operator-managed retention.

## Offline acceptance rehearsal

Build once, disconnect networking manually, run full preflight, launch, import a
synthetic fixture library in a separate data directory, batch/review/export, restart,
and verify state. The automated `municipal_smoke.py` blocks Python socket connects
while running a real frozen curated example:

```bash
python scripts/municipal_smoke.py --example 'Robbery: correct cascade example (curated)' --output outputs/municipal/new-offline-smoke.json
```

Choose a new output name: existing results are never overwritten. This is a
qualitative smoke check, not a new benchmark score. Physical network disconnection,
Windows/NVIDIA driver behavior, real pilot codecs and full reboot remain explicit
on-device rehearsal checks. Preserve software versions and preflight receipts.
