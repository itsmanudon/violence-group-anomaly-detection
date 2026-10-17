# Local municipal evaluation pilot runbook

## Prepare the pilot machine before going offline

Use Python 3.11+ (validated Windows interpreter: 3.13.5). Install Git and a matched
PyTorch/TorchVision build. The recorded primary Windows environment is CUDA 12.6,
torch 2.13.0+cu126, torchvision 0.28.0+cu126 and Gradio 6.29.1. The full observed
environment is pinned in `requirements/municipal-windows-v1.txt`. It contains no
editable Git/network dependency. Install the local project separately:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/municipal-windows-v1.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Installation requires previously obtained wheels or internet access. For an isolated
machine, prepare the matching platform/architecture wheelhouse in advance, transfer
it by approved local media, and use pip `--no-index --find-links <wheelhouse>`.
Do not attempt to install Windows CUDA wheels on a Mac. The secondary direct
dependency profile is `requirements/municipal-macos-v1.txt`; resolve compatible Mac
wheels before the rehearsal. Mac hardware was unavailable for this phase, so Mac
execution and exact wheel availability remain unverified. Use `--device cpu` there;
MPS is not validated. See `macos-replication.md` for scientific asset handoff details.

Copy the exact external assets named in `configs/surveillance_demo.yaml`:

| File | Role |
| --- | --- |
| runs/ucf-crime/sultani_shared_safe_v1/seed_0/best.pt | frozen selected Sultani scorer |
| runs/dcsass/human_rgb_detected_v1/random_init_seed_0/best.pt | frozen selected RGB behavior model |
| checkpoints/c3d_fc6_openmmlab_v1.pt | approved C3D export |
| checkpoints/i3d_mixed4f_collective_rgb_v1.pt | approved I3D Mixed_4f export |
| checkpoints/external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth | frozen Faster R-CNN detector |

Also copy each trained model's sibling `selection.json`, original receipts and
export sidecars. Never replace a missing model with another checkpoint. Optional
cached examples require the separately transferred videos/results pinned by
`configs/demo_examples_ucf_v1.json`. None are bundled into Git or imported on launch.
Run the existing `scripts/verify_demo_assets.py` for full tensor loading if desired;
municipal preflight performs hash and selection-receipt checks without scoring data.

```powershell
.\.venv\Scripts\python.exe -m surveillance.municipal --preflight
```

Missing/hash-mismatched assets produce a nonzero exit and named asset status. The
library, reviews and installed UI can still open without model assets. Genuine
inference fails explicitly until the correct assets are installed. No model is
downloaded at runtime. Preflight verifies hashes; the existing loader additionally
validates model/export compatibility on first inference.

## Launch and operate

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m surveillance.municipal
# Equivalent Windows wrapper:
.\scripts\launch_municipal.ps1
```

On Mac: `sh scripts/launch_municipal.sh --device cpu`. Open
`http://127.0.0.1:7861`. Stop with Ctrl+C; let the current clip finish cleanly.
The original professor demo remains `python -m surveillance.demo`, port 7860.
Do not run two municipal workers against the same catalog. Different catalogs may
run independently but would compete for GPU memory; use one pilot instance.

Useful options: `--data-dir <local-folder>`, `--port 7861`, `--device cpu|cuda|auto`,
`--max-upload-mib 512`, `--concurrency 1`, `--retention-hours 24`.
Keep concurrency one on the pilot GPU. Even configured worker concurrency above one
serializes inference. The local API is `GET /api/health`; docs are at `/docs`.
The listener binds localhost unless --host is explicitly supplied. Public hosting
and multi-user authentication are outside this local pilot.

1. Dashboard: check model readiness and population overview. The initial library
   is empty; unknown labels display no metric rather than invented accuracy.
2. Library: expand Import to add operator-selected footage with source/license.
   Uploads remain arbitrary_upload, assignment upload, unlabeled. Use the manifest
   CLI for curated datasets and protected evaluation membership.
3. Refresh, select one or more clip IDs, choose actual inference or explicit
   installed example cache, and queue. The queue refreshes every three seconds.
4. Processing queue: inspect each item, failure details, attempts and mode. Cancel
   stops remaining clips after the current one; retry makes a new auditable job.
5. Refresh and open a completed item in Analysis. Check video playback, the 32-score
   timeline, routed suspicious times, reference-frame actor boxes, probabilities,
   disagreements and recorded latency. A cache's recorded inference time is old.
6. Human review: enter operator ID, assessment, notes and optional start/end seconds
   in the observed-event timing table. Save before revealing supplied references. Reveal is audited;
   later reviews of that item are marked unblinded.
7. Results: choose cohort and execution mode. Inspect coverage, abstentions, support,
   source diversity and error examples. Export all items or paste one item ID.
   CSV/JSON retain provenance; HTML is a printable report (browser Print → PDF).
   Reference-inclusive exports explicitly unblind exported prediction items.

No-actor anomalies preserve the generic anomaly alert and abstain on behavior. A
behavior prediction of Normal in a routed anomaly remains a visible disagreement.
No suspicious window means behavior was skipped, not a verified nonviolent label.
Probabilities are not calibrated risk estimates. Decisions require human judgment;
outputs cannot establish identity, intent, guilt or grounds for enforcement.

## Backup, retention and recovery

The predictable default location is `outputs/municipal/catalog.sqlite3`. The UI
backup button and CLI use SQLite's online backup, including jobs/results, review
history and reference versions. Backups do not include video; copy media separately.

```powershell
.\.venv\Scripts\python.exe -m surveillance.municipal --backup
.\.venv\Scripts\python.exe -m surveillance.municipal --purge-temporary --retention-hours 24
```

Temporary upload files and generated UI intermediates are retained only within the
application temporary tree and cleaned at startup or by the command. Symlinks are
skipped. Owned catalog videos and exports are not deleted by retention. Treat them
according to the site's rights/privacy policy. For a full portable backup, stop the
app and copy its entire data directory. Online SQLite backup alone protects the
database consistently; do not copy a live database without its WAL using file copy.

Restore into the same configured data directory, or explicitly relocate owned paths
under controlled administration; imported media paths are absolute. Keep originals
and their hashes. A newer database version requires a compatible app; it is never
silently downgraded. A stale worker.lock file is harmless: the OS lease is released
when the process exits. Restart requeues interrupted items, retains completed ones
and respects cancellation. An interrupted inference may run again; it is not claimed
to be a new completed result until it succeeds.

## Acceptance and troubleshooting

Run `python -m pytest -q`, `python -m ruff check .`,
`python -m ruff format --check .` and `git diff --check`. On Windows a restricted
execution sandbox can block asyncio loopback sockets: run local HTTP tests in a
normal authorized local process; do not disable host/origin checks in the app.
The frontend is the installed prebuilt Gradio React distribution. There is no npm
project/build/typecheck to invoke; the acceptance suite mounts it, requests built
JS/CSS and verifies the six tabs and disabled analytics.

Corrupt files, unsupported codecs, oversized videos, changed hashes and protected
split conflicts are rejected. OpenCV inspection validates a real first frame and
container metadata; full decode consistency is checked by the inference pipeline.
A codec readable by OpenCV may still fail in the browser. Supply a permitted
browser-playable derivative as a separately identified/hash-pinned clip; do not
overwrite evidence bytes. Check both video and audio in the actual pilot browser.

For an explicit real-model smoke (does not populate the pilot catalog):

```powershell
.\.venv\Scripts\python.exe scripts/smoke_municipal.py --video data/examples/robbery_cascade_success_ucf_v1.mp4 --output outputs/municipal-smoke-new
```

Use a new output path. This is an engineering smoke, not benchmark scoring. All
model hashes remain frozen. Indian data acquisition, annotation verification,
licensing and untouched evaluation registration happen later under the import
contract; this phase does not download or merge footage.
