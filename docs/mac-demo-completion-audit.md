# Mac deployment audit — awaiting frozen assets

**Presentation validation is incomplete.** Source, React build and local UI checks
are verified. Frozen model loading, live inference, cached playback and external
qualitative rehearsal await the user's SSD asset handoff. This document does not
claim completion of the asset-dependent phases.

## Observed Mac and environment

| Item | Observed value |
|---|---|
| Model | MacBook Pro, Mac16,7 |
| Architecture / CPU | arm64, Apple M4 Pro, 14 cores (10 performance + 4 efficiency) |
| RAM | 48 GB |
| macOS | 26.6.2, build 25G83 |
| System Python | 3.12.8 |
| Demo `.venv` Python | 3.12.7, existing Anaconda-derived venv; project supports Python >=3.11 |
| torch / torchvision | 2.14.1 / 0.29.1, normal macOS wheels; no CUDA installation |
| Gradio fallback | 6.29.1 |
| React / Vite | 19.2.0 / 7.2.2 |
| Node / npm | 22.17.1 / 11.7.0; build-time tools only |
| MPS built / available | true / true outside sandbox; sandbox reports unavailable |
| Component policy | CPU for Sultani, C3D, Faster R-CNN, I3D and Actor-Transformer; actual model execution pending |

The existing `.venv` was reused, pip bootstrapped/upgraded and project demo/dev
extras installed using editable metadata. `runs/demo_mac/environment.json` and
`requirements.txt` record the installed snapshot; npm dependencies have a committed
lockfile. No Windows environment, datasets or model files were assumed present.

## Evidence and outstanding validation

| Check | Status / evidence |
|---|---|
| Initial Git | `main`, `415bbd0`; untracked pre-existing `uv.lock` preserved |
| Baseline source suite | 542 passed, 2 skipped, 236 warnings; 50.07s |
| Latest full source suite | 559 passed, 2 skipped, 236 warnings; 51.81s (`runs/demo_mac/pytest-final.log`) |
| Ruff / format / diff | Passed; current Ruff also formats Markdown Python snippets |
| React build | Passed, Vite production build ~0.7s; bundled JS ~213 kB (67 kB gzip), CSS ~16 kB |
| API behavior | Tests cover upload/cache separation, media allowlist, wrong origin, invalid input, serialized inference and failure recovery |
| Browser desktop | React console renders; three modes, missing assets and blank prediction states inspected |
| Browser mobile | 390×844 inspected; content width equals 390, no document horizontal overflow; viewport restored |
| Gradio fallback launch | Passed in actual Chrome; empty-input error displayed |
| Preflight | **FAIL**: five model files and all nine clip/cache pairs absent; output directory writable |
| Component CPU/MPS smoke | Pending assets; `demo_mac_smoke.py --probe-mps` records finite/repeat tests, latency and fallback reasons |
| End-to-end / no-actor Mac execution | Pending genuine models/clip; source routing/abstention tests are not a substitute |
| Cached examples | Pending byte-pinned clip/cache copies; scientific identity verified on acceptance |
| Genuine live upload | Transport verified with unit fixtures; real frozen inference pending |
| External clips tested / successes / limitations | Zero supplied, zero inferred; no invented clip labels/predictions |
| Live / CPU / MPS / cache latency | Not measured on frozen models yet; historical Windows latency is not a Mac measurement |
| Offline | Built page scripts/styles reference localhost only; no CDN/fonts/analytics. Full cached/live/limitation offline rehearsal pending |
| Fresh boot / shell | Exact commands documented; real fresh-shell asset rehearsal and reboot pending |
| Model / threshold changes from rehearsal | **NO**; no retraining, tuning, checkpoint selection or benchmark changes |
| Push | Not performed; required Mac model/demo validation remains incomplete |

A first post-change suite exposed two loader validation-order regressions. The
established order was restored and the subsequent full suite passed. Both failed
and passing logs remain under ignored `runs/demo_mac/`.

## Required local assets

See [complete inventory](mac-demo-assets.md) and
[per-file machine-readable pins](../configs/demo/mac_assets.example.yaml).
Copy the exact UCF Sultani scorer, DCSASS behavior checkpoint, C3D export,
I3D export, Faster R-CNN checkpoint, and both sibling `selection.json` receipts.
For presentation playback also copy the nine indexed clips and genuine caches.
Full research datasets are optional and unnecessary for normal demo operation.

## Launch and presentation

```bash
source .venv/bin/activate
npm ci --prefix frontend
npm run build --prefix frontend
python scripts/demo_preflight.py
python -m surveillance.demo --profile professor --device cpu
```

The frontend build is a setup step; after it exists, only Python is needed to launch.
Open `http://127.0.0.1:7860`. An unavailable cached pair cannot be opened as a cached
case. Live upload always invokes the existing frozen service. CPU remains the safe
policy while per-component MPS execution is unvalidated.

The [professor runbook](mac-professor-demo-runbook.md) gives the sequence:
preflight → verified normal-label case (explain any false alert) → curated correct
anomaly case → timeline and actor reasoning → live external qualitative upload →
known limitation → measured limitations. Disclose cached execution, disagreement,
predeclared expectations and prior rehearsal. Use “live unseen” only before the
clip's first-ever inference.

Fallbacks: exact local caches in the same React UI, then genuinely pre-rendered
videos/screenshots/overlays. The old local Gradio UI is available with `--ui gradio`.
Pre-rendered presentation artifacts still need to be prepared from supplied assets.

## Remaining work after SSD handoff

Verify hashes/receipts, run CPU component and full cascade tests, probe MPS without
scientific rewrites, render actual video/boxes/timeline, verify no-actor behavior,
measure cold/warm/cache requests, rehearse 5–8 authorized external clips with
expectations set first, record successes/failures, prepare fallback images/videos,
exercise all three modes offline and in a fresh shell, then update this audit.
No deployment readiness or production violence-detection claim is made.
