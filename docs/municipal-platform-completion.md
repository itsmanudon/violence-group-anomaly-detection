# Municipal evaluation platform — Phase 1 handover

Local delivery date: 18 October 2026 (Asia/Kolkata).
Base: clean local main `415bbd0`.
Branch: `feat/municipal-evaluation-platform`.
No push, merge, remote modification, dataset download or model training was performed.

## Delivered operator workflow

Six screens extend the existing Gradio React console: Dashboard, Video library,
Processing queue, Analysis workspace, Human review, Results and exports. The library
starts empty. Operators can import local videos, inspect provenance and annotation
status, select multiple clips, queue actual inference, inspect failures/cancel/retry,
play evidence with actor reference frames and a Sultani timeline, independently
review predictions and export traceable results. The default workflow stays local.

The source checkout has no standalone React source/package.json. The existing
Gradio 6.29.1 static React frontend is reused and served locally. There is no separate
frontend build/typecheck command; built JS/CSS, six tab definitions and evidence/
timed-review callbacks are validated directly. This compatibility decision and
the exact old/new API shapes are recorded in the architecture and scoped plan.

## Delivered backend and data contract

The local API exposes health/preflight, catalog/search/video, bounded uploads,
batch submission/status/cancel/retry, stored items, audited reference reveal,
append-only reviews/reference versions, population evaluation, CSV/JSON/printable
reports and online SQLite backup. See municipal-platform-architecture.md for all
method/path/body/response contracts.

SQLite schema v3 stores clips, canonical source membership and aliases, durable jobs
and per-clip items, frozen prediction JSON, reference versions, review history and
audit records. Video metadata and licenses remain source-backed; unknown annotations
and geographic origin remain unknown. Canonical source assignment and byte
deduplication prevent known split leakage; conflicting duplicate rights are rejected.
Imports copy and verify owned media with bounded writes and actual decoding.

The queue defaults to one worker and serializes GPU calls. It claims work
transactionally, recovers interrupted items after restart, preserves successful
results, cancels between clips, retries only bounded transient errors and supports
idempotent requests. Reruns with a new key are explicit. Actual inference never
uses a cache; cache mode requires an installed byte-pinned example and never
silently falls back. Every successful prediction retains execution mode, 32 anomaly
scores, windows, actors, disagreements/abstentions, timings and frozen model/profile/
canonical configuration hashes. Scientific inference implementations are unchanged.

References and reviews are separate authorities. References require supplied
verification evidence, author, reason and optimistic version checks. SQL triggers
preserve history. Ground truth stays hidden by default; reference reveal and
reference-inclusive export are audited and mark later reviews unblinded. Reviews
never become reference labels. Catalog annotation status reflects reference presence
without exposing the label itself.

Metrics are reference-only and selected by cohort, execution mode and profile, with
latest submission per clip. Reports retain failed/pending population counts, unknown
label counts, classified support, abstentions, coverage, source diversity and selection
method. Accuracy, macro F1, per-class precision/recall, conservative binary violence
mapping and temporal false-alert measures have explicit denominators. Synthetic
fixtures and curated demonstrations are not unbiased benchmark evidence. Printable
HTML provides browser PDF output without embedding video or facial imagery.

## Validation and asset requirements

The existing suite passed **544 tests** without modifying its tests. The final
combined suite passed **576 tests** (544 existing + 32 new), with 325 existing
TorchScript deprecation warnings, in 323.24 seconds. The local JUnit receipt is
`outputs/municipal-acceptance.xml`. Latest-code municipal checks also validate the
final catalog and editable timing-table refinements.
The latest municipal run passed **32 tests**; focused frontend checks also cover
the timing table's nondeprecated Gradio column interface.
Ruff lint, Ruff formatting, git diff whitespace checks and pip dependency consistency
checks passed. A 100-clip fixture import test and a separate CLI dry run succeeded;
the dry-run catalog remained empty. HTTP tests cover uploads, corruption/size/path/
origin restrictions, queue/review/reference/export flows and backup. Frontend
tests cover local static assets, six screens, evidence rendering and timed review.

Five frozen local assets and both completed selection receipts passed preflight.
Two real frozen-model queue smokes completed, including one with actual routed
Faster R-CNN, I3D and Actor-Transformer execution. Their receipts and timing scope
are listed in municipal-platform-acceptance.md. Neither smoke is benchmark evidence.
The default pilot library remains empty; smoke/fixture catalogs are isolated ignored
outputs. No checkpoints, media, training runs, historical results or profile pins
were added to Git or overwritten.

Windows NVIDIA/CUDA launch and localhost health were verified. Mac CPU rehearsal
uses the same entry point and a separate direct dependency profile but was not
hardware-tested. Prepare platform-compatible wheels and exact model assets before
going offline. Runtime does not require internet; installation/provisioning was
not reproduced on a fresh disconnected machine. Browser codec compatibility must
be checked with actual incoming footage; OpenCV decoding is not a universal browser
playability guarantee. The app is a trusted local workstation application, without
multi-user authentication or public-hosting guarantees.

Launch: `python -m surveillance.municipal` → http://127.0.0.1:7861.
Preflight: `python -m surveillance.municipal --preflight`.
Backup: `python -m surveillance.municipal --backup`.
Temporary cleanup: `python -m surveillance.municipal --purge-temporary --retention-hours 24`.
Default database: `outputs/municipal/catalog.sqlite3`; media and exports are adjacent.
See municipal-demo-runbook.md for installation, offline handoff, recovery and rights.

## Source changes and commits

New code: src/surveillance/municipal/{catalog,queue,service,api,console,__main__}.py
plus package initialization. New scripts: launch_municipal.ps1/.sh,
municipal_fixture_library.py, smoke_municipal.py. New tests:
test_municipal_platform.py, test_municipal_api.py. Added direct municipal extras and
versioned Windows/Mac requirement profiles. README links the workflow. Six requested
handover documents are present. The original pipeline, selected configs, loaders,
checkpoints, protocols and historical research reports have no changes.
The shell launcher is pinned to LF line endings through .gitattributes for Mac handoff.

Local implementation milestone: **208c75f** (`feat: add offline municipal evaluation platform`).
This completion report follows in a separate local documentation commit. No remote
operations are part of this handover. Final source checks: Ruff lint clean,
215 Python files formatted, git diff whitespace clean and pip dependency checks clean.
The final timing-table constructor is verified on the installed Gradio runtime;
the current column API is used without unsupported column-limit settings.

## Remaining dataset-dependent work and model limitations

Acquire Indian data through legitimate channels; establish geographic provenance
from evidence, canonical recording IDs, rights/training/municipal handover permission,
actual annotation authority and untouched evaluation membership. Audit re-encoded
source overlap against prior training/evaluation populations. Transform supplied
metadata under dataset-import-contract.md and dry-run before explicit import.
Add verified labels only with evidence and versioned curator changes; keep arbitrary
uploads and curated demos separate from evaluation. Rehearse media codecs,
operator review and offline provisioning on the intended municipal/Mac hardware.

Historical UCF ROC-AUC **0.7441**, behavior macro F1 **0.2140**, cascade alert F1
**0.4733** remain material weaknesses. Uncalibrated scores, detector misses,
no-actor abstentions, routing bypasses, distribution shift and model disagreement
limit use. This platform supports human-reviewed research evaluation; it does not
establish crime, guilt, intent or identity and adds no enforcement integration.
