# Municipal evaluation platform — Phase 1 plan

Starting point: clean local `main` / `01d1ef18aa17354587e43e6cd16fa7c6a8e00a66`;
branch `feat/municipal-evaluation-platform`. No remote operations.

## Audit and reuse

Existing FastAPI `create_api` serves React from `frontend/dist` in the same Python
process. Routes: GET `/api/status` (assets, examples, profile/device), POST
`/api/upload` multipart `file` → upload_id/video_url; POST `/api/analyze`
`{example?,upload_id?,use_cache:true}` → 202 job_id; GET `/api/jobs/{job_id}`
(queued/running/complete/failed, result/media links); GET `/api/media/{token}`.
Jobs/media registrations are in memory, single active inference returns 409.
Upload has 500 MB/extension limits but no decoding check at ingestion.

`DemoService.analyze(video, example, use_cache)` validates example bytes and
pipeline identity; live calls lazily load `SurveillancePipeline`. Output schema 1
contains video metadata, anomaly.segment_scores/timestamps, behavior.windows
(boxes, counts, class probabilities, disagreement/no-actor status), alerts,
provenance and timings. Live execution_mode is `on_demand`, cached is
`cached_example`. `RGBBehaviorAnalyzer` verifies detector/I3D hashes against the
behavior checkpoint; no-actor windows abstain. Sultani is fixed to 32 C3D segments.
Reuse all of these without training/policy changes. Professor config is locked;
Mac preflight checks five hashes and selection receipts. Mac defaults to CPU.
Frontend is React 19/Vite 7, plain JSX; no existing typecheck/test scripts.

## Incremental implementation and validation

1. Audit baseline pytest; retain scientific source/config/checkpoint identities.
2. SQLite migrations v1→v2: clips, immutable reference versions, jobs/items,
   reviews and audit. Versioned manifest, explicit source grouping, root-contained
   paths, full media decoding/hash checks, transactional import/dedup/dry run.
3. Durable per-clip queue with atomic claims, one worker by default, shared model
   lock, crash recovery, bounded transient retries and between-clip cancellation.
   Persist complete raw results and frozen config/model identities.
4. Mount `/api/platform` on the existing API. Extend React with dashboard, library,
   queue, analysis/review and results/printable export; retain professor console.
5. Reference versioning/blinded review and population-specific metrics; exports
   default to omitting reference labels and respect external-handover permissions.
6. Local launcher/preflight, backup/retention, adapter contract and Windows/Mac
   runbook. No video/weights committed and no automatic imports/downloads.
7. Focused integration tests after backend phases, full pytest, Ruff lint/format,
   frontend build/tests, diff check, local HTTP launch/preflight and real frozen
   model smoke if installed assets permit. Record exact evidence and limitations.

## Invariants

Unknown labels stay unknown. No individual guilt/intent labels. References are
append-only with an audited reason/version; reviews never become ground truth.
Source recording groups cannot cross assignments. Evaluation is opt-in, excludes
synthetic/upload/demo/cache populations, and reports coverage with denominators.
No output overwrites research runs, selected weights, protocols or historical
results. Local SQLite lives under outputs/municipal by default. No external
services/telemetry. Cancellation finishes the active clip and stops pending clips.

## Migration/recovery

New additive SQLite database only; reject newer schema versions. Back up through
SQLite's online backup API; restore offline with the app stopped. Existing demo
routes/results remain compatible except invalid uploads now fail before inference.

## Audit/phase checkpoints

Baseline: `.venv/bin/python -m pytest -q` → 561 passed, 2 skipped.
Catalog/queue checkpoint → 11 new tests passed. API/review/metrics checkpoint →
23 focused tests passed. Migration/concurrency/security checkpoint → 26 focused
tests passed. Final evidence is maintained in municipal-platform-acceptance.md.
A shared-lock contention issue found in integration testing was fixed by separating
legacy request admission from the common model execution lock. No research code
or scientific configuration was changed.
