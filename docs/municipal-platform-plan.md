# Municipal evaluation platform — Phase 1 plan

Base: clean local `main`, commit `415bbd0`. Working branch:
`feat/municipal-evaluation-platform`. No remote operations authorized.

## Audit and compatibility

The repository has no standalone React/package.json frontend. Its professor console
is `surveillance.demo`, Gradio 6.29.1 (React runtime). Extend this installed runtime
and reuse DemoService, SurveillancePipeline and demo_rendering instead of inventing
a second frontend toolchain. Existing routes are Gradio named functions
`/analyze_video(video, source, cached, revision)` (ten outputs) and
`/preview_example(name, revision)` (video plus cleared evidence). Public generated
HTTP discovery is `/gradio_api/info`; queued calls use Gradio's call protocol.
The old command and interfaces remain available.

DemoService.analyze(video, example, use_cache) already distinguishes cached_example
and on_demand. SurveillancePipeline.predict_video returns schema_version,
provenance, video_metadata, anomaly (32 scores/timestamps), behavior.windows,
alerts/final_alert, timings and policy. RGBBehaviorAnalyzer preserves no_actors
and detector boxes; Sultani uses the approved C3D/32-segment contract. Existing
loading validates five byte-pinned assets and completed model selection receipts.
No checkpoint, selection, benchmark, training, threshold or model changes planned.

Mac runbook requires separate local assets and CPU rehearsal; Windows CUDA is
primary. Existing demo lacks durable storage, catalog, review/reference separation,
batch jobs, uploads with application-level validation and population-safe exports.

## Implementation sequence

1. Run existing tests before modifications. Add versioned SQLite migrations,
   safe decoded imports, dedup/source membership rules and fixture-only tests.
2. Add transactional durable per-clip queue, single-process ownership, sequential
   default worker, cancellation boundaries, bounded transient retries and recovery.
3. Add append-only reference versions and review audit, explicit metric populations,
   abstention/coverage and temporal false-alert metrics; CSV/JSON/print exports.
4. Extend existing React-backed Gradio console with Dashboard, Library, Queue,
   Analysis, Review and Results; mount scoped local HTTP API on its existing stack.
5. Add one-command launcher, health/preflight, SQLite backup, scoped temporary-file
   retention and dry-run dataset adapter contract. Verify offline static assets.
6. Run new tests at each stage; full pytest, Ruff check/format, frontend runtime
   checks, git diff --check and real frozen-model smoke when assets are installed.
7. Write architecture, acceptance evidence, demo runbook, import contract and
   completion report. Do not push/merge/download or import actual datasets.

## Interface design

Audience: municipal reviewers assessing fallible observable-behavior predictions.
Job: inspect evidence, record independent assessments and export traceable results.
Palette: paper #F0F4F7, ink #213C4E, signal #287C8E, caution #B77B27,
border #CBD6DE, surface #FFFFFF. Display: Bahnschrift; body: Segoe UI; data:
Consolas (all local fallbacks). Six functional tabs, dense readable evidence tables,
one signature: a persistent evidence strip naming execution mode and review status.
No decorative hero, risk gauge, external fonts, analytics or automated enforcement.

## Validation boundaries

Synthetic videos/results validate software only and never count as benchmark data.
No dataset acquisition depends on this phase. Geographic provenance must be supplied
with evidence; appearance is insufficient. Empty labels remain unknown. Historical
weaknesses remain visible: UCF ROC-AUC .7441, behavior macro F1 .2140,
cascade alert F1 .4733. CPU/Mac speed and field reliability are not established.
