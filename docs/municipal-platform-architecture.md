# Municipal evaluation platform architecture

## Scope and research boundary

This is a local human-review research prototype. It preserves Sultani et al.
CVPR 2018 (C3D, 32 temporal segments, MIL and temporal localization) and
Gavrilyuk et al. CVPR 2020 (Faster R-CNN actors, I3D Mixed_4f, RoIAlign and
Actor-Transformer). Existing implementation adaptations and historical protocols
remain authoritative. No model, checkpoint, threshold, training population or
benchmark result was changed. Known historical weaknesses remain visible:
UCF ROC-AUC .7441, DCSASS behavior macro F1 .2140, cascade alert F1 .4733.

The inspected checkout contains a Gradio React runtime, not a standalone React
source tree. `surveillance.municipal.console` extends the same components and
`demo_rendering` helpers. `surveillance.demo` and its professor API remain intact.
The municipal app mounts Gradio on the existing FastAPI stack; SQLite and a local
thread worker run inside that process. There is no additional service to deploy.

## Runtime flow

1. An operator imports arbitrary videos in the UI/API, or a curator explicitly
   imports a manifest through the CLI with an authorized media root.
2. The catalog checks extension, size, media metadata, actual first-frame decoding,
   SHA256 and source/split membership. It copies bytes into owned storage under a
   generated filename and verifies the copied hash. Unknown annotations stay unknown.
3. The operator submits clip IDs and an execution mode. SQLite atomically creates
   a job and unique per-clip items; a caller idempotency key binds the full request.
4. A transactional claim selects a queued item. The worker verifies the video
   hash, then calls the existing DemoService / SurveillancePipeline. The original
   loading path checks every checkpoint hash and selection receipt. No download,
   alternate checkpoint or preflight model is used.
5. Completed results retain all 32 scores, timestamps, routed windows, actor boxes,
   behavior probabilities, disagreements, abstentions, metadata, stage timings,
   model identity, execution mode, config hash and profile hash.
6. Independent reviews append to review history. Verified reference versions have
   separate storage and change reasons. Metrics read references, never reviewer
   opinions as ground truth. Exports contain no raw media by default.

## Storage and migrations

Default root: `outputs/municipal/` (override with `--data-dir`). SQLite location:
`catalog.sqlite3`; media: `media/`; upload/generated intermediates: `temporary/`;
explicit reports and backups: `exports/`. These are excluded from Git.
SQLite uses foreign keys, WAL, busy timeout and short transactions, with
transactional migrations through `PRAGMA user_version=3`. A newer schema is rejected.

| Table | Authority / important fields |
| --- | --- |
| clips | immutable imported identity, source ID, SHA256 UNIQUE, assignment, cohort, owned path, metadata JSON, imported time, status |
| source_recordings | canonical source ID with protected assignment/cohort |
| clip_sources | preserves alternate source IDs encountered during byte deduplication |
| jobs | idempotency key UNIQUE, request/profile/config hashes, frozen provenance, execution mode, submission time, cancel flag |
| items | item/job/clip IDs, state, attempts, start/finish times, sanitized failure and full prediction JSON; UNIQUE job/clip |
| reference_versions | clip/version primary key, supplied verified payload, author, reason, creation time |
| reviews | prediction item, operator, decision, notes, observed intervals, blinded flag and time |
| audit | append-only imports, aliases, jobs, references, reviews, reveals, exports and worker recovery |

Metadata includes duration, fps, resolution, frame count/codec, decoding/playability
qualification, source provenance, geography confidence/evidence, rights, handover
and training permissions, original creation time, selection method, status and notes.
Reference annotation status is authoritative in reference_versions; import status
starts `unlabeled`, and model outputs never change it into a reference label.

Migrations add source aliases (v2) and the full configuration hash (v3). Review,
reference and audit tables reject UPDATE and DELETE using SQLite triggers. A new
reference version requires the current expected version, an author and reason;
stale writes fail. This protects application integrity, not against an administrator
with direct filesystem/database control. This is a trusted single-workstation app,
not an authenticated multi-user deployment.

## Queue semantics

States: queued → running → completed / failed; queued work can become cancelled.
Batch state is derived from its items. Progress counts all terminal items. A running
clip finishes before cancellation; remaining clips are cancelled. Cancelled work
never silently restarts. Retry creates a new job for failed/cancelled clips, retaining
the original record. TimeoutError/ConnectionError get at most one automatic retry
(constructor configurable 0–3); validation, missing assets and OOM are not transient.

An OS file lease (`worker.lock`, released on exit/crash) prevents multiple worker
processes owning the catalog. Running items are recovered to queued after restart,
or cancelled if their batch was cancelled. This is at-least-once execution for a
crash during inference, with one durable completed result per item. Completed items
are not rerun. The default worker concurrency is one; optional 1–4 worker threads
still serialize model calls through FrozenRunner's lock to protect one GPU.

Idempotency means repeated submission with the same key returns the same job;
reuse with different clips/profile/mode fails. Deliberately new keys allow reruns.
Clip hashes deduplicate storage. Cached_example requests require an installed,
byte-pinned example cache with matching provenance; they never fall back to live
inference. On_demand never reads example caches. Synthetic execution is test-only.

## Review and evaluation integrity

The default analysis and catalog responses contain no verified reference labels.
Reference reveal is explicit and audited. Any later review of that prediction item
is conservatively marked unblinded, including after a reference-inclusive export.
Review timing is an observation, never a reference annotation. Reference edits are
new immutable versions. Persisting the full model output does not mutate references.

Evaluation selects a profile, cohort and execution mode, then the latest submitted
item per clip. A failed rerun stays in the population and cannot resurrect an old
success. Defaults: untouched_evaluation / on_demand / current frozen profile.
Curated_demo, arbitrary_upload, synthetic_fixture and cached_example are separately
selectable populations. Reports expose item IDs, completion counts, labeled support,
abstentions and coverage. Unknown labels contribute no fabricated ground truth.

Accuracy and six-class macro F1 include abstentions as incorrect for supplied
behavior references; covered-only metrics are separate. Macro F1 averages the
fixed six-class vocabulary (absent-class F1 = 0). Undefined precision/recall is null.
No anomaly/routing bypass is a behavior abstention, never an inferred Normal class.
Clip behavior is the final selected behavior where available, otherwise the
highest-anomaly available behavior window (including a Normal disagreement).
Binary violence metrics use explicit boolean reference labels, Fighting=true and
Normal=false; other behavior classes conservatively abstain. This restricted mapping
is printed in the report and does not equate all anomalies with violence.

Temporal false alerts require explicitly complete verified event intervals. Measures
include non-overlapping alert count, false-alert fraction, false alerts per negative
hour and false-positive time fraction using merged interval durations. These evaluate
generic anomaly review windows, not proof of crimes or guilt. Selection method,
source count, assignments and geographic confidence are reported alongside metrics.
A curated 100-clip demonstration is not automatically an unbiased benchmark.

## Security and distribution

Default listener is 127.0.0.1. Host/Origin/Sec-Fetch-Site checks restrict local browser
access; cross-site frames are denied and browser connections are restricted to self.
Raw uploads need a custom request header, a safe filename, supported extension,
streamed size limit and actual decode. User paths resolve inside explicit roots;
media responses accept clip IDs only. SQL is parameterized. CSV cells neutralize
spreadsheet formulas; printable HTML escapes metadata. No raw video is embedded
in reports or logged. No analytics, cloud uploads or automated enforcement are added.

The operator can explicitly change --host, but remote exposure requires a separate
authentication/privacy deployment design. The app is not hardened for public hosting.
OpenCV decoding does not guarantee every browser can play a codec; containers are
recorded and playback is checked in the pilot browser without silently transcoding
evidence. Full decoding consistency remains enforced by the existing inference path.

Temporary retention is scoped to temporary/, checks resolved paths, skips symlinks
and runs at startup. Explicit SQLite online backup includes metadata, references,
jobs/results and audit; media must be copied separately. Media and exports are not
automatically deleted. Backups are local and subject to the same access/retention policy.

## HTTP contract

All API routes precede the Gradio root mount. `/docs` and `/openapi.json` describe
the typed bodies. JSON writes reject unknown fields for jobs, reviews and reference
versions. Validation failures return 400, typed body failures 422, oversize upload
413, and denied browser origin/host 403. Missing files return 404. No filesystem
import endpoint is exposed: curator imports require the explicit local CLI root.

| Method/path | Input → output |
| --- | --- |
| GET /api/health | status, schema, library/composition, pending reviews, job counts, worker/assets/profile |
| POST /api/preflight | no body → byte/selection verification by asset key, readiness/config/profile hashes |
| GET /api/clips | search, assignment query → metadata list without references, notes or filesystem paths |
| GET /api/clips/{clip_id}/video | known clip ID → hash-verified owned local media with range support |
| POST /api/uploads?filename=clip.avi | raw bounded video body; X-Municipal-Request: local-upload → clip_id, deduplicated |
| POST /api/jobs | {clip_ids:[...], idempotency_key, execution_mode:on_demand or cached_example} → job_id |
| GET /api/jobs | job list with item IDs, states, attempts, timestamps, failures, mode and progress fraction |
| POST /api/jobs/{job_id}/cancel | no body → cancel_requested, boundary after current clip |
| POST /api/jobs/{job_id}/retry | {idempotency_key} → new job_id for failed/cancelled items |
| GET /api/items/{item_id} | stored item/result without video_path or reference labels |
| POST /api/items/{item_id}/reveal-reference | no body → current immutable reference version or null; audit reveal |
| POST /api/reviews | {item_id,reviewer,decision,notes,event_intervals} → review_id |
| POST /api/references/{clip_id} | {payload,author,reason,expected_version} → new immutable reference version |
| GET /api/evaluation | cohort, mode query → explicit population, metrics, coverage and composition |
| GET /api/exports/{json,csv,html} | optional include_references=false, item_id → local download, no media |
| POST /api/backup | no body → online SQLite backup download |

Decision values are correct/incorrect/uncertain/not_assessable. Blinding is computed
server-side from reveal history; it cannot be set by a review client. Reference-
inclusive export is explicit and marks exported items revealed. HTML prints the
aggregate current untouched-evaluation summary plus processing inventory; the
per-item filter applies to CSV/JSON. The original professor named Gradio functions
remain unchanged in its separate entry point. Municipal Gradio callbacks are private
to the UI (`api_name=False`); scripts should use the stable /api routes above.
