# Municipal platform architecture

Phase 1 extends the existing React/Vite console and `demo_api.create_api` in a
single loopback FastAPI process. No cloud, broker, worker service, telemetry,
external fonts, analytics, police integration or automated enforcement is added.

## Scientific boundary

The approved Sultani CVPR 2018 path remains C3D FC6 → 32 temporal segments → MIL
anomaly localization. The approved Gavrilyuk CVPR 2020 path remains Faster R-CNN
actors → I3D Mixed_4f → RoIAlign → Actor-Transformer. The frozen professor profile,
all five model hashes, selection receipts, thresholds and preprocessing are reused.
No model, training, test protocol or historical result is changed. The measured
weaknesses remain UCF ROC-AUC 0.7441, DCSASS behavior macro F1 0.2140 and cascade
alert F1 0.4733. This is an uncertain human-review prototype.

## Components and storage

- `municipal/catalog.py`: validated manifest v1, complete decoding, content hashes,
  root containment, transactional import/dry-run/dedup, immutable source membership,
  versioned reference labels, append-only review/audit, online backup, upload deletion.
- `municipal/queue.py`: durable jobs/items, atomic claims, exclusive process owner,
  recovery and bounded retry, shared model lock and persisted results.
- `municipal/reporting.py`: blinded records, separate metric populations and exports.
- `municipal/api.py`: mounted `/api/platform` routes, existing service reused.
- `municipal/__main__.py`: local launcher, preflight and explicit administration.
- `frontend/src/Municipal.jsx`: new workflow, reuses original `Timeline` and preserves
  the professor console. System fonts and built assets run without Node at runtime.

Default launcher storage: `outputs/municipal/catalog.sqlite3`; originals are referenced
inside `outputs/municipal/imports` (or explicit `--import-root`); uploads use random
names under `outputs/municipal/uploads`. SQLite WAL/foreign keys/busy timeout enabled.
The original professor launcher stores its added catalog under its configured
`outputs/municipal` subdirectory; use the municipal launcher for the canonical path.
Never run two app processes against one catalog: OS file ownership prevents this.

| Table | Key and authoritative contents |
| --- | --- |
| clips | clip_id PK; unique SHA256; source recording, validated path, measured metadata, manifest spec, assignment/population, import time, status, upload ownership |
| reference_versions | (clip_id,version) PK; verified payload, author, reason, timestamp; no update/delete |
| jobs | job_id PK; unique idempotency key, sorted clip request, frozen profile, state, cancellation, timestamps |
| items | item_id PK; unique (job_id,clip_id); state, attempts, timestamps, error, complete raw prediction JSON |
| reviews | review_id PK; item_id, decision, notes, reviewer, optional observed intervals, time; append-only |
| audit | append-only actions, subject, actor, details, time |

`PRAGMA user_version` migrates atomically 0→1 (schema)→2 (prediction/identity and
source-membership triggers). Newer versions are rejected. v1→v2 is additive and
tested with an existing audit row. Back up before upgrade. No automatic downgrade;
stop the application and restore a matching app/database backup for rollback.
Database administrators can bypass SQLite triggers; the audit is an operational
history, not a cryptographic tamper-proof ledger.

## API contract

The original `/api/status`, `/api/upload`, `/api/analyze`, `/api/jobs/{id}` and
`/api/media/{token}` remain. Legacy status uses `complete`; municipal state uses
`completed`. Legacy uploads now require decoding and enter managed retention.
Legacy professor example jobs remain transient/explicit caches; municipal batch
jobs persist across restarts. The professor's cached results are not silently
imported into the municipal catalog or evaluation population.

All routes below start `/api/platform`. JSON errors use `detail`; validation errors
use 422, domain/storage validation 400, missing predictions 404, premature reference
reveal 409, oversized uploads 413. Mutations accept only local same-origin requests
when an Origin is supplied. Trusted hosts prevent DNS rebinding hostnames.

| Method / path | Shape / behavior |
| --- | --- |
| GET /health | schema, worker state, profile/model hashes, concurrency, timestamp |
| GET /dashboard | library count, batch states, pending reviews, evaluation |
| GET /clips | q/population/assignment/offset/limit → total, clips (max 1000/page) |
| POST /import?dry_run=true | manifest v1 → imported IDs, duplicates, dry_run; false commits atomically |
| POST /uploads | multipart file → clip_id, video_url; full decode + 500 MB default |
| GET /clips/{id}/video | registered contained local video only; HTTP range playback |
| DELETE /clips/{id}/upload | deletes owned inactive upload bytes only, retains audit/results |
| POST /jobs | {clip_ids:[...],idempotency_key} → 202 job_id |
| GET /jobs | offset/limit → total, jobs with item summaries |
| GET /jobs/{id} | state, items/raw results, profile, completed-item fraction |
| POST /jobs/{id}/cancel | cancel queued items; finish active item |
| POST /jobs/{id}/retry | {idempotency_key} → new job for failed/cancelled items |
| GET /items/{id} | prediction, clip provenance, review history, media/overlay URLs; no reference |
| GET /items/{id}/overlay/{index} | PNG generated on demand; boxes belong to this reference frame |
| POST /items/{id}/reviews | {decision,reviewer,notes,intervals:[[start,end]]} → review_id |
| GET /items/{id}/reference | explicit audited reveal after at least one review |
| GET /results | offset/limit → total, result/review summaries without reference |
| GET /evaluation | metric definitions, denominators, reference versions, exclusions/composition |
| GET /export | format=json/csv, references=false, external=false, optional item_id |
| POST /backup | SQLite online backup download, includes private references/paths |
| GET /audit | latest bounded audit entries; limit up to 1000 |

References can only be added by explicit local CLI administration. There is no
browser reference-edit route. Reviewer names are operator attestations, not
identity-provider accounts. This single-workstation pilot does not provide remote
multi-user authentication. Protect OS account and disk; do not expose it on a LAN.

## Queue semantics

Default one worker; `--concurrency 1..4` controls dispatch threads, but all calls to
the shared `DemoService` remain serialized (including professor requests). This
keeps GPU concurrency at one and avoids unsafe concurrent model initialization.
A batch records queued/running/completed/failed/cancelled. Individual clips finish
independently. Any failure makes a terminal batch failed; otherwise pending
cancellation makes it cancelled. Progress is finished clips / selected clips,
not simulated within-clip progress. A hung model call requires process restart;
cancellation is deliberately between clips.

Idempotency keys bind the sorted clip set and frozen profile; conflicting reuse
fails. Distinct keys permit intentional new inference. Duplicate video bytes map
to one catalog identity; retries are explicit new jobs. Timeouts receive at most
one automatic retry by default (constructor supports 0..3). Other failures are
terminal. Crashed running items requeue within the same attempt budget; completed
items are never rerun by recovery. A partial execution may run again after crash,
but only the committed completed result is exposed. Changed profiles require
explicit resubmission; recovery never substitutes checkpoints.

Results preserve original video metadata, 32 scores/timestamps, routed intervals,
actor boxes/counts/classes/probabilities, disagreements, abstentions, stage timing,
video SHA256, frozen provenance and profile hash. Batch inference requires
`execution_mode=on_demand`; `cached_example` is rejected by this path. Cached
professor requests keep their original timing and label.

## Evaluation and privacy

Prediction, reference and review are separate roles in storage. Reviews never
supply ground truth. References require optimistic expected-version checking,
author/reason, interval bounds and an audit entry; old versions persist. Analysis
is blinded until saved review + explicit reveal. Explicit reference exports are
an administration action and can reveal labels outside the blinded workflow.

Metrics select the latest completed actual inference per protected `test` clip
and current frozen profile. Evaluation admission requires a predeclared source-safe
training-overlap exclusion and evidence. All other populations, failures, cached
outputs and duplicate runs are excluded. Behavior accuracy/F1 are conditional on
class availability; no anomaly/no actors is not fabricated Normal. Macro F1 uses
all six approved classes, zero-support class F1=0. Denominators, coverage,
abstentions, source diversity and selection methods are visible. Binary violence
uses explicitly verified boolean references and the documented class mapping;
other classes abstain. Temporal false-alert duration uses merged thresholded
anomaly intervals minus overlap with complete verified event intervals, divided
by reference-negative seconds. Clip-alert metrics only use complete temporal
references. This is not the historical cascade benchmark protocol.

Uploads are size/extension/full-decode validated, randomly named and never used
as paths from a client. Relative import paths reject traversal/absolute paths and
symlink escape; workers recheck containment and byte identity. Metadata/notes are
rendered by React as text; CSV formula prefixes are escaped. Raw video/actor
imagery is absent from exports/logs. Overlay frames are generated on demand with
no persistent frame cache. JSON/CSV omit local video paths; backups intentionally
retain them and must be treated as private. Imported originals are never deleted.
Managed uploads have explicit local deletion and dry-run retention commands.
Secure erase on SSDs is not claimed. No automatic retention runs in the background.
