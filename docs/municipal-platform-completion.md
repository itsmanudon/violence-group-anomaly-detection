# Municipal evaluation platform — Phase 1 handover

Phase 1 implements the dataset-independent local evaluation workflow on the
existing React/FastAPI/Python application. No training, threshold tuning, model
replacement, remote service, Indian footage acquisition or historical result
rewrite occurred. The [plan](municipal-platform-plan.md),
[architecture/API/schema](municipal-platform-architecture.md),
[acceptance evidence](municipal-platform-acceptance.md),
[operator runbook](municipal-demo-runbook.md) and
[adapter contract](dataset-import-contract.md) are the handover set.

## Delivered screens and behavior

Dashboard counts library/jobs/pending review and shows evaluation eligibility and
composition. Video library searches/filters, shows source/rights/geography,
supports manifest dry-run/commit, multiple uploads, selection/batch submission and
managed upload deletion. Queue exposes per-clip state, attempts, progress, failure,
cancellation and explicit retry. Analysis reuses video/timeline and shows routed
windows, boxes, actor counts, class scores, disagreement, abstentions, latency
and frozen hashes. Human review captures four decisions, notes and optional event
intervals, with blinded reference reveal after save. Results displays feedback and
error examples, per-clip/full exports and a printable browser report. The professor
console remains available for explicitly cached curated examples.

SQLite schema v2 contains clips, jobs/items, append-only references/reviews/audit.
Manifest v1 supports source-safe group membership and conservative rights/labels.
The queue defaults to one worker, one GPU call, durable atomic state, exclusive
process ownership, one automatic timeout/restart retry and safe between-clip
cancellation. Fresh results persist the entire frozen model output and metadata.
Cached professor outputs are separate and excluded from municipal evaluation.
References and operator feedback never silently substitute for one another.

Metrics use only legitimately labeled, protected test clips with a predeclared
source-safe exclusion declaration, current frozen profile and actual completed
inference. Repeated jobs do not inflate the population. Class/binary/temporal
metrics explicitly report eligibility, coverage and abstentions. JSON exports
include the aggregate report; CSV contains per-clip records. Both omit raw media
and allow deliberate reference inclusion and rights-filtered external handover.

## Verification and runtime

See the acceptance document for case-level evidence. Final gate results are
recorded below after the final run. Model preflight passed for all five installed
frozen assets and receipts, and real offline cascade inference passed on CPU.
The default storage is ignored `outputs/municipal`; frontend static assets and
all dependencies/models must be prepared locally before demo time. The default
launcher binds 127.0.0.1 only. No data is automatically imported.

Measured model limitations remain **UCF ROC-AUC 0.7441, DCSASS macro F1 0.2140,
cascade alert F1 0.4733**. Scores are uncalibrated; source bias, domain shift,
missed actors, false alerts and disagreement remain material. Nothing in this
software phase establishes reliable crime detection, individual guilt or intent.

## Source scope and Git

New `surveillance.municipal` catalog/queue/reporting/router/CLI, municipal React
page/styles/utilities/tests, fixture and offline smoke scripts, Windows/Mac
launchers, Mac environment snapshot and six handover docs. Existing integration
changes are limited to demo API mounting/serialization/upload validation, React
entrypoint/navigation, package test script, HTML title, README and ignore rules.
One existing upload test now creates a real valid video; no assertions were removed.
No research pipeline, model config, checkpoint, evaluation protocol or historical
results file was modified. No datasets, weights, databases, videos or built bundles
are committed. Branch: `feat/municipal-evaluation-platform`; no push/merge/remotes.

## Remaining dataset and pilot tasks

- Acquire authorized Indian footage separately; establish documented geographic
  evidence rather than inferring location from appearance.
- Audit source recordings/alternate encodes and existing training/validation/test
  overlap; freeze independent evaluation membership and selection before inference.
- Verify class/binary/complete temporal annotations independently, and record
  reference versions with author/reason. Unknown labels remain unknown.
- Resolve licensing, training permission, municipal redistribution and privacy
  handling; use filtered exports for handover.
- Rehearse the supplied launcher/dependencies/CUDA/codec path on actual Windows +
  NVIDIA hardware, then physically disconnected operation and reboot recovery.
  Mac execution does not certify that hardware path.

None of these tasks authorize retraining, retuning or replacing the selected models.

## Final gate record

- Full pytest: **583 passed, 2 skipped**, 236 existing TorchScript warnings,
  59.92 seconds. Baseline was 561 passed, 2 skipped; **22 new municipal cases**.
- Focused final backend/API run: **28 passed**.
- Ruff lint: PASS; Ruff format check: PASS (271 Python files).
- Frontend production build: PASS; frontend Node tests: **2 passed**.
  No existing TypeScript/typecheck script was available.
- `git diff --check`: PASS.
- Full five-asset/model-loading preflight: PASS, CPU; genuine network-blocked
  frozen cascade smoke: PASS, 32 segments / 3 windows.
- Local browser walkthrough: PASS for synthetic catalog → fresh queue inference →
  analysis → independent review → Results. This is not benchmark evidence.

All source is local. A milestone commit is recorded in Git history; no push or
merge is authorized or performed. Remaining target-hardware/dataset steps above
are explicit handover work, not silently assumed validation.
