# Phase 3 — Automated Dataset Intelligence and Review Queues

Execute after Phase 2 in the same checkout. Teammates are busy for 1–2 days. Automate everything that can be done without assigning false human provenance judgments.

## Authoritative starting evidence (verify locally)

Last reported catalog: **521 videos plus 2 audio-only files**; retired 536 estimate does not prove missing media. Quality screening: **504 candidates = 39 violence + 465 nonviolence**, **17 quarantined**, **0 source-approved Indian train/evaluation clips**. Previously reported inventory path: `D:\violence-datasets\_manifests\v1-20261021\master-clip-inventory.csv`. Reconcile with actual disk and `docs/indian-domain/implementation-handoff-20261025.md` rather than inventing unavailable sources.

Original annotation files may include `reviewer_1.csv`, `reviewer_2.csv`, `reviewer_3.csv`, and derivatives `indian_annotations_combined_unverified.csv` / `indian_annotations_priority_review.csv`; locate actual local paths. Historical CSV auditing reported 48 original violence labels, 465 nonviolence, seven `not_sure`, one blank; the *39 candidate positives* are after quality screening. Do not conflate these populations.

## Work A: source/provenance candidates

Using metadata, existing catalog, media paths, file names, any actual notes/URLs, source hashes, temporal similarity, and perceptual fingerprints:
- exact SHA-256 duplicates;
- near-duplicates / re-encoded videos;
- likely crop/trim/time-overlap derivatives;
- possible shared incident between collections;
- visually similar but **not** proven same incident;
- unknown original source.

Capture observations and confidence/method. Suggest groupings, but **do not mark suggestions as verified incident/source IDs**. Do not use unique `clip_id` as proof of independent incidents. Do not use face identity/recognition. If media cannot be read, report accurately.

## Work B: automated video/annotation checks

Reuse previously implemented validators/fingerprinting before adding anything. Do lightweight reproducible frame sampling only when needed; cache/reuse derived fingerprints and record configurations. Detect decoding failures, missing paths, audio-only items, duration/fps mismatches, MM:SS versus numeric timestamp parsing, end-times slightly exceeding duration, contradictory or impossible ranges, nonviolence with violence intervals, uncertain labels, missing values, and conflicting media or provenance statuses.

**Never alter original CSVs, reviewer labels, timestamps, media status, original media, or checkpoint files.** Store normalized time fields and repair proposals separately; no fabricated reviewer decisions. `not_sure` is quarantine, not a trainable third class. Do not interpret `needs_review` as proof the violence label is wrong; it may concern provenance.

## Work C: high-value human review queues

Create a **deduplicated, actionable priority queue** with actual record IDs and supported evidence, covering:
- all currently eligible violence positives for second-look review;
- 17 quality holds and unavailable/uncertain media;
- high-confidence near-duplicate/incident candidates;
- unverified source/incident provenance;
- temporal anomalies or contradictory labels;
- candidate hard negatives **only if actual existing prediction scores support it**.

Useful columns: `clip_id`, `file_path`, collection, original label, quality status, specific reason, original reviewer, observed evidence, algorithm/score, suggested group, review action, and separate blank human-disposition fields. Distinguish one record with several flags from several records. If a required input is unavailable, say so.

## Work D: source/label confounding and split readiness

Report source-by-label counts, original incident diversity *where verified*, unknown source fraction, near-duplicate clusters, leakage risks, and feasible data partition constraints. The prior collections were strongly source-label-correlated; verify this from current CSV instead of assuming it remains identical.

Do not approve a source-safe train/validation/test split, perform supervised Indian training, or produce an Indian generalization metric. Identify **the smallest set of manual decisions** that could unlock a defensible source-safe training population and protected test evaluation.

## Work E: outputs and tests

Write:
- `docs/indian-domain/sprint-20261026/dataset-quality-report.md`
- `docs/indian-domain/sprint-20261026/source-overlap-report.md`
- `docs/indian-domain/sprint-20261026/source-readiness.md`

Write private candidate/review CSVs in the repository's established ignored local artifact directory; in docs use *paths*, counts, scripts and regeneration commands without exposing unnecessary personal data/frames. Do not upload raw CCTV or identifiable frames to external services.

Run focused deterministic tests for missing provenance, source conflicts, exact/near duplicates, ambiguous near-matches, timestamp normalization, uncertainty, quarantine preservation, and no accidental automatic approval. Record actual counts and limits, then **proceed to Phase 4 automatically.**
