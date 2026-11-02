# Pose to Protect — implementation and experiment plan

25 October continuation: the actual annotations are now audited in the
[data-readiness report](data-readiness-20261025.md). Use its gates and the
[controlled training matrix](training-matrix-20261025.md) for the next stage.
The original milestone schedule below is retained; dates are not completion claims.

Execution policy now requires the local RTX 4070 only. The authoritative inventory
is 521 videos plus two audio-only exclusions; the 536 verbal estimate is retired.

Baseline: `4d7c27ad0d7412752307a37eff273db58f93386b`. Deadline: 10 November
2026. Extend this repository; preserve the functioning six-class cascade and all
existing receipts. No federated learning, repository migration, remote changes,
or automatic enforcement. Predictions remain human-review aids.

## Milestones, files and acceptance gates

| Target | Stage and file-level changes | Acceptance criteria |
| --- | --- | --- |
| 23 Oct | P0: `docs/indian-domain/repository-audit.md`, this plan | Read implementations/manifests/tests, record baseline/local checkpoint hashes; distinguish implementation from actual execution |
| 23–24 Oct | P1: new `datasets/indian_domain.py`, `scripts/prepare_indian_domain.py`, `configs/indian-domain/columns.example.json`, `docs/indian-domain/data-contract.md`, `tests/test_indian_domain.py` | Inspect actual header before mapping; strict labels; quarantine uncertainty and invalid/conflicting rows; full decode/hash; retain all raw provenance/temporal values; deterministic quality report; exclusive output directory; synthetic integration tests |
| 24–25 Oct, conditional on annotations | P2: new `datasets/indian_split.py`, `scripts/freeze_indian_split.py`, tests in `tests/test_indian_split.py` | Source/incident/content connected groups; benchmark exclusion inventory; near-duplicate review dispositions; explicit provenance review; seeded approximate source-group 70/15/15 with class support; actual clip/source/class counts; immutable test membership and hash receipt; no final split if unresolved |
| 25–27 Oct | P3/P6: new binary module/trainer/CLI and tests, using frozen C3D bags first; later actor cache adapter reusing `dcsass_actor_inputs.py`, `inference/rgb_behavior.py`, I3D and detector | Distinct binary checkpoint type; features bound to frozen split; train-only optimization, validation-only selection; CPU/GPU forward/backward; smoke/save/resume determinism; preserve original six-class loader tests. C3D head is an explicitly separate whole-clip baseline, not an Actor-Transformer claim |
| 27–30 Oct | Register A/B/C protocols under new `configs/experiments/indian_*`; adapt `experiments/environment.py`, hashing and selection patterns; new actor binary group-only adapter | A frozen cascade with anomaly and behavior outputs reported separately; never assume all non-Normal behavior is violence. B frozen representations + binary head; C selectively tune classifier representation only. Same held-out clip IDs, truth and routing/coverage reporting; detector failures remain in denominator; stop if insufficient data |
| 30 Oct–3 Nov | P4: new `evaluation/indian_binary.py` and temporal-rule module/tests; validation prediction/calibration receipt | Independent hard-negative, threshold, causal smoothing, minimum-duration, hysteresis/persistence, abstention ablations. Select by F1 subject to declared recall floor. Train-only hard negatives; never mine protected test. ROC-AUC, PR-AUC definition, precision/recall/F1/FPR/confusion and coverage. Continuous camera-hours only with representative continuous footage; temporal latency only with onset truth |
| 3–5 Nov | P5: at most 1–2 controlled variants in new experimental configs/modules | Gate on A/B/C and failure evidence; choose improved temporal aggregation or actor aggregation only if justified. State hypothesis, baseline, one changed variable, parameter/compute budget, seed/settings, validation outcome and saved failures. No novelty claim from component composition |
| 5–7 Nov | P7/final preparation: regression and GPU execution receipts | Local RTX 4070 smoke, VRAM/throughput profiling, cache validation and targeted tests. Record GPU/software/CUDA/torch, data/checkpoint/config/code hashes, seeds, duration, metrics and paths. No expensive launch without successful smoke and source-safe data approval |
| 7–8 Nov | P8: final freeze and one protected test evaluation, new `docs/indian-domain/final-evaluation.md` and immutable results directory | Validation-selected checkpoint, preprocessing and thresholds frozen before test. Recheck test independence/hashes. All A/B/C comparisons on compatible populations; six-class metrics only where genuine six-class truth exists. Preserve failure examples and latency scope. Do not tune after seeing test |
| 9–10 Nov | P9: new `architecture.md`, `dataset-provenance.md`, `experiment-comparison.md`, `methodology.md`, `professor-demo.md`, `pilot-readiness.md` | Source-backed reports, reproducible commands and limitations. Mark unperformed experiments/metrics unavailable; demonstrate human review, disagreement and abstention; no intent/guilt/culpability or deployment-guarantee claims |

Dates are planning targets, not completed experiments. If labels arrive late,
protect P1–P4 quality and reduce architecture experiments first.

## Immediate preparation scope

Implement P1/P2 plus independently testable binary-head and evaluation foundations
where feasible in this change. Keep a stage verification log with exact passed,
failed and unrun commands. Real ingestion, split approval and research results
remain blocked until media, annotations and provenance are supplied. Do not create
a fake 536-row manifest or expected 375/80/81 split.

## Data arrival procedure

1. Inspect real CSV header, unique labels and reviewer/playback values; configure
   canonical-to-actual column names only after inspection. Retain original CSV hash.
2. Import into a fresh version directory; review quality/quarantine and conflicting
   labels. Missing temporal fields remain unknown. Resolve source, incident,
   origin/rights and review evidence with annotators.
3. Supply a benchmark inventory covering all previously used UCF/DCSASS/Collective
   sources, aliases and available content hashes; record scope/evidence. Review
   visual candidates including re-encodes/crops manually. Unresolved provenance or
   duplicate/overlap reviews block final split approval.
4. Freeze groups once, inspect actual counts/distributions; if source support is
   inadequate, revise collection before final membership is selected. After
   freezing, never move test clips to improve training or metrics.
5. Extract frozen features into a new directory; bind checkpoint/preprocessing,
   video and feature hashes; run smoke before real optimization. Train/validate
   without accessing test predictions or labels for model selection.

## Research controls

Use source-level confidence intervals/seed variability where sample size permits;
report all source/clip denominators. Freeze a validation recall floor before
calibrating thresholds and show precision–recall tradeoffs. Binary annotation does
not provide six behavior labels, actor supervision or complete temporal truth.
Balanced clip evaluation cannot estimate operational false alarms/camera-hour.
Keep baseline population definitions unchanged, including uncovered actors and
non-routed clips; report conditional and all-clip coverage separately.
