# Controlled A0/A1/A2 study — completed 26 October 2026

**Outcome C: diagnostic capability measured; no alert-decision improvement.**
Coverage/agreement changes the availability of a six-class suggestion, not the
anomaly alert. We did not invent a suppression policy. All **1,022** binary
decisions were identical under A0 legacy, A1 coverage+agreement and A2 coverage-only.
This is a historical validation diagnostic, not Indian performance or an
independent confirmatory test. No weights, threshold or routing budget changed.

## Inputs and eligibility

Original manifest `data/manifests/dcsass_human_centric_v1_final.jsonl`, SHA256
`f3e7fc11159fce191f61b8ecf1d0d667cb68c9f949afa3ab84758c0532ee7258`.
All selected validation rows: **1,022 clips / 32 original source recordings**,
591 Normal / 431 non-Normal, six-class supports 591/68/82/9/232/40.
Each source is `val` in both the human split receipt and the selected UCF Sultani
protocol. No selected clip hash crosses human partitions; known source aliases
and declared source/content overlap checks pass. All 1,022 files existed and
matched recorded bytes before and after inference.

Checked original inputs:

- `runs/dcsass/human_centric_v1/final_split_receipt.json`, SHA256
  `522110e5a930ff1aed598e05ac413cb2fa64e8211db4f489e84892ed6febea74`.
- `runs/ucf-crime/sultani_shared_safe_v1/protocol.json`, SHA256
  `a163f05823825daf647622eb3f776ca3de4cefaf77c55ccf0aac6b2ecbf40c9d`.
- Both selected-checkpoint receipts, behavior training registration and embedded
  checkpoint manifest identity. No validation source was used for optimization
  of either learned head. Frozen C3D/detector/I3D remain unchanged.

Validation **was already used for checkpoint selection**: anomaly bag ROC-AUC,
behavior macro F1/initialization selection. Threshold 0.5 was a prospective fixed
default, not selected here. These are original validation sources, not the
protected 991-clip test population. Original-recording grouping is checked;
shared incidents across different recordings are not annotated. Backbone
pretraining overlap is not newly certified. No new rights claim is made.

The former missing joint-validation-input blocker was resolved: one authorized
inference-only pass generated the missing per-window outputs. No historical
test inference, Indian footage access, parameter tuning or training occurred.
Binary labels mean abnormal/non-Normal weak clip labels, **not certified violence**.
They do not establish actor or temporal action truth.

## Fixed configuration and identical inputs

Config `configs/surveillance_demo_ucf_v1.yaml`, SHA256
`e5e6b5d6960af2f7790911cbd38708639aca1182a9d0a3ded9f57ac330c56ebb`.
Study `configs/indian-domain/controlled-review-20261026.json` fixes the existing
three modes, threshold 0.5, three windows, bootstrap seed 20261026 and 2,000
replicates before scoring. All modes consume identical saved predictions and
labels through the existing comparison function. Assessment source is unchanged.

| Checkpoint | SHA256 |
| --- | --- |
| C3D FC6 | `beac4ff065de3663bf5c6bab36aedca4fe56ec59b231c66bf5374c23eda14831` |
| Sultani MIL | `7e5b548ea6260614209a727a947e6270a1a1af2185d29accf1778f91b4af7c66` |
| Faster R-CNN | `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384` |
| I3D | `fe7fc30ca6f26430232e2e0f6bdffd8514478a071db065e452bff46f60f4e0c4` |
| Six-class Actor-Transformer | `7870e512787e74cd76263df28e40c8bc39ecb8a3d0a5b2b8b6a7e72a218a8c69` |

## Actual results

All three modes: **TP 285, FP 364, TN 227, FN 146**. Precision 0.439137,
recall 0.661253, F1 0.527778, specificity 0.384095, FPR 0.615905, FNR 0.338747,
ROC-AUC 0.541820, trapezoidal PR-AUC 0.455655 (AP 0.457681).
Raw binary decision coverage is 100%; binary abstentions are zero.
Correct/incorrect alert changes, false alerts prevented, true abnormal detections
lost and false negatives rescued are **all zero**. Violence-specific recall/loss
is unavailable from this binary annotation semantics.

| Mode | Covered behavior / 1,022 | Behavior abstained | Conditional accuracy | Correct suggestions / all clips | Conditional six-class macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 legacy | 464 (45.40%) | 558 (54.60%) | 197/464 = 42.46% | 197/1,022 = 19.28% | 0.260661 |
| A1 coverage+agreement | 214 (20.94%) | 808 (79.06%) | 125/214 = 58.41% | 125/1,022 = 12.23% | 0.266787 |
| A2 coverage only | 333 (32.58%) | 689 (67.42%) | 148/333 = 44.44% | 148/1,022 = 14.48% | 0.287617 |

A1 withholds **178 wrong and 72 correct** legacy suggestions, leaving 89 wrong
suggestions. A2 withholds **82 wrong and 49 correct**, leaving 185 wrong. Higher
conditional accuracy compares selected subsets and is not improved full-population
classification. Abstentions are neither correct negatives nor silently removed
from the 1,022 denominator. A1 retains no Fighting-class suggestion on these data;
its zero covered support is not evidence of class correctness.

Paired original-source cluster bootstrap (32 clusters, 2,000 resamples, seed
20261026), A1 minus A0, descriptive 95% percentile intervals:
conditional-accuracy delta **+7.33 to +22.45 percentage points**;
coverage delta **−32.49 to −16.83 points**;
correct-suggestion/full-population fraction delta **−10.06 to −4.21 points**.
A2 conditional-accuracy delta interval is **−1.78 to +5.64 points**. These are
conditional on recording-level independence, reused model-selection validation
and the fixed modes. No clip-independent p-value, causal or confirmatory claim.
Binary decision delta is exactly zero by contract, not a significance finding.

## Actual execution and runtime

Artifact root: `runs/indian-domain/controlled-review-20261026-v1/`.
`validation-inference/` contains immutable registration, all 1,022 individual
results, `predictions.json`, the all-split comparison schema adapter, metrics,
runtime and completion with hashes. `comparison/` contains matched comparison,
diagnostics, 1,022-row comparison CSV, zero-row `actual-alert-changes.csv`, outcome
counts, cached-policy timing and examples. No frames/footage exported.

Fresh GPU inference was needed because standalone behavior outputs could not
reconstruct cascade routing/multiple-window coverage. RTX 4070 Laptop, torch
2.13.0+cu126, CUDA 12.6; all clips 30 FPS, 30–120 frames (1–4 seconds). Model
startup **1.876730 s**; inference wall sum **697.013806 s**; loop including
serialization **716.003672 s**. Per-clip wall median **0.736612 s**, p95
**1.366841 s**. Peak PyTorch allocation **972.15 MiB**, reservation **1,234 MiB**.
First required clip includes cold effects; no extra warmup/repeated GPU benchmark.

Separately, five timed CPU batches over resident cached predictions (one full
batch warmup/mode): A1 median **18.356 ms per 1,022 applications**; individual
median **3.8 microseconds**, p95 **46.7 microseconds**. Most clips bypass or abstain
quickly. This excludes JSON loading, decoding, all networks and bootstrap; it is
not a speedup comparison against GPU/video inference. No camera-hour false-alarm
rate, real-time CCTV throughput or event detection latency is claimed.

## Commands and focused verification

```powershell
# Already completed once; do not repeat GPU inference merely to reproduce a table.
.venv/Scripts/python.exe scripts/run_controlled_review_experiment.py infer --output runs/indian-domain/controlled-review-20261026-v1/validation-inference
# Safe cache-only reproduction into a NEW directory:
.venv/Scripts/python.exe scripts/run_controlled_review_experiment.py analyze --predictions-dir runs/indian-domain/controlled-review-20261026-v1/validation-inference --output runs/indian-domain/controlled-review-20261026-v2-comparison
```

Focused controlled-mode/source/six-class tests: **41 passed in 31.18 s**.
No protected test tuning or unrelated suite. Source hashes and all previous
uncommitted files/checkpoints are checked in the final preservation receipt.
See [decision changes](decision-change-analysis.md) and
[recommendation](next-experiment-recommendation.md).

Final combined affected run (including synthetic Indian proposal tests): **64
passed in 22.04 s**, no skips/failures. Six new Python files pass lint/format;
`git diff --check` passes. Reproduce in a new basetemp:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_controlled_review.py tests/test_cascade_review.py tests/test_cascade_evaluation.py tests/test_rgb_behavior_loading.py tests/test_indian_development.py tests/test_indian_split.py tests/test_indian_review.py -p no:cacheprovider --basetemp runs/indian-domain/controlled-review-final-new-version
```

Branch `feat/municipal-evaluation-platform`, HEAD
`4d7c27ad0d7412752307a37eff273db58f93386b`. All **59** pre-existing uncommitted
files are byte-identical to this turn's startup snapshot. All checkpoint and five
original/copy annotation hashes match. New code/config snapshots, experiment
output hashes, final test XML, status and preservation evidence live under
`runs/indian-domain/controlled-review-20261026-v1/`. No commit/push/merge/deploy.
