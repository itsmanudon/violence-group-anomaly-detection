# Implementation handoff — local RTX 4070, 25 October 2026

Branch: `feat/municipal-evaluation-platform`. Baseline commit:
`4d7c27ad0d7412752307a37eff273db58f93386b`. All prior and current work remains
local/uncommitted. No reset, clean, original annotation edit, research-checkpoint
overwrite, remote mutation, push, merge or publication occurred.

## Completed implementation

- Staged **five byte-identical annotation CSVs** in
  `data/indian-domain/annotations-20261025-v1/`, with source/copy hashes and proof
  that the three reviewer files' union exactly equals the combined 521 records.
  These raw local inputs are gitignored. No labels, intervals or statuses changed.
- Added receipt-based `prepare_indian_domain.py --reuse-audit`. It checks prior
  input/output receipts, current media SHA256, original-catalog IDs/paths and
  presentation durations, then emits new reports/queues without decoding footage
  again. Removed the obsolete default of 536; a reference-count difference is
  never turned into fabricated missing records.
- Reused the existing Phase 2A timing parser for the **one new Snatch timing case**:
  615 encoded samples, **599 presented samples and 599 sequentially decoded
  frames**. Stored the timestamp/count evidence separately. Count agreement does
  not prove decoded-frame identity; the strict existing loader is unchanged and
  this clip remains on hold for its current training path.
- Extended the existing binary trainer with training-only class weighting,
  balanced sampling, validation-based early stopping, FP16 AMP/scaler resume and
  gradient accumulation. Defaults retain FP32/uniform/unweighted training. Valid
  negatives remain in the training pool; no fabricated hard-negative review was
  performed. All new settings are bound to registration/resume identity.
- Checkpoints now retain validation predictions, source-wise metrics, actual
  feature-loading timing and GPU peak-memory observations. Extraction records
  model-load and per-video timing when a legitimate cache is eventually built.
  Accuracy and false-negative rate supplement existing binary metrics.
- Added guarded `train_indian_binary.py calibrate`: requires a registered selected
  `best.pt`, the matching frozen split, and exactly its validation membership,
  labels and source IDs. Writes a new immutable threshold receipt, includes
  recall/FPR tradeoffs, refuses test membership and existing output paths.
- Added bounded `profile` command, measured on the local RTX with synthetic
  resident C3D-shaped tensors. This does not load original research checkpoints,
  extract footage features, create an Indian model, or measure recognition quality.

Files affected: `.gitignore`; `datasets/indian_readiness.py`;
`training/indian_binary.py`; `evaluation/indian_binary.py`;
`scripts/prepare_indian_domain.py`; `scripts/train_indian_binary.py`;
`tests/test_indian_readiness.py`; `tests/test_indian_binary.py`; updated Indian
runbook/contract/matrix/readiness/plan and this handoff. Existing six-class model
and five-model inference implementations were not modified.

## Authoritative data evidence

Confirmed original inventory exists at
`D:/violence-datasets/_manifests/v1-20261021/master-clip-inventory.csv`.
It agrees with the handover catalog and prior physical-media reconciliation:
**523 media = 521 videos + 2 audio-only exclusions**. All 521 video IDs have
annotation rows and current byte-hash matches. **Missing video annotations: 0.**
The 536 figure is retired; no 15-record missing list is required or justified.

| Gate | Actual count |
| --- | ---: |
| Violence / nonviolence / uncertain / blank | 48 / 465 / 7 / 1 |
| Strict existing-loader qualified | 520 |
| Annotation and media candidates | **504: 39 positive + 465 negative** |
| Quality holds | **17: 16 annotation priority + 1 loader/timing hold** |
| Independent source/incident clearance | **0** |
| Training / protected-test eligible | **0 / 0** |
| New train/validation/test assignments | **None; all 521 unassigned** |

Reviewer counts reproduced from actual CSV bytes:

| Reviewer | Total | Violence | Nonviolence | Uncertain/blank |
| --- | ---: | ---: | ---: | ---: |
| 1 | 174 | 17 | 154 | 3 |
| 2 | 178 | 12 | 162 | 4 |
| 3 | 169 | 19 | 149 | 1 |

Presentation-duration sums verified against the original catalog: **7,002.026 s**
for all 521 videos, **6,982.009 s** for the 520 strict-qualified videos, and
**6,802.291 s (113.37 minutes)** for the 504 annotation/media candidates. These
are clip-duration sums, not representative continuous camera-hours or evidence
for an operational false-alarm rate. Catalog durations remain distinct from
decoded frame-count/FPS clocks; no temporal localization accuracy is claimed.

All 278 original `needs_review` media values remain verbatim. Original-source,
playback and annotation statuses are not interchangeable. Original source IDs,
URLs and review timestamps are still blank. All 48 positive clips have unknown
catalogued geography; source/label confounding remains as previously reported.
The Archive visual pair and historical transformed-footage/source overlap remain
uncleared. Preserve the 14 HCI municipal reservations outside optimization.

## Measured local GPU engineering profile

Observed GPU: **NVIDIA GeForce RTX 4070 Laptop GPU**; total VRAM
8,585,216,000 bytes (about 8 GiB), free at observation 7,411,335,168 bytes
(6.90 GiB). Python 3.13.5, torch 2.13.0+cu126, torchvision 0.28.0+cu126,
CUDA build 12.6. No environment reinstall or remote GPU dependency.

Each profile used three warm-up steps followed by ten synchronized timed
forward/backward/Adam steps. Inputs were resident synthetic `[B,32,4096]` tensors.
Peak allocation is PyTorch allocator memory, not a system-wide VRAM peak.

| Precision | Batch | Timed seconds, 10 steps | Synthetic examples/s | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1 | 0.009031 | 1,107 | 17.72 |
| FP32 | 8 | 0.009077 | 8,813 | 27.35 |
| FP32 | 32 | 0.010925 | 29,290 | 60.35 |
| FP16 AMP | 1 | 0.012148 | 823 | 17.72 |
| FP16 AMP | 8 | 0.012607 | 6,346 | 27.35 |
| FP16 AMP | 32 | 0.014679 | 21,800 | 60.35 |

These short engineering diagnostics show no AMP speed benefit for this tiny head;
retain FP32 by default. They are **not C3D/I3D extraction throughput, video FPS,
full-cascade latency or Indian-domain recognition results**. Real extraction and
dataset-throughput measurements remain unrun until the data gate passes.

## Verification

First affected run: **72 passed in 35.94 s**. After adding selected-checkpoint
calibration: **75 passed in 24.74 s**, no skips/failures. The final run included
ingestion, MM:SS, receipt reuse/mutation rejection, source splitting, binary
shapes, CPU/CUDA exact resume, balanced/weighted losses, AMP+accumulation,
early-stopping protection, validation/test membership isolation and six-class
RGB loading checks. These are software fixtures, not new Indian experiments.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_indian_readiness.py tests/test_indian_domain.py tests/test_indian_split.py tests/test_indian_binary.py tests/test_rgb_behavior_loading.py -p no:cacheprovider --basetemp runs/indian-rtx-implementation-tests-20261025-v2
```

Initial lint identified formatting and a profiling closure/deletion issue; both
were corrected. Final affected lint/format checks and whitespace checks passed.
The prior 112-test evidence remains dated evidence, not rerun or enlarged by
addition. No unrelated full suite or repeated Gradio investigation was run.

## Precise blockers and next executable steps

Human evidence is still required: adjudicate the 17 priority/media holds, supply
original recording and incident mappings, establish permitted research use, and
clear within-corpus/historical overlap. Review candidate footage and source
context without changing uncertain labels to negatives. Collection/tour guards
are not proof of independent original recordings. There is insufficient cleared
positive-source diversity for a defensible 70/15/15 split today.

The immediately actionable files are:

- [17-row priority queue](../../runs/indian-domain/readiness-authoritative-20261025-v1/priority-review-queue.csv)
- [521-row source evidence queue](../../runs/indian-domain/readiness-authoritative-20261025-v1/source-provenance-queue.csv)
- [Authoritative readiness/duration receipt](../../runs/indian-domain/readiness-authoritative-20261025-v1/readiness.json)
- [Snatch timing evidence](../../runs/indian-domain/snatch1-timing-20261025-v1/timing.json)

Fill the new queues with actual adjudicator/date/evidence; preserve raw CSV
snapshots. Do not mark rows approved merely to make splitting pass. Once evidence
is supplied, create a reviewed import/version and use the existing guarded split
freezer. Run the frozen baseline on the appropriate validation population before
choosing a new model. No Indian baseline or improvement metric is currently
measured; C/D remain conditional on A/B failure evidence.

Commands that are runnable now for engineering verification (choose new output
versions; current receipts are preserved):

```powershell
.venv/Scripts/python.exe scripts/train_indian_binary.py profile --output runs/indian-domain/rtx4070-head-fp32-20261025-v2 --batch-sizes 1 8 32 --steps 10 --precision fp32
.venv/Scripts/python.exe scripts/prepare_indian_domain.py --csv data/indian-domain/annotations-20261025-v1/indian_annotations_combined_unverified.csv --reuse-audit runs/indian-domain/readiness-20261025-v1 --original-catalog 'D:/violence-datasets/_manifests/v1-20261021/master-clip-inventory.csv' --output runs/indian-domain/readiness-authoritative-20261025-v2
```

Do not repeat those unchanged measurements without a reason. After legitimate
source-safe approval and frozen-baseline evidence, A/B commands are:

```powershell
# Future split/cache paths: these have NOT been created or approved.
.venv/Scripts/python.exe scripts/train_indian_binary.py cache --split-dir data/indian-domain/split-v1 --checkpoint checkpoints/c3d_fc6_openmmlab_v1.pt --output data/indian-domain/c3d-cache-v1 --device cuda:0 --batch-size 2
.venv/Scripts/python.exe scripts/train_indian_binary.py train --split-dir data/indian-domain/split-v1 --cache data/indian-domain/c3d-cache-v1 --output runs/indian-domain/a-seed7 --device cuda:0 --steps 100 --seed 7 --batch-size 8 --patience 20
# Compare one training-only imbalance remedy; do not combine factors initially.
.venv/Scripts/python.exe scripts/train_indian_binary.py train --split-dir data/indian-domain/split-v1 --cache data/indian-domain/c3d-cache-v1 --output runs/indian-domain/b-weighted-seed7 --device cuda:0 --steps 100 --seed 7 --batch-size 8 --patience 20 --class-weighting balanced
.venv/Scripts/python.exe scripts/train_indian_binary.py calibrate --split-dir data/indian-domain/split-v1 --checkpoint runs/indian-domain/a-seed7/best.pt --output runs/indian-domain/a-validation-threshold-v1 --min-recall 0.9 --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

These are prospective bounded settings, not completed experiments. Use the same
approved populations for comparisons, record recall alongside false-positive
changes, and keep test inference outside model selection/calibration. None of
this establishes SMC pilot readiness, intent, guilt or enforcement authority.
