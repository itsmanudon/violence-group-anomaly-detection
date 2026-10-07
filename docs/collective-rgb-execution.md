# Collective / RGB / GT boxes / seed 0

This execution uses **our frozen Collective protocol**, not an exact reproduction
of Gavrilyuk et al. The pose experiment and its receipts/caches remain separate.
No fusion, detected boxes, surveillance adaptation or test-driven tuning is part
of this run. Generated reports, images, features and checkpoints stay ignored.

## Assets and compatibility boundary

The supplied `piergiaj/pytorch-i3d` snapshot matches revision
`05783d11f9632b25fe3d50395a9c9bb51f848d6d` through the expected source and Git-blob
fingerprints. It is a verified file snapshot, not a full Git checkout.
`models/rgb_imagenet.pt` is **50,883,138 bytes**, SHA256
`2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538`.
These are **converted DeepMind ImageNet+Kinetics RGB weights**, not official
PyTorch I3D weights.

The [export boundary](i3d-export.md) imports the supplied architecture, validates
the complete 400-class model strictly, then selects its native blocks through
`Mixed_4f`. The only checkpoint compatibility addition initializes 57 historical
BatchNorm integer counters absent from PyTorch 0.3 weights. No learned tensor or
running statistic is omitted. No upstream layer or Actor-Transformer math changes.
Fixed temporal/spatial tracing is guarded at runtime; unsupported inputs fail.

Archive: `checkpoints/i3d_mixed4f_collective_rgb_v1.pt`, SHA256
`fe7fc30ca6f26430232e2e0f6bdffd8514478a071db065e452bff46f60f4e0c4`.
Embedded metadata binds source/checkpoint hashes, revision, endpoint, preprocessing,
exporter version and PyTorch version. The serialized convolutions retain cuDNN
with full FP32 semantics, matching the validated HRNet export policy. Caller
backend flags are preserved. Unoptimized inference-only JIT execution avoids
cold-versus-warm graph folding changing cached feature rounding.

## Feature contract and validation

Ten RGB frames, offsets -5 through +4, use frame index 5 as the reference and
replicate sequence edges. Existing input decoding resizes to 480×720, converts
BGR to RGB and scales to [0,1]. The adapter normalizes once with mean/std 0.5.

`[B,3,10,480,720] -> [B,832,3,30,45] Mixed_4f -> temporal mean ->
90×160 bilinear resize -> aligned 5×5 RoIAlign -> [B,N,20800]`.

Full-resolution synthetic checks passed: native/export CUDA maximum difference
0; CPU/CUDA difference `1.52587890625e-5` within `rtol=1e-3, atol=1e-4`.
Repeated eval, finite values, padded actors, pooling, resize, RoI geometry and
existing RGB-only Actor-Transformer acceptance passed. These are contract checks,
not benchmark metrics.

Ten real **training-only** scenes (`seq04:0001` through `seq04:0091`, 73 actors)
passed frame-window/path correspondence, channel order, labels, 20,800-dimensional
features, repeatability, actor permutation and coordinate checks. Visual review
of reference overlays at frames 1, 41 and 91 found boxes aligned to the annotated
people, including overlaps/occlusion; no systematic offset was observed. GT boxes
do not necessarily annotate every visible person. Action semantics are retained
from dataset annotations, not relabeled by visual interpretation.

## GPU feasibility

Torch 2.13.0+cu126 / torchvision 0.28.0+cu126 / CUDA 12.6, RTX 4070 Laptop GPU.
Conservative batch size **one clip**, unchanged 480×720 geometry:

| Measurement | Value |
| --- | ---: |
| Clips / actors | 10 / 73 |
| Transfer + extraction | 0.962 seconds |
| Clips / actors per second | 10.39 / 75.86 |
| Peak CUDA allocated / reserved | 401 / 614 MiB |
| Decoding + collation + transfer + extraction | 1.696 seconds |
| End-to-end clips per second | 5.90 |
| Projected 2,547-clip extraction | 432 seconds, before cache-writing overhead |

Warmup is excluded. All outputs were finite; no OOM occurred. The bounded sample
comes from one source, so runtime projection is approximate. CPU/CUDA validation
was completed separately before real extraction; no CPU full-extraction run.

## Experiment identity

[collective_rgb_gt_v1.yaml](../configs/experiments/collective_rgb_gt_v1.yaml)
declares only seed 0 / `rgb_gt`, with a separate output root and feature manifest.
It retains the existing split: training 1,687 scenes / 9,027 actors; validation
85 / 427; test 775 / 3,420. Validation sources remain sequences 1, 2 and 3.
Validation group counts are crossing 24, waiting 0, queueing 0, walking 60,
talking 1. This known imbalance is not repaired during this experiment.

RGB-only training uses the existing batch size 8, 20,000 iterations, Adam 1e-4,
10× LR drops at 5k/10k, gradient clipping 1, validation every 100 iterations,
and highest validation group accuracy with earliest-maximum tie retention.
The transformer, positional encoding and joint losses are unchanged.
Pose seed 0 used batch size 16; this existing modality-specific difference is
declared when comparing results. A single-seed difference cannot isolate a
causal effect of motion information or establish statistical significance.

## Commands used

Each stage is reviewed before proceeding. Do not run this block blindly or
overwrite existing run identities. Existing outputs are refused by the staged
workflow; a deliberate rerun needs a distinct protocol/output identity.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe scripts/run_collective_experiment.py --protocol configs/experiments/collective_rgb_gt_v1.yaml --stage inspect --experiment rgb_gt --max-scenes 10
.\.venv\Scripts\python.exe scripts/run_collective_experiment.py --protocol configs/experiments/collective_rgb_gt_v1.yaml --stage preflight --experiment rgb_gt --seed 0 --max-scenes 10 --max-iterations 5
.\.venv\Scripts\python.exe scripts/run_collective_experiment.py --protocol configs/experiments/collective_rgb_gt_v1.yaml --stage extract --experiment rgb_gt
.\.venv\Scripts\python.exe scripts/run_collective_experiment.py --protocol configs/experiments/collective_rgb_gt_v1.yaml --stage freeze --experiment rgb_gt
.\.venv\Scripts\python.exe scripts/run_collective_experiment.py --protocol configs/experiments/collective_rgb_gt_v1.yaml --stage run --experiment rgb_gt --seed 0
```

The bounded GPU benchmark runs after inspection and before preflight/full
extraction. The full run CLI performs validation-selected checkpoint loading,
then one held-out evaluation and saved-prediction error analysis. No separate
evaluation invocation is used to repeat the held-out test.

## Execution reports

Ignored artifacts:

- `runs/collective/rgb_gt_v1_audit/asset_verification_acquired_20261004T165626Z.json`
- `runs/collective/rgb_gt_v1_execution/backbone_validation.json`
- `runs/collective/rgb_gt_v1/inspection/rgb_gt/rgb.json`
- `runs/collective/rgb_gt_v1/inspection/rgb_gt/temporal_visual_audit.json`
- `runs/collective/rgb_gt_v1/inspection/rgb_gt/gpu_benchmark.json`
- Local debug overlays under the same inspection directory; dataset images are not committed.

The real five-iteration preflight passed on 20 train/validation scenes / 112
actors, with no test population. Total losses were 2.7771, 1.7229, 1.2150,
0.9472 and 0.7397; group/actor components, optimizer and parameters were finite.
LR remained 1e-4. Validation executed on ten scenes / 39 actors; checkpoint
save/reload passed on CUDA. These results are **REAL RGB PREFLIGHT — NOT
BENCHMARK**. Their small subset accuracy is not used for tuning.

Full extraction completed on 2,547 scenes / 12,874 actors. All 2,547 arrays and
sidecars passed the repository's schema-v2 numeric/provenance validation, with
no errors or warnings and one common extraction configuration. Ordered scene,
box, label and split identities match the original annotation manifest.
Feature manifest SHA256:
`a538b87b65a123cbb5864f557f7b9f0f067c0c4b3385c55565f6d6f5edc565a3`.

The separate RGB freeze completed and binds **30,336 artifacts**. Protocol hash:
`16a056eccf041c11685b3a2191834d6fdaf844955d8f7bca38bfd02a5973ce13`.
Freeze hash:
`ee41d3279f4cf73740be1e07074fa01640462cf087f491b5b761c669e3732a9e`.
Extraction configuration hash:
`6bdd840d75d03d5d262b9662caf22d725ce904feee81ce97915b9f6226ae8323`.
All structural dataset checks passed. The raw-data report preserves the existing
warning: 442 boundary boxes clipped under `clip_to_image`, including 352 selected
supervised actors. Raw annotations were not modified; pose/RGB use the same policy.

Receipt verification passed. Source base commit:
`527a88f4aea3e3fbc0f4c1733afa04f26ac7706c`, with the uncommitted export/config
boundary explicitly captured in `protocols/rgb_gt_execution.json` (source tree
hash `aaca2bb84e3e9037bb22ece7ffab1be9e0648c9b4b9643bd64797541df8af459`).
No source or protocol changes occurred during training. Pose seed-0 artifacts,
its receipt and feature manifest remain byte-for-byte unchanged.

## Completed real seed-0 result (2026-10-05)

All **20,000 iterations / 200 validation checks** completed on CUDA. The LR
schedule dropped at 5k/10k; every recorded loss/metric, saved parameter and
optimizer state was finite. Optimizer step counters equal 20,000.
The first/last 100-iteration mean total losses were **1.340843 / 0.0000725823**.
Training accuracy rapidly approached one, while validation remained weak.

The highest validation group accuracy selected **iteration 1,300**, retaining
the earliest maximum. Its identity was recorded without a model forward before
saved test metrics; checkpoint SHA256:
`40a5fc3687c277914335916bd46dff72ccde5e4b6518c677f193178090de7fcd`.
This is an early selected checkpoint, similar to pose seed 0's iteration 1,100.
No later checkpoint was selected using test results.

| Selected-checkpoint validation metric | Value |
| --- | ---: |
| Group accuracy / macro F1 | 32.94% / 0.1357 |
| Actor accuracy / macro F1 | 47.31% / 0.2607 |
| Scenes / actors | 85 / 427 |

The existing orchestrator performed **one held-out evaluation**. Subsequent
verification recomputed metrics only from saved predictions. The ordered test
and validation populations match pose seed 0 exactly.

| Held-out metric | Pose seed 0 | RGB seed 0 | RGB minus pose |
| --- | ---: | ---: | ---: |
| Group accuracy | 68.00% | **77.55%** | **+9.55 percentage points** |
| Group macro F1 | 0.6548 | **0.7947** | **+0.1399** |
| Actor accuracy | 59.01% | **78.22%** | **+19.21 percentage points** |
| Actor macro F1 | 0.5698 | **0.7923** | **+0.2225** |

Held-out population: **775 scenes / 3,420 actors**, zero invalid/abstained
samples. Correct groups: 601; correct actors: 2,675. Both heads predicted all
five classes. Group accuracy exceeds the unchanged 28.13% test majority baseline.
This is a measured **Collective / RGB / GT boxes / seed 0 / frozen protocol**
result, not an exact paper reproduction or a surveillance violence benchmark.

## Class recall and confusion

Recall is diagonal count divided by true-class support. Class ordering is
crossing, waiting, queueing, walking, talking.

| Class | Group pose | Group RGB | Actor pose | Actor RGB |
| --- | ---: | ---: | ---: | ---: |
| crossing | 62.59% | 71.43% | 54.91% | 67.49% |
| waiting | 41.48% | 65.93% | 32.67% | 63.35% |
| queueing | 58.06% | 100.00% | 46.36% | 92.93% |
| walking | 68.81% | 60.55% | 66.15% | 71.65% |
| talking | 96.15% | 100.00% | 78.52% | 96.29% |

RGB group confusion matrix (rows true; columns predicted):

```text
105   1   0  41   0
 41  89   0   5   0
  0   0  93   0   0
 67  14   3 132   2
  0   0   0   0 182
```

RGB actor confusion matrix:

```text
467  22   0 203   0
 95 318   1  88   0
  0   4 447  29   1
229  29   7 690   8
  0  20   9   0 753
```

| Directed confusion | Group pose → RGB | Actor pose → RGB |
| --- | ---: | ---: |
| crossing → walking | 55 → 41 | 296 → 203 |
| walking → crossing | 51 → 67 | 232 → 229 |
| waiting → crossing | 40 → 41 | 115 → 95 |

Group crossing/walking swaps total **106 → 108**: RGB does **not** resolve that
group-level pair. Actor swaps improve **528 → 432**. Waiting recall improves,
but its group-level waiting→crossing confusion does not; actor errors in that
direction decrease. These are observed outcomes, not proof that temporal motion
alone caused the gains: backbones and existing training batch sizes also differ.

There are 174 group mistakes and 745 actor mistakes. Under the existing fixed
0.8 descriptive confidence cutoff, 91 group mistakes are high-confidence.
Mean confidence is 0.9130 for correct groups and 0.7775 for incorrect groups.
The highest-confidence mistakes include `seq15:0381` and nearby scenes from
`seq07`, with walking predicted as crossing. No behavior cause is inferred.

## Final verification and limitations

**413 pytest tests passed / 325 warnings / 150.79 seconds**: all 392 earlier
tests plus 21 offline export-boundary cases. Ruff lint passed; all 132 Python
files pass formatting; `git diff --check` passed with ordinary LF/CRLF notices.
Python 3.13.5 and the existing CUDA packages worked without dependency changes.
Expected TorchScript/tracing deprecation warnings remain; input guards constrain
the traced padding branches to their validated geometry.

Two supplemental audit-path mistakes were corrected by resolving existing pose
artifacts from their recorded run/receipt paths. They did not affect research
source, data, caches, checkpoints or protocol. No Actor-Transformer mathematics
bug was discovered. No commits, pushes, fusion or additional RGB seeds ran.

The large train/validation gap, absent validation waiting/queueing groups and
early checkpoint selection remain scientific limitations. This is one RGB seed;
RGB variance and causal attribution to motion are not established. A predefined
pose+RGB late-fusion comparison is a reasonable next experiment because pose
retains higher group walking recall, but gains are not guaranteed. Select fusion
settings using training/validation only. Any validation redesign requires a new
protocol identity; preserve this measured v1 baseline.

Completed artifacts under `runs/collective/rgb_gt_v1/rgb_gt/seed_0/`:

- `metrics.json`, `resolved_config.yaml`, `resolved_protocol.json`, `environment.json`
- `best.pt`, `last.pt`, `history.jsonl`, `tensorboard/`
- `checkpoint_selection_pretest.json`, `execution_verification.json`
- `predictions.json`, `errors.json`, `errors.md`, `actor_errors.json`
- `pose_comparison.json`, `pose_comparison.md`
- `training_validation_curves.png`, `confusion_matrices.png`

`protocols/rgb_gt.json`, `rgb_gt_dataset.json`, `rgb_gt_features.json` and
`rgb_gt_execution.json` retain freeze and execution provenance. All outputs and
dataset imagery remain ignored by Git.
