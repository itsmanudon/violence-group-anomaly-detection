# Milestone 2C-R1: first real Collective pose baseline

Result name: **Collective / Pose / GT boxes / seed 0 / frozen protocol**.
The tooling remains verified on artificial fixtures. The first genuine real
seed-0 experiment has now completed: **68.00% group accuracy / 0.65482 macro F1;
59.01% actor accuracy / 0.56979 macro F1**. See the
[real execution record](#first-real-execution-2026-10-0304). This is our frozen
Collective protocol, not an exact paper reproduction; seeds 1/2 have not run.

## Assets and protocol

Acquire the original **dataset.ver1** manually from the
[authors' dataset page](https://cvgl.stanford.edu/projects/collective/collectiveActivity.html)
(University of Michigan authors; page now hosted at Stanford). Use its original
five-class, 44-sequence release, not the augmented dancing/jogging variant.
The authors document every-tenth-frame annotations and the seven-column format.
Install all 44 Collective sequences from that release in
`data/raw/collective/seq01` through `seq44`. See the exact
[data layout](../data/README.md#collective-activity-milestone-2a). Each source needs
consecutive `frame0001.jpg`, ... and `annotations.txt`. Annotation rows are
`frame x y width height class_id pose_id`; the existing parser consumes the first
six columns and ignores the trailing pose ID. Annotation centers remain every
tenth frame starting at 1. Do not fabricate annotations or silence validation
errors. No data, upstream repository or checkpoint is downloaded automatically.

Supply the official COCO `pose_hrnet_w32_256x192.pth` and a vetted local official
HRNet checkout. Export with [hrnet-export.md](hrnet-export.md). Random, unrelated
and synthetic checkpoints are not real feature assets.

The existing [frozen protocol](collective-protocol.md) stays unchanged:

- Test sequences: **5,6,7,8,9,10,11,15,16,25,28,29**.
- Training-side sources: the other **32** sequences in 1..44.
- Validation: **1,2,3**, removed from optimization, leaving **29** train sources.
- Ten frames, offsets -5..4 with edge replication; pose uses reference index 5.
- All GT actors in manifest order; majority actor class defines group class,
  lowest class ID on a tie; crossing/waiting/queueing/walking/talking.
- HRNet pre-final feature map flattened to 98304; no additional L2 normalization.
- Embedding 128; one encoder layer/head; FFN 256; dropout .1; 2D positions;
  joint group/actor cross entropy with weights 1/1.
- Adam 1e-4, betas .9/.999; 20,000 iterations; LR drops at 5,000 and 10,000;
  batch 16; gradient clip 1; validation/checkpoint interval 100.
- Select checkpoint on **validation group accuracy**, earliest checkpoint on ties.
- Declared seeds **0,1,2**; inspect seed-0 health before starting seeds 1 and 2.

Call this **our frozen Collective protocol**. Exact equivalence of historical
paper split IDs is unverified; direct box stretching also differs from upstream
HRNet pose preprocessing. No architecture/hyperparameter retuning is introduced.

## Visible boundary actors in the installed release

Real annotation inspection exposed positive-area actors partly outside the frame,
including one-pixel boundary overruns. Strict parsing remains the default for
existing protocols. The explicitly authorized execution policy is
`dataset.annotation_box_policy: clip_to_image`: intersect each raw xywh box with
the source image, retain the visible actor, and preserve annotation order and
labels. Nonpositive, fully outside and duplicate clipped boxes still fail.
The raw dataset is never edited. Validation reports include every correction's
source path, annotation line, frame, raw xywh, clipped xyxy and supervised-selection
flag in `box_corrections`, plus a summary warning. Nonselected/NA rows are audited
as well. This is a declared annotation handling choice, not test-set tuning.

Use a new protocol ID (`collective_pose_visible_boxes_v1`) for this policy before
creating a freeze receipt; previously frozen strict protocols stay unchanged.
The locally installed archive contains an `ActivityDataset` wrapper, so its local
protocol root is `../../data/raw/collective/ActivityDataset`. Split IDs, label
mapping, model mathematics and training settings remain unchanged.

For this installed release, use the explicit policy when revalidating:

```powershell
.\.venv\Scripts\python.exe scripts/validate_collective.py --root data/raw/collective/ActivityDataset --manifest data/manifests/collective.jsonl --box-policy clip_to_image --report runs/collective/dataset_revalidation.json
```

## Commands in order

Run from the repository root with the installed environment (`python` below can
be replaced by `.venv/Scripts/python.exe`). First export the official checkpoint:

```powershell
python scripts/export_hrnet_features.py --hrnet-repo "D:/path/to/deep-high-resolution-net.pytorch" --checkpoint checkpoints/pose_hrnet_w32_256x192.pth --output checkpoints/hrnet_w32_features.pt
Copy-Item configs/experiments/collective_protocol.yaml configs/experiments/collective_pose_local.yaml
(Get-FileHash checkpoints/hrnet_w32_features.pt -Algorithm SHA256).Hash.ToLower()
```

The local copy stays next to the candidate so YAML-relative paths remain valid;
it is gitignored. Declare any boundary policy and protocol ID as described above
before freezing. Configure these asset fields in that copy, using the
feature archive hash printed above (or `archive_sha256` from its export report):

```yaml
features:
  # Retain normalization, rgb and the other existing pose fields.
  pose:
    architecture: pose_hrnet_w32
    endpoint: pre_final_layer
    checkpoint: ../../checkpoints/hrnet_w32_features.pt
    checkpoint_sha256: <feature-archive-SHA256>
```

This is a partial configuration excerpt, not a replacement YAML. Retain all split,
seed, preprocessing and training values. RGB/detector paths may remain null;
every command below explicitly selects **pose_gt**, so they are not required.
The copied candidate becomes bound to actual data and caches by the freeze receipt.

```powershell
python scripts/run_collective_experiment.py --protocol configs/experiments/collective_pose_local.yaml --stage prepare
python scripts/run_collective_experiment.py --protocol configs/experiments/collective_pose_local.yaml --stage inspect --experiment pose_gt --max-scenes 10
# Optional local GT-box/index overlays for the SAME training-only inspection subset:
python scripts/inspect_actor_features.py --manifest runs/collective/inspection/pose_gt/training_subset.jsonl --checkpoint checkpoints/hrnet_w32_features.pt --modality pose --max-scenes 10 --image-size 480 720 --report runs/collective/inspection/pose_gt/pose_debug.json --debug-dir runs/collective/inspection/pose_gt/debug
python scripts/run_collective_experiment.py --protocol configs/experiments/collective_pose_local.yaml --stage preflight --experiment pose_gt --seed 0 --max-scenes 10 --max-iterations 5
```

`prepare` validates all 44 sequences and emits
`runs/collective/dataset_validation.json` plus the actor JSONL. If that manifest
already exists, validate it instead of overwriting it:

```powershell
python scripts/validate_collective.py --root data/raw/collective --manifest data/manifests/collective.jsonl --val-sequences 1 2 3 --report runs/collective/dataset_validation.json
```

Inspect the dataset counts/distributions and warnings, `inspection/pose_gt/pose.json`
and optional overlays before continuing. Inspection checks finite values, correct
shape, repeated eval, reverse actor ordering and box-coordinate mapping. It records
crop size, processed reference-image size, source/frame ID, boxes, actor indices and
labels. Overlays show the processed reference frame with GT indices, not identities;
they belong under ignored runs and must not be committed or redistributed.

The preflight validates the full data tree but bounds expensive extraction to ten
train and ten validation scenes, with five optimization steps and **no test
evaluation**. Check finite losses, actor labels, validation, LR, checkpoint saves
and environment metadata under `runs/collective/preflight/`. These outputs are
explicitly **NON-BENCHMARK**. Existing preflight/cache/run paths are refused; use
a new documented output root for a separately versioned retry, never erase results
to hide a failure or overwrite a frozen experiment.

After inspection/preflight pass:

```powershell
python scripts/run_collective_experiment.py --protocol configs/experiments/collective_pose_local.yaml --stage extract --experiment pose_gt
python scripts/run_collective_experiment.py --protocol configs/experiments/collective_pose_local.yaml --stage freeze --experiment pose_gt
python scripts/run_collective_experiment.py --protocol configs/experiments/collective_pose_local.yaml --stage run --experiment pose_gt --seed 0
# Read the saved evaluation; this does NOT run test inference a second time:
python -m json.tool runs/collective/pose_gt/seed_0/metrics.json
python scripts/analyze_collective_errors.py --gt runs/collective/pose_gt/seed_0/predictions.json --output runs/collective/pose_gt/seed_0/pose_error_analysis
```

Extraction creates `data/manifests/collective_pose.jsonl` and pose arrays with
schema-v2 sidecars under `runs/collective/features/pose_gt/`. Source, frame,
box/actor order, annotation labels, full HRNet export metadata, original weight
SHA, archive SHA, preprocessing/config hash and source-image-content hashes are
bound into provenance. An actor's row index within its scene is the actor ID;
no tracking identity is inferred. Stale/synthetic/mismatched caches are rejected
before freeze/run. Dataset validation failures are reported, not skipped.

`run` reloads the validation-selected `best.pt`, verifies the freeze receipt and
evaluates the held-out test once. Do not invoke a separate test evaluator to choose
another checkpoint. Read its saved metrics and predictions for reporting. The run
directory cannot be overwritten, though software cannot prevent deliberate manual
re-evaluation outside this workflow. Never tune using test mistakes.

Inspect seed 0 for implementation health: finite training/validation history,
five-class supervision, cache identities, nonempty counts, correct validation
checkpoint selection and device consistency. Retain its test errors as observations,
not a reason to tune this protocol. A demonstrable bug needs a regression test,
explicit invalidation note and new versioned experiment.

Only after seed 0 is healthy, run the other already declared seeds:

```powershell
python scripts/run_collective_experiment.py --protocol configs/experiments/collective_pose_local.yaml --stage run --experiment pose_gt --seed 1
python scripts/run_collective_experiment.py --protocol configs/experiments/collective_pose_local.yaml --stage run --experiment pose_gt --seed 2
python scripts/aggregate_experiments.py --runs runs/collective/pose_gt/seed_0/metrics.json runs/collective/pose_gt/seed_1/metrics.json runs/collective/pose_gt/seed_2/metrics.json --output runs/collective/pose_gt_summary
```

The aggregate retains individual seeds, means and **sample standard deviation
(ddof=1)**. It refuses incompatible protocols, evidence kinds or scene/actor
populations. Do not mix preflight results into real metrics. A single seed has
unknown standard deviation, not zero spread.

## Saved evidence and original tooling verification (before real assets)

Each real seed directory contains resolved config/protocol, environment (Python,
torch, torchvision, CUDA availability, device, OS, Git commit, seed/config hash),
`history.jsonl`, TensorBoard logs, checkpoints, checkpoint SHA256, `metrics.json`,
`predictions.json`, errors JSON/Markdown and summary. Protocol receipts under
`runs/collective/protocols/` bind annotations, image contents, manifests, feature
arrays/sidecars and local feature export. Metrics include group/actor accuracy,
macro F1, per-class accuracy/F1/support, confusion matrices, class distributions,
scene/actor counts and abstention/invalid counts. Structural failures abort rather
than silently reducing the evaluation population.

Error tooling records source/scene IDs, true/predicted group labels, confidence,
high-confidence mistakes, class confusions and actor mistakes. Original test
predictions retain actor boxes, true/predicted action classes and probabilities.
No imagery is copied by error reporting, and no behavioral causes are invented.

Initial state was clean at `319b7bb` on `feat/collective-real-benchmark`.
All **347** tests passed before edits; the requested branch did not exist and was
created as `feat/collective-pose-real-baseline`. No benchmark implementation bug
was identified, no model/loss mathematics changed and prior synthetic results
remain valid. The only adapter restriction added rejects fine-tuning an archive
explicitly marked inference-only; existing scripted training exports still work.

At inspection, `data/raw/collective/`, the official upstream checkout and official
checkpoint were absent from repository/configured paths. Consequently **real
dataset validation, official export, ten-real-scene inspection, real preflight,
full seed-0 training, held-out metrics, seeds 1/2 and real error analysis have not
run**. No benchmark values or confusion findings are available.

New fixture tests cover local upstream imports, complete checkpoint validation,
metadata, input resolution, endpoint equality, deterministic/variable-batch eval,
inference-only rejection, missing assets, no-overwrite behavior, CLI conversion,
GT inspection/overlays, extraction sidecars and changed-checkpoint rejection.
A separate integration test runs export -> GT features -> five-step preflight ->
full fixture extraction -> freeze -> five-step training -> checkpoint reload ->
single fixture test evaluation -> errors. It uses variable actor counts 1/2 and
a tiny **non-HRNet external network**; results are explicitly **NON-BENCHMARK**.
No detector/RGB experiment is added or launched.

Python 3.13.5 / torch 2.14.1+cpu / torchvision 0.29.1+cpu on Windows are the
verified local environment. CUDA is unavailable and remains unverified. Upstream
TorchScript deprecation warnings are preserved. The official checkout/checkpoint's
compatibility with this environment still requires the actual export command.

Run software verification without external assets:

```powershell
python -m pytest --basetemp .pytest_cache/real-pose-verification-new -q
ruff check .
ruff format --check .
git diff --check
```

Verification actually run for this change:

- Before edits: **347 passed** (106.07 seconds).
- `.venv/Scripts/python.exe -m pytest --basetemp .pytest_cache/m2cr1-full-final -q`:
  **376 passed**, **155 upstream TorchScript FutureWarnings**, 157.25 seconds.
  All original 347 remain green; 29 new cases exercise export/integration contracts.
- `ruff check .`: passed. `ruff format --check .`: passed, 127 Python files.
- `git diff --check`: passed. New untracked files were also reviewed and checked
  using `git diff --no-index --check`. Ordinary Windows LF/CRLF notices are not errors.
- Export CLI help: passed. Missing-assets export exited 1 with local-asset instructions.
- Missing-data validation exited 1 and saved
  `outputs/m2cr1-missing-collective.json`, listing all 44 missing sequences and zero
  installed sources/scenes/actors. This is an availability diagnostic, not a real
  annotation-validation result. No assets were downloaded.
- The fixture export/cache/preflight/train/reload/evaluate/error workflow described
  above passed, including five optimization steps and counts 1/2. Generated outputs
  remain under ignored pytest directories and are not real benchmark evidence.

Exact authored inventory (six modified, five added):

```text
Modified:
  .gitignore
  README.md
  data/README.md
  docs/actor-backbones.md
  scripts/inspect_actor_features.py
  src/surveillance/features/actor_backbones.py
Added:
  docs/hrnet-export.md
  docs/milestone-2c-real-pose.md
  scripts/export_hrnet_features.py
  src/surveillance/features/hrnet_export.py
  tests/test_hrnet_export.py
```

The branch is `feat/collective-pose-real-baseline` at parent checkpoint `319b7bb`.
All eleven files are uncommitted; nothing was staged, committed or pushed. The
existing experiment YAML, dataset parser, model, loss and trainer are unchanged.
The only additional ignore rule is the documented local pose protocol copy.

Remaining work is to supply those three official assets and execute the commands
above. Establish the measured pose/GT baseline before any detector or RGB phase.
No commit, push, merge, remote modification or history rewrite is part of R1.

## First real execution: 2026-10-03/04

The official local HRNet source/configuration and supplied
`checkpoints/external/pose_hrnet_w32_256x192.pth` loaded with complete strict
weight validation. The source checkpoint SHA256 is
`19bc083708bb8d873211e50d85d56344c10290c6e8b564c813fdde09645c4c1c`.
The final export is `checkpoints/hrnet_w32_features_fp32_cudnn.pt`, SHA256
`d4bc248337ff4d7a0d60681552a6e73cd0cce7e46e9a5c37350e0b5601227220`.
It returns **[N,32,64,48] pre_final_layer** maps from normalized RGB 256x192
crops; flattening to 98,304 remains an implementation consequence.

The local candidate protocol is `configs/experiments/collective_pose_local.yaml`,
ID `collective_pose_visible_boxes_fp32_cudnn_v2`. Its model/loss/split/schedule
settings were preserved. Generated evidence is under
`runs/collective/fp32_cudnn_v2/`; older diagnostic exports and preflights remain
separate. The original 376-test baseline passed; the current suite has **392
passing tests**, including ten boundary-annotation and six export/execution
regressions. Ruff lint, Ruff formatting and `git diff --check` pass.

The initial real annotation audit found 442 boxes partially outside image
boundaries, including 352 selected supervised actors. Explicit
`clip_to_image` handling preserves positive visible intersections and actor
order/labels, records every correction, rejects invalid/fully outside geometry,
and leaves raw files untouched. Preparation validated all 44 sources, 2,547
scenes and 12,874 actors, with no missing frames, invalid boxes, duplicate IDs
or leakage. Stage 1 was not regenerated during the subsequent Stage 2 resumption.
The frozen split remains 29 optimization sources / three validation sources
(1,2,3) / 12 test sources. Historical paper-exact source-ID equivalence remains
unverified.

Actual CUDA execution exposed two numerical portability problems: profiling
rewrote an inference-only trace after its cold forward, and CPU tracing embedded
TF32 permission into its CUDA convolution calls. The pose adapter now keeps
inference-only archives on an unoptimized JIT graph, bound into cache provenance.
Exporter version 2 disables TF32 while explicitly preserving cuDNN's execution
flags. A temporary diagnostic export that inherited the context manager's
cuDNN-disabled default was corrected before benchmarking/full extraction.
Weights, geometry, endpoint, model mathematics and preprocessing were unchanged;
earlier synthetic results and real benchmark claims were not invalidated (no
real benchmark existed before these corrections). See [hrnet-export.md](hrnet-export.md).

The repository venv uses Python 3.13.5, torch **2.13.0+cu126**, torchvision
**0.28.0+cu126**, CUDA 12.6 and the RTX 4070 Laptop GPU. Package snapshots and the
official-index install receipt are retained under `runs/collective/`. CUDA
commands set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts.

Completed stage evidence:

| Stage | Verified result |
| --- | --- |
| Export/CUDA gate | Strict weights, finite/repeatable endpoint; original source provenance retained |
| Ten-scene inspection | 73 real training actors; shapes, ordering and coordinate correspondence pass |
| Visual inspection | GT boxes align; documented stretching/overlap remains; no systematic frame offset |
| Five-step preflight | 20 real train/validation scenes, 112 actors; finite losses/optimizer state, CUDA reload passes; **not benchmark performance** |
| GPU feasibility | 144 actors / 21 training scenes in 5.334 s; 27.0 actors/s; peak 274 MiB allocated / 406 MiB reserved; initial batch four, then scene batches 5–12 actors; no OOM |
| CPU/GPU agreement | Two fixed real actors; max absolute difference 1.10e-5, rtol=1e-3/atol=1e-4 unchanged |
| Full extraction | All 2,547 scenes / 12,874 actors; actors per scene 1–13; no failed scenes or skipped actors; every cache/provenance check passes |
| Freeze | 30,336 artifacts bound and verified; receipt `protocols/pose_gt.json` |
| Seed 0 | All 20,000 iterations; scheduled drops at 5k/10k; validation-only selection; one held-out evaluation |

The bounded throughput estimate was approximately eight minutes for extraction
of 12,874 actors, excluding subsequent full integrity/freeze/training guards and
subject to disk/scene-density differences. This is compute feasibility, not
classification performance.

Receipt freeze hash:
`9e023a1caf9323ae0f18e4a01d8ac71c067d7d3226c22fb0d85ffab1fe9d41ee`.
Resolved protocol hash:
`8ecc63d9f81fe817a645fe249894f5bd2d15a1e3565922405f8995c661737314`.
The selected `pose_gt/seed_0/best.pt` is iteration **1,100**, SHA256
`f87a60f5d3ef07f99d50335588056beaa3f0d154fde6f548a21a391111c49630`.
Selection used validation group accuracy only, retaining the earliest maximum.
The last checkpoint records iteration 20,000; it was not selected using test data.

Result name: **Collective / Pose / GT boxes / seed 0 / our frozen protocol**.

| Population | Scenes / actors | Group accuracy | Group macro F1 | Actor accuracy | Actor macro F1 |
| --- | --- | --- | --- | --- | --- |
| Validation, selected checkpoint | 85 / 427 | 50.59% | 0.21578 | 46.14% | 0.21960 |
| Held-out test | 775 / 3,420 | **68.00%** | **0.65482** | **59.01%** | **0.56979** |

No scenes were invalid or abstained. Both prediction heads cover all five
classes. Held-out group per-class recall/accuracy: crossing 62.59%, waiting
41.48%, queueing 58.06%, walking 68.81%, talking 96.15%. Actor per-class
recall/accuracy: 54.91%, 32.67%, 46.36%, 66.15%, 78.52%, in the same order.
Complete class supports/F1/confusion matrices are in the saved metrics.

Training loss decreased (first/last 100-iteration means **1.59047 / 0.001501**),
but validation remained weak: its best group accuracy is below its 70.59%
walking-majority baseline, and validation contains no waiting/queueing scenes
and only one talking scene. Held-out accuracy exceeds its own 28.13% majority
baseline; these populations must not be conflated. This is a technically valid
run with a large train/validation gap, requiring scientific review.

There are 527 correctly classified and 248 incorrect test scenes; 137 incorrect
scenes have confidence >=0.8 under the existing diagnostic policy. The largest
group confusions are crossing -> walking (55), walking -> crossing (51), and
waiting -> crossing (40). The largest actor confusions include crossing ->
walking (296) and walking -> crossing (232). These are observed prediction errors,
not established behavioral or feature-extraction causes. Error records retain
scene IDs and actor labels/predictions. No test-based tuning followed evaluation.

Important artifacts below the execution output root:

- `cuda_environment_gate.json`, `inspection/pose_gt/pose.json` and ignored debug crops/overlays.
- `preflight_verification.json`, `gpu_benchmark/pose_gt/report.json`.
- `full_pose_cache_verification.json`, `freeze_verification.json`.
- `pose_gt/seed_0/metrics.json`, `predictions.json`, `errors.json`, `errors.md`.
- `pose_gt/seed_0/checkpoint_selection_verification.json`, `execution_verification.json`.
- `pose_gt/seed_0/training_validation_curves.png`, `confusion_matrices.png`, TensorBoard logs.

The post-run check validates both saved checkpoints, all 20k loss records, all
200 validation checks, optimizer finiteness/step count, probability validity and
metrics recomputed from saved predictions. It never repeats held-out inference.
No detector/RGB experiment ran. No seeds 1/2 ran. No commit or push occurred;
the uncommitted execution corrections' source hashes and Git parent are saved in
`freeze_verification.json`. Review seed 0 before authorizing additional seeds.
Box-stretch preprocessing and unverified historical split equivalence continue
to preclude an exact Gavrilyuk et al. reproduction claim.
