# Research execution and handoff readiness

**25 October continuation:** see the [actual data-readiness report](data-readiness-20261025.md)
and [controlled matrix](training-matrix-20261025.md). The initial preparation
status below describes 23 October; it is not a claim that the newly supplied CSV
has remained unavailable. Original model/experiment receipts stay unchanged.

Read [the audit](repository-audit.md), [milestone plan](implementation-plan.md),
[data contract](data-contract.md) and [verification log](verification-log.md).
The current deliverable is preparation software, not an Indian-domain trained
model or completed research evaluation. No real Indian CSV/media/source records
have been verified. Human review remains mandatory.

## Architecture and implemented boundaries

The deployed Video→C3D→MIL→suspicious windows→Faster R-CNN→I3D actor RoIs→six-class
Actor-Transformer→human-review interface is unchanged. All new code is additive.

The preparation path is CSV→header/mapping validation→full video decode/hash→
unsplit binary staging + quarantine→curator provenance/overlap/candidate review→
source/incident/content connected groups→frozen common `Record` JSONL + receipt.

Two binary model foundations are available:

1. `FrozenBagBinaryHead`: mean of the existing 32 normalized frozen C3D segments
   → two-output linear layer. This is a weak clip-label supervised baseline.
   It has no temporal localization claim and does not consume the MIL score.
2. `BinaryActorAdapter`: copies an RGB-only precomputed ActorTransformerSystem,
   pools contextual actor embeddings and adds a new two-output group head.
   Frozen mode keeps the representation in eval mode, including dropout. Optional
   classifier tuning enables only RGB projection and encoder gradients. Original
   six-class heads are retained inside the copy, frozen and unused for binary
   supervision; the original model is not mutated. No actor labels are invented.

The runnable CLI/trainer currently supports the **C3D bag baseline**. The actor
adapter has shape/gradient/regression tests, but does not yet have an Indian actor
cache/training orchestration or separate saved actor-binary checkpoint loader.
That is the next bounded implementation stage after examining real duration,
detector coverage and annotation data. Do not report the C3D experiment as an
Actor-Transformer fine-tune or claim A/B/C comparison is complete.

## Frozen-feature development baseline

Only run after actual annotation/provenance review and a frozen split. Use new
output directories. Cached features cover train and validation only; test feature
extraction/evaluation needs a separately frozen final-evaluation protocol later.

```powershell
.venv/Scripts/python.exe scripts/train_indian_binary.py cache --split-dir data/indian-domain/split-v1 --checkpoint checkpoints/c3d_fc6_openmmlab_v1.pt --output data/indian-domain/c3d-cache-v1 --device cuda --batch-size 2
.venv/Scripts/python.exe scripts/train_indian_binary.py train --split-dir data/indian-domain/split-v1 --cache data/indian-domain/c3d-cache-v1 --output runs/indian-domain/c3d-smoke-v1 --device cuda --steps 3
```

The cache fixes the installed C3D preprocessing: RGB, mean [104,117,128], 32
L2-normalized segments. Video hashes are checked before and after extraction;
checkpoint and feature bytes are bound in `registration.json`. Training rejects
missing/mutated features, a changed frozen split, changed media, changed code or
configuration on resume. Test feature entries are refused by this trainer.

Registration includes git commit, tracked diff hash, explicit new source-file
hashes (also covers untracked new modules), software/GPU/CUDA, cache registry,
split receipt, seed, batch size, learning rate and threshold. Adam optimizes
cross-entropy, clipping gradients to 1.0. Sampling is seeded per step. Validation
selection maximizes trapezoidal PR-AUC then binary F1 at 0.5; ties keep the earlier
checkpoint. All validation curves/results are preserved in checkpoint history.
`last.pt`/`best.pt` use distinct `indian_frozen_bag_binary_v1` type and explicit
`[non_violence,violence]` classes. The strict six-class loader rejects them.

Only the run's own best/last checkpoints are atomically replaced. An existing
output directory is refused unless explicitly resuming its own last checkpoint:

```powershell
.venv/Scripts/python.exe scripts/train_indian_binary.py train --split-dir data/indian-domain/split-v1 --cache data/indian-domain/c3d-cache-v1 --output runs/indian-domain/c3d-smoke-v1 --device cuda --steps 6 --resume runs/indian-domain/c3d-smoke-v1/last.pt
```

Resume may extend the step horizon only; changing code, seed, batch size, device,
learning rate, cache or split requires a fresh experiment. Do not mutate source
files during a run. Training duration in the checkpoint covers the optimization
loop/validation/checkpoint work, excluding cache construction and setup. A failed
process can leave `.tmp` checkpoint files; preserve the directory for inspection
and use a fresh run if recovery is uncertain. Do not delete existing artifacts.

Validation is evaluated each step in this development trainer. Limit horizons
prospectively, and register final research settings before expensive runs;
frequent validation on a small source population increases selection uncertainty.
The local RTX 4070 is the only research training/evaluation device. CPU fixtures
remain useful software tests. Use the new measured profile and keep expensive
extraction behind the approved-data gate; no remote compute is required.

## Controlled research sequence

| Experiment | Variable | Required controls and evidence |
| --- | --- | --- |
| A: frozen existing cascade | None | Score the same independent Indian clip IDs; report anomaly alerts separately from six behavior probabilities. Existing anomaly or non-Normal behavior is not automatically binary violence truth. Define observable output mapping prospectively; retain all-clip routing and no-actor coverage |
| B0: C3D binary head | Supervised binary clip head on fixed C3D features | Current trainer; same frozen partitions; validation-only checkpoint/threshold selection; no localization claim |
| B1: actor binary head | New binary group head on frozen pretrained actor representation | New actor-cache adapter needed; identical window policy for paired A/B1/C; all-clip and covered-population results separately; do not train on selectively routed positives alone |
| C: classifier tuning | Unfreeze RGB projection/encoder only | Match B1 data, sampled windows, optimizer budget and seed policy; original I3D/detector frozen; optional backbone tuning is a separate experiment gated on trainable export support |

Original binary annotations cannot support per-behavior six-class F1 or actor
metrics. These must be marked unavailable unless separately annotated. Never
merge Indian denominators with UCF frame or DCSASS clip results.

Prioritize false positives after baselines. Add a single independent change per
experiment: train-only hard-negative mining; threshold calibration subject to a
prospectively declared validation recall floor; causal EMA; minimum contiguous
alert duration; hysteresis; confidence abstention. `evaluation/indian_binary.py`
provides metrics/calibration and temporal-rule primitives, not measured ablation
results. Do not silently enable these in deployment. Mining/threshold decisions
must not use final test errors. Always report violence recall changes and the
precision–recall tradeoff, not accuracy alone.

Temporal rules accept ordered nonoverlapping `[start,end)` intervals with real
seconds; gaps reset persistence and smoothing. Decision time is when a scored
interval ends and the minimum-duration rule is met, not a backdated onset.
Application assumes the interval score is available at that end; whole-video
segment extraction is offline and cannot establish online detection latency.
Measure decoding/model delay separately. `None` scores in abstention evaluation
mean unavailable, and low-confidence probabilities can abstain. Both stay in a
two-row/three-column confusion table; conditional recall and full positive
population recall are reported separately.

Only after baseline/failure analysis choose at most one or two changes to temporal
aggregation or actor aggregation, with a written hypothesis, baseline, single
variable, compute/parameter cost, settings, seeds, validation metrics and failure
examples. Combining existing components is not evidence of architectural novelty.

## Metrics and failure review

Report precision, recall, F1, ROC-AUC, trapezoidal PR-AUC **and** average precision
(different definitions), confusion matrix and FPR with exact source/clip support.
One-class ROC/PR-AUC is undefined. Clip-level operational false alarms/camera-hour
and temporal detection latency remain null. Operational measurements require
continuous representative footage, actual camera-hours, complete event annotations
and a registered event-matching rule. Balanced clips cannot establish them.

For failures record true label, prediction, source, lighting/viewpoint, occlusion,
benign interaction type, routing score, detector count/coverage, no-actor failure,
and short qualitative description. Keep identity/intent/guilt claims out of
annotations and UI. Compare group-level uncertainty where enough sources exist.
Measure inference latency with warm-up, synchronized CUDA timing, sample count,
device, resolution, duration and p50/p95; distinguish cold load, extraction, head,
full pipeline and UI time. No new real latency measurement has been made here.

## Local RTX 4070 execution

This session verified RTX 4070 availability, deployed checkpoint loading and
synthetic binary CPU/CUDA forward/backward, short training and exact resume.
This is not real C3D/Indian training validation. The latest continuation adds
bounded FP32/AMP head profiles and resource instrumentation; see the
[implementation handoff](implementation-handoff-20261025.md). Preserve the working
CUDA environment and use allowlisted `environment_metadata`. Do not dump secrets
or environment variables. Keep the existing local paths and hash-bound receipts;
no cloud, distributed training or remote GPU provisioning belongs to this scope.

## Final evaluation and professor demonstration

Before testing: choose checkpoint using validation only; freeze model bytes,
preprocessing, routing/window aggregation, thresholds and temporal rules; record
all hashes/versions and verify source/content/incident independence. Register a
separate one-time protected test execution. The present CLI deliberately has no
test-tuning/evaluation option. Final metrics, A/B/C tables, ablations, failure
examples and latency remain pending; preserve failed/negative experiments too.

For the professor, demonstrate the existing verified asset preflight and local
human-review console using its existing runbook, explaining its historical
populations and weak performance. Show this preparation workflow and its test
log as readiness evidence. Do not show synthetic software metrics as Indian
results or imply that an untrained binary head is deployed.

Municipal research-pilot readiness is **not established**. Required evidence still
includes representative permissions/provenance, independent Indian evaluation,
continuous false-alert burden, recall, detector misses, latency, reviewer workflow
validation and scoped pilot governance. This system must remain a human-review
aid and must never infer intent, guilt or culpability or initiate enforcement.
