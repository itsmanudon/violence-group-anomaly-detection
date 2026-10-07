# Collective / predefined Pose+RGB late fusion / GT / seed 0

This tests the existing Milestone 2A fusion definition under **our frozen
Collective protocol**, not an exact reproduction of Gavrilyuk et al. No new
architecture, preprocessing, split or test-driven hyperparameter tuning is used.
The frozen seed-0 experiment completed on 2026-10-05: **group accuracy 71.48%,
group macro F1 0.6919, actor accuracy 62.11%, actor macro F1 0.6079**. The
predefined 2:1 probability-level late-fusion configuration did not improve upon
RGB under our frozen protocol. It improved walking group recall and reduced
crossing/walking group swaps. No settings were changed after test evaluation.

## Registered implementation and inputs

Source: `7a4cc29ea887c99a55cf2e1a8008aa6ae9aabca8`, parent
`527a88f4aea3e3fbc0f4c1733afa04f26ac7706c`, branch
`feat/collective-pose-rgb-late-fusion`. The committed source contains the successful
RGB export boundary. The new experiment configuration is separately bound by
its freeze receipt; no additional commit or push is authorized.

[collective_pose_rgb_late_gt_v1.yaml](../configs/experiments/collective_pose_rgb_late_gt_v1.yaml)
declares seed 0 only, GT boxes and cached features. Model/loss/trainer mathematics
are unchanged. Each modality has an independent linear projection to 128,
2D position encoding, one transformer encoder layer/head, FFN 256, dropout 0.1,
actor classifier and group classifier after masked max pooling. There is no raw
feature concatenation, cross-attention, gating or learned modality weighting.

The existing actor **and** group prediction rule is:

```text
p_fused = (2 * softmax(pose_logits) + softmax(rgb_logits)) / 3
```

Stable log probabilities feed the existing joint group-plus-actor cross entropy
with weights 1/1. The existing design trains both branches jointly from seed-0
initialization; it does not load the separately trained Pose/RGB classifier
checkpoints as an ensemble. HRNet and I3D are fixed cached backbones, not trainable
modules in this run.

Predefined benchmark training: batch **16**, Adam 1e-4, betas .9/.999, epsilon
1e-8, gradient clip 1, 20,000 iterations, LR times .1 at 5,000/10,000,
validation/checkpoint every 100. Selection uses **validation group accuracy**,
earliest maximum on ties. RGB-only used its existing batch 8; this difference
is disclosed rather than changed after viewing its test result.

- HRNet W32 / COCO pre-final features: 98,304 values per actor, stretched RGB
  crops 256x192, no extra feature normalization. Original checkpoint SHA256
  `19bc083708bb8d873211e50d85d56344c10290c6e8b564c813fdde09645c4c1c`;
  archive SHA256 `d4bc248337ff4d7a0d60681552a6e73cd0cce7e46e9a5c37350e0b5601227220`.
- Converted DeepMind ImageNet+Kinetics I3D RGB weights, not official PyTorch
  weights; revision `05783d11f9632b25fe3d50395a9c9bb51f848d6d`.
  Original checkpoint SHA256
  `2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538`;
  archive SHA256 `fe7fc30ca6f26430232e2e0f6bdffd8514478a071db065e452bff46f60f4e0c4`.
  Ten RGB frames at 480x720, Mixed_4f, temporal mean, resize 90x160,
  RoIAlign 5x5, 20,800 values per actor.
- Environment: project `.venv/Scripts/python.exe`, Python 3.13.5,
  torch 2.13.0+cu126 / torchvision 0.28.0+cu126 / CUDA 12.6,
  RTX 4070 Laptop GPU; existing deterministic/TF32 policies preserved.

## Population and alignment

All 44 sequences retain the existing source split. Validation sequences 1,2,3
come only from the 32 training-side sources, leaving 29 optimization sources.
Test sources remain 5,6,7,8,9,10,11,15,16,25,28,29.

| Split | Scenes | Actors |
| --- | ---: | ---: |
| Train | 1,687 | 9,027 |
| Validation | 85 | 427 |
| Test | 775 | 3,420 |
| Total | 2,547 | 12,874 |

Join keys are dataset/source sequence/scene/reference frame, not storage order.
Every sample's split, complete frame window, labels, ordered boxes and actor
counts matched both modalities and the annotation authority exactly. All missing,
extra, box, label, order and source-frame counters are **zero**. Relative/absolute
path aliases are compared after resolution. The earlier diagnostic that treated
equivalent `..` spellings as mismatches is retained, along with its correction.

All **5,094** reused numeric arrays and schema-v2 sidecars passed source-image,
array/checkpoint fingerprints, dimensions, finiteness and extraction-configuration
checks. `data/manifests/collective_pose_rgb_late_gt_v1.jsonl` links the original
arrays; no cache was overwritten or feature re-extracted.

## Inspection and preflight

Ten real training scenes (`seq04:0001` through `seq04:0091`) / **73 actors** passed
both 128-d projections, actor/group probability equality to the fixed mixture,
deterministic eval, extra-NaN-padding invariance and valid modality attention
`[10,1,1,8,8]`. Gradients reached both projections, encoders and heads. No raw
backbone was instantiated. The inspection's training diagnostic did not request
attention; its separately verified eval attention records establish that contract.

**REAL LATE-FUSION PREFLIGHT - NOT BENCHMARK:** ten cached train scenes / ten
cached validation scenes (73/39 actors), five steps, LR 1e-4. Total losses:
3.43385744, 1.87144279, 1.14182067, 0.83030373, 0.61992621.
Group and actor components, optimizer state and model parameters were finite;
validation executed and `best.pt` reloaded on CUDA. No test prediction occurred.
The dry-run orchestration uses a bounded cached subset and the unchanged full
annotation authority; native raw-feature preflight would re-extract features.

Artifacts and the execution ledger live under ignored
`runs/collective/late_gt_v1_execution/`. Baseline preservation identities were
recorded before any fusion configuration was created.

## Commands and result policy

From the repository root, with the existing fixed caches and assets:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe scripts/run_collective_experiment.py --protocol configs/experiments/collective_pose_rgb_late_gt_v1.yaml --stage freeze --experiment late_gt
.\.venv\Scripts\python.exe scripts/run_collective_experiment.py --protocol configs/experiments/collective_pose_rgb_late_gt_v1.yaml --stage run --experiment late_gt --seed 0
```

These are original-run commands, not instructions to overwrite an existing run.
The runner refuses existing run output and validates immutable receipts around
held-out evaluation. It loads the validation-selected checkpoint, evaluates test
once, and saves predictions/metrics. Analysis reads saved predictions; no further
held-out model forwards or retraining are authorized.

## Freeze and completed execution

Experiment identity: `collective_pose_rgb_late_gt_v1`, experiment `late_gt`, seed
0. The new receipt is
`runs/collective/late_gt_v1/protocols/late_gt.json`; the supplemental source,
environment and backbone identity record is `late_gt_execution.json` alongside
it. Pose and RGB receipts remain unchanged.

| Identity | SHA256 |
| --- | --- |
| Protocol | `5872bc7a05158fa85765097f3593a03aaba39e109c53de67a2de0eea8e3827c2` |
| Freeze | `3616b556ea0777cd3eecded325dd0a9a9879c52a3fc8ad94bd51b6ac958cd6d7` |
| Receipt file | `5df05e1e98c324215ae3928bd398283ed2f0240f69cbd12a622eadb5bcad32ee` |
| Aligned manifest | `32905bd7737855980919600c56137e367cc3a7e320e16a2d2f5be8e1aa510f61` |
| Scientific source/configuration set | `f804ca7029c5f123c61bf84b11a94026f37c74fa828c080eb9e5bb38a05b6edf` |
| Selected checkpoint | `0d4ae638244d4b9a47bc43ea7bbcad0346908f5bde39cb939c0bcb834fa2a550` |

The receipt binds **35,431 artifacts**, including both caches and sidecars,
source frames/annotations, aligned manifest and exported backbones. Native
receipt validation succeeded before and around held-out inference. Final
preservation checks passed for 142 scientific source/configuration files, 39
preserved baseline files, original checkpoints, receipt, manifest and selected
checkpoint/history. Neither baseline's metrics nor artifacts changed.

The native runner completed **20,000 optimizer steps and 200 validation checks**
with finite losses, parameters and optimizer state. LR milestones occurred at
5,000 and 10,000 as registered. Mean total training loss over the first/last
100 steps fell from **1.3358 to 0.0114**. Training accuracy approached one, while
validation remained weak: group accuracy was 27.06% at iteration 100, peaked
at 45.88%, and ended at 31.76%. This divergence is reported without early stopping
or a protocol change; even the peak is below the validation majority baseline
of 70.59%.

Validation selected **iteration 2,900**, the earliest maximum of group accuracy.
The checkpoint identity and selection history were sealed in
`checkpoint_selection_pretest.json` before test metrics were saved. At selection:

| Validation metric | Value |
| --- | ---: |
| Group accuracy | 45.88% |
| Group macro F1 | 0.1877 |
| Actor accuracy | 48.71% |
| Actor macro F1 | 0.2068 |

The existing runner then evaluated the held-out population **once**. All later
metric/confusion analysis reads saved predictions without additional test model
forwards. Test contains **775 scenes / 3,420 actors**, with **zero invalid scenes
and zero abstentions**. Both heads predict all five classes. Group prediction
counts are `[176,61,112,241,185]`; actor counts are `[743,221,535,1294,627]`.

## Measured three-way comparison

All rows compare seed 0 on exactly the same ordered validation/test populations.
Accuracy deltas use percentage points; F1 deltas use the 0-1 scale.

| Metric | Pose | RGB | Late fusion | Fusion minus RGB | Fusion minus Pose |
| --- | ---: | ---: | ---: | ---: | ---: |
| Group accuracy | 68.00% | 77.55% | 71.48% | -6.06 pp | +3.48 pp |
| Group macro F1 | 0.6548 | 0.7947 | 0.6919 | -0.1028 | +0.0371 |
| Actor accuracy | 59.01% | 78.22% | 62.11% | -16.11 pp | +3.10 pp |
| Actor macro F1 | 0.5698 | 0.7923 | 0.6079 | -0.1844 | +0.0381 |

Group recall:

| Class | Test support | Pose | RGB | Late fusion |
| --- | ---: | ---: | ---: | ---: |
| crossing | 147 | 62.59% | 71.43% | 65.99% |
| waiting | 135 | 41.48% | 65.93% | 40.00% |
| queueing | 93 | 58.06% | 100.00% | 72.04% |
| walking | 218 | 68.81% | 60.55% | 74.31% |
| talking | 182 | 96.15% | 100.00% | 95.60% |

Actor recall:

| Class | Test support | Pose | RGB | Late fusion |
| --- | ---: | ---: | ---: | ---: |
| crossing | 692 | 54.91% | 67.49% | 57.95% |
| waiting | 502 | 32.67% | 63.35% | 32.67% |
| queueing | 481 | 46.36% | 92.93% | 63.20% |
| walking | 963 | 66.15% | 71.65% | 68.74% |
| talking | 782 | 78.52% | 96.29% | 75.83% |

Fusion group confusion matrix, rows true and columns predicted; order is
crossing/waiting/queueing/walking/talking:

```text
[[97,  0,  3,  47,   0],
 [33, 54, 31,  17,   0],
 [ 0,  0, 67,  15,  11],
 [46,  7,  3, 162,   0],
 [ 0,  0,  8,   0, 174]]
```

| Group confusion | Pose | RGB | Late fusion |
| --- | ---: | ---: | ---: |
| crossing -> walking | 55 | 41 | 47 |
| walking -> crossing | 51 | 67 | 46 |
| Combined crossing/walking swaps | 106 | 108 | 93 |
| waiting -> crossing | 40 | 41 | 33 |

Walking group recall recovered beyond both baselines (+13.76 points versus RGB,
+5.50 versus Pose). RGB's perfect queueing/talking group recognition was **not**
retained. Although waiting -> crossing dropped, total waiting recall worsened:
waiting -> queueing/walking became 31/17. Actor crossing/walking swaps are
506 (Pose 528, RGB 432), so the group improvement does not extend to actor swaps.

There are **554 correct / 221 incorrect group scenes**, including 50 mistakes
at the existing confidence threshold 0.8, and **1,296 actor mistakes**. Mean
confidence is 0.8639 for correct and 0.6805 for incorrect group predictions;
confidence is not a guarantee of correctness. The test majority baseline is
28.13%, but exceeding it does not establish deployment robustness.

## Descriptive attention and saved evidence

Post-run attention used **validation only**, with no extra held-out forward or
selection changes. Among 85 validation scenes: 25 Pose-correct/RGB-wrong,
10 RGB-correct/Pose-wrong, 24 Fusion-correct with a baseline wrong, and 12
Fusion-wrong/RGB-correct. The report saves two examples per category.
`seq01:0001` and `seq01:0041` are walking scenes corrected by fusion relative to
RGB; `seq01:0071` and `seq01:0081` are crossing scenes that RGB gets right but
fusion predicts waiting/walking. Modality attention is finite and row-stochastic;
the trained fusion branches' matrices differ from standalone matrices (mean
absolute difference 0.103-0.240 pose, 0.025-0.051 RGB in these four examples).
There is no cross-modal attention. These observations are descriptive, not a
causal explanation of success or failure.

The ignored run directory `runs/collective/late_gt_v1/late_gt/seed_0/` contains:

- `metrics.json`, resolved configuration, environment metadata, best/last
  checkpoints and TensorBoard logs;
- `history.jsonl`, `checkpoint_selection_pretest.json`,
  `execution_verification.json`, saved `predictions.json` and `errors.json`;
- `three_way_comparison.json` / `.md` and `actor_errors.json`;
- [training/validation curves](../runs/collective/late_gt_v1/late_gt/seed_0/training_validation_curves.png)
  and [three-way confusion matrices](../runs/collective/late_gt_v1/late_gt/seed_0/three_way_confusion_matrices.png);
- `attention_validation.json` and visualization-ready validation attention PNGs.

## Limitations and next scientific decision

The unchanged validation group distribution is **[24,0,0,60,1]** in class order
crossing/waiting/queueing/walking/talking. It is walking dominated and lacks waiting
and queueing groups. No validation repair is allowed in this comparable v1 run.
Historical paper-exact split equivalence and preprocessing equivalence remain
unproven. Fixed pre-head HRNet cropping, converted I3D weights, model capacity,
joint training and batch differences limit attribution to temporal information.
One fusion seed does not establish variance or significance. Attention is
descriptive; optional post-run attention uses validation scenes, preserving the
single held-out evaluation.

The predefined 2:1 probability-level late-fusion configuration did not improve
upon RGB under our frozen protocol. This does **not** show that Pose and RGB can
never benefit from fusion. Any different weighting/fusion design requires a
separately registered experiment, not a retest of this run. **Recommend RGB as
the current GT-box reference** for a subsequent authorized detected-box study,
given its stronger aggregate group and actor metrics. Preserve this fusion
result as evidence of class-specific trade-offs.

Initial verification passed **413 tests** (explicit pytest process exit 0).
Fresh final verification passed **413 tests / 325 warnings in 116.57 seconds**,
Ruff lint passed, Ruff formatting passed (132 files), and `git diff --check`
passed. The warnings are retained in the ignored pytest log; the only Git
diagnostic is its existing LF-to-CRLF normalization notice for Markdown. No model
or training source was changed, and no new architecture tests were needed.

Source-tree changes: modified `README.md`; added the dedicated experiment YAML
and this execution report. Data, feature links/caches, checkpoints, run reports
and figures remain ignored. No commit/push, additional seeds or subsequent
experiments were performed. Final branch/HEAD remain
`feat/collective-pose-rgb-late-fusion` / `7a4cc29`.
