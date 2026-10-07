# Collective / RGB / detected boxes / same frozen checkpoint

This is a deployment-oriented actor-box substitution experiment, not a new model
training run or an exact Actor-Transformers paper reproduction. RGB was explicitly
selected by the user after the completed GT Pose/RGB/fusion comparison. Detector
confidence selection uses validation localization only, not test group scores.

Branch: `feat/collective-rgb-detected-box-eval`; base commit
`cc65c42463c1c966df55ec10c7b708881ff65b61`. A proven detector-loading correction
is recorded in the new source snapshot; Actor-Transformer, I3D, preprocessing,
split and checkpoint-selection mathematics remain unchanged.

## Preserved RGB reference

The measured RGB GT seed-0 result remains group accuracy **77.55%**, group macro
F1 **0.7947**, actor accuracy **78.22%**, actor macro F1 **0.7923** on 775 scenes
and 3,420 actors. Use only
`runs/collective/rgb_gt_v1/rgb_gt/seed_0/best.pt`, selected at iteration 1,300.
Checkpoint SHA256:
`40a5fc3687c277914335916bd46dff72ccde5e4b6518c677f193178090de7fcd`.
The original RGB receipt and 30,336 artifact bytes were verified unchanged.

I3D export: `checkpoints/i3d_mixed4f_collective_rgb_v1.pt`, SHA256
`fe7fc30ca6f26430232e2e0f6bdffd8514478a071db065e452bff46f60f4e0c4`.
These are converted DeepMind ImageNet+Kinetics weights, not official PyTorch I3D
weights. Ten RGB frames (-5..+4, reference index 5), Mixed_4f, temporal mean,
90x160 resize and aligned 5x5 RoIAlign remain fixed: **20,800 values per actor**.
Neither backbone nor classifier is trained in this experiment.

## Official detector and compatibility correction

Local asset:
`checkpoints/external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`.
Size **167,502,836 bytes**, SHA256
`258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`.
Model: `torchvision.models.detection.fasterrcnn_resnet50_fpn`, weights identity
`FasterRCNN_ResNet50_FPN_Weights.COCO_V1`, 91 classes, person ID 1.
Both pretrained factory arguments stay `None`; no download is allowed.
Official COCO FrozenBatchNorm epsilon 0 is restored before strict local loading.

The original adapter's exact pre-load key check incorrectly rejected **18
legacy FPN/RPN names** used by the supplied official checkpoint. TorchVision's
native `load_state_dict(strict=True)` accepted the same file with all keys matched.
The minimal correction validates the post-migration name set while letting the
native loader perform its existing versioned migration. No learned tensor is
skipped, reshaped or replaced, and no detector layer definition changes.
Ambiguous aliases and discarded buffers remain errors.

A small real FPN/RPN fixture reproduced the failure before the correction.
Seven new regression cases verify exact tensor preservation and rejection of
missing, extra, wrong-shape, duplicate-alias, discarded-buffer and incompatible
version metadata. Full suite after the correction: **420 tests passed**, Ruff
lint/format and `git diff --check` passed.

The actual pretrained detector passed CUDA synthetic forward checks with finite
outputs and bitwise-identical repeated boxes/scores/labels. Model is in eval mode.
Torch 2.13.0+cu126 / torchvision 0.28.0+cu126 / CUDA 12.6,
RTX 4070 Laptop GPU; no dependency or model-family change.

## Training-only visual sanity

Ten real training scenes `seq04:0001` through `seq04:0091`, 73 GT actors, produced
93 detections at the original default confidence 0.5. This is a geometry sanity
check, not threshold selection. Native images are read once as RGB [0,1].
Coordinates are absolute pixel xyxy image edges; final boxes are clipped, validated
and canonically ordered left-to-right by center x, then center y/coordinates/score.

All ten overlays were reviewed, with full-resolution review at frames 41 and 91.
Boxes correspond to people and no systematic offset was observed. Occluded actors
can be missed; partial-body duplicates and distant unannotated people were visible.
No geometry, NMS, model or preprocessing setting was changed from those examples.
Raw outputs, filtered counts, scores, matches and local PNG overlays remain ignored
under `runs/collective/rgb_detected_v1_execution/inspection/`.

## Validation-only confidence selection

Only **85 validation scenes / 427 GT actors**, from sources 1, 2 and 3, enter
the existing threshold tool. No test detector output is used for selection.
Objective: maximum micro detector F1 subject to recall >= 0.5; ties prefer higher
recall then higher confidence. Sweep and all non-confidence settings are inherited.

| Confidence | Precision | Recall | F1 | Feasible |
| --- | ---: | ---: | ---: | --- |
| 0.3 | 41.996% | 89.696% | 0.5721 | yes |
| 0.4 | 47.925% | 89.227% | 0.6236 | yes |
| 0.5 | 51.007% | 88.993% | 0.6485 | yes |
| 0.6 | 55.233% | 88.993% | 0.6816 | yes |
| **0.7** | **58.733%** | **88.993%** | **0.7076** | **selected** |

The sweep was not expanded. Frozen NMS IoU=0.5, matching IoU=0.5 (inclusive),
max actors=20, minimum width/height=4 pixels, minimum area=0. Matching maximizes
eligible bipartite match cardinality, then summed IoU. Unmatched actors never
receive fabricated action labels. All retained actors influence group prediction.

## Separate settings freeze and population

[collective_rgb_detected_v1.yaml](../configs/experiments/collective_rgb_detected_v1.yaml)
registers the unchanged RGB actor configuration plus the detector selection.
Settings receipt: `runs/collective/rgb_detected_v1/protocols/detected_rgb_settings.json`.
Protocol hash:
`f28e0211ff98244ed4b235b37c08aaf24169b6470806184cb3cd2e165d829ab5`.
Settings freeze hash:
`61ba724ef826f538258ff19fc5b307951fe2d2f446dfa276c5d6d327d267264f`.
It binds 30,488 artifacts, including original dataset/GT cache identities, both
pretrained checkpoints, selected RGB classifier, confidence receipt and scientific
source/configuration. Environment/source snapshot is in `settings_execution.json`.

| Split | Scenes | GT actors | Retained detections |
| --- | ---: | ---: | ---: |
| Train | 1,687 | 9,027 | 14,410 |
| Validation | 85 | 427 | 647 |
| Test | 775 | 3,420 | 5,293 |
| Total | 2,547 | 12,874 | 20,350 |

All 2,547 records passed source/reference image fingerprints, declared filters,
selection/checkpoint hashes, canonical actor order and exact scene coverage.
There are **zero empty scenes and zero truncated scenes**. Full detection took
606.45 seconds. Versioned JSONL retains pre/post-filter counts, suppression and
truncation metadata. Original GT/pose/fusion caches are not overwritten.

## Frozen detector results

These are actor-annotation localization metrics at IoU >=0.5, not COCO AP or a
measure of how many unmatched boxes are truly non-person. Dataset annotation
selection excludes NA actors; an unmatched detection can be a real unannotated
person, a duplicate, or a poorly aligned box.

| Metric | Validation | Test |
| --- | ---: | ---: |
| Precision | 58.73% | 55.92% |
| Recall / matched coverage | 88.99% | 86.55% |
| F1 | 0.7076 | 0.6794 |
| Mean matched IoU | 0.6814 | 0.6551 |
| Matched actors | 380 | 2,960 |
| Missed GT actors | 47 | 460 |
| Unmatched detections | 267 | 2,333 |
| Mean detections / scene | 7.61 | 6.83 |
| Empty scenes | 0 | 0 |

No setting changes follow test detector metrics. Known validation limitations
remain: walking dominated, with no waiting/queueing group scenes.

## Cached extraction and same-checkpoint evaluation policy

Detected RGB extraction completed on **2,547 scenes / 20,350 actor instances**,
using independent schema-v2 arrays and sidecars under
`runs/collective/rgb_detected_v1/features/detections/rgb/`. Every `[N,20800]` array
passed numeric, scene, source, ordered detection, image-content and I3D identity
checks, with zero errors or warnings. Original GT caches are preserved.

The feature receipt binds **35,588 artifacts**. Feature freeze hash:
`06ca8b5d55533125821cbf4b6283c41b0894f075cbb0f0bb3af8fbb221e526f0`.
Feature manifest SHA256:
`f3f3130b290a471a3480d29c37609ce0a3aa0312560cb7a3f5d2a79417d1260d`.
Classifier evaluation completed after successful verification of this receipt.

Existing package functions and CLIs are orchestrated directly because the generic
full experiment runner requires GT training and detection evaluation to share a
protocol hash/run root. We do **not** relabel or retrain the older GT baseline to
satisfy that assumption. Saved original GT predictions are reused for comparison;
the detected classifier receives a single test pass with the same `best.pt`.
No fake actor is inserted for empty scenes. Actor metrics use matched detections
only; paired GT metrics on that same matched subset distinguish selection effects.

Original command boundaries (not instructions to overwrite existing artifacts):

```powershell
.\.venv\Scripts\python.exe scripts/tune_collective_detector.py --manifest data/manifests/collective.jsonl --collect-candidates --checkpoint checkpoints/external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth --candidate-output data/manifests/collective_rgb_detected_v1_validation_candidates.jsonl --output runs/collective/rgb_detected_v1/protocols/detector_selection.json --confidences 0.3 0.4 0.5 0.6 0.7 --min-recall 0.5 --device cuda
.\.venv\Scripts\python.exe scripts/extract_i3d_features.py --manifest data/manifests/collective.jsonl --detections data/manifests/collective_rgb_detected_v1_detections.jsonl --box-source detections --checkpoint checkpoints/i3d_mixed4f_collective_rgb_v1.pt --image-size 480 720 --device cuda --feature-dir runs/collective/rgb_detected_v1/features --output-manifest data/manifests/collective_rgb_detected_v1_rgb_features.jsonl
```

Detection uses existing `preparation.detect_people(protocol)`; settings and cache
seals use `freeze_receipt` / `verify_receipt`. Classifier evaluation uses
`DetectedGroupActivityPipeline.predict_manifest`, comparison uses saved GT outputs
with `compare_predictions`, and errors use `analyze_predictions`. The ignored
execution ledger/scripts/logs are under `runs/collective/rgb_detected_v1_execution/`.
Do not call the new protocol's GT training stage. No fine-tuning, tracking,
surveillance adaptation, cascade, commit or push is authorized in this execution.

## Completed same-checkpoint held-out result

One detected-box test pass completed on **775 scenes / 5,293 detections**.
The same RGB checkpoint is used, without training or checkpoint reselection.
All original GT predictions are reused from their immutable saved artifact;
no new GT test forward is performed. All five group classes are predicted.
There are **zero invalid scenes, zero abstentions and zero empty scenes**.
Only **2,960 matched actors** enter individual-action metrics; all 5,293 detections
contribute to group reasoning. The 2,333 unmatched actors retain null labels.

| Metric | GT boxes | Detected boxes | Detected minus GT |
| --- | ---: | ---: | ---: |
| Group accuracy | 77.55% | **74.32%** | -3.23 pp |
| Group macro F1 | 0.7947 | **0.7383** | -0.0564 |
| Actor accuracy | 78.22% | **75.64%** | -2.57 pp |
| Actor macro F1 | 0.7923 | **0.7756** | -0.0167 |

The raw actor rows above compare different populations (all 3,420 GT actors
versus 2,960 matched actors); those deltas are descriptive, not isolated actor
degradation. On the **identical matched subset**, GT accuracy/F1 are
**76.55% / 0.7943**, versus detected **75.64% / 0.7756**: paired delta
**-0.91 points / -0.0187 F1**. Misses remain separately visible in 86.55% coverage.
All group scenes are supported, so group populations are identical.

| Group class | Support | GT recall | Detected recall | Delta |
| --- | ---: | ---: | ---: | ---: |
| crossing | 147 | 71.43% | 61.90% | -9.52 pp |
| waiting | 135 | 65.93% | 37.78% | -28.15 pp |
| queueing | 93 | 100.00% | 100.00% | 0.00 pp |
| walking | 218 | 60.55% | 72.94% | +12.39 pp |
| talking | 182 | 100.00% | 100.00% | 0.00 pp |

| Actor class | All-GT recall | Paired-GT recall | Detected recall | Matched / GT actors | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| crossing | 67.49% | 68.45% | 62.35% | 656 / 692 | 94.80% |
| waiting | 63.35% | 64.95% | 54.64% | 485 / 502 | 96.61% |
| queueing | 92.93% | 92.49% | 95.98% | 373 / 481 | 77.55% |
| walking | 71.65% | 71.51% | 78.66% | 923 / 963 | 95.85% |
| talking | 96.29% | 95.03% | 91.97% | 523 / 782 | 66.88% |

Queueing/talking group recalls remain perfect, but coverage of their actors is
lower. Queueing group precision declines because walking -> queueing increases
from 3 to 17; perfect recall does not imply perfect class performance.

Detected group confusion matrix (rows true, columns predicted; order
crossing/waiting/queueing/walking/talking):

```text
91   1   0  55   0
67  51   1  16   0
 0   0  93   0   0
42   0  17 159   0
 0   0   0   0 182
```

## Error attribution and interpretation

| Paired outcome | Scenes |
| --- | ---: |
| GT correct / detected wrong | 73 |
| Both correct | 528 |
| Both wrong | 126 |
| GT wrong / detected correct | 48 |
| No actors detected | 0 |

There are 73 new errors and 48 corrections: net 25 fewer correct groups.
The new errors contain 39 waiting, 15 crossing and 19 walking scenes. Of those
73 scenes, **70 contain unmatched detections and 11 contain missed GT actors**;
categories can overlap. This association does not prove which actor caused an
error. `gt_correct_detected_wrong.json` retains every scene's true/GT/detected
classes, actor counts, coverage, IoUs, misses and extra indices. No relevance or
behavioral cause is invented from attention or matching alone.

Waiting -> crossing rises **41 -> 67**; waiting -> walking **5 -> 16**.
Crossing -> walking rises **41 -> 55**, while walking -> crossing falls
**67 -> 42**. Combined crossing/walking swaps fall **108 -> 97**. Better walking
recognition offsets some crossing/waiting losses; aggregate accuracy hides these
class-specific vulnerabilities.

The aggregate group gap is modest in absolute accuracy (3.23 points), and paired
actor degradation is modest (0.91 points), but **waiting degradation is severe**.
This does not establish acceptable CCTV/violence performance: the dataset is
Collective, labels are the five benchmark activities, actor annotation coverage
is partial, and this is one frozen seed/checkpoint with the same imperfect
validation split. RGB was user-selected from earlier GT results; this is an
exploratory box-source sensitivity experiment, not blind modality selection.

Detected-box fine-tuning is worth a separately registered study for class-level
robustness, but is not automatically justified as necessary for every class.
First recommend a fixed-checkpoint diagnostic ablation of unmatched-actor
influence versus localization/coverage, then a separately registered detected-box
fine-tuning comparison if needed. No such follow-up is started here.

Completed artifacts under `runs/collective/rgb_detected_v1/detected_rgb/seed_0/`:
`metrics.json`, `comparison.json`, `detected_predictions.json`, `errors.json` /
`.md`, `gt_correct_detected_wrong.json`, `result_analysis.json` and
`confusion_matrices.png`. Detector split reports, cache validation, settings/cache
receipts and provenance live above that run directory. All generated artifacts
remain ignored; final tests, preservation checks and optional validation-only
attention observations are recorded after completion.

## Descriptive attention and final verification

Optional attention comparison used **validation only** after the frozen test
evaluation, with no additional test forward or threshold/model selection. Four
validation scenes were GT-correct/detected-wrong; three examples were saved with
GT actor axes and detected axes mapped to matched GT indices or `extra`.
`seq03:0021` changes walking -> crossing with 6 GT / 7 detected actors and mean
attention mass 0.2962 to unmatched actors. `seq03:0141` changes crossing -> walking
with 5 GT / 8 detections and mass 0.4327. `seq03:0131` also changes crossing ->
walking despite all five actors matching and no extra actor. Thus extra-actor
attention is not a complete explanation. Matrices and populations differ; these
are descriptive observations, not causal evidence. See `attention_validation.json`
and local heatmaps under the run directory.

Fresh final verification: **420 tests passed / 325 warnings / 132.05 seconds**;
Ruff lint passed, Ruff formatting passed (132 files), `git diff --check` passed.
Preservation checks confirm 142 frozen scientific source files, all 39 recorded
Pose/RGB seed-0 artifact files, original RGB classifier, detector checkpoint,
registered protocol and settings receipt remain unchanged after freezing.
All original GT results are preserved. No Actor-Transformer training occurred.

Source-tree changes are `README.md`, `docs/person-detection.md`, the detector
compatibility correction and seven regression cases in
`tests/test_detector_backend.py`, plus the new registered YAML and this execution
report. All images, detections, features, checkpoints, numeric reports and local
execution helpers remain ignored. Branch/HEAD are
`feat/collective-rgb-detected-box-eval` / `cc65c42`; **no commit or push** occurred.
