# Person detection and detected-box robustness

The Actor-Transformer paper baseline uses annotated actors. Automatic detection
is a deployment-oriented adaptation, not a reproduction of that evaluation
setting and not a contribution attributed to either project paper. Milestone 2B
measures the gap using the same trained checkpoint and unchanged transformer.
No identity inference, tracking, violence classifier, cascade or UI is added.

## Coordinates and filtering

`PersonDetector.detect(image)` accepts finite float RGB `[3,H,W]` in `[0,1]`.
`DetectionResult` contains CPU tensors `boxes [N,4]`, `scores [N]`, `class_ids [N]`,
`image_size=(height,width)` and JSON-compatible metadata. Coordinates are absolute
xyxy **image edges**; x is in `[0,width]`, y in `[0,height]`. A person box is not
an identity. Empty tensors have shapes `[0,4]`, `[0]`, `[0]`.

Filtering is separate from the backend and deterministic:

1. Validate aligned tensor shapes, finite probability scores in `[0,1]` and
   nonnegative integer class IDs; malformed score arrays fail clearly.
2. Keep only the configured person class, then confidence **>= threshold**.
3. Drop nonfinite, reversed and zero-area geometry; clamp remaining boxes to image
   boundaries. Drop boxes fully outside or below the configured width/height/area.
4. Greedy NMS in descending confidence, with coordinates breaking score ties.
   Suppress only IoU **> NMS threshold**. CPU float64 comparisons avoid GPU tie
   ordering differences. Detector-side NMS and matching thresholds are distinct.
5. Keep the highest-confidence `max_actors` survivors. Record truncation counts.
6. Sort retained actors by center x, center y, xyxy coordinates, descending score,
   then class ID. This exact ordering is also the feature-row/attention-axis order.

`configs/person_detector.yaml` exposes these implementation defaults: person ID 1,
confidence 0.5, NMS IoU 0.5, max actors 20, minimum width/height 4 pixels, minimum
area 0. These are not Gavrilyuk paper constants. Metadata includes per-stage
rejection counts, before/after counts, truncation, and effective filter settings.
Counts refer to candidates supplied by the detector, not every internal proposal.

Normalized actor inputs are produced at one explicit boundary:
`[x1/W,y1/H,x2/W,y2/H]`. Existing HRNet crops and I3D RoIAlign then multiply by
their own image/feature grid dimensions. The native reference image dimensions
are checked against detection metadata during raw extraction. Anisotropic scaling
is deliberate; no hidden xywh/yxyx conversion is performed.

## Local TorchVision backend

`TorchvisionPersonDetector` wraps COCO **Faster R-CNN ResNet-50 FPN v1**, 91 classes
including background, person ID 1. The original
[TorchVision interface](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
accepts RGB tensors and returns pixel boxes, scores and labels. The constructor
passes **both `weights=None` and `weights_backbone=None`**, preventing a hidden
ImageNet backbone download. It restores FrozenBatchNorm2d with epsilon 0 for the
official COCO v1 architecture before strictly loading the local state dictionary.

Supply a trusted, bare PyTorch tensor `state_dict` saved from the compatible
pretrained v1 model (for example an already locally acquired official
`fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`). There is no downloader. v2, MobileNet,
custom class heads, Lightning wrappers, or `module.`-prefixed dictionaries require
an explicit, independently verified conversion; they are not silently accepted.
Loading uses `weights_only=True`, exact keys and strict parameter shapes.

In a trusted environment where the correct pretrained model is already loaded:

```python
torch.save(model.state_dict(), "checkpoints/fasterrcnn_resnet50_fpn_coco_v1.pt")
```

Then set `detector.checkpoint: ../checkpoints/fasterrcnn_resnet50_fpn_coco_v1.pt`
in `configs/person_detector.yaml`. Paths resolve relative to the YAML directory.
The adapter records checkpoint SHA256, architecture, torch/torchvision versions
and filtering settings. Compatible shapes do not prove pretrained provenance;
the user must verify the weight source and license.

The ROI detector's score cutoff is set to 0 and NMS to 1 so the exposed person
filters govern final selection. A documented 1,000-result backend candidate cap
and the model's RPN proposal limits still apply. Counts before filtering start
after those backend limits. These are not raw proposal recall measurements.
`allow_untrained=True` permits structural API tests only; the CLI provides no
random-weight inference option. Missing checkpoints fail before model allocation.

## Offline detection and feature artifacts

```powershell
python scripts/detect_people.py --manifest data/manifests/collective.jsonl --config configs/person_detector.yaml --output data/manifests/collective_detections.jsonl
# Validate/copy an existing artifact without loading detector weights:
python scripts/detect_people.py --manifest data/manifests/collective.jsonl --precomputed data/manifests/collective_detections.jsonl --output data/manifests/validated_detections.jsonl
```

Detection uses native `frame_paths[5]`; frames -5..+4 remain the temporal clip.
Center-frame boxes supply both static crops and dynamic RoIs. No temporal tracking
is required by this architecture. Precomputed copying preserves existing filtering;
it does not reapply current YAML thresholds to cached output.

The [versioned JSONL format](../data/README.md#person-detections-and-feature-caches-milestone-2b)
preserves every scene, including empty detections. Parsers reject missing/unknown
schema keys, duplicate clip/source-center IDs, wrong coordinate declarations and
invalid geometry. Uncached actors are canonicalized deterministically; an unsorted
record with feature paths is rejected to prevent changing feature associations.
Persisted records contain only the declared person class (default 1).

Both existing feature CLIs accept `--box-source detections --detections PATH`.
The `--manifest` remains the original GT scene manifest; the output is a detection
JSONL with feature references. Running RGB extraction on the pose output preserves
pose references, labels in the original manifest, detection order and empty scenes.
Artifacts live beneath `<feature-dir>/detections/<modality>/`.

Each cached modality carries a hash of ordered boxes/scores/classes/image size,
source-reference identity, feature-file SHA256, extraction image size, backbone
metadata and checkpoint SHA256. Cache names include the source/detection/checkpoint
identity and image size. Loading verifies box/source/file hashes and tensor shape;
it cannot reuse GT features implicitly. Native source image *contents* are not
hashed, so preserve image files, versions and export provenance. Manually imported
arrays must supply the same box/source/file hashes; extraction-size metadata, when
present, must match the configured size. A missing size for an imported artifact
means its preprocessing provenance remains the importer's responsibility.

## Matching and label policy

IoU matrices use continuous xyxy edges (no `+1` pixel area). GT boxes are converted
to native pixels before matching. An eligible edge has IoU **>= threshold**, with
threshold in `(0,1]`. Assignment maximizes eligible match cardinality first, then
total IoU using SciPy's
[linear assignment solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
This avoids losing eligible matches by assigning first and thresholding later.
Fixed canonical input ordering gives deterministic tie behavior.

Each detection and GT actor is used at most once. A matched detector inherits that
GT actor's label; unmatched actors use internal ignore value `-100`, serialized
as `label: null` in inference. Missed GT actors are explicitly listed. All retained
detections participate in group attention/pooling, including unmatched actors.
The benchmark annotation subset excludes NA actors: an unmatched detection may
be a real unannotated person, so report **unmatched detections**, not presumed
non-person errors. Matching transfers benchmark labels, not identity.

```powershell
python scripts/match_actor_boxes.py --manifest data/manifests/collective.jsonl --detections data/manifests/collective_detections.jsonl --split test --iou-threshold 0.5 --output outputs/actor_matches.json
python scripts/evaluate_detector.py --manifest data/manifests/collective.jsonl --detections data/manifests/collective_detections.jsonl --split test --output outputs/detector_metrics.json
```

Both CLIs read `matching.iou_threshold` from `--config` (default detector YAML),
unless overridden by `--iou-threshold`. Precision, recall and F1 aggregate counts
across scenes, not per-frame percentages. Mean IoU uses matched pairs. Undefined
ratios are null; no detections with nonempty GT means recall/F1 0 and precision
null. Counts and mean actors/frame include empty scenes. This is not COCO AP.

## Inference and downstream comparison

`DetectedGroupActivityPipeline` wraps `GroupActivityPipeline`; it never forks the
transformer. `predict_detections(result, pose_features=..., rgb_features=...)`
accepts `[N,D]` features without any labels. `predict_clip(frames, detector,
pose_extractor=..., rgb_extractor=...)` detects the center frame and extracts
required features from `[10,3,H,W]` RGB frames. Raw checkpoints use their saved
extractors; precomputed checkpoints require supplied local feature adapters.
`predict_manifest` joins cached detections to the GT manifest for evaluation.

Inference returns `status: ok` or `status: no_actors_detected`. Empty cases have
null group predictions/probabilities, an empty actor list, and empty attention.
They never enter self-attention or fabricate an actor. For nonempty scenes, each
actor includes index, normalized box, absolute `box_pixels`, `detection_score`,
detector class and predictions. Optional attention axes use these same indices.
Ground-truth matching metadata is attached after prediction and never feeds the
model. A detector label is not an individual action label.

`evaluate_box_robustness` uses one checkpoint with separate GT/detected features.
Its machine-readable report includes:

- `ground_truth_boxes`: all-scene group metrics and all annotated actor metrics.
- `detected_boxes`: supported-scene group metrics, all-scene accuracy with
  abstentions incorrect, matched-only actor metrics, detection and scene coverage.
- `paired_ground_truth`: GT metrics restricted to the same supported scenes and
  matched actor subset, making conditional deltas interpretable.
- `delta`: detected minus GT on the explicitly named populations; null if undefined.
- Both prediction lists, checkpoint, split, IoU threshold and metric policy.

Tune thresholds on validation sources, freeze the protocol, then evaluate test
sources. Do not select detector thresholds using test accuracy. Detection-only
evaluation and downstream classification always identify the altered box setting.
The existing GT trainer/evaluator remains unchanged; detected-box fine-tuning is
not introduced by this robustness milestone. Cached detection/feature artifacts
can be reused without running the detector each epoch in a future adapted trainer.

## Verification and limitations

Tests use synthetic images, feature archives, injected detectors and structural
TorchVision models; no downloads or CUDA are required. `scripts/smoke_detection.py`
explicitly uses crop-mean fixture features, never claiming those are HRNet/I3D.
Actual HRNet/I3D crop/coordinate contracts are separately covered with synthetic
local exports. The original 123 tests remain required. Real detector/Collective
accuracy is unmeasured until the user supplies the data and vetted checkpoints.

Occlusion, center-frame misses, crowded-scene truncation, unannotated people,
calibration and distribution shift can affect results. Attention is descriptive,
not proof of causality or intent. No identity association is maintained across
clips. GPU determinism and real pretrained checkpoint behavior remain unverified.
