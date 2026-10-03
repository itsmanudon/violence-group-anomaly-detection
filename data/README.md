# Local datasets and manifests

No script downloads data. Acquire DCSASS, UCF-Crime, and Collective Activity from authorized sources and
unpack locally. Videos/features/checkpoints/manifests/split lists are ignored by
Git. Preserve dataset versions, licenses, and split provenance in experiment notes.

## Expected directories

```text
data/raw/dcsass/
  Fighting/Fighting001_x264_0.mp4
  Fighting/Fighting001_x264_1.mp4
  labels.csv
data/raw/ucf_crime/
  Fighting/Fighting001_x264.mp4
  Robbery/Robbery001_x264.mp4
  Normal_Videos/Normal_Videos001_x264.mp4
data/splits/
  ucf_train.txt
  ucf_test.txt
  ucf_temporal.txt
data/manifests/
data/features/c3d/
```

Nested folders and .avi/.mov/.mkv are supported. UCF recognizes the 13 canonical
category names, including `RoadAccidents`, and `Normal`, `Normal_Videos`, or
`NormalVideos`. Unknown categories and missing directories fail clearly.

## DCSASS labels

DCSASS clips inside an anomaly category can be normal. Do **not** infer binary
labels from category folders. Local Kaggle exports vary; normalize actual clip
annotations into a CSV with header `path,label` and optional `source_video_id`
and `anomaly_type` columns. The CLI consumes this documented normalized schema,
not an assumed positional raw export:

```csv
path,label,source_video_id,anomaly_type
Fighting/Fighting001_x264_0.mp4,0,fighting001,Fighting
Fighting/Fighting001_x264_1.mp4,1,fighting001,Fighting
```

Paths are relative to dataset root. Labels are 0 (normal) or 1 (abnormal). Every
clip needs a label; duplicates, missing labels, and nonexistent listed clips fail.

```powershell
python scripts/prepare_dcsass.py --root data/raw/dcsass --labels data/raw/dcsass/labels.csv --output data/manifests/dcsass.jsonl --seed 7 --ratios 0.7 0.15 0.15
```

## Source leakage and splits

**DCSASS is derived from UCF-Crime material. These are not independent cross-dataset
evidence.** Use the same original source IDs across both datasets.
Names such as `Fighting001_x264_12` and `Fighting001_x264` recover `fighting001`.
Preparation case-folds canonical IDs. Unrecoverable filenames require an explicit
`--source-map` JSON object (paths relative to the dataset root):

```json
{"Fighting/renamed_clip.mp4": "fighting001"}
```

The optional source ID in the DCSASS labels CSV takes precedence. Recover IDs from
provenance; do not invent unique clip IDs that hide shared originals. Manually
authored manifests remain the user's responsibility: leakage checks only compare
the IDs provided.

Splitting shuffles sorted unique source IDs with a seeded local RNG, assigns whole
groups, and is independent of record ordering. Ratios refer to source counts, not
clip counts. Largest-remainder rounding keeps zero-ratio splits empty; group sizes
affect clip counts, and classes are not stratified.
Inspect printed `(split,label)` counts; training requires both labels.

For combined **research splits**, jointly split manifests and automatically rebase paths:

```powershell
python scripts/split_manifests.py --manifests data/manifests/dcsass.jsonl data/manifests/ucf_crime.jsonl --output data/manifests/joint.jsonl --seed 7
```

This command replaces membership with a new research split; it does not preserve
official UCF test sources. For official benchmark work, preserve those test sources
and keep all corresponding DCSASS clips out of training. `check_leakage` rejects
cross-split sources even across dataset names. Training/evaluation invoke it on the
whole supplied manifest. It cannot identify the same video under incorrectly different IDs.

## UCF official split and annotations

```powershell
python scripts/prepare_ucf_crime.py --root data/raw/ucf_crime --output data/manifests/ucf_crime.jsonl --train-list data/splits/ucf_train.txt --test-list data/splits/ucf_test.txt --annotations data/splits/ucf_temporal.txt --val-fraction 0.15 --seed 7
```

Split lists contain video filename/path in the first whitespace-delimited field.
Both lists must be supplied, not overlap, and exactly cover installed videos.
Validation sources come only from official training sources. Without lists, a
seeded research split is generated, not the official benchmark protocol.

Optional temporal annotation format:

```text
Fighting001_x264.mp4 Fighting 100 200 400 500
Normal_Videos001_x264.mp4 Normal -1 -1 -1 -1
```

The parser explicitly assumes **1-based inclusive frame indices**. `[100,200]`
becomes `[99/fps,200/fps)` seconds. Check the convention for your annotation version
and convert before use if necessary. `-1 -1` means absent interval. Annotations beyond
the probed frame count fail. Unlisted abnormal videos keep unknown intervals (`null`);
frame evaluation refuses them rather than inventing all-positive labels. Weak normal
video labels imply normal frames during evaluation. Container metadata may be approximate.

## JSONL schema

One JSON object per line. Required fields:

| Field | Meaning |
|---|---|
| dataset | Dataset name |
| video_id | Unique within dataset, normally relative path without extension |
| source_video_id | Canonical original source shared across derived datasets |
| path | Manifest-relative video path, or absolute |
| split | `train`, `val`, or `test` |
| label | Binary normal 0 / abnormal 1 |

Optional: `anomaly_type`, `duration_sec`, `fps`, `num_frames`,
`temporal_annotations`, `feature_path`. Unknown metadata is `null`. Intervals are
`[start,end)` seconds; `null` means unknown, `[]` means explicitly no anomaly.
Feature paths resolve relative to the manifest directory too. Example:

```json
{"dataset":"ucf_crime","video_id":"Fighting/Fighting001_x264","source_video_id":"fighting001","path":"../raw/ucf_crime/Fighting/Fighting001_x264.mp4","split":"train","label":1,"anomaly_type":"Fighting","duration_sec":120.0,"fps":25.0,"num_frames":3000,"temporal_annotations":null,"feature_path":"../features/c3d/ucf_crime/Fighting/Fighting001_x264.npy"}
```

Parsing checks required fields, labels, split, positive metadata, interval structure,
and duplicate dataset/video IDs. Feature-only experiments need no installed video.

## Precomputed feature bags

Files contain finite `[32,4096]` C3D FC6 bags, already averaged and L2-normalized.
Formats: `.npy`, tensor `.pt`/`.pth`, or whitespace `.txt`. The loader does not
silently convert backbones/dimensions or normalize arbitrary imported features.
Record checkpoint/preprocessing provenance. To attach imported arrays:

```python
from dataclasses import replace
from pathlib import Path
from surveillance.datasets.common import read_manifest, write_manifest

manifest = Path("data/manifests/ucf_crime.jsonl")
rows = [replace(r, feature_path=f"../features/c3d/{r.dataset}/{r.video_id}.npy")
        for r in read_manifest(manifest)]
write_manifest(rows, Path("data/manifests/ucf_features.jsonl"))
```

For synthetic development use `dataset="synthetic"`, correct distinct source IDs,
balanced labels, and real feature paths. `path` may reference an unavailable video
when only features are used.

## Local C3D checkpoint

`C3DFC6` implements convolutions `conv1`, `conv2`, `conv3a/b`, `conv4a/b`,
`conv5a/b`, and `fc6`. Pooling yields `[512,1,4,4]`; FC6 maps 8192 to 4096 and
emits post-ReLU activations. Supply a PyTorch state dict or `{"state_dict": ...}`
with exactly these compatible keys/shapes. Remove fc7/fc8 classifier parameters
explicitly when converting full checkpoints. Foreign layouts/naming require explicit
conversion; another 4096-output backbone is not silently accepted as C3D.

Preprocessing resizes decoded BGR frames to 171x128, center crops 112x112, optionally
converts RGB, and subtracts three supplied means on raw 0..255 scale. Native FPS is
used. Verify against the chosen pretrained model. Spatial mean files or other
scaling require adapting the transform; no guessed pretrained constants are supplied.

```powershell
python scripts/extract_c3d.py --manifest data/manifests/ucf_crime.jsonl --output-manifest data/manifests/ucf_features.jsonl --feature-dir data/features/c3d --c3d-checkpoint checkpoints/c3d_fc6.pt --mean 0 0 0 --channel-order bgr --batch-size 4 --device auto
```

Zero means above are a syntax example, not verified pretrained preprocessing.
Missing/incompatible weights fail before extraction. Tests/scorer development do
not require them. Extraction streams 16-frame clips, pads the final one, stores
embeddings, partitions temporal units, averages, L2-normalizes, and preserves source
IDs/splits in the output manifest.

## Collective Activity (Milestone 2A)

The actor task has its own `ActorRecord` JSONL schema; do not pass these records
to the Milestone 1 binary anomaly scripts. Acquire and unpack the original
Collective Activity dataset locally; no downloader is provided. Expected layout:

```text
data/raw/collective/
  seq01/
    annotations.txt
    frame0001.jpg
    frame0002.jpg
    ...
  seq02/
    annotations.txt
    frame0001.jpg
    ...
  ...
  seq44/
    annotations.txt
    frame0001.jpg
    ...
data/features/collective/
  pose/                    [actors,98304] float .npy arrays and provenance .json
  rgb/                     [actors,20800] float .npy arrays and provenance .json
```

Each annotation row starts with six whitespace-separated integer columns:
`frame x y width height action`; extra trailing metadata is ignored. For example
`1 20 30 40 80 2` identifies a crossing actor in frame 1 with xywh box
`[20,30,40,80]`. This is the actual annotation layout verified against a
[released Collective loader](https://github.com/wjchaoGit/Group-Activity-Recognition/blob/master/collective.py),
not an inferred folder-label scheme. Frames are numbered from 1 and must be
consecutive. Annotation centers are 1,11,21,...; each manifest clip contains
center-5 through center+4, clamped at sequence boundaries. Raw action 1 is NA and
excluded; 2..6 map to crossing/waiting/queueing/walking/talking IDs 0..4.

The group label is the majority valid actor action (ties use the lowest class ID).
An all-NA scene, malformed label, nonpositive/out-of-bounds box, missing image,
or unexpected annotation layout fails with a diagnostic. Source image dimensions
determine normalization; boxes become `[x/W,y/H,(x+w)/W,(y+h)/H]`. No actor identity,
tracking or box interpolation is invented. Details and protocol citations are in
[docs/collective-format.md](../docs/collective-format.md).

```powershell
python scripts/prepare_collective.py --root data/raw/collective --output data/manifests/collective.jsonl
# Optional development holdout from training sources only:
python scripts/prepare_collective.py --root data/raw/collective --output data/manifests/collective.jsonl --val-sequences 1 2 3
```

Default test source IDs are **5,6,7,8,9,10,11,15,16,25,28,29**; the other IDs
in 1..44 are training. This 32/12 split is verified from the related authors'
[configuration](https://github.com/wjchaoGit/Group-Activity-Recognition/blob/master/config.py).
The Actor-Transformers paper does not enumerate IDs, so exact identity with its
split is not asserted. `--train-sequences` and `--test-sequences` accept explicit
integer lists. `--allow-subset` is for development fixtures and relaxes 32/12 counts;
it does not create benchmark evidence. Validation subtracts sources from training.
All clips from a source stay together. Manifest validation also rejects reused
physical frame/feature paths across splits and duplicate source/center records.

Required actor fields: `dataset`, `video_id`, `source_video_id`, `clip_id`, `split`,
ten `frame_paths`, ten ordered `frame_indices`, normalized `actor_boxes` `[N,4]`,
`actor_labels` `[N]`, and scalar `group_label`. Optional `pose_feature_path` and
`rgb_feature_path` default to null. Example single JSONL record (wrapped here):

```json
{
  "dataset": "collective", "video_id": "seq01", "source_video_id": "seq01",
  "clip_id": "seq01:6", "split": "train",
  "frame_paths": ["../raw/collective/seq01/frame0001.jpg", "../raw/collective/seq01/frame0002.jpg", "../raw/collective/seq01/frame0003.jpg", "../raw/collective/seq01/frame0004.jpg", "../raw/collective/seq01/frame0005.jpg", "../raw/collective/seq01/frame0006.jpg", "../raw/collective/seq01/frame0007.jpg", "../raw/collective/seq01/frame0008.jpg", "../raw/collective/seq01/frame0009.jpg", "../raw/collective/seq01/frame0010.jpg"],
  "frame_indices": [1,2,3,4,5,6,7,8,9,10],
  "actor_boxes": [[0.1,0.2,0.3,0.8],[0.5,0.1,0.7,0.9]],
  "actor_labels": [0,0], "group_label": 0,
  "pose_feature_path": "../features/collective/pose/example.npy",
  "rgb_feature_path": null
}
```

The example illustrates the generic schema; the preparation script emits centers
1,11,21,... and absolute image paths. Serialize each full object on one line.
Paths in hand-authored manifests resolve relative to their JSONL file. Extraction
rebases existing paths when writing a new manifest, retaining absolute references
when Windows drive letters differ. Precomputed-only training does not require the
image paths to exist, but source/split checks still apply.

Import finite float `.npy` arrays `[N,D]` in exactly the same actor order as the
boxes/labels, or run the extractors after preparing vetted local exports:

```powershell
python scripts/extract_pose_features.py --manifest data/manifests/collective.jsonl --output-manifest data/manifests/collective_pose.jsonl --feature-dir data/features/collective --checkpoint checkpoints/hrnet_w32_features.pt
python scripts/extract_i3d_features.py --manifest data/manifests/collective_pose.jsonl --output-manifest data/manifests/collective_both.jsonl --feature-dir data/features/collective --checkpoint checkpoints/i3d_mixed4f.pt
```

See [local backbone exports](../docs/actor-backbones.md) for HRNet COCO weight
provenance, I3D Mixed_4f export metadata and preprocessing requirements. Arbitrary
state dictionaries are not accepted by these feature-only archive adapters.
Each extraction creates a sidecar recording checkpoint SHA256, metadata, actor
boxes, image size and feature shape. Preserve those sidecars when importing or
sharing arrays. No actual pretrained HRNet/I3D checkpoint is included or downloaded.

## Person detections and feature caches (Milestone 2B)

Keep the original actor manifest as the GT annotation authority. Person detections
are a separate **version 1 JSONL** artifact, one record per scene, including empty
results. No script downloads detector weights or data. For example (one JSONL line):

```json
{"version":1,"coordinate_system":"absolute_xyxy","dataset":"collective","video_id":"seq01","source_video_id":"seq01","clip_id":"seq01:11","frame_index":11,"image_height":480,"image_width":720,"detections":[{"box":[20,30,60,110],"score":0.92,"class_id":1}],"metadata":{"person_class_id":1},"pose_feature_path":null,"rgb_feature_path":null,"feature_metadata":{}}
```

`frame_index` must equal the GT record's `frame_indices[5]`; dataset/clip/video/source
identities must agree. `image_width/height` describe the **native reference image**.
Boxes are finite absolute xyxy edge coordinates, bounded by those dimensions,
with positive area. Confidence is in `[0,1]`. Only the declared person class is
accepted (metadata `person_class_id`, default 1). `detections: []` is a valid
no-person result. Missing scene records are errors, not equivalent to zero boxes.

Writer order is deterministic by `(dataset,clip_id)`. Actors are left-to-right
by center x, then center y, coordinates, descending confidence and class. Uncached
records can be reordered on read; cached unsorted records fail rather than corrupt
feature row alignment. Duplicate clip/source-center records and incorrect
coordinate declarations fail. Metadata stores checkpoint SHA256, filter settings,
before/after counts and per-stage rejection/truncation counts when produced locally.
The parser does not rerun filters on an already finalized detection artifact.

```powershell
python scripts/detect_people.py --manifest data/manifests/collective.jsonl --config configs/person_detector.yaml --output data/manifests/collective_detections.jsonl
python scripts/extract_pose_features.py --manifest data/manifests/collective.jsonl --box-source detections --detections data/manifests/collective_detections.jsonl --output-manifest data/manifests/collective_detected_pose.jsonl --feature-dir data/features/collective --checkpoint checkpoints/hrnet_w32_features.pt
python scripts/extract_i3d_features.py --manifest data/manifests/collective.jsonl --box-source detections --detections data/manifests/collective_detected_pose.jsonl --output-manifest data/manifests/collective_detected_both.jsonl --feature-dir data/features/collective --checkpoint checkpoints/i3d_mixed4f.pt
```

Set the local Faster R-CNN checkpoint in the detector YAML. Checkpoint paths in
YAML resolve beside that YAML; feature paths resolve beside their detection JSONL.
The dataset layout is unchanged. Detection cache files belong under
`data/manifests/`; arrays/provenance belong under `data/features/` (both ignored).

The extraction outputs preserve detection records and add `pose_feature_path`
and/or `rgb_feature_path` plus a `feature_metadata` mapping keyed by modality.
Each mapping contains `detection_fingerprint`, `source_fingerprint`,
`feature_sha256`, checkpoint SHA256, backbone metadata, extraction `image_size`,
shape and box source. Loaded arrays must be finite float `[N,D]` in detection order.
Ordered box/source/content fingerprints are mandatory when importing cached arrays;
use `detection_fingerprint(result)` and `source_fingerprint(gt_row,gt_manifest)` to
compute them. SHA256 hashes identify the saved `.npy` bytes. A source fingerprint
identifies ordered resolved paths and frame indices, not actual source image bytes.
Preserve image data and provenance separately. Empty scenes remain records and
bypass feature/model calls at inference.

Ground-truth and detected feature files are never interchangeable, even after
matching. Matching transfers only class supervision for **evaluation**, never GT
crop features. Unmatched action labels stay unknown; group labels remain the
original scene annotations. See [person-detection.md](../docs/person-detection.md)
for matching, metric denominators, empty-scene policy and local checkpoint contract.

## Collective real benchmark validation (Milestone 2C)

The strict benchmark path expects all `seq01` through `seq44`, each containing
`annotations.txt` and consecutive `frame0001.jpg`, `frame0002.jpg`, ... files.
It reuses the exact annotation parser described above. No guessed annotation
format, download, silent label replacement or silent box clipping is introduced.
Install data in `data/raw/collective/` under its applicable license.

```powershell
python scripts/validate_collective.py --root data/raw/collective --report runs/collective/dataset_validation.json --val-sequences 1 2 3
python scripts/run_collective_experiment.py --stage prepare
```

The protocol preserves the released 32/12 IDs, explicitly holds out training
sequences 1,2,3 for validation, and verifies original train/test source assignment,
duplicate scenes, referenced frames, labels, dimensions, finite in-bounds boxes,
group-majority supervision, temporal windows and physical-path leakage.
Validation reports include counts/class distributions by split, actor-count
warnings and structural failures. A supplied `--manifest` is compared to the
parsed raw annotations. Failures save reports and exit nonzero. The benchmark
requires all 44 sources; `--allow-subset` is only for non-benchmark fixtures.

See [collective-protocol.md](../docs/collective-protocol.md) for the precise split,
preprocessing, local exports, staged commands and metric populations. Current
feature extraction writes schema-v2 provenance sidecars and detected JSONL metadata.
Benchmark validation rejects older caches, changed boxes/order, different checkpoint
fingerprints, altered image size/preprocessing, wrong feature dimensions and changed
feature bytes. Re-extract old caches or import complete verified provenance; do not
invent metadata for unknown features. The legacy standalone trainer remains usable
for model development on synthetic/precomputed arrays.

Detector confidence is selected only from broad validation candidates with explicit
local pretrained-weight and backend-cap provenance. Final detections reference
the selected filters' receipt hash. Test sources cannot be relabeled as validation
for tuning. All extraction, selection, freeze and run artifacts belong under the
gitignored data/run directories; do not redistribute dataset imagery in reports.
