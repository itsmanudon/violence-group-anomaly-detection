# Local datasets and manifests

No script downloads data. Acquire DCSASS and UCF-Crime from authorized sources and
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
