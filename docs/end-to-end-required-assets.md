# Required local assets for the surveillance MVP

The initial 2026-10-06 inspection found Collective data/models but no installed
DCSASS, UCF-Crime, C3D bags/weights or real Sultani checkpoint. The user subsequently
supplied DCSASS and modern C3D weights. DCSASS is fully audited, frozen and evaluated
for behavior; C3D conversion is validated. UCF-Crime is now installed externally,
with completed seed-0 training and registered frame evaluation. The separate
historical DCSASS Sultani run also completed training and evaluation.
The five pinned deployment assets and genuine example caches are installed. This document
retains acquisition instructions for a fresh installation.

Current installed dataset root:
`data/raw/dcsass/DCSASS Dataset/DCSASS Dataset/`. Use the frozen
`data/manifests/dcsass_human_centric_v1_final.jsonl`, not a newly randomized clip
split. Actual audits and experiment receipts are linked from
[the protocol](dcsass-human-centric-protocol.md) and
[the measured results](dcsass-surveillance-results-v1.md).

## DCSASS — installed; fresh acquisition instructions

1. Open the repository's documented
   [DCSASS Kaggle dataset](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset).
   Download through the dataset page using your account as required. Preserve the
   archive, original annotation files, dataset version and usage terms. The page
   exists, but this audit's web reader did not expose its file listing or license;
   no archive format, count or label-column convention has been assumed.
2. Extract the videos and original annotations under
   `D:/Github Repos/violence-group-anomaly-detection/data/raw/dcsass/`.
   Retain all categories; excluded categories must remain on disk.
   An existing installation elsewhere can be used directly with `--root`; avoid
   copying a large dataset unnecessarily.
3. Inspect the actual annotations before converting them to the existing loader's
   normalized CSV. Preserve a path for every installed clip and its actual binary
   label. Do not assign a positive label merely because the folder is Fighting,
   Abuse, Assault, Robbery or Vandalism. The normalized contract is:

   ```csv
   path,label,source_video_id,anomaly_type
   Fighting/Fighting001_x264_0.mp4,0,fighting001,Fighting
   Fighting/Fighting001_x264_1.mp4,1,fighting001,Fighting
   ```

   These two rows illustrate the schema; they are not observed local annotations.
   `path` is relative to the dataset root, `label` is 0/1, and optional source IDs
   must identify the original video, not the individual derived clip.
4. The existing preparation command is:

   ```powershell
   .\.venv\Scripts\python.exe scripts/prepare_dcsass.py --root data/raw/dcsass --labels data/raw/dcsass/labels.csv --output data/manifests/dcsass.jsonl --seed 7 --ratios 0.7 0.15 0.15
   ```

   This command creates a generic binary research manifest. It does **not** yet
   implement the requested class-aware six-class Human-Centric v1 audit/freeze.
   Its split must not be treated as the final surveillance protocol. The audited
   Human-Centric v1 protocol has now been frozen separately with the new audit/freeze
   scripts. Do not replace its scientific population or assignments.

The existing parser recognizes `Fighting001_x264_12` and `Fighting001_x264` as
`fighting001`. Renamed/ambiguous clips require a verified source-map JSON. Source
recovery must be sampled manually and tested; unresolved source identity blocks
training.

Human-Centric v1 labels: **Normal, Abuse, Assault, Fighting, Robbery, Vandalism**.
A binary-normal clip from a selected anomaly category maps to Normal. Arrest,
Arson, Accident/RoadAccidents, Burglary, Explosion, Shooting, Stealing and
Shoplifting are excluded from this actor task, retained for generic anomaly work.
Actual category spellings and aliases must be verified against the export first.

## Sultani — mandatory for real raw-video anomaly inference

Acquire UCF-Crime and its split/temporal annotation provenance through the
[Sultani authors' repository](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018).
Its README links dataset acquisition locations, publishes `Anomaly_Train.txt` and
`Temporal_Anomaly_Annotation.txt`, and identifies C3D-v1.0 as the original feature
extractor. The linked UCF project page returned HTTP 502 during this audit;
archive availability has not been verified by downloading it.

Expected installation:

The user completed extraction at `E:\anomaly-detection-dataset-UCF`. Use this
external root directly; no second repository copy is needed. All 1,900 author-listed
anomaly videos are present and audited. The separate 50 event-recognition Normal
videos and ZIPs are retained. See [the frozen source/content protocol](ucf-sultani-execution.md).

```text
data/raw/ucf_crime/<category>/<original_video>.mp4
data/splits/ucf_train.txt
data/splits/ucf_test.txt
data/splits/ucf_temporal.txt
checkpoints/c3d_fc6.pt
```

Supply trusted pretrained C3D weights with documented channel order, mean
subtraction, spatial preprocessing and conversion provenance. The existing adapter
accepts PyTorch `conv1`, `conv2`, `conv3a/b`, `conv4a/b`, `conv5a/b`, `fc6`
parameters. Caffe/foreign layouts require explicit, verified conversion. An arbitrary
4096-dimensional network is not a compatible substitute. The author's
`weights_L1L2.mat` contains anomaly-scoring weights, **not** the C3D feature backbone.

Verified pretrained C3D `[32,4096]` bags can unblock training/evaluation after
manifest and protocol validation. They do not alone enable uploaded-video inference;
that deliverable also requires the C3D extraction model and its verified preprocessing.
The user-supplied official OpenMMLab Sports1M archive has been verified and converted
to `checkpoints/c3d_fc6_openmmlab_v1.pt`. Its raw SHA256, pinned upstream source,
FC6 equivalence and channel-mean caveat are in
[the conversion validation](c3d-openmmlab-validation.md). It is a modern compatible
backbone, with preprocessing explicitly distinguished from the original Caffe model.
See [the existing C3D contract](../data/README.md#local-c3d-checkpoint).

Before freezing DCSASS, identify the UCF held-out sources. The datasets share
original videos. DCSASS training/validation must not expose sources reserved for
the unified held-out evaluation. If official UCF splits are unavailable, use a
documented common research split and label results accordingly.

## Already installed — preserve these assets

| Asset | Local path |
|---|---|
| Selected Collective RGB transfer initialization | `runs/collective/rgb_gt_v1/rgb_gt/seed_0/best.pt` |
| Frozen I3D Mixed_4f archive | `checkpoints/i3d_mixed4f_collective_rgb_v1.pt` |
| Converted DeepMind RGB weights | `checkpoints/external/pytorch-i3d/models/rgb_imagenet.pt` |
| COCO_V1 Faster R-CNN | `checkpoints/external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` |
| Converted supplied C3D FC6 | `checkpoints/c3d_fc6_openmmlab_v1.pt` |
| Validation-selected DCSASS behavior control | `runs/dcsass/human_rgb_detected_v1/random_init_seed_0/best.pt` |
| Validation-selected DCSASS Sultani scorer | `runs/dcsass/sultani_generic_v1/seed_0/best.pt` |
| Default validation-selected UCF Sultani scorer | `runs/ucf-crime/sultani_shared_safe_v1/seed_0/best.pt` |

Run `python scripts/verify_demo_assets.py` to verify hashes and load all five
interfaces. See [the deployed UCF evidence](sultani-ucf-results-v1.md) and
[the preserved historical DCSASS evidence](sultani-dcsass-results-v1.md).

Do not download replacements or change the matched torch/torchvision/CUDA setup.
For fresh installations, follow acquisition requirements before starting the
commands in [the execution journal](end-to-end-execution-log.md). The current
local installation is complete; acquisition does not require changing architectures.
