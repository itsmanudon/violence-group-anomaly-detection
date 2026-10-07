# Real Sultani baseline on DCSASS clip bags

This is a real-data, source-separated implementation of the Sultani weak MIL
baseline using the supplied modern C3D weights. It is a **DCSASS clip/bag
experiment**, not a UCF-Crime frame benchmark or paper-exact reproduction.

## Data and settings

All 16,590 valid binary-labelled clips from 519 original sources are included,
including actor-excluded categories and no-actor clips. The 49 unresolved labels
are quarantined. All 203 sources shared with the human-centric actor population
retain their frozen train/validation/test assignments. Published UCF test sources
are reserved; no original source crosses splits.

| Split | Sources | Clips | Normal | Anomaly |
|---|---:|---:|---:|---:|
| Train | 333 | 10,642 | 5,680 | 4,962 |
| Validation | 81 | 2,590 | 1,608 | 982 |
| Test | 105 | 3,358 | 2,212 | 1,146 |

The model uses 32 normalized 4,096-dimensional C3D FC6 segments, a 512/32 scoring
network with dropout 0.6, MIL ranking margin 1, sparsity and temporal smoothness
coefficients 0.00008, weight L2 0.001 and Adagrad learning rate 0.001. Seed 0 ran
20 epochs with 30 positive/normal bag pairs per batch. The five-iteration real
preflight verified finite gradients/losses and exact checkpoint reload.

The OpenMMLab Sports1M export SHA256 is
`beac4ff065de3663bf5c6bab36aedca4fe56ec59b231c66bf5374c23eda14831`.
RGB scalar means `[104,117,128]` are explicit. This differs from the original
Caffe volume mean; see [C3D validation](c3d-openmmlab-validation.md).

The complete feature manifest SHA256 is
`f7124bb4903285531e1bd4a4214514eb1e0d7705a3d59a06da5de58831820893`.
Every cached bag is finite `[32,4096]`, with source/video/backbone identities,
actual frame-unit boundaries and feature-byte checksums.

## Frozen selection and one held-out pass

Validation bag ROC-AUC selected **epoch 2**, with **0.6705405001**. Training
completed all 20 epochs; the selected checkpoint was reloaded and its validation
ROC-AUC reproduced before test. Its SHA256 is
`cb51fdd02c3ba2e258c648a59673d9aa701e7f5132470338832096a3ac159872`.

Threshold **0.5** was defined prospectively and remains unchanged. Neither test
nor demonstration clips were used for checkpoint or threshold selection.

| Held-out clip/bag metric | Result |
|---|---:|
| ROC-AUC | 0.6583 |
| Precision | 0.4569 |
| Recall | 0.5742 |
| F1 | 0.5089 |
| False-positive rate | 35.35% |
| Clips scored | 3,358 |

Confusion matrix (rows actual Normal/anomaly, columns predicted Normal/anomaly):
`[[1430, 782], [488, 658]]`. All clips are scored, regardless of person detection.
There are 782 normal false alerts and 488 missed anomalous bags. This is a weak
baseline and requires human review. It must not be described as reliable deployment.

Artifacts are in `runs/dcsass/sultani_generic_v1/seed_0/`: best/last checkpoints,
TensorBoard history, frozen `selection.json`, and `held_out/` registration,
metrics and per-clip 32-score predictions. Later timeline/error analysis uses these
saved predictions, rather than rerunning the held-out benchmark.

## Timelines and limitations

Raw-video timestamps follow the actual nonoverlapping 16-frame C3D units, including
explicit short-video replication and clamping of the final padded unit. Suspicious
intervals merge overlapping/consecutive thresholded ranges; the cascade analyzes
up to three distinct top windows. A decoded/reported frame-count mismatch is
rejected, so a truncated upload cannot silently receive false exact timestamps.

DCSASS supplies clip labels, not event onset/offset annotations. Thus interval
locations are model predictions; **localization ROC-AUC, temporal IoU and event
timing accuracy are not measured**. Many clips are only two or three seconds long,
so repeated 32-bin features cannot create extra temporal resolution. Normal clips
often show context from anomalous original sources, rather than independent
ordinary surveillance videos. These domain/label differences limit comparison
with Sultani's original benchmark.

UCF-Crime acquisition remains in progress at `C:/Users/manan/Downloads`; it can
be prepared from that external location without copying it into the repository.
Any future UCF training must preserve held-out shared DCSASS sources for cascade
evaluation, and disclose deviations from the published protocol.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/run_sultani_dcsass.py --stage prepare
.\.venv\Scripts\python.exe scripts/run_sultani_dcsass.py --stage extract
.\.venv\Scripts\python.exe scripts/train_sultani.py --config configs/experiments/dcsass_sultani_generic_v1.yaml --manifest data/manifests/dcsass_sultani_generic_features_v1.jsonl --output runs/dcsass/sultani_generic_v1/seed_0
.\.venv\Scripts\python.exe scripts/freeze_sultani.py --run runs/dcsass/sultani_generic_v1/seed_0 --manifest data/manifests/dcsass_sultani_generic_features_v1.jsonl --population dcsass_sultani_generic_v1
.\.venv\Scripts\python.exe scripts/evaluate_registered_sultani.py --run runs/dcsass/sultani_generic_v1/seed_0 --manifest data/manifests/dcsass_sultani_generic_features_v1.jsonl
```

Preserve the existing frozen artifacts. Use a separately registered run when
reproducing training; the selection and held-out commands refuse repeated output.
