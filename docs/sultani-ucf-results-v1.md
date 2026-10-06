# Real UCF-Crime Sultani baseline v1

The source/content-safe seed-0 baseline is trained on real original UCF-Crime
videos using the verified modern C3D FC6 backbone and Sultani MIL scorer. The
single registered held-out pass gives **frame ROC-AUC 0.7441**. This is a compatible
implementation, not an exact reproduction of the paper's data/training/Caffe
preprocessing protocol. Predictions require human review.

## Population and fixed settings

All 1,900 author-listed videos are installed on `E:\anomaly-detection-dataset-UCF`.
The separate 50 event-recognition Normal videos and ZIP archives remain untouched.
Twenty duplicate training copies and 27 sources already held out in DCSASS are
excluded from optimization. Retained splits are 1,309 train / 254 validation /
290 test; zero source or exact-content overlap crosses partitions. The authors'
full test list is preserved, including the Normal_Videos_936/937 duplicate pair:
290 entries represent 289 unique contents. Primary metrics retain its weighting.

The [prospective protocol and audit](ucf-sultani-execution.md) records five original
annotation ends extending one or two frames beyond verified footage. Their
bounded intersection with available frames was declared before scoring; original
videos/annotations remain unchanged and larger inconsistencies are rejected.

Extraction uses every nonoverlapping 16-frame C3D unit, mean aggregation into 32
normalized 4,096-dimensional segments, RGB channel means `[104,117,128]`, resize
128x171 and center crop 112x112. Channel means differ from the original Caffe
volume mean. All 1,853 retained videos fully decoded with frame-count checks;
extraction took 17,284.65 seconds (4.80 hours), batch 4, RTX 4070 Laptop. Every
bag's hash, finite shape and source/backbone provenance passed a second validation.

The existing 512/32 scoring network, dropout 0.6, MIL ranking margin 1, sparsity
and smoothness 0.00008, weight L2 0.001, Adagrad LR 0.001, 30 bag pairs/batch and
20 epochs were fixed before training. Seed 0 only. Loss decreased from 1.7540 to
0.8944. The best checkpoint is epoch 20, selected solely by validation bag ROC-AUC.

## Validation selection

Reloaded validation bag ROC-AUC is **0.9109** on 254 videos. At the fixed 0.5
threshold: precision 0.9076, recall 0.7883, F1 0.8438, normal-video false-positive
rate 9.40%, confusion `[[106,11],[29,108]]` (rows Normal/anomaly). This is bag
selection evidence, not validation frame localization evidence.

The frozen selection binds checkpoint, feature manifest and prospective source
protocol, with evaluation mode `frame`, projection `c3d_units` and threshold 0.5.
No held-out output was used for checkpoint or threshold selection.

## Primary held-out frame evaluation

Scores project onto the actual C3D-unit boundaries; short repeated-unit scores
are averaged. Ground truth uses the supplied temporal annotations. One registered
pass covers **290 videos / 1,111,808 frames**: 1,027,470 Normal-labelled frames and
84,338 anomaly-labelled frames. Non-event portions of abnormal videos also belong
to the negative-frame denominator.

| Metric | Value |
|---|---:|
| Frame ROC-AUC | 0.744053 |
| Precision at 0.5 | 0.191198 |
| Recall at 0.5 | 0.518865 |
| F1 at 0.5 | 0.279428 |
| False-positive rate at 0.5 | 0.180164 |

Confusion matrix, rows actual Normal/anomaly and columns predicted:

```text
842357  185113
 40578   43760
```

The low threshold precision and 18.02% negative-frame false-positive rate are
material limitations. ROC-AUC does not establish calibrated confidence or safe
operational violence detection. The threshold remains 0.5; it is not retuned from
these results or presentation clips.

## Secondary bag view and illustrations

Using the same saved 32 scores, without another model pass, maximum-score bag
ROC-AUC is **0.8539**, accuracy **76.55%**, precision 0.8214, recall 0.6571 and
F1 0.7302. Confusion is `[[130,20],[48,92]]`: 20/150 Normal videos alert and
48/140 abnormal videos miss the fixed threshold. These bag metrics must not be
substituted for frame ROC-AUC.

Fourteen timeline examples were fixed before scoring: first held-out video by ID
in each category, including Normal. They overlay supplied annotations and exact
C3D boundaries and retain failures. The first Fighting example detects only part
of the annotated event and also crosses the threshold before its start. Figures
and the frame ROC plot are under `seed_0/held_out/analysis/`; analysis reads saved
predictions only and performs no additional optimization or model scoring.

## Artifacts and reproducibility

Run root: `runs/ucf-crime/sultani_shared_safe_v1/`.

| Identity | SHA256 |
|---|---|
| Selected epoch-20 checkpoint | `7e5b548ea6260614209a727a947e6270a1a1af2185d29accf1778f91b4af7c66` |
| Feature manifest | `2a888b0e0da7b0115efab7a8641f67490ef0a7d4cf9ef17238aac5ca9c9302e0` |
| Prospective source protocol | `a163f05823825daf647622eb3f776ca3de4cefaf77c55ccf0aac6b2ecbf40c9d` |
| Training config | `3fe7c63a81e6b5bdd1ec121903c1a6d2f4c5bca31bfa0a86d6852614a09b4394` |

`seed_0/selection.json` records completed training and reloaded validation.
`seed_0/held_out/evaluation_registration.json`, `metrics.json` and
`predictions.json` preserve the single test pass. `analysis/summary.json` binds
the saved-score analysis and predeclared timeline selection. The registered test
CLI refuses a second pass into that run.

Deployment is versioned in `configs/surveillance_demo_ucf_v1.yaml`, retaining the
same frozen RGB Actor-Transformer, detector and I3D assets. Separate DCSASS
standalone and Collective measurements remain unchanged. The
[991-clip cascade protocol](cascade-evaluation-protocol-v1.md) measures coarse
clip alerts and conditional behavior on protected sources; it must remain
separate from this frame benchmark.
