# DCSASS human-centric v1: measured evidence

This is a source-separated surveillance adaptation of the RGB Actor-Transformer,
using frozen I3D features and automatic COCO_V1 Faster R-CNN actors. These are
clip-level observable-behavior labels, not labels of a person's identity or
character. DCSASS has no individual actor action targets or actor box ground truth.

## Population and frozen protocol

One complete local installation contains 16,639 readable clips from 520 original
sources. A second installation is byte-identical and is not counted again. Three
identical repeated CSV annotations were collapsed; 49 clips without an accepted
binary annotation were quarantined. The human-centric population contains 6,491
clips from 203 sources. Other categories remain installed and explicitly excluded
from this actor-model population.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Normal | 2,116 | 591 | 557 |
| Abuse | 435 | 68 | 65 |
| Assault | 164 | 82 | 32 |
| Fighting | 67 | 9 | 10 |
| Robbery | 1,384 | 232 | 275 |
| Vandalism | 312 | 40 | 52 |
| Total | 4,478 | 1,022 | 991 |
| Original sources | 140 | 32 | 31 |

Sources are recovered from original-video directory names and clip filename
stems. Sources and exact content do not overlap between splits. Installed sources
appearing in the authors' official UCF-Crime test list are reserved for test.
Binary-normal clips in selected categories map to Normal, rather than inheriting
their source category. See [the frozen protocol](dcsass-human-centric-protocol.md).

## Actor coverage

Confidence remains 0.7; it has not been tuned against held-out results. Empty
clips are excluded from actor optimization and explicitly abstained at evaluation.

| Split | All clips | Covered | No actors | Coverage |
|---|---:|---:|---:|---:|
| Train | 4,478 | 3,211 | 1,267 | 71.71% |
| Validation | 1,022 | 692 | 330 | 67.71% |
| Test | 991 | 686 | 305 | 69.22% |
| Overall | 6,491 | 4,589 | 1,902 | 70.70% |

Covered training counts are 1,398 / 306 / 103 / 62 / 1,135 / 207 in class order.
Inverse-frequency cross-entropy weights use only these training counts. Covered
validation includes **only two Fighting clips from one source**. Test has ten
Fighting clips, also from one source. No precision, recall or IoU is claimed for
the detector because DCSASS supplies no actor box ground truth.

## Seed-0 initialization comparison

Both runs use identical cached features, source splits, class weights, optimizer,
learning rate, iteration budget and architecture. The five-class Collective group
head is replaced with a new six-class head. The unused actor head is frozen and
has no classification loss. I3D remains frozen.

| Initialization | Selected iteration | Validation accuracy | Validation macro F1 |
|---|---:|---:|---:|
| Collective RGB transfer | 1,900 | 50.14% | 0.3060 |
| Random Actor-Transformer control | 2,900 | 49.13% | 0.4112 |

The predeclared E-C comparison selects random initialization by validation macro
F1 alone. This is the same Actor-Transformer architecture, not another model
family. Transfer remains a reported experiment. The higher random-control macro
F1 is sensitive to correctly predicting one of the two validation Fighting clips;
it is not evidence of stable class-wide recognition. No further tuning or extra
seed is justified for the MVP at this gate.

## Transfer baseline: one held-out pass

These metrics belong specifically to the transfer checkpoint, not the selected
random control. Classification metrics are conditional on the 686 covered test
clips. The 305 uncovered clips abstain and are not labelled Normal.

Accuracy: **52.04%**; macro F1: **0.2464**; balanced accuracy: **24.45%**.
All-population accounting: 357 covered correct, 329 covered incorrect, 305 abstained.

| Class | Precision | Recall | F1 | Covered support |
|---|---:|---:|---:|---:|
| Normal | 0.5543 | 0.5915 | 0.5723 | 328 |
| Abuse | 0.1795 | 0.1321 | 0.1522 | 53 |
| Assault | 0.1000 | 0.0417 | 0.0588 | 24 |
| Fighting | 0.0000 | 0.0000 | 0.0000 | 10 |
| Robbery | 0.5646 | 0.6349 | 0.5977 | 241 |
| Vandalism | 0.1818 | 0.0667 | 0.0976 | 30 |

The binary view uses `P(anomaly) = 1 - P(Normal)` with a fixed 0.5 threshold:
accuracy **58.16%**, precision **0.6035**, recall **0.5782**, F1 **0.5906**,
ROC-AUC **0.6303**. Binary confusion matrix (rows actual Normal/anomaly) is
`[[192, 136], [151, 207]]`; false-positive rate is **41.46%**.

Structured mistakes and actor-count strata are retained under
`runs/dcsass/human_rgb_detected_v1/seed_0/held_out/`. These results are weak,
particularly for aggressive minority categories, and do not support deployment
without human review.

## Selected random control: one held-out pass

Selection was frozen in `initialization_selection.json` before this evaluation.
The better validation result did not carry through to held-out macro F1. We retain
the validation-selected control; switching back based on test would be test tuning.

Conditional accuracy is **51.02%**, macro F1 **0.2140**, balanced accuracy **22.87%**.
Coverage remains **686 / 991 = 69.22%**. All-population accounting is 350 covered
correct, 336 covered incorrect and 305 abstained. No actor accuracy is reported.

| Class | Precision | Recall | F1 | Covered support |
|---|---:|---:|---:|---:|
| Normal | 0.5611 | 0.5457 | 0.5533 | 328 |
| Abuse | 0.0000 | 0.0000 | 0.0000 | 53 |
| Assault | 0.0000 | 0.0000 | 0.0000 | 24 |
| Fighting | 0.0000 | 0.0000 | 0.0000 | 10 |
| Robbery | 0.5219 | 0.6929 | 0.5954 | 241 |
| Vandalism | 0.1379 | 0.1333 | 0.1356 | 30 |

The confusion matrix below uses class order Normal, Abuse, Assault, Fighting,
Robbery, Vandalism; rows are ground truth and columns are predictions.

```text
179   2   5   1 130  11
 41   0   0   0   6   6
 20   0   0   0   3   1
  6   0   0   0   3   1
 62   3   0   3 167   6
 11   1   2   1  11   4
```

Abuse and Assault frequently become Normal; Normal frequently becomes Robbery.
The model predicts all six classes somewhere, but has no correct held-out Abuse,
Assault or Fighting prediction. It therefore cannot be presented as reliable
six-class violence recognition. The frozen pipeline supports an honest research
demonstration of weak generalization and model disagreement.

The fixed-threshold binary view has accuracy **58.31%**, precision **0.5963**,
recall **0.6229**, F1 **0.6093**, ROC-AUC **0.6307**, and false-positive rate
**46.04%**. Confusion matrix is `[[177, 151], [135, 223]]`. Multiclass and binary
decisions differ because the latter uses `1 - P(Normal) >= 0.5` rather than argmax.
Saved metrics, predictions, actor-count strata, uncovered clips and high-confidence
mistakes are under `random_init_seed_0/held_out/`. There will be no further
test-informed tuning in this protocol.

## Separate Sultani population

While UCF-Crime is downloading, a provisional Sultani MIL baseline uses all 16,590
valid DCSASS binary-labelled clips from 519 sources, including non-human-centric
categories and no-actor clips. All 203 shared actor sources retain the same split.
Generic train/validation/test counts are 10,642 / 2,590 / 3,358 clips and
333 / 81 / 105 sources. The larger test share reserves official UCF test sources.

This population provides clip/bag labels, not temporal ground truth. It can support
bag ROC-AUC and a working timeline demo, **not UCF-Crime frame ROC-AUC or a
paper-exact reproduction claim**. Many clips are about two seconds long; repeated
32-segment bins cannot create temporal detail beyond the actual C3D units.
