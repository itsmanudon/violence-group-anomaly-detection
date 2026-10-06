# Source-safe UCF → DCSASS cascade results v1

The working cascade was measured once on all **991 DCSASS human-centric test
clips from 31 held-out sources**. It performs poorly on this population: clip-alert
ROC-AUC **0.5118**, F1 **0.4733**, and conditional routed behavior macro F1
**0.1888**. It is a functioning research demonstration, not reliable violence
detection. No parameters were changed after these results.

## Population and registered inference

The [prospective protocol](cascade-evaluation-protocol-v1.md) binds the selected
UCF Sultani checkpoint, selected DCSASS RGB Actor-Transformer, frozen detector
and I3D, human manifest and common source guard. Fourteen sources belong to the
official UCF test list; seventeen other sources were explicitly excluded from
UCF optimization. Human test labels comprise 557 Normal, 65 Abuse, 32 Assault,
10 Fighting, 275 Robbery and 52 Vandalism clips.

Threshold is 0.5, with at most three distinct suspicious windows per clip.
Sultani normal results bypass actor inference. No detected actors preserves a
generic anomaly alert; covered Normal behavior preserves model disagreement.
Conditional behavior uses the final-alert covered window, or the first covered
window if necessary, matching the interface's fixed rule.

These are weak **clip labels**, not interval or actor ground truth. They cannot
measure localization within DCSASS clips or actor action accuracy. Neither UCF
frame metrics nor standalone DCSASS behavior metrics share this denominator.

The first process encountered a Windows atomic progress-replacement lock. After
verifying all 479 serialized results and preserving both progress versions, the
same registration resumed only the remaining 512 clips. Final 991 result hashes,
source/labels, original input hashes and model identities were independently
verified; recomputation from saved results exactly matches the final metrics.
There was no repeat population model pass or test tuning.

## Clip-alert metrics on all 991 clips

An alert is any preserved Sultani anomaly, including disagreement or no actors.

| Metric | Value |
|---|---:|
| Accuracy | 54.19% |
| Precision | 0.4766 |
| Recall | 0.4700 |
| F1 | 0.4733 |
| ROC-AUC | 0.5118 |
| Normal-clip false-positive rate | 40.22% |

Confusion, rows actual Normal/anomaly: `[[333,224],[230,204]]`.
224/557 normal clips alert, and 230/434 abnormal clips are bypassed. The
near-chance clip ROC-AUC is consistent with limited short-clip generalization
or domain/label differences; their individual effects were not measured. It
does not invalidate or replace the separate
UCF frame benchmark (0.7441 ROC-AUC).

## Coverage and conditional behavior

| Accounting | Clips/windows |
|---|---:|
| Total test clips / sources | 991 / 31 |
| Routed clips | 428 |
| Bypassed clips (actors unexamined) | 563 |
| Covered routed clips | 311 |
| Routed clips with no actors in any selected window | 117 |
| All-population behavior abstentions | 680 |
| Analyzed windows / no-actor windows | 1,020 / 339 |

Behavior coverage is **31.38% overall**, **72.66% among routed clips**. This
differs from the standalone middle-frame detector's 69.22% coverage: the cascade
examines different reference frames and deliberately skips 563 clips. Bypassed
clips must not be described as detector failures or assigned Normal behavior.

On 311 covered routed clips, accuracy is **45.66%**, macro F1 **0.1888**;
142 correct, 169 incorrect, 680 unclassified across the full population.

| Class | Recall | F1 | Covered support |
|---|---:|---:|---:|
| Normal | 42.28% | 0.4903 | 149 |
| Abuse | 0.00% | 0.0000 | 29 |
| Assault | 0.00% | 0.0000 | 7 |
| Fighting | 0.00% | 0.0000 | 7 |
| Robbery | 79.59% | 0.5756 | 98 |
| Vandalism | 4.76% | 0.0667 | 21 |

Confusion uses Normal, Abuse, Assault, Fighting, Robbery, Vandalism:

```text
63 6 2 4 69 5
18 0 0 0 10 1
 2 0 0 0  5 0
 4 0 0 0  3 0
10 3 0 5 78 2
11 1 0 0  8 1
```

Normal frequently becomes Robbery; Abuse/Fighting often become Normal or Robbery;
Assault mostly becomes Robbery. There are no correct Abuse, Assault or Fighting
classifications here. Review the saved clip records rather than treating a
high softmax probability as calibrated reliability.

## Measured RTX 4070 Laptop latency

These timings cover the actual 991-clip pass, excluding model loading, JSON
serialization and UI rendering. Means include bypasses and no-actor skips.
Median total is **0.201 seconds**, mean **0.505 seconds**, p95 **1.346 seconds**.
The short DCSASS population cannot establish full-length video real-time FPS.

| Stage | Mean seconds | p95 seconds |
|---|---:|---:|
| Video decoding (including analyzed windows) | 0.11614 | 0.24158 |
| C3D preprocessing | 0.02718 | 0.04857 |
| C3D forward | 0.04747 | 0.07489 |
| Sultani scorer | 0.00177 | 0.00256 |
| Actor preprocessing | 0.02460 | 0.07752 |
| Person detection | 0.18263 | 0.56660 |
| I3D | 0.06427 | 0.28205 |
| Actor-Transformer | 0.00317 | 0.01505 |
| Total | 0.50509 | 1.34632 |

Sultani/anomaly total (including its decode/backbone work) averages 0.17445
seconds. Stage quantiles are separate distributions and must not be added.

Fixed-input presentation validation on the four-second curated Robbery case
completed the full five-model path: warm GPU 1.09095 seconds; CPU 33.21159 seconds.
GPU model loading took 2.05069 seconds and CPU loading 1.50277 seconds separately.
Two GPU passes had exactly equal scores, boxes, probabilities and alert. CPU
completed with the same alert but slightly different floating-point probabilities;
CPU/GPU equality is not claimed. Receipt:
`runs/ucf-crime/deployment-runtime-v1.json`. Six genuine cached/live/upload API
requests passed in `runs/ucf-crime/demo-api-ucf-final-v2.json`, including live
Robbery actor inference, both normal and Robbery uploads, and an empty-input error.

## Presentation examples and provenance

All eight original demonstration clips remain; their new genuine UCF-model
results expose false alerts, missed anomalies and disagreement. A ninth,
**Robbery: correct cascade example (curated)**, is a disclosed post-hoc selection:
first correctly alerted true Robbery clip by ID in saved registered results,
`Robbery020_x264_10`, source `robbery020`. Its cached result is copied from genuine
registered inference, without rescoring or changing any model. This illustration
is not additional benchmark evidence. Both original Robbery and Assault examples
remain misses. Normal and aggressive examples retain their wrong predictions.

Artifacts: `runs/integration/ucf_sultani_human_v1/{registration,progress,metrics,completion}.json`
and `clips/`. Final metrics SHA256:
`58eb76d828dbd1cd519437cb76f07a592d2b85fb8156d971b9be7f5d6f0e3a51`.
Full saved-result verification/supplemental-selection receipt:
`runs/ucf-crime/demo-supplemental-correct-robbery-v1.json`.
Deployment/configuration: `configs/surveillance_demo_ucf_v1.yaml` and
`configs/demo_examples_ucf_v1.json`; launch instructions in [the runbook](demo-runbook.md).
