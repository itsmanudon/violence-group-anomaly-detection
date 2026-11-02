# Phase 1: registered historical failure attribution

26 October 2026. Branch `feat/municipal-evaluation-platform`, HEAD
`4d7c27ad0d7412752307a37eff273db58f93386b`. Startup hashes of all 32 existing
changed files and five pretrained assets: `runs/indian-domain/sprint-20261026-v1/startup.json`.
No model was rescored. The analyzer checked all 991 progress-registered prediction
hashes, registration provenance and completion/metrics hash, then reproduced the
entire saved metrics dictionary exactly.

Evidence: `runs/indian-domain/sprint-20261026-v1/historical-attribution/failures.json`.
Input: `runs/integration/ucf_sultani_human_v1`, fixed threshold 0.5, three windows.
Population: 991 DCSASS weakly labeled clips, 31 held-out sources, 557 normal and
434 abnormal. These are not Indian binary violence labels or interval truth.

| Attribution | Direct observation | Denominator / qualification |
| --- | --- | --- |
| Anomaly false alerts | 224 | 557 normal clips; FPR 0.402154 |
| Anomaly/routing misses | 230 | 434 abnormal clips; all bypassed, same failures counted once |
| Window routing | 428 routed, 563 bypassed | 991 clips; bypassed actors are unexamined |
| Person/actor availability | 117 routed clips with no actors; 339 no-actor windows | 428 routed clips; 1,020 windows; not detector recall without boxes truth |
| Conditional behavior | 169 wrong, 142 right | 311 covered routed clips; macro F1 0.1888 |
| Mixed actor coverage | 56 clips | 311 covered clips have both available and no-actor windows |
| Different classes across covered windows | 83 clips | 311 covered clips; observational disagreement, not temporal truth |
| Chosen maximum softmax | Wrong median 0.949687; correct median 0.980900 | 169 wrong / 142 correct; not calibrated reliability |
| Temporal smoothing/calibration causal effect | NOT MEASURED | No eligible paired cascade validation predictions located |
| Unsupported/failed media rate | NOT MEASURED | Completed prediction population cannot establish ingestion failure rate |

Covered supports Normal/Abuse/Assault/Fighting/Robbery/Vandalism are
149/29/7/7/98/21. Correct counts are 63/0/0/0/78/1. Small class/source supports
limit interpretation. Binary confusion is `[[333,224],[230,204]]`; clip ROC-AUC
0.511794 and F1 0.473318 remain separate from UCF frame ROC-AUC 0.7441.

Examples (first by saved clip ID, not incident interpretations): normal false
alert and wrong behavior `Abuse004_x264_0`; no actors `Abuse004_x264_15`;
abnormal bypass `Abuse022_x264_10`. IDs do not determine truth; saved labels do.

Inspected `evaluation/cascade.py`, `inference/{surveillance_pipeline,anomaly_pipeline,
rgb_behavior,loading,provenance}.py`, C3D extraction and both historical protocol
documents. Legacy alerts preserve the anomaly even when actors are unavailable
or behavior predicts Normal. Selecting a representative covered window loses
the fact that other windows can be unavailable or disagree.

Proxies: missing actors could reflect viewpoint, timing, occlusion or actual
absence; class disagreement could reflect changing content. None of those causes
was directly established. Anomaly domain mismatch and training imbalance remain
hypotheses. We do not re-score held-out clips or sweep thresholds here.

Development evidence: the existing behavior selection receipt contains 692
covered validation examples and summary metrics, not joint per-window cascade
outputs. That supports loader/selection provenance, not A0/A1/A2 comparison.
No legitimate complete cascade validation artifact was found in the scoped run.
Performance selection is therefore NOT PERFORMED.

Reproduce: `.venv/Scripts/python.exe scripts/analyze_cascade_failures.py --registered-run runs/integration/ucf_sultani_human_v1 --output runs/indian-domain/sprint-attribution-new-version`.
Focused tests: **9 passed in 8.14 s** (new analyzer plus existing cascade accounting).
