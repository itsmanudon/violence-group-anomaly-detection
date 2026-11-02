# E0–E4 future experiments: exact interfaces and honest limits

Machine-readable plan: `configs/indian-domain/sprint-20261026.json`. It is a
declarative research plan, **not a new trainer config format**. Command strings
match actual CLI help. Every `split-approved-v1`, cache and real run path below
is prospective; none exists or has been approved. No experiment in this file
was launched on Indian footage. No H100/cloud compute dependencies.

Common gate: resolve human review, actual original recording/shared incident,
rights and duplicate/historical source evidence; retain reservations; use existing
import and `freeze_indian_split.py freeze` with evidence-bound review. Check actual
class/source support rather than imposing clip ratios. Freeze test membership
once before model selection. Do not invent a review JSON with approvals merely
to make the command run. Existing `verify_split` rechecks membership, source/content
groups, inputs and media. No test inference command is added in this sprint.

## E0: frozen pipeline baseline

New minimal `scripts/evaluate_indian_frozen.py run` reuses the existing loader and
`verify_split`. Only validation videos are inferred. Registration binds split,
config, code, five model hashes, seed/environment and fixed anomaly-score proxy.
Metrics include precision, recall, F1, ROC/PR-AUC/AP, FPR, confusion and per-source
performance. Predictions retain the original six-class behavior outputs, but no
six-class ground truth is manufactured from binary labels. Partial failures keep
incomplete outputs; completion is written only after input/model rechecks.

```powershell
.venv/Scripts/python.exe scripts/evaluate_indian_frozen.py run --split-dir data/indian-domain/split-approved-v1 --config configs/surveillance_demo_ucf_v1.yaml --output runs/indian-domain/e0-approved-val-v1
```

Expected new artifacts: registration, per-clip result/hash records, metrics,
completion. Frozen pretrained anomaly threshold 0.5, three behavior windows,
C3D batch four; no selection/optimization. This is a disclosed anomaly-versus-
violence proxy baseline, not an assertion that anomaly equals violence. Actual
RTX demo loading/inference passed; long Indian clip capacity remains unmeasured.

## E1: frozen C3D binary head

```powershell
.venv/Scripts/python.exe scripts/train_indian_binary.py cache --split-dir data/indian-domain/split-approved-v1 --checkpoint checkpoints/c3d_fc6_openmmlab_v1.pt --output data/indian-domain/c3d-approved-cache-v1 --device cuda:0 --batch-size 2
.venv/Scripts/python.exe scripts/train_indian_binary.py train --split-dir data/indian-domain/split-approved-v1 --cache data/indian-domain/c3d-approved-cache-v1 --output runs/indian-domain/e1-seed7 --device cuda:0 --steps 3 --seed 7 --learning-rate 0.001 --batch-size 8 --patience 20 --precision fp32 --sampling uniform --class-weighting none
# After successful real-data smoke, resume the SAME run/config, extending horizon only.
.venv/Scripts/python.exe scripts/train_indian_binary.py train --split-dir data/indian-domain/split-approved-v1 --cache data/indian-domain/c3d-approved-cache-v1 --output runs/indian-domain/e1-seed7 --device cuda:0 --steps 100 --seed 7 --learning-rate 0.001 --batch-size 8 --patience 20 --precision fp32 --sampling uniform --class-weighting none --resume runs/indian-domain/e1-seed7/last.pt
.venv/Scripts/python.exe scripts/train_indian_binary.py calibrate --split-dir data/indian-domain/split-approved-v1 --checkpoint runs/indian-domain/e1-seed7/best.pt --output runs/indian-domain/e1-val-threshold-v1 --min-recall 0.9 --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

Cache contains train+validation only, mean [104,117,128], RGB, 32 L2-normalized
4096-D segments. Two-output head, Adam, max grad norm one, validation PR-AUC then
F1 checkpoint selection, early stopping patience 20 steps. Recall floor 0.9 is a
prospective constraint, not achieved evidence. If no candidate meets it, calibration
fails. Register seeds/budgets before launch; do not extend after inspecting test.
Artifacts: cache registration/feature hashes, run registration, best/last weights,
history, source metrics, calibrated threshold receipt. Existing tests verify exact
CPU/CUDA resume and AMP/accumulation, not real Indian feature extraction. FP32
default retained; no real training throughput or backward backbone memory claim.

## E2: hard-negative-aware training, conditional

No Indian train prediction scores or reviewed hard negatives currently exist.
Register train-only negative ranking/budget, bind clip/model/cache hashes, retain
all legitimate negatives, and change only per-clip sampling against E1. Validation
selects checkpoint/threshold; validation/test examples must never be mined.
The current sampler supports uniform and class-balanced sampling, **not reviewed
per-clip hard-negative weights**. There is deliberately no fabricated CLI option
or runnable E2 launch command. Plan records exact missing implementation and
inputs. A separate `--class-weighting balanced` run is an available imbalance
control, not E2 and not evidence that hard negatives were reviewed. Defer until
approved E1 failures justify that bounded extension.

## E3: chosen review-evidence comparison

```powershell
.venv/Scripts/python.exe scripts/evaluate_indian_frozen.py review --split-dir data/indian-domain/split-approved-v1 --baseline runs/indian-domain/e0-approved-val-v1 --output runs/indian-domain/e3-review-val-v1
```

Verifies full exact validation membership, split/model/result identities, original
binary labels and video hashes; uses the **same saved E0 results**, no rerunning
backbones. A0 legacy/A1 coverage+agreement/A2 coverage-only. CPU state counts,
abstentions and overhead; original binary precision/recall/F1/AUC/FPR/confusion
are identical by construction. There is no binary false-alert reduction claim.
If genuine six-class validation truth is available, the separate
`compare_cascade_review.py` command reports conditional six-class macro F1 and
all-population class recall/coverage. Never use held-out historical truth for tuning.

## E4: selective actor classifier tuning, conditional

Existing API `BinaryActorAdapter(system, classifier_tuning=True)` can unfreeze
RGB projection and encoder while preserving detector/I3D and original six-class
heads. Its gradient/shape behavior is tested. It lacks a registered real actor
cache/trainer/dedicated binary save-loader orchestration. No runnable E4 CLI or
trained actor model is claimed. Require approved matched actor validation baseline,
failure evidence and an RTX one-batch backward/save-resume smoke before choosing
the exact optimizer budget/learning rate. Do not substitute C3D-head training for
actor fine-tuning. Skip E4 if these prerequisites cannot be met before deadline.

Final testing is outside E0–E4 model selection: freeze validation-selected model,
all preprocessing, routing, threshold/assessment policy and hashes; independently
verify protected test sources; register one test evaluation and retain its result
without tuning again. Continuous camera-hour burden and event latency require
representative continuous footage and complete temporal truth, still unavailable.

Timing supplement: 341 presentation-versus-frame/FPS clock flags are recorded
separately. Clip-only E1 labels do not acquire interval semantics from this
diagnostic. Temporal localization/latency remains blocked until sample clocks are
validated; the flag does not itself add binary quality quarantines.
