# Controlled Indian-domain experiments — proposal, 25 October 2026

Deadline: 10 November 2026. These are prospective experiments, not completed
training or evaluation. Reuse the existing models, loaders, binary-head trainer,
evaluation primitives and original six-class deployment. Develop on RTX 4070.

The current implementation adds `--class-weighting`, `--sampling`, `--patience`,
`--precision` and `--accumulation-steps`, plus a validation-selected `calibrate`
subcommand. See [the latest handoff](implementation-handoff-20261025.md) for
measured local resources, evidence gates and executable commands. Only the local
RTX 4070 is in scope.

## Entry gate and comparison population

Do not run F0/A–D until the [readiness report](data-readiness-20261025.md) blockers
are cleared: approved annotations, original recording/incident identities,
source/visual overlap review, and a source-safe split. Record source permission
evidence without assuming that a private upload grants training or redistribution
rights. Preserve the 14 catalogued HCI municipal reservations outside optimization.

Freeze source-connected train/validation/test membership once, before model
selection. Target 70/15/15 by independent group only where both classes have
adequate independent sources; report the actual resulting clip and source counts.
Current groups do not justify a split or an Indian-positive generalization claim.
Do not generate unique original-source IDs from clip IDs or filename stems.

F0 and A–C must use the same approved clip IDs, labels and source groups in each
comparison. Preserve full-population routing/coverage; do not silently drop no-
person clips or unrouted windows. For a separate actor-covered analysis report its
IDs and denominator. Binary labels do not provide six-class behavior truth.

## Matrix

| ID | Hypothesis / changed variable | Fixed controls and validation selection | Status |
| --- | --- | --- | --- |
| F0: frozen existing cascade | Measure existing anomaly alerts against the new binary violence annotations | `configs/surveillance_demo_ucf_v1.yaml`, exact pinned weights, preprocessing, threshold 0.5 and three-window routing. Report anomaly score/alert metrics explicitly as an anomaly-based violence screening proxy; keep six-class outputs separate, without equating every non-Normal class to violence. Validation diagnostic first; protected test only after final freeze | Blocked by data/source gate; no new inference performed |
| A: frozen C3D + binary head | Binary supervision on the existing frozen representation may improve violence precision/recall | Existing 32-segment normalized C3D, mean aggregation, two-logit linear head, cross-entropy/Adam; original C3D frozen. Existing trainer selects validation trapezoidal PR-AUC then F1 at 0.5. First run three-step development smoke, then a prospectively bounded horizon; new output directory per setting. Reuse seeds 7,17,27 for final paired variability if source support/time permit | CLI implemented; real cache/training blocked |
| B1: threshold only | Validation calibration can reduce false positives without a large recall loss | Same selected A checkpoint, predictions and preprocessing. Small predeclared threshold grid, maximize F1 subject to proposed minimum validation recall 0.90. Fix the recall floor before sweeping; if infeasible, record failure rather than lower the floor after seeing test | Guarded calibration CLI and immutable receipt implemented/tested; real validation predictions pending |
| B2–B4: temporal rules, one at a time | Causal EMA, duration persistence or hysteresis may reject transient spikes | Same checkpoint, validation clips, thresholds and temporal score/window policy. Separate one-variable trials, then only a justified combination. Genuine presentation-time ground truth and preprocessing time mapping required. Whole-clip mean-head scores cannot provide event timing; do not fabricate a temporal stream from them | Primitives implemented; blocked by temporal review and a registered temporal scoring path |
| C: selective classifier tuning | A pretrained actor representation with binary supervision may improve interaction errors | First establish a frozen binary actor-head comparator. Compare frozen versus projection/encoder tuning with identical detected actor features, temporal windows, batches, head initialization and budget. I3D, person detector and original six-class model remain frozen/preserved. Do not compare C directly with A and attribute all differences to tuning: C also changes representation | Binary actor adapter implemented/tested; Indian actor cache, training and checkpoint orchestration pending |
| D: at most one architecture change | Select one aggregation change only if validation failure cases implicate aggregation | Written failure analysis, one hypothesis/changed variable, measured compute/parameter cost and controlled seed/settings. Keep sources, selected representation, training budget and preprocessing fixed. Defer if time/source support is inadequate | Not justified yet; no implementation or novelty claim |

Hard-negative mining is an optional separate A ablation using **training errors
only**, preserving a fixed sampling/weighting budget and source distribution.
Never mine validation/test failures into training. Match benign interaction and
camera conditions within positive-bearing collections where genuine annotations
and independent source evidence permit; do not infer new labels from filenames.
For the requested false-positive-aware experiment B, run one imbalance remedy at
a time: `--class-weighting balanced` or `--sampling balanced`. Neither discards
the valid negative pool. Treat threshold calibration and temporal rules as
separate validation ablations, not evidence that hard-negative training occurred.

## Metrics and receipts

Report precision, recall, F1, ROC-AUC, trapezoidal PR-AUC plus average precision,
FPR and confusion matrices, both pooled and by actual source/collection. Include
support, source count, actor coverage, abstentions and source-group uncertainty
where estimable. Distinguish conditional recall from recall on all violence clips.
Undefined one-class AUC is null, not zero. Accuracy is supplementary.

Record hardware/software/CUDA, baseline commit and dirty-code hashes, input CSV/
catalog/split hashes, feature/checkpoint/preprocessing hashes, hyperparameters,
seeds, actual duration, validation history, selected checkpoint and artifact paths.
Compare cold load, extraction, head and full-pipeline latency separately; synchronize
CUDA for timing. Short-clip datasets cannot supply operational camera-hour false-
alarm rates. Temporal latency requires verified onset and presentation timebase.

Validate the final checkpoint, threshold, preprocessing and window rules before
one protected-test execution. No repeated tuning against its results. Historical
UCF frame metrics and DCSASS clip/cascade metrics stay in their own populations.

## Commands after the entry gate passes

The paths below are **future approved artifacts**, not files created by this
readiness run. They intentionally fail until a legitimate frozen split exists.
Never feed `records.json` or `priority-quarantine.json` to training.

```powershell
# Verify the actual approved source/content-safe split first.
.venv/Scripts/python.exe -c "from pathlib import Path; from surveillance.datasets.indian_split import verify_split; rows, receipt = verify_split(Path('data/indian-domain/split-v1')); print(receipt['clip_counts'], receipt['class_counts'])"

# A: create a new frozen train/validation cache, then a three-step RTX smoke.
.venv/Scripts/python.exe scripts/train_indian_binary.py cache --split-dir data/indian-domain/split-v1 --checkpoint checkpoints/c3d_fc6_openmmlab_v1.pt --output data/indian-domain/c3d-cache-v1 --device cuda --batch-size 2
.venv/Scripts/python.exe scripts/train_indian_binary.py train --split-dir data/indian-domain/split-v1 --cache data/indian-domain/c3d-cache-v1 --output runs/indian-domain/c3d-development-v1 --device cuda --steps 3 --seed 7 --batch-size 8 --learning-rate 0.001

# Verify save/resume before a bounded larger run; only extend this run's horizon.
.venv/Scripts/python.exe scripts/train_indian_binary.py train --split-dir data/indian-domain/split-v1 --cache data/indian-domain/c3d-cache-v1 --output runs/indian-domain/c3d-development-v1 --device cuda --steps 6 --seed 7 --batch-size 8 --learning-rate 0.001 --resume runs/indian-domain/c3d-development-v1/last.pt
```

These development steps do not replace final experiment registration. All work
runs locally on RTX 4070. Do not reinstall the working CUDA environment or
initiate remote compute.

## Remaining-time priorities

25–27 October: resolve the priority queue and source/
permission evidence; obtain within-source positive/negative coverage. 28 October–
2 November: F0 and A, then validation-only B if eligible. 3–6 November: C only if
the frozen actor comparator is ready. D is the first item to drop. 7–10 November:
final freeze/test, source-wise limitations and professor handoff, only if the data
gate is satisfied. A blocked readiness report is preferable to a contaminated
result under the deadline.
