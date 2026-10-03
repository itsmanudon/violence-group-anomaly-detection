# Milestone 2B verification record

## Initial state and execution

Initial branch was clean `feat/actor-transformer-baseline`, commit `d64d8f7`,
tracking its matching origin branch. The requested branch did not exist.
Read the Milestone 2A documents, README/data instructions, configs, model,
dataset, batching and inference/extraction interfaces. Ran the complete existing
suite: **123 passed before edits**. Created `feat/automatic-actor-detection`.

Followed [milestone-2b-plan.md](milestone-2b-plan.md): independent detector and
cache contract; deterministic filters; assignment and detection metrics; detected
input adapter; shared extraction; empty-aware inference and paired evaluation;
tests and documentation. No commit, push, merge, remote change or PR was made.

## Delivered behavior

- PersonDetector protocol; validated CPU absolute-pixel xyxy results and v1 JSONL.
- Optional local COCO Faster R-CNN ResNet-50 FPN v1, strict state-dict loading,
  explicit no-download construction and checkpoint/config/version provenance.
- Confidence/person/geometry filtering, clipping, size/area limits, deterministic
  greedy NMS, confidence cap and canonical center-x/center-y actor ordering.
- Threshold-gated assignment maximizes eligible match count, then IoU, using
  SciPy linear assignment. IoU >= threshold, threshold in (0,1].
- Matched-only action labels; extras remain unknown but participate in groups.
- Absolute-to-normalized adapter reuses the existing pose/RGB extraction and
  RoIAlign. No Actor-Transformer/model/loss/trainer/GT batching math changes.
- Ordered-box/source/file fingerprints protect detected-feature caches.
- No-person scenes return null group predictions and remain in evaluation.
- Detection precision/recall/F1, matched IoU, counts/misses/extras/mean actors.
- One-checkpoint GT/detected comparison, paired populations, explicit supported
  and all-scene group metrics, actor and scene coverage, and attention mappings.

Full policies, formats, API usage and real-data commands are in
[person-detection.md](person-detection.md), [README](../README.md#milestone-2b-automatic-actors-and-box-robustness)
and [data/README](../data/README.md#person-detections-and-feature-caches-milestone-2b).

## Final verification

| Check | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -q --basetemp .pytest_cache/m2b-final` | **234 passed**, 81 upstream TorchScript warnings, 51.12s |
| `ruff check .` | Pass |
| `ruff format --check .` | Pass, 100 Python files formatted |
| `git diff --check` | Pass (ordinary LF/CRLF notices only) |
| Original 123 tests | All retained and passing, no test edits |
| Model/trainer/GT dataset/batching diff | Empty |

111 new test cases cover results/config/records, clamping and invalid geometry,
person/confidence/NMS/cap behavior, deterministic permutations, assignment
thresholds/adversarial cardinality, label transfer, metrics, no-person scenes,
native center-frame detection, local structural backend, cache corruption/source
and image-size mismatch, pose and RGB CLI extraction, coordinates and attention,
and the same-checkpoint robustness evaluator. An exact-threshold float32 rounding
inconsistency was caught and fixed with consistent float64 GT coordinate conversion.
An original sidecar serialization regression was caught by an unchanged test and
fixed by preserving original GT coordinates.

Environment remains Python **3.13.5**, torch **2.14.1+cpu**, torchvision
**0.29.1+cpu** on Windows. No Python compatibility failures. SciPy is now an
explicit dependency (already present transitively). GPU behavior is untested.
TorchScript warnings come from existing/local synthetic backbone export tests;
the new detector uses state dictionaries. Official COCO FrozenBatchNorm is
restored explicitly because modern torchvision otherwise chooses BatchNorm when
both pretrained flags are None. Actual pretrained weights were not downloaded
or exercised; structural and injected-backend tests do not prove real accuracy.

## Synthetic workflows

```powershell
python scripts/smoke_detection.py --output outputs/m2b_final
python scripts/compare_actor_boxes.py --checkpoint outputs/m2b_final/run/last.pt --manifest outputs/m2b_final/gt.jsonl --detections outputs/m2b_final/detections.jsonl --device cpu --output outputs/m2b_final/cli_comparison.json
python scripts/infer_group_activity.py --checkpoint outputs/m2b_final/run/last.pt --manifest outputs/m2b_final/gt.jsonl --box-source detections --detections outputs/m2b_final/detections.jsonl --device cpu --attention --output outputs/m2b_final/cli_inference.json
```

All commands passed. A: mock detection on a synthetic RGB center frame, crop-mean
fixture features, and Actor-Transformer prediction. B: GT and perturbed detector
boxes, matching/metrics and one-checkpoint comparison. C: zero people returns
`no_actors_detected` without invoking attention. These fixtures are explicitly
synthetic, not HRNet/I3D or pretrained detector substitutes for research.

The test split contains three scenes, six GT actors, four detections, three
matches, three missed actors and one unmatched detection. Prediction actor counts
are `[2,2,0]`; only the three matched actors enter actor accuracy/F1. The empty
scene remains in the all-scene denominator. These are fixture arithmetic checks,
not real dataset performance measurements. Real pose/RGB adapter contracts also
pass independent tests with synthetic local feature-only exports.

## Exact file inventory

Modified:

```text
README.md
data/README.md
pyproject.toml
scripts/_actor_features.py
scripts/infer_group_activity.py
```

Added:

```text
configs/person_detector.yaml
docs/milestone-2b-plan.md
docs/milestone-2b-verification.md
docs/person-detection.md
scripts/compare_actor_boxes.py
scripts/detect_people.py
scripts/evaluate_detector.py
scripts/match_actor_boxes.py
scripts/smoke_detection.py
src/surveillance/datasets/detected_actors.py
src/surveillance/detection/__init__.py
src/surveillance/detection/config.py
src/surveillance/detection/matching.py
src/surveillance/detection/person_detector.py
src/surveillance/detection/records.py
src/surveillance/detection/torchvision_detector.py
src/surveillance/evaluation/box_robustness.py
src/surveillance/evaluation/detection_metrics.py
src/surveillance/inference/detected_group_activity.py
tests/test_actor_box_matching.py
tests/test_box_robustness.py
tests/test_detected_actor_inputs.py
tests/test_detected_extraction.py
tests/test_detection_metrics.py
tests/test_detection_records.py
tests/test_detector_backend.py
tests/test_person_detector.py
```

## Remaining dependencies and next milestone

Real runs require Collective frames/annotations, a trained Actor-Transformer,
GT and detected actor features or vetted HRNet/I3D exports, and local compatible
detector weights or finalized precomputed detections. The backbone export contract
from Milestone 2A remains in force. Source fingerprints do not hash image bytes;
preserve dataset versions and source contents. Imported cache preprocessing still
requires trustworthy provenance. Unmatched detections may be unannotated people.

Recommended Milestone 2C: acquire/version the actual artifacts, run annotated-box
and detected-box Collective benchmarks, tune thresholds on validation only, freeze
the test protocol and report class-level errors plus coverage. Use measured gaps
to decide whether detected-box fine-tuning or annotation work is needed before
surveillance adaptation. The cascade and UI remain later work.

Final branch: `feat/automatic-actor-detection`, based on `d64d8f7`. Five modified
tracked files and 27 new files, all unstaged. Test/output artifacts are ignored.
No commit or push was performed.
