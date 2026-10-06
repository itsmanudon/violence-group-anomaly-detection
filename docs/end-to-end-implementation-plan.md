# End-to-end implementation gates

The user's A–O specification is the design authority. Execution is autonomous
in the current repository and uses the existing Sultani and RGB Actor-Transformer
implementations. Each implementation increment gets targeted tests; each major
execution gets a full suite, Ruff, formatting and diff check before expensive work.
Local milestone commits follow verified phase boundaries; no remote operations.

## Current executable work

- [x] Verify branch, prior evidence, environment and mandatory-asset availability.
- [x] Convert the supplied OpenMMLab C3D checkpoint with a strict tensor/hash map;
  verify post-ReLU FC6 equivalence to pinned native upstream source on GPU.
- [x] Reproduce/fix video-directory discovery; implement raw CSV parsing and the
  six-class observable-behavior taxonomy, with explicit invalid-label quarantine.
- [x] Verify full software suite and ten-clip decode preflight.
- [x] Fully decode/hash all files in one explicit DCSASS installation root.
- [x] Inspect source identity samples, missing/invalid annotations and exact
  duplicates. Verify identical second-copy files separately without doubling the
  research population. Quarantine unresolved labels and content conflicts.
- [x] Freeze deterministic category-aware original-source splits with all published
  UCF test sources reserved. Verify zero original-source/content overlap and
  training/validation class support. Save manifests, split identity and audit report.
- [x] Detect on clip-local middle frames using existing COCO_V1 Faster R-CNN at 0.7.
  Store separate per-clip caches with actual local indices; do not invent original
  timestamps or adapt Collective's source-center uniqueness by renaming sources.
- [x] Validate coverage without GT detector metrics. Preserve all no-actor records;
  exclude them only from actor-model optimization and conditional evaluation.
- [x] Cache deterministic middle-centered ten-frame windows with short-clip edge
  replication, the existing RGB preprocessing, I3D Mixed_4f and 5x5 RoIAlign.
  Bind video bytes, sample indices, detector output, backbone and feature bytes.
- [x] Transfer projection/encoder/position/group representation from the selected
  Collective RGB checkpoint; reset the six-class group head. Retain/freeze the
  unused actor head structurally; do not manufacture actor supervision or metrics.
- [x] Derive class weights from covered training clips only, run a real five-step
  preflight/reload, then freeze optimization and select on validation macro F1.
- [x] Run seed 0, freeze selected checkpoint, then perform one held-out pass.
  Save multiclass/binary metrics, coverage accounting and structured errors.
- [x] Apply phase E's evidence gate without post-test tuning.

The E-C random initialization control wins on the predeclared validation macro F1
metric. Its comparison receipt and single held-out evaluation are complete. The
poor held-out minority-class performance is documented; keep the validation-selected
model and proceed to an honest demonstration without further test-informed tuning.

While UCF downloads, the separately identified DCSASS generic Sultani population
is source-safe and has passed its real five-step preflight. Full C3D extraction
is in progress. It supports bag evaluation, not frame-ground-truth reproduction.

## Subsequent gates

When the UCF copy is stable, validate official source assignments/annotations,
extract real C3D bags, preflight and train/evaluate Sultani. Establish threshold
policy on validation only and preserve C3D frame-unit boundaries for inference.

Then implement `SurveillancePipeline` as a small cascade over suspicious windows,
with both model outputs visible, no-anomaly bypass and anomaly-preserving no-actor
fallback. Test interval boundaries, serialization, short videos and deterministic
evaluation before real integrated inference and latency measurement.

Build the offline local Gradio interface over that API. Include upload and local
example modes, timeline/interval markers, actor overlays, probabilities and errors.
Support cached example results alongside genuine uploaded-video processing. UI
labels describe behavior and require human review. Include a limitation example.

Finish with separate subsystem/integration evidence, updated README, architecture
diagram, demo runbook and concise professor talk track. Verify the final whole
suite and clean committed working tree. Any missing mandatory input is recorded
in the execution journal; no unmeasured result is called a benchmark.
