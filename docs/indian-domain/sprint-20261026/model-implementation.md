# Phase 2 implementation

Added `inference/cascade_review.py`: immutable named policy plus an opt-in
`ReviewAwarePipeline` wrapper. Default legacy mode returns the original object.
Coverage/consistency modes add a sidecar with explicit abstention reasons and
preserve all original alerts, boxes, six probabilities, scores and provenance.
No detector weights, model heads, preprocessing, frontend, default routing or
thresholds changed. No new dependency, checkpoint or architecture training.

`evaluation/cascade_review.py` and `scripts/compare_cascade_review.py` require
complete validation membership matched to an all-split source manifest, video
hashes, labels and exact model/preprocessing identity. Cross-split source leakage,
test/undeclared predictions and mismatches are rejected. The report retains
abstentions in a six-by-seven confusion matrix, coverage, conditional errors and
all-clip recall. Raw binary alert metrics cannot change under this assessment.

Validation and CPU boundaries: missing actors, partial actor coverage, differing
classes, invalid numerical probabilities, class/shape/count errors, nonempty time
intervals, stable repeated outputs, unchanged legacy serialization, test refusal.
Existing six-class loading and cascade regression paths remain intact.

Focused run: **41 passed in 13.59 s**, no failures/skips. Initial lint found
overlong lines; formatted code and fixed the remaining descriptive string. Final
lint/format verification is recorded in the integrated handoff. Runtime profiler
loaded all five real pinned checkpoints on CUDA and reproduced scores, boxes,
probabilities and alerts exactly across five repetitions.

Missing/invalid input video and checkpoint handling remains in the unchanged
loader/decoder; the wrapper does not manufacture a result for an invalid video.
Assessment works on offline selected windows, including ranked/nonchronological
ones. It does not impose causal smoothing, infer continuous streaming coverage,
or reset a streaming clock. Existing temporal primitives remain separate.
