# Milestone 2B plan

Initial repository: clean `feat/actor-transformer-baseline` at `d64d8f7`, tracking
origin. All 123 existing tests passed before edits. Created the requested
`feat/automatic-actor-detection` branch. No commits or pushes are authorized.

1. Implement detector-independent absolute-pixel xyxy results, strict JSONL
   caches, deterministic filtering and local TorchVision Faster R-CNN inference.
2. Implement threshold-gated maximum-cardinality/maximum-IoU bipartite assignment,
   matched labels and detector precision/recall/coverage metrics.
3. Adapt detected boxes to the existing normalized actor inputs, sharing raw
   HRNet/I3D extraction and preserving source identity and actor order.
4. Add explicit empty-scene inference and same-checkpoint GT/detected comparison.
   Report matched-only actor accuracy alongside coverage and scene abstentions;
   never silently drop failed scenes or infer labels for unmatched detections.
5. Verify filtering, coordinates, cache alignment, extraction, attention mapping,
   complete model workflows, and previous functionality. Document all defaults.

The transformer, positional encoding, pooling, heads and losses remain unchanged.
The detector uses only the center frame (index 5); tracking is unnecessary for
the existing center-box RoI pipeline. Precomputed detections/features work offline.
Local model parameters are never downloaded implicitly. Random detector weights
are reserved for explicitly requested structural tests, never research inference.

Detector records are separate from annotated ActorRecord manifests so original
labels and empty detection failures cannot be overwritten or fabricated. Feature
caches retain ordered detection fingerprints and source metadata. GT labels are
joined at evaluation time, never used as detector input. All retained detections
contribute to group prediction; only IoU-matched detections enter actor metrics.

The comparison includes common-supported-scene metrics and all-scene accuracy
with abstentions counted as incorrect, plus explicit coverage. Matched actor
metrics use the same GT actor subset for paired deltas; full-GT actor metrics
are still reported independently. No real benchmark numbers are invented.
