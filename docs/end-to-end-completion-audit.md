# End-to-end completion audit

All twenty requested MVP deliverables have concrete evidence below. Completion
means a reproducible research implementation with measured limitations, not
reliable violence recognition. Predictions require human review. The frozen
UCF frame, standalone behavior and cascade populations are reported separately.

| Required deliverable | Verified evidence |
|---|---|
| 1. Real Sultani detector | All 1,853 retained UCF videos fully decoded to verified C3D bags; fixed 20-epoch seed-0 training; validation-selected epoch 20; one registered 290-video frame test, ROC-AUC 0.7441 |
| 2. Real DCSASS RGB Actor-Transformer | Collective transfer and one controlled random initialization trained; random control selected solely by validation macro F1; group-only six-class supervision, frozen cached I3D |
| 3. Automatic people | Local COCO_V1 Faster R-CNN, confidence 0.7; real middle-frame caches and actual interval detections verified with immutable detector hash |
| 4. Source-safe DCSASS protocol | 6,491 valid human clips / 203 sources, 140/32/31 source splits; zero source/content overlap; binary-normal mapping and exclusions exhaustively tested |
| 5. Held-out behavior metrics | 686/991 covered; accuracy 0.510204, macro F1 0.214047, balanced accuracy 0.228669; fixed-threshold binary ROC-AUC 0.630723; all confusion/per-class/errors retained |
| 6. Unified cascade | SurveillancePipeline uses real C3D/Sultani → suspicious windows → Faster R-CNN/I3D/RoIAlign/Actor-Transformer; all five pinned assets; full registered 991-clip run |
| 7. Video timeline | 32 scores with exact C3D-unit ranges; actual browser/API plot; fourteen predeclared UCF annotation-overlay illustrations |
| 8. Suspicious localization | Fixed threshold 0.5, at most three distinct analyzed windows; frame/time boundary tests and real UCF frame evaluation; no DCSASS interval-GT claim |
| 9. Observable behavior | Six-class probabilities, actor counts, interval evidence; no identity or character inference; unsupervised actor head outputs not reported |
| 10. No-actor fallback | Real validation abstention and registered 117 routed no-actor clips preserve generic anomaly alerts; 563 bypasses remain unexamined rather than detector failures |
| 11. Offline Gradio | One local loopback process, share false, analytics off, local/system assets; uploaded-video on-demand inference and explicit example caches; six real API checks including uploaded Robbery |
| 12. Bundled examples | Nine byte-pinned real local clips/caches; all eight original cases, including failures, retained plus disclosed post-hoc correct Robbery illustration; no synthetic benchmark clips |
| 13. End-to-end tests | 542 tests pass; parsing/leakage/taxonomy/cache/transfer/group loss/routing/no actors/short videos/serialization/determinism/input races/receipt preservation covered |
| 14. README | Professor entry point, architecture diagram, measured Collective/DCSASS/UCF/cascade evidence, install/train/infer/demo commands and limitations |
| 15. Runbook | docs/demo-runbook.md: asset verification, one-command launch, offline/cached/live procedure, CPU fallback and useful errors |
| 16. Talk track | docs/professor-demo-talk-track.md: two papers, data, normal/correct case/disagreement, evidence and future work |
| 17. Limitations | Seed scope, Collective imbalance, converted weights, detector errors, source structure, missing actor/interval labels, weak supervision, domain shift, false alarms, UCF duplicate weighting |
| 18. No fabricated metrics | Reported values checked against completed real receipts; preflight/synthetic tests/curated demonstrations clearly separated |
| 19. No hidden test tuning | Frozen source/initialization/UCF selection and threshold; canonical one-pass frame/cascade registrations; crash recovery verified saved data and scored pending clips only |
| 20. Local reproducibility | Completed assets/receipts retained locally; all quality gates pass; read-only review; local milestone commit only, no push or merge |

## Evidence and scope

- [UCF baseline](sultani-ucf-results-v1.md): 1,309 train / 254 validation / 290
  test, no cross-split source/content overlap. Authors' test entries retained,
  including one duplicate pair (289 unique contents). Validation bag ROC-AUC
  0.9109; frame ROC-AUC 0.7441, threshold precision 0.1912 and F1 0.2794.
- [DCSASS behavior](dcsass-surveillance-results-v1.md): 350 covered correct,
  336 incorrect, 305 abstained. No correct held-out Abuse, Assault or Fighting
  prediction. No actor-level accuracy or unsupported detector GT metrics.
- [Registered cascade](cascade-results-ucf-v1.md): 991 clips / 31 protected
  sources; fourteen official UCF test sources and seventeen excluded from UCF
  optimization. Clip-alert F1 0.4733 / ROC-AUC 0.5118. Covered routed behavior
  142 correct, 169 wrong, 680 abstained; macro F1 0.1888. Weak clip labels do not
  establish correctness of individual intervals.
- Final registered result hashes and original input/source/label/provenance
  verified; saved metric recomputation matched exactly. Original failed log and
  before/staged progress receipts preserved under the canonical run's recovery/.
  Metrics SHA256 58eb76d828dbd1cd519437cb76f07a592d2b85fb8156d971b9be7f5d6f0e3a51.
- Default config configs/surveillance_demo.yaml matches the UCF version. Older
  DCSASS deployment/config/cache/report remain intact. Real GPU repeat exact;
  full CPU inference completes. The nine-example deployment's API receipt is
  runs/ucf-crime/demo-api-ucf-final-v2.json. Uploaded Normal bypass and uploaded
  Robbery full actor inference both passed; missing-input error returned safely.
- Final full suite receipt runs/ucf-crime/pytest-final-mvp-v1.log: 542 passed,
  325 upstream warnings, 145.48 seconds. Ruff and format checks passed (202 files).
  Receipt-preservation/atomic-retry tests include failed-before-fix evidence.
- Browser confirms actual false alert, disagreement, timeline and overlays;
  final correct-case/live screenshots are recorded in the execution journal.
  Automated browser upload remains limited by extension file-access permission;
  server upload is tested and the runbook retains a manual pre-class playback check.

Raw footage remains at its existing location, including E:\anomaly-detection-dataset-UCF.
The installed Windows demo needs local examples/checkpoints/receipts, not the full
raw UCF archive. Windows virtual environments are not portable to a Mac; the
validated presentation environment remains this RTX 4070 Laptop system.
