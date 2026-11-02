# Phase 2 — Selected Model Improvement and RTX 4070 Measurement

Execute immediately after Phase 1 in the *same* checkout. Read `model-improvement-decision.md` and implement the highest-ranked supportable improvement, rather than beginning another research survey.

## Scope and compatibility

Retain original C3D / MIL / Faster R-CNN / I3D / Actor-Transformer models, historical checkpoints and their hashes, current default/legacy mode, six behavior classes, API and frontend compatibility, and existing Indian binary-train infrastructure. Prefer extension of current inference, calibration, and temporal alert modules. The chosen improvement must be switchable/configurable, and its behavior must be deterministic for the same inputs. Do not modify original historical metrics or overwrite model checkpoints.

If Phase 1's selection depends on unavailable data, choose its documented fallback where supportable; never silently invent metric improvements. A model-comparison harness plus tested implementation is still valuable when scientific validation is blocked.

## Work A: implementation

- Identify smallest relevant codepaths; implement only the selected intervention and absolutely required correctness fixes.
- Preserve old behavior as an accessible baseline and avoid changing default thresholds merely to improve-looking metrics.
- Handle short/empty clips, missing actor detections, numerical bounds, invalid or absent checkpoints, and streaming/window boundary behavior appropriate to the chosen feature.
- Add focused unit and affected integration tests for both legacy and selected modes.

## Work B: comparison/ablation

Prepare and, if eligible validation artifacts exist, run matched comparisons:
- **A0** legacy scoring/alert behavior;
- **A1** selected improvement;
- **A2** one small ablation if scientifically meaningful.

Use identical decoding, clips, models, and feature caches. Choose tuning on legitimate development/validation data, not protected test examples. Keep previously frozen UCF/DCSASS test evidence unchanged. Report precision, recall, F1, PR-AUC, ROC-AUC when appropriate, specificity/false alarms, coverage, and latency with denominators and uncertainty where meaningful. If no legitimate validation exists, report **NOT MEASURED** instead of claiming a recognition improvement. Synthetic boundary tests are engineering evidence only.

## Work C: RTX 4070-only profiling

**No H100, cloud compute, multi-GPU, or distributed training.** The laptop RTX 4070 was reported with approximately 8 GiB VRAM; discover actual available memory and stack.

Measure where checkpoints and permissible clips actually exist:
- PyTorch version, CUDA, device name, total/available memory;
- peak allocated/reserved PyTorch memory, and device memory if available;
- cold checkpoint/model start-up separately from warm inferencing;
- C3D extraction, Sultani head, person detector, I3D, actor transformer, end-to-end inference;
- cached versus uncached execution, when runnable;
- repeated-run median/p95 latency and throughput with clip frame count, resolution, temporal sample rate, batch size, and warmup method.

Mark unavailable components **NOT MEASURED**. The prior **10.93 ms / 60.35 MiB** measurement was 10 synthetic binary-head training steps, not video inference—do not present it as full-model latency.

Reuse existing AMP, deterministic seed/resume, early stopping, gradient accumulation and feature caching rather than duplicating. Add only measured worthwhile optimizations (e.g., `inference_mode`, safe mixed precision, avoiding needless host/device transfers or reloads). Profile after the changes. Do not perform Indian supervised training.

## Work D: verification and deliverables

Focused tests should cover logits/probability validity, temporal window boundaries, threshold hysteresis if used, no-actor/no-person fallback, existing six-class behavior, deterministic predictions, checkpoint/cache compatibility, CPU/CUDA smoke, missing dependencies, and bounded VRAM. Run lint/format checks on touched files. Do not re-run the known stalled full Gradio suite unless a reproducible affected failure requires it.

Write:
- `docs/indian-domain/sprint-20261026/model-implementation.md`
- `docs/indian-domain/sprint-20261026/gpu-benchmarks.md`
- `docs/indian-domain/sprint-20261026/model-comparison.md`

Every asserted performance effect must have exact dataset/split/checkpoint/config/measurement provenance. Explicitly separate engineering tests, historical diagnostics, eligible validation comparisons, and final held-out evidence. Add phase checkpoint; **then proceed to Phase 3 automatically.**
