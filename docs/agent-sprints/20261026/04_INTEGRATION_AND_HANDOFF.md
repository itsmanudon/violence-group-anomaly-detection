# Phase 4 — Focused Integration and November Experiment Readiness

Execute after Phases 1–3. This is the **end of the same autonomous sequential run**, not a fresh broad review.

## Work A: preserve and reconcile

- Review the working tree and the three phase documents; separate existing pre-sprint uncommitted code from new sprint changes as best the local diff permits.
- Verify that original five-model inference, six classes, historical checkpoints, cached features, annotation source files, and previous experiments have not been overwritten.
- Fix only concrete integration defects in the affected surface; do not launch architecture expansions or a generic audit.

## Work B: targeted verification

Run focused local tests for affected components: old/new inference mode, temporal scoring/decision boundaries, actor coverage behavior, six-class compatibility, binary head/resume/calibration compatibility where touched, cache/checkpoint compatibility, local RTX 4070 smoke tests if runnable, input validation, duplicate/source grouping proposals, and quarantines. Reuse unaffected prior evidence only when inputs truly remain unchanged. Run touched-file lint/format/whitespace. Investigate actual failures; don't bypass or weaken tests. The historical Gradio full-suite stall alone is not reason to rerun everything.

## Work C: scientific evidence summary

Consolidate measured outcomes and explicit uncertainty:
- historical UCF-Crime vs DCSASS **different** evaluation populations;
- observed cascade-stage failure attribution;
- verified architecture references and selection rationale;
- exactly what model behavior changed and legacy switch;
- actual component latency/VRAM results and benchmarks not runnable;
- controlled validation comparisons, or **NOT MEASURED** if labels/eligible data missing;
- current 521-video / two-audio inventory and 504/17 quality status **only if reconfirmed**, unknown source/incident identities, candidate near-duplicates, source–label bias, and 0 approved clips if still true.

Never create misleading Indian-domain accuracy/final test results or false camera-hour alert rates from short clips.

## Work D: future experiment readiness — config, not execution

Prepare precise runnable **templates**, following real repo CLI/config conventions (verify commands with `--help` or source), for:
- **E0:** frozen existing pipeline baseline;
- **E1:** cached C3D + lightweight binary head;
- **E2:** hard-negative-aware training;
- **E3:** chosen temporal/alert improvement comparison;
- **E4:** selective fine-tuning, only if justified by validation and feasible in 8 GiB VRAM.

For each: required approved manifest, grouped split ID, checkpoint/features, hyperparameters and seed, local RTX 4070 capacity, specific commands, expected artifacts, validation-only model selection, and final untouched test protocol. All Indian-domain training and official evaluation should remain on hold until verified source/incident grouping and source-safe split approval exist. Do not insert H100/cloud-distributed paths.

## Work E: write the single authoritative handoff

Create: `docs/indian-domain/sprint-20261026/sprint-handoff.md`.

Include:
1. Branch and exact `HEAD`, `git status`, and uncommitted preservation status.
2. Phases completed versus limited, and exact changed files.
3. Tests passed/failed/not run with commands and result locations.
4. Measured RTX 4070 profiles with configuration; missing measurements labeled.
5. Model improvement, failure it addresses, ablation and validation evidence (if any).
6. Dataset review/provenance queue artifact paths, counts, and recommended human actions.
7. Exact next **safe** executable command, then commands contingent on explicit data approval.
8. Short October 28–November 10 priorities, classifying ready now / needs human provenance or labels / optional.
9. Specific remaining blockers, not speculative new review gates.

Do not commit/push/merge/deploy. Provide one concise final message to the user with the above, not a new set of instructions requiring another agent.
