# Pose to Protect — Single-Agent Sequential Sprint Orchestrator

**Execution window:** October 26–27, 2026; handoff target October 28.  
**Final academic deadline:** November 10, 2026.  
**Repository:** `D:\Github Repos\violence-group-anomaly-detection` (`itsmanudon/violence-group-anomaly-detection`).  
**Reported branch:** `feat/municipal-evaluation-platform`.  
**Reported baseline revision:** `4d7c27ad0d7412752307a37eff273db58f93386b`; verify current `HEAD`, do not assume it is unchanged.

## Your instruction

You are the **sole implementation and integration agent** for this sprint. Read this file and **execute the following four files in numerical order, in the same checkout and agent session**:

1. `01_FAILURE_ANALYSIS_AND_RESEARCH.md`
2. `02_MODEL_OPTIMIZATION_AND_GPU.md`
3. `03_DATASET_QUALITY_AUTOMATION.md`
4. `04_INTEGRATION_AND_HANDOFF.md`

**These are execution instructions, not four separate jobs requiring four user messages.** Open each next phase file automatically after completing the preceding one. Do not ask for approval between phases. Do not stop after planning. Execute useful, authorized code changes, measurements, tests, and documentation within the written scope. You may keep an internal progress checklist and write intermediate phase handoffs to disk. Return **one final consolidated sprint report** when finished, unless a genuine blocker prevents all remaining meaningful work. If a phase is partly blocked, document the blocker and continue with its independent tasks and later phases that remain safe.

## Start-up rules: preserve the existing checkout

- Run `git status --short`, `git branch --show-current`, `git rev-parse HEAD`, and inventory **existing uncommitted changes**. Do not reset, clean, checkout over, stash destructively, or overwrite them. Preserve original human annotations, raw videos, pretrained checkpoints, experimental evidence, and six-class functionality.
- Read existing context, if present:
  - `docs/indian-domain/repository-audit.md`
  - `docs/indian-domain/implementation-plan.md`
  - `docs/indian-domain/research-runbook.md`
  - `docs/indian-domain/verification-log.md`
  - `docs/indian-domain/implementation-handoff-20261025.md`
  - `docs/cascade-results-ucf-v1.md`
  - `docs/cascade-evaluation-protocol-v1.md`
- Reuse earlier implementations rather than rebuilding: CSV ingest/quarantine, hashing/deduplication, source-safe split proposals, cached-feature binary training, resume, deterministic balancing, gradient accumulation, optional AMP, early stopping, calibration, and temporal alerts. Earlier handoffs reported 112 then 75 passing focused tests; treat these as **reported historical evidence**, not a substitute for tests after changed code.
- Work only in this repository. Do not start `quality-aware-violence-detection` or restore federated learning.
- Do not commit, push, merge, deploy, upload footage, overwrite experiment outputs, or publish without explicit user authorization. Local source edits and new local documentation/tests are authorized.

## Hard scientific/data boundaries

The last reported authoritative inventory was **521 videos plus 2 audio-only exclusions**, not 536 videos. The old 536 estimate does **not** prove 15 missing files. The last reported quality screening found 504 candidates (39 violence / 465 nonviolence), 17 quarantined records, and **0 source-approved Indian-domain training/evaluation clips**. Verify actual local evidence rather than treating reports as fresh measurements. `D:\violence-datasets\_manifests\v1-20261021\master-clip-inventory.csv` is a *previously reported* path and must be checked for existence.

No Indian-domain supervised training, frozen evaluation split, or official baseline on these clips until media, provenance/incident groups, duplicates, and source-safe split approval are established. You **may** test synthetic engineering contracts, run authorized inference-only diagnostics, and evaluate historical *development/validation* artifacts under their original protocols. Do not use any protected UCF/DCSASS/Indian test labels to select architecture, threshold, stopping point, or hyperparameters. Analyzing historical held-out predictions to describe failures is allowed; tuning to them is not.

Source-to-label confounding and limited independent positive incidents are major research risks. An inferred duplicate/incident grouping is **a candidate, not verified provenance**. Human-review decisions remain human decisions. Never fabricate labels, media availability, timestamps, URLs, source IDs, positive clips, approved splits, metrics, or successful tests.

## Hardware and budget

**RTX 4070 Laptop GPU only (reported approximately 8 GiB VRAM).** No institute H100, cloud GPU, distributed/multi-GPU training, or mandatory remote hardware. Profile actual available resources. Prefer reuse/caching, frozen representations, small batches, inference-mode, mixed precision where safe, checkpoint resume, and local focused tests. The previous 10-step synthetic binary-head timing was **10.93 ms**, with **60.35 MiB peak PyTorch allocation**; this is not a real video inference benchmark.

## Scope discipline and sequencing

- **Phase 1** measures/explains observed historical errors, surveys feasible methods using verified references if accessible, and selects exactly **one** bounded primary improvement plus a fallback. It must not start unapproved training or expand into a whole-repo audit.
- **Phase 2** implements the selected improvement (or a fallback supported by the evidence), makes it optional/configurable with a legacy-compatible default or mode, tests only affected paths, and profiles actual RTX 4070 components if runnable.
- **Phase 3** runs automated media/annotation/provenance screening, creates review queues, and avoids silent relabeling or inventing source identity.
- **Phase 4** integrates, verifies affected code, prepares executable *future* training configs without violating data gates, and writes the final handoff.

Write outputs under `docs/indian-domain/sprint-20261026/`; create this folder as needed. Use existing project conventions for local/ignored diagnostic CSVs, feature caches and private media. Do not commit identifiable raw footage or derived personally identifiable frames.

## Execution and interruptions

- Do not stop at the end of any phase; automatically open the next file.
- If online research is unavailable, clearly state what could not be checked, use local papers/code where available, and avoid unsupported source claims; proceed with defensible alternatives.
- If existing predictions/checkpoints/media are missing, produce explicit `NOT AVAILABLE`/`NOT MEASURED` entries, run useful unit/integration work where feasible, then continue.
- The previously observed Gradio full-suite stall is not a mandate to rerun the full suite. Run targeted changed-component tests, affected compatibility checks, and lint/format checks. Do not weaken tests or bypass genuine failures.
- Prefer a single consolidated final response. At each phase boundary, write a concise checkpoint to `docs/indian-domain/sprint-20261026/phase-status.md` with completed, not completed, tests, blockers, next phase.

## Final response requirements

Provide (1) branch + `HEAD`, (2) preserved uncommitted status, (3) phase-by-phase work performed, (4) changed files, (5) actual reproducible test outcomes, (6) measured GPU/runtime metrics versus unmeasured items, (7) scientific findings with sources/artifacts, (8) current data/split status, (9) exact next commands and teammate review queue, (10) only precise remaining blockers. Do not claim Indian-domain performance if no source-approved evaluation happened.

**Begin immediately with Phase 1 and proceed through Phase 4 autonomously.**
