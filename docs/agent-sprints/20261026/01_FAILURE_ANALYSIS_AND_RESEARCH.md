# Phase 1 — Cascade Failure Analysis and Focused Architecture Research

Execute this phase as directed by `00_START_HERE.md`. Read the relevant project documentation and existing predictions first. Do not repeat the full audit already completed.

## Objective

Determine which portion of the legacy five-model surveillance cascade contributes to false alarms, missed violence, poor actor coverage, and behavior misclassification; choose one implementable, well-motivated lightweight improvement for Phase 2.

Existing cascade: **C3D FC6 → Sultani MIL (32 segments) → suspicious windows → Faster R-CNN persons → I3D actor-region features → Actor-Transformer** with **Normal, Abuse, Assault, Fighting, Robbery, Vandalism** classes. Preserve it.

Historical *reported* benchmark facts to verify against local artifacts: 991 DCSASS clips, 224/557 normal clips falsely alerted (40.22%), 230/434 abnormal clips bypassed routing, 311/991 actor-covered, 0.4733 clip F1, 0.5118 clip ROC-AUC, and 0.1888 covered/routed six-class macro F1; Abuse/Assault/Fighting had zero recall. A separate UCF-Crime frame ROC-AUC of 0.7441 belongs to a different evaluation population. Do not mix these denominators or claim that these are Indian-domain metrics.

## Work A: evidence-based failure attribution

1. Locate frozen cascade prediction files, source-safe historical evaluation protocols, and the relevant scoring/window/person/actor/inference code.
2. Produce an attribution table with denominators and availability for:
   - anomaly-stage false alarms and misses;
   - suspect-window selection/bypass;
   - person detector/no-actor coverage;
   - incorrect conditional six-class predictions;
   - calibration/alert temporal aggregation effects;
   - unsupported or failed media/pipeline runs.
3. Give example video IDs **only from existing evidence**; do not infer individual incident facts from filenames.
4. Explicitly separate directly measured failures, proxy observations, and untested explanations.
5. If a necessary statistic is missing, implement only a minimal deterministic read-only local artifact analyzer and focused tests. Do not rerun full cascades without a concrete need and suitable checkpoints/inputs.
6. Protect benchmark integrity: historical held-out predictions can support descriptive failure characterization, **not** model/threshold selection. Find eligible development or validation evidence for selecting the improvement. Otherwise note that performance selection is unmeasured.

## Work B: targeted research

Compare plausible alternatives for a **~8 GiB RTX 4070 laptop** and an early-November deadline:
- smarter score aggregation, hysteresis, temporal consistency, and calibration;
- coverage-aware abstention and person-routing safeguards;
- lightweight temporal attention or video heads on existing cached embeddings;
- interaction/skeleton/pose features, if actual repo input contracts can support them;
- stronger pretrained video representations and weakly supervised anomaly methods as future possibilities.

Use verified paper/open-source references only where actual web or existing bibliography access exists. Capture author/title/year, method, benchmark/task fit, verified licensing when available, inference cost, checkpoint/dependency access, expected engineering risk, need for approved labels, and how it differs from the current approach. Mark anything unverified. Avoid hallucinated paper results or novelty claims. **No checkpoint downloads or model rebuild in this phase.**

## Work C: select one bounded implementation

Rank alternatives and select **one primary Phase-2 intervention** and **one fallback**. Decision must specify:
- supported observed failure;
- affected file/module boundaries;
- behavior contract and backward compatibility;
- what must be configurable;
- comparison/ablation and legitimate eligible validation source;
- known risks and evidence still missing;
- why it can fit approximately a working day on local hardware.

Do not ask for approval of the routine implementation choice inside the agreed scope. Do not use the historical test set as tuning validation.

## Required deliverables

- `docs/indian-domain/sprint-20261026/failure-analysis.md`
- `docs/indian-domain/sprint-20261026/architecture-research.md`
- `docs/indian-domain/sprint-20261026/model-improvement-decision.md`

Add to `phase-status.md`: artifacts inspected, actual diagnostics computed, papers actually verified, any code/tests changed, recommendation, precise Phase-2 handoff. **Proceed directly to Phase 2.**
