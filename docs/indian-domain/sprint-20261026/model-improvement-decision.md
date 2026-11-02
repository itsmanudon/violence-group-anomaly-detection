# Phase 1 decision: optional evidence-aware behavior assessment

Primary: add a deterministic **coverage-and-consistency review assessment** as a
separate optional wrapper around the existing cascade result. Rationale is the
existing output-contract limitation: a single chosen window can hide missing
coverage or contradictory classes elsewhere. Historical failures describe that
limitation; they do not establish that abstention improves recognition.

No eligible full-cascade validation population exists locally in the scoped
artifacts, so no performance-based policy selection is possible. The intervention
is an engineering/review-evidence experiment, not a trained architecture change
or a promised reduction in binary false alarms.

Contract: `legacy` returns original results unchanged. `coverage_consistency`
adds a separate assessment: retain the original anomaly alert, abstain from a
single six-class suggestion if any routed window lacks actors or covered classes
disagree. Even accepted suggestions require human review and do not claim binary
violence, intent or culpability. Bypassed clips say unexamined, not no persons.
No thresholds are selected; softmax is not treated as correctness confidence.
Invalid numerical probabilities/intervals fail explicitly; no temporal duration
or stream-latency guarantee is introduced by offline windows.

Fallback: `coverage` mode checks only complete routed actor coverage and retains
the legacy representative class. It is also the single-factor A2 ablation. A0 is
legacy; A1 is coverage+agreement. Settings are named, serialized and separate from
frozen cascade provenance, so old caches/models/API schemas remain intact.

Files: new `inference/cascade_review.py`, `evaluation/cascade_review.py`, CLI and
focused tests. Wrapper avoids modifying historical loaders, checkpoints or UI.
Use `evaluation/cascade.py`'s representative-window rule. At most three tiny
six-class vectors, no new GPU model, feasible in a day. Invalid/no/short clips
remain subject to existing decoder; assessment operates on valid outputs only.

Acceptance: no input mutation; byte-equivalent legacy object serialization;
raw anomaly decisions preserved in every mode; deterministic outputs; explicit
no-actor/partial-coverage/disagreement states; validated probability/boundary
contracts; comparison refuses test records and mismatched populations/provenance.
Future matched validation comparison reports six-class risk/coverage, all-population
abstentions, full-population alert metrics and paired clip/source support. Binary
alert metrics should be identical by construction. No selective metric substitutes
for unconditional coverage. Calibration/continuous false alarms remain unmeasured.

Risk: unanimity is conservative and can reject correct changing activities;
agreement can also be unanimously wrong. It cannot repair the 230 historical
routing misses, person detector recall, six-class weights or data confounding.
Adoption requires approved paired validation; until then legacy remains default.
