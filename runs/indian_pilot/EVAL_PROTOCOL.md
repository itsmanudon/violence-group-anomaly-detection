# Indian pilot — one-time test evaluation protocol

Committed before any test feature is extracted or any test clip is scored. Test is
evaluated exactly once with `scripts/pilot_test_eval.py`, which refuses to overwrite its
output. Split v1, freeze hash `ae9a0fab918eb20992b3202093a0982af16f7e3bc28a57c9e28ff2598818b32f`:
test = 72 clips (6 violence, 66 non-violence), 17 groups.

## Framing (decided 2026-11-03 after validation; see VALIDATION.md and DECISIONS.md D17–D18)

The pilot is presented as a **confound finding**: the pipeline runs on Indian footage and a
head can be fine-tuned, but with the current data it cannot be told apart from a detector of
which source a clip came from. Test numbers are reported with that framing, never as
evidence of Indian violence detection.

## Test features

C3D FC6 32-segment bags for the 72 test clips: same checkpoint
(`beac4ff0…`), same preprocessing, MPS, non-strict decoding (D5, D6), extracted by
`cache_c3d(..., splits=("test",), test_protocol=<this file>)` into
`runs/indian_pilot/features/c3d_test_v1/`.

## Models and fixed operating points

| id | model | threshold |
|---|---|---|
| HEAD | `experiments/violence_f100_s7/best.pt` (deployed; fixed in D15) | **0.40** (validation calibration: max F1 s.t. recall ≥ 0.8) |
| CURVE | `violence_f{025,050,075,100}_s{7,8,9}` (12 heads) | threshold-free metrics only |
| MITIG | `mitigation_unknown_only_s{7,8,9}` | threshold-free; documented-Indian false-positive rate at 0.40 |
| PROBE | `probe_geography_all_s7` (target: unknown geography) | threshold-free |

## Baselines

| id | baseline | operating point |
|---|---|---|
| SULTANI | zero-shot UCF Sultani clip score (max of 32 segments) on the same test bags | 0.5 (the cascade's fixed routing threshold) |
| CASCADE | zero-shot five-stage cascade alert (`final_alert.state != no_anomaly`), CPU, via `scripts/pilot_cascade_baseline.py`; refused clips = no alert | binary |
| SOURCE | source-only rule: score 1 if the clip's collection contains violence in train (HCI-User-Collection, Snatch-1.0, Unidentified-Refined), else 0 | 0.5 |

## Metrics

For HEAD, SULTANI, CASCADE and SOURCE:
- PR-AUC (trapezoidal; average precision reported separately) and ROC-AUC, where the score
  is continuous (CASCADE and SOURCE are binary, so their AUCs reflect one operating point).
- Precision, recall, F1, false-positive rate and the confusion matrix at the fixed operating
  point.
- Recall at FPR ≤ 0.05 (the best recall over score thresholds whose test FPR ≤ 0.05).
- **95% bootstrap CIs:** 2000 resamples of test clips, stratified by label (6 violence and 66
  non-violence per resample), seed 20261103, percentile intervals. Clip-level resampling
  ignores within-group correlation, so the intervals are optimistic; this is stated in the
  report.

## Populations (kept separate)

1. **Overall:** all 72 test clips.
2. **Unknown geography:** 21 clips (6 violence; 15 Unidentified-Archive non-violence).
3. **Documented-Indian non-violence:** 51 clips; false-positive rate only.

There are no non-violence test clips from a violence-bearing collection, so a within-source
test metric does not exist. SOURCE is expected to score ROC-AUC 1.0 on test by construction.

## Learning curve on test

For each train fraction (25/50/75/100%): mean ± std over seeds 7/8/9 of test PR-AUC and
ROC-AUC (overall and unknown geography). Mean over seeds, not best seed.

## Pre-registered reading of the results

- HEAD is said to show evidence **beyond source identity** only if its test PR-AUC exceeds
  SOURCE's, with the bootstrap 95% CI of the paired difference (HEAD − SOURCE) excluding 0.
  This is not expected to happen, because SOURCE is perfect by construction on this split.
- HEAD vs SULTANI and CASCADE is reported as "the fine-tuned head separates the test
  sources better/worse than the zero-shot pipeline", not as violence detection.
- A rising learning curve is reported as "source separation improves with more data", not
  violence detection, unless the beyond-source criterion holds.
- PROBE and MITIG test numbers are reported as confound diagnostics.

## Examples for the report

From HEAD's test scores: correct = the up-to-5 highest-scoring violence clips at or above
0.40 and the lowest-scoring non-violence clips; failures = all violence clips below 0.40,
then the highest-scoring non-violence clips, up to 5 in total. Each gets a one-line likely
cause from its thumbnails.

## Demo check

HEAD is wired in as the optional "Indian pilot" score (`--indian-pilot`). Live upload scores
for 3 consistently decoding test clips must match the offline test scores within 1e-3
(differences come only from CPU strict decoding live vs MPS non-strict offline).
