# Indian pilot — validation results and hard stop

Date: 2026-11-03. Split v1 (`ae9a0fab…`), validation only: 79 clips (6 violence, 73
non-violence). Test features have not been extracted; the test split is untouched.
Machine-readable results: [`validation_results.json`](validation_results.json).

## Headline numbers (validation)

| model / rule | population | PR-AUC | ROC-AUC | at operating point |
|---|---|---|---|---|
| Fine-tuned head, seed 7 (deployed) | all 79 | 0.974 | 0.998 | thr 0.40: 6/6 violence, 1 false alarm |
| Fine-tuned head, seed 7 | unknown geography (24) | 1.000 | 1.000 | 6/6, 0 false alarms |
| **Source-only rule** (violence iff the clip's collection has violence in train) | all 79 | 0.875 | **0.986** | 6/6, 2 false alarms |
| Source-only rule | unknown geography (24) | 0.875 | 0.944 | 6/6, 2 false alarms |
| Zero-shot UCF Sultani score (same C3D bags) | all 79 | 0.847 | 0.963 | — |
| Zero-shot five-stage cascade alert | all 79 | — | — | 4/6, 1 false alarm (2 clips refused) |

## Learning curve (validation, mean ± std over seeds 7/8/9)

| train fraction | train clips (V) | PR-AUC | ROC-AUC |
|---|---|---|---|
| 25% | 112 (6) | 0.974 ± 0.000 | 0.998 ± 0.000 |
| 50% | 135 (12) | 0.991 ± 0.012 | 0.999 ± 0.001 |
| 75% | 306 (17) | 0.991 ± 0.012 | 0.999 ± 0.001 |
| 100% | 353 (27) | 0.983 ± 0.012 | 0.998 ± 0.001 |

The curve is flat at the ceiling from 25% (6 violence clips) upward. Validation cannot show
improvement with more data.

## Collection-shortcut probe

- **Probe A**, same head trained to predict geography (documented-Indian vs unknown): val
  PR-AUC 0.97–0.99. Collection is near-perfectly readable from the C3D features.
- **Probe B**, the same, trained on non-violence clips only (no violence signal available):
  val PR-AUC 0.94–0.95. The features carry strong source identity on their own.
- **Mitigation**, violence head trained only within unknown geography: perfect within
  unknown geography, but it flags **89–100%** of the documented-Indian non-violence clips
  (WalkIndia, IPID street walks) as violent. A model that never saw those collections calls
  them violent: it learned "not one of my non-violence sources", not violence.

## Interpretation

1. A rule that knows only which collection a clip came from reaches ROC-AUC 0.986 on
   validation. The head (0.998) adds very little beyond source identity.
2. The only evidence beyond source identity: within Snatch-1.0 (4 violence vs 2
   non-violence, same dataset), the head ranks all 4 violence clips above both
   non-violence clips (0.44–0.57 vs 0.36–0.37), where the source rule fails. Six clips, and
   the two non-violence clips come from a distinct sub-network (`s_*.avi`), so this is an
   anecdote, not evidence.
3. The frozen test split contains **no non-violence clips from any collection that contains
   violence** (test violence is Snatch 4, HCI 1, Refined 1; test non-violence is Archive,
   WalkIndia, IPID). On test, the source-only rule will score ROC-AUC 1.0 by construction,
   so a test result cannot separate violence detection from source detection.
4. This meets handover hard stop 5: the probe shows source identity is near-perfectly
   learnable, and the one mitigation tried fails to generalise. Results are reported here
   and not presented as Indian violence detection pending a decision.

## Notes

- Operating-point metrics for non-deployed runs in `validation_results.json` use the seed-7
  threshold and are not meaningful (their scores are uncalibrated early-step checkpoints);
  compare those runs by PR-AUC/ROC-AUC only.
- Validation numbers for the deployed head are optimistic: the checkpoint and threshold were
  both selected on these same 6 positives.
