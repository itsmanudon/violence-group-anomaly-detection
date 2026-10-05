# Collective RGB matched-only actor-set diagnostic

This is a **diagnostic / oracle-assisted actor-set ablation**, not a deployable
detector configuration. It uses annotated GT matching to remove unmatched
detections from an already frozen detected-box scene. An unmatched detection may
be a real unannotated person; it is not necessarily a non-person detection.

## Fixed inputs and method

The source checkpoint is `9797e1114dc7f2b6b2258687311902a1d937ba98` on
`feat/collective-detected-actor-ablation`. The existing all-detected result was
verified from saved predictions and `comparison.json`, without a new deployment
test pass. All 35,588 artifact hashes in the original detected-feature receipt,
142 scientific source files, and 39 preserved Pose/RGB baseline artifacts were
verified before diagnostic inference.

The RGB Actor-Transformer remains the GT-trained model selected at iteration
1300. Its `best.pt` SHA256 is
`40a5fc3687c277914335916bd46dff72ccde5e4b6518c677f193178090de7fcd`.
The original detected-feature freeze hash is
`06ca8b5d55533125821cbf4b6283c41b0894f075cbb0f0bb3af8fbb221e526f0`.
Faster R-CNN COCO v1 confidence 0.7, NMS IoU 0.5, matching IoU 0.5, max actors
20, geometry filtering, RGB/I3D preprocessing, and dataset splits are unchanged.

For each of the 775 test scenes, the diagnostic:

1. Joins the saved deployment prediction and caches by dataset, clip ID and
   source ID; verifies frame indices, group target, boxes, scores and labels.
2. Checks that the saved matching equals the unchanged inclusive-IoU assignment
   policy. The saved matched indices are the selection authority.
3. Sorts retained original detector indices ascending, preserving detector actor
   order rather than the assignment solver's GT order.
4. Slices detected boxes and their existing `[N,20800]` RGB feature rows together.
   Every retained feature row and box is checked against the original tensor.
   GT boxes/features are never substituted and missed GT actors are not restored.
5. Remaps matching indices into the compact retained actor set, then calls the
   existing `DetectedGroupActivityPipeline` with the same checkpoint in eval mode.

No training, extraction, detector inference, threshold selection or model
mathematics changes occur. Removing actors can change attention, pooling and the
predictions for retained actors; their classification metrics are measured anew.

The original population contains 3,420 GT actors, 5,293 detections, 2,960 matches,
460 misses and 2,333 unmatched detections. The diagnostic retains 2,960 actors.
All 775 scenes have a nonempty matched set. Empty matched scenes would retain
the existing explicit abstention behavior, counted incorrect in all-scene group
accuracy; supported-scene macro-F1 would identify its population separately.

## Artifact locations

Generated artifacts remain ignored under `runs/collective/rgb_matched_only_v1/`:

- `registration.json`: diagnostic input hashes, original freeze identity,
  checkpoint, selection rule, source commit and CUDA settings, saved before
  inference. It does not replace either original GT/detected receipt.
- `selection.json`: original matching and retained detector row indices.
- `matched_only_predictions.json`: group/actor probabilities and original actor
  index mapping for every scene.
- `scene_records.json`: GT, all-detected and matched-only predictions/counts,
  misses, extras and preservation checks.
- `comparison.json`: original repository metric definitions and paired GT actor
  metrics on the same retained identities.
- `analysis.json`: three-way metrics, deltas, regression recovery, miss/extra
  strata, confidence distributions and directed confusion counts.
- `original_73_regressions.json`: outcomes for every original GT-correct /
  all-detected-wrong scene.
- `errors.json` / `errors.md` and `confusion_matrices.png`: error analysis and
  visualization-ready results, without copied dataset imagery.

Execution helpers and fresh check reports are retained under the separate ignored
`runs/collective/matched_actor_ablation_v1_execution/` folder. Evaluation refuses
to overwrite an existing diagnostic run. Analysis can be rerun from its saved
predictions without another model forward pass.

## Interpretation limits

This is our frozen Collective protocol, not an exact Gavrilyuk et al.
reproduction. The imperfect validation population and converted I3D weights
remain unchanged. This diagnostic is performed after observing the deployment
gap, using known test annotations; it does not define or tune a deployable
actor-quality rule. Performance recovered after oracle-assisted removal is
evidence about this intervention, not definitive causal proof that every
unmatched person is harmful or that matching quality is ideal.

Full GT actor metrics use 3,420 actors; both detected conditions evaluate the
same 2,960 matched actors. Paired GT actor metrics must accompany raw differences
to avoid interpreting a population change as classification degradation.

## Measured result

All 775 scenes were evaluated with 2,960 retained actors. Invalid scenes and
matched-only abstentions are both zero. The retained actors cover 86.55% of the
GT actor population, with the same mean matched IoU 0.6551 and 460 misses as
before. The oracle subset's apparent precision of 1 is a consequence of GT
selection, not an improvement in the deployable detector.

| Metric | GT boxes | All detected | Matched-only detected | Matched minus all | Matched minus GT |
|---|---:|---:|---:|---:|---:|
| Group accuracy | 77.55% | 74.32% | 72.65% | -1.68 pp | -4.90 pp |
| Group macro F1 | 0.7947 | 0.7383 | 0.7536 | +0.0153 | -0.0410 |
| Actor accuracy | 78.22% | 75.64% | 73.68% | -1.96 pp | -4.53 pp* |
| Actor macro F1 | 0.7923 | 0.7756 | 0.7643 | -0.0113 | -0.0279* |

*Raw GT actor differences compare different populations. On the identical 2,960
matched GT identities, GT accuracy is **76.55%**, macro F1 **0.7943**. Matched-only
minus this paired GT condition is **-2.87 pp** accuracy and **-0.0300** macro F1.
Both detected conditions evaluate the same actors; their changes are directly
paired. Predictions for all five classes remain present.

Group recall:

| Class | GT | All detected | Matched-only |
|---|---:|---:|---:|
| crossing | 71.43% | 61.90% | 71.43% |
| waiting | 65.93% | 37.78% | 57.04% |
| queueing | 100.00% | 100.00% | 100.00% |
| walking | 60.55% | 72.94% | 52.29% |
| talking | 100.00% | 100.00% | 95.60% |

Actor recall:

| Class | GT full | GT matched subset | All detected | Matched-only |
|---|---:|---:|---:|---:|
| crossing | 67.49% | 68.45% | 62.35% | 69.51% |
| waiting | 63.35% | 64.95% | 54.64% | 55.88% |
| queueing | 92.93% | 92.49% | 95.98% | 94.64% |
| walking | 71.65% | 71.51% | 78.66% | 68.15% |
| talking | 96.29% | 95.03% | 91.97% | 90.25% |

The matched-only group confusion matrix uses rows=true, columns=predicted, in
order crossing, waiting, queueing, walking, talking:

```text
105   1   0   41    0
 48  77   0   10    0
  0   0  93    0    0
 97   6   1  114    0
  0   5   3    0  174
```

Waiting recall recovers **19.26 pp** from all-detected, but remains **8.89 pp**
below GT. Waiting-to-crossing errors decline from **67 to 48**, versus GT 41.
Crossing-to-walking declines **55 to 41**, while walking-to-crossing increases
**42 to 97**. Combined swaps rise **97 to 138** (GT 108). The recovered waiting
and crossing predictions coexist with 45 fewer correct walking scenes and eight
fewer correct talking scenes; blanket removal does not improve overall accuracy.

## Original failures and descriptive strata

Of the original **73 GT-correct / all-detected-wrong scenes**, performance is
recovered in **42**, **31** remain wrong and **0** abstain. The 42 recovered
scenes comprise crossing 10, waiting 25, walking 7. Of all 73, 70 contain extras,
11 contain misses, and all 11 miss-containing scenes also contain extras. Four
of these 11 are recovered and seven remain wrong. These overlapping counts do
not prove that a particular missed or unmatched person caused an error.

Across the complete test set, **51** all-detected errors become correct, but
**64** previously correct predictions become wrong, a net loss of 13 correct
scenes (576 to 563). The 61 scenes with no extras retain their original group
predictions, providing a check of unchanged inputs and model behavior.

| Stratum | Scenes | GT group accuracy | All detected | Matched-only |
|---|---:|---:|---:|---:|
| No extras | 61 | 73.77% | 70.49% | 70.49% |
| 1 extra | 113 | 74.34% | 74.34% | 69.91% |
| 2+ extras | 601 | 78.54% | 74.71% | 73.38% |
| No misses | 456 | 67.76% | 64.04% | 61.84% |
| 1 miss | 200 | 88.50% | 85.50% | 87.00% |
| 2+ misses | 119 | 96.64% | 94.96% | 89.92% |

These are different class/scene populations, not matched experiments between
strata. Their class supports and per-scene actor counts are saved in
`analysis.json` / `scene_records.json`. In particular, higher accuracy in a
miss-containing stratum does not show that missing actors improves recognition.

| Confidence statistic | Matched (2,960) | Unmatched (2,333) |
|---|---:|---:|
| Mean | 0.9940 | 0.9304 |
| Median | 0.9985 | 0.9725 |
| Q1 | 0.9960 | 0.8858 |
| Q3 | 0.9993 | 0.9948 |
| Minimum | 0.7139 | 0.7004 |
| Maximum | 0.9998 | 0.9998 |

Quartiles use NumPy's linear definition. Confidence is person detection
confidence, not membership in the dataset's annotated activity group. These test
statistics are descriptive and were not used to derive a new threshold.

## Diagnostic conclusion

The result is closest to **Case B at the aggregate level**: matched-only remains
below GT and does not recover deployment accuracy. It improves macro F1 and
waiting/crossing recall, demonstrating class-specific actor-set sensitivity,
while substantially reducing walking recall. Unmatched actors are therefore not
a uniformly harmful set in this measured intervention. The experiment cannot
separate localization, missed actors and actor-context distribution mismatch.

A separately registered **detected-box fine-tuning** study is better justified
next than blanket removal or an immediate confidence increase. A conservative
actor-quality study remains plausible for waiting, but must use validation data
and account for the observed walking tradeoff. Neither study has been started.

The diagnostic registration hash is
`645184f9891803355da2d54c2077c26241e28d22815e85147dabd5f9868a5227`.
Inference used the existing CUDA environment (torch 2.13.0+cu126,
torchvision 0.28.0+cu126, RTX 4070 Laptop GPU) and preserved its default eval
settings. The registration and analysis record the exact identities and settings.

Any proposed detector-filtering or fine-tuning study must be registered separately
and selected using training/validation data. No subsequent experiment is launched
as part of this diagnostic.

## Final verification

Fresh checks on 2026-10-06 passed before inference (420 tests) and after analysis:
**420 passed, 325 warnings in 246.63 seconds**; Ruff lint passed; Ruff formatting
check reported 132 files already formatted; `git diff --check` passed. Warnings
include existing TorchScript deprecations from checkpoint fixture tests. No
production source, model, detector, configuration or test modifications were
required. Only this report and the README are changed in tracked/untracked source
status; generated outputs and execution helpers remain ignored. Nothing is
staged, committed or pushed.

The separate execution audit's `final_verification.json` seals output hashes and
checks that original baseline artifacts, the scientific source and frozen
checkpoint remain unchanged. Saved predictions can be reanalyzed without another
inference pass:

```powershell
.\.venv\Scripts\python.exe runs/collective/matched_actor_ablation_v1_execution/analyze.py
```
