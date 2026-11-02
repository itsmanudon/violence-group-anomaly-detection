# Indian pilot — report

Pilot fine-tune of a binary violence head on Indian-domain footage, for the Nov 10, 2026
presentation. Branch `pilot/indian-binary`. Every decision is logged in
[`DECISIONS.md`](DECISIONS.md); the test protocol was committed before test features existed
([`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md), commit `0a17a50`).

## One-paragraph summary

The existing five-stage pipeline runs end to end on the team's Indian-domain clips, and a
small head on its frozen C3D features can be fine-tuned and served live in the review
console. But **with the current data, the fine-tuned head cannot be told apart from a
detector of which source a clip came from.** A rule that knows only the clip's collection
matches it on test (both perfect) and nearly matches it on validation (ROC-AUC 0.986 vs
0.998). The same features predict the collection almost perfectly (PR-AUC 0.94–0.995), and a
head trained without the Indian street-walk collections calls most of them violent. The
honest result is a measured confound, and the path forward is specific: non-violence clips
from the **same cameras and phone sources** as the violence clips.

## 1. Setup and data

| item | value |
|---|---|
| Inventory | 523 items: 521 videos + 2 audio-only (all SHA256-verified after transfer) |
| Team labels (unadjudicated) | 48 violence, 465 non-violence, 7 not sure, 1 blank |
| Eligible for the pilot | 504 clips: 39 violence, 465 non-violence |
| Excluded (with reasons) | 17 quality holds (9 violence, 7 not sure, 1 blank) and 2 audio-only items |
| Features | frozen C3D FC6, 32-segment bags (the pipeline's own), extracted on MPS (CPU parity ≤ 6e-6) |
| Head | one linear layer on the mean-pooled bag; class-balanced cross-entropy; 300 steps |
| Hardware | MacBook M4 Pro, 48 GB; no CUDA |

| split | clips | source groups | violence | unknown-geo non-violence | documented-Indian non-violence |
|---|---|---|---|---|---|
| train | 353 | 77 | 27 | 80 | 246 |
| val | 79 | 24 | 6 | 18 | 55 |
| test | 72 | 17 | 6 | 15 | 51 |

## 2. What "pilot" means here

- **Readiness override.** The repository's readiness gates mark all 521 clips as not
  training-eligible because provenance is unresolved. The pilot overrides this explicitly by
  name (`pilot_override: team_labels_unadjudicated` in the split receipt); the gates
  themselves were not edited.
- **Labels are team annotations, not adjudicated ground truth.** One visible example of
  label noise: `Refined Clips/Video4.mp4` is annotated non-violence but shows a police shield
  line and thrown objects. It was kept as annotated and forced into train.
- **Geography is mostly unknown.** All 39 violence clips have unknown geography. Five staged
  clips (SHAWNRTV watermark) look non-Indian. "Indian-domain" describes the collection effort,
  not verified location.

## 3. How source groups were reconstructed

Provenance columns were empty, so groups were rebuilt so that no source crosses splits:
perceptual hashes of ~1 frame/s (linked at ≤ 8 bits on ≥ 2 frames; the first unrelated
cross-collection match appears at 10 bits), filename conventions (WalkIndia scene chunks,
HCI `krishna-clip-*`), capture times within 5 minutes, and a **visual review of every clip in
every collection that contains violence** (48 clips). The review merged 5 sets (for example,
5 staged clips sharing one watermark) and **rejected** a filename rule that looked right
(Snatch `N` / `N_0`) but showed different footage in all 5 pairs. Result: 118 groups, 32 with
violence. Audit trail: `manifests/v2-pilot/violence_cluster_review.md`.

## 4. The confound, measured

| diagnostic | result |
|---|---|
| Same head trained to predict geography (all clips) | val PR-AUC 0.97–0.99; test PR-AUC 0.995 |
| Same, trained on non-violence clips only (no violence signal exists) | val PR-AUC 0.94–0.95 |
| Violence head trained only within unknown geography | flags 89–100% (val) and 59–100% (test) of documented-Indian street walks as violent |
| Source-only rule (violence iff the collection has violence in train) | val ROC-AUC 0.986; test ROC-AUC 1.000 |

Why: in this data, violence and non-violence come from **different sources**. Violence comes
from CCTV snatching clips and phone/news screen recordings (often portrait, often with
overlays); non-violence comes from 1080p walking tours, a fixed-camera archive and a
pedestrian dataset. The C3D features encode source very strongly, so any classifier can score
well by recognising the source.

## 5. Results

### Validation (79 clips, 6 violence; model and threshold selected here, so optimistic)

| model | PR-AUC | ROC-AUC | at operating point |
|---|---|---|---|
| Fine-tuned head (thr 0.40) | 0.974 | 0.998 | 6/6 violence, 1 false alarm / 73 |
| Source-only rule | 0.875 | 0.986 | 6/6, 2 false alarms |
| Zero-shot UCF Sultani score | 0.847 | 0.963 | — |
| Zero-shot five-stage cascade alert | — | — | 4/6, 1 false alarm |

The only evidence beyond source identity: within Snatch-1.0 (4 violence vs 2 non-violence
from the same dataset), the head ranks all violence above all non-violence where the source
rule cannot. That is 6 clips, and the non-violence ones come from a distinct camera network.
It is an anecdote, not evidence.

### Test (72 clips, 6 violence; evaluated once; 95% bootstrap CIs, clip-level, label-stratified)

| model | PR-AUC [95% CI] | ROC-AUC | at operating point | recall at FPR ≤ 0.05 |
|---|---|---|---|---|
| Fine-tuned head (thr 0.40) | 1.000 [1.000, 1.000] | 1.000 | 6/6, 0 false alarms / 66 | 1.00 |
| Source-only rule | 1.000 [1.000, 1.000] | 1.000 | 6/6, 0 false alarms | 1.00 |
| Zero-shot UCF Sultani score | 1.000 [1.000, 1.000] | 1.000 | 3/6 at 0.5 [CI 1–5 of 6], 0 false alarms | 1.00 |
| Zero-shot cascade alert | 0.771 [0.618, 0.924] | 0.750 | 3/6, 0 false alarms (1 clip refused) | 0.50 |

- **Pre-registered criterion for "beyond source identity": not met.** Paired head − source
  PR-AUC difference 0.000, CI [0, 0].
- The test split is **saturated**: even the untuned UCF score ranks every test violence clip
  above every test non-violence clip. The head's only measurable gain over the zero-shot
  pipeline is the operating point (6/6 vs 3/6 detected at fixed thresholds).
- Within unknown geography (21 clips, 6 violence) every model except the cascade scores 1.000.
- With 6 positives, one clip moves recall by 17 points; the clip-level CIs ignore
  within-group correlation and are optimistic.

## 6. Learning curve (mean ± std over 3 seeds; same nested group subsets for every seed)

| train fraction | train clips (violence) | val PR-AUC | test PR-AUC |
|---|---|---|---|
| 25% | 112 (6) | 0.974 ± 0.000 | 0.923 ± 0.052 |
| 50% | 135 (12) | 0.991 ± 0.012 | 0.952 ± 0.032 |
| 75% | 306 (17) | 0.991 ± 0.012 | 1.000 ± 0.000 |
| 100% | 353 (27) | 0.983 ± 0.012 | 1.000 ± 0.000 |

Test PR-AUC rises with data, but per the protocol this is **source separation improving with
data**, not violence detection: within unknown geography the curve is flat at 1.000, and
validation is at the ceiling from 6 positives. It shows the training loop and evaluation
work end to end; it cannot show that violence understanding improves.

## 7. Examples (test, head scores)

Correctly flagged violence (all also pass the source rule):

| clip | head | zero-shot | likely reason |
|---|---|---|---|
| HCI `Screen Recording …191404` (NDTV clip) | 0.636 | 0.89, cascade: review | portrait phone recording of news footage: men with sticks; source cue and motion both present |
| Refined `Video6` | 0.630 | 0.67, cascade: review | portrait phone video of a hospital-corridor scuffle; close-range motion blur |
| Snatch `17.mp4` | 0.584 | 0.46, cascade: **refused** | CCTV motorbike snatch; the strict cascade rejected the file (header overstates frames) |
| Snatch `21_0` | 0.528 | 0.37, cascade: no alert | low-res street CCTV with broadcaster overlay, like Snatch training clips; zero-shot missed it |
| Snatch `9_0` | 0.518 | 0.83, cascade: review | CCTV with timestamp overlay, pedestrian snatch near parked bikes |

Closest calls (highest-scoring non-violence; all correctly below 0.40, so no test errors):

| clip | head | likely reason |
|---|---|---|
| WalkIndia `temple-Scene-005-scene001` | 0.376 | dense crowd at a temple gate, saturated colour |
| Archive `video68.MOV` | 0.337 | roadside traffic with a burnt-in timestamp, like CCTV snatch clips |
| Archive `video62.MOV` | 0.326 | same camera family and timestamp overlay |
| WalkIndia `metro-Scene-001-scene001` | 0.320 | close, fast-moving commuter crowd |
| Archive `video70.MOV` | 0.312 | street traffic with timestamp overlay |

The highest-scoring non-violence clips share **visual source cues** (timestamp overlays,
CCTV-like framing, dense crowds) with the violence clips. That is consistent with the
confound.

## 8. What more and better data would fix

1. **Matched negatives (most important).** Non-violence clips from the same cameras and phone
   sources as the violence clips: ordinary Snatch-style CCTV street footage, and peaceful
   protests and crowd scenes recorded on phones or screen-recorded from news and social
   media. The project already defines peaceful protests as negatives; they are exactly the
   hard negatives this pilot lacks. With them, a within-source test can tell violence from
   source.
2. **More violence groups.** 39 clips in 32 groups give 6 per evaluation split; any metric
   moves 17 points per clip. Roughly 30+ violence groups per evaluation split would make CIs
   informative.
3. **Adjudicated labels** (two annotators plus a tie-break) and recorded **provenance**
   (source URL or camera, incident), so groups need not be reconstructed.
4. **Documented geography** for violence clips, so "Indian-domain" can be claimed.

## 9. Demo

The fine-tuned head is an optional add-on score in the existing console: it reuses the
cascade's own C3D bag, never changes the cascade alert, and is labelled for human review only.
Live upload scores match the offline test scores to within 6e-8 for 3 test clips (2 violence
at 0.636 and 0.630 flagged "Review"; 1 non-violence at 0.376 "Below 0.400"), verified by a
Playwright run in Chrome.

```bash
# from the repo root, on the Mac (CPU; frozen models; nothing leaves the machine)
PYTHONPATH=src .venv/bin/python -m surveillance.municipal --device cpu \
  --indian-pilot runs/indian_pilot/demo/indian_pilot_spec.json serve
# open http://127.0.0.1:7860 → Technical details → Professor console → Choose a video
```

The standalone console (`python -m surveillance.demo --profile professor --device cpu
--indian-pilot runs/indian_pilot/demo/indian_pilot_spec.json`) takes the same flag. The 11
clips whose headers overstate their frame count are rejected by the live cascade (strict
decoding), so pick demo clips outside that list.

## 10. Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/pilot_source_groups.py fingerprint
PYTHONPATH=src .venv/bin/python scripts/pilot_source_groups.py group --threshold 8 --min-matches 2 \
  --review manifests/v2-pilot/review_decisions.json
PYTHONPATH=src .venv/bin/python scripts/pilot_freeze_split.py --forced-train-groups G096   # refuses to overwrite
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv/bin/python scripts/train_indian_binary.py cache \
  --split-dir manifests/v2-pilot/split_v1 --checkpoint checkpoints/c3d_fc6_openmmlab_v1.pt \
  --output runs/indian_pilot/features/c3d_trainval_v1 --device mps --batch-size 8 --splits train val \
  --non-strict-frame-count
PYTHONPATH=src .venv/bin/python scripts/pilot_experiments.py train && \
  PYTHONPATH=src .venv/bin/python scripts/pilot_experiments.py summarize
PYTHONPATH=src .venv/bin/python scripts/pilot_cascade_baseline.py --split val --output runs/indian_pilot/baselines/cascade_val
# test (once, protocol-gated): cache --splits test --test-protocol runs/indian_pilot/EVAL_PROTOCOL.md,
# pilot_cascade_baseline.py --split test --test-protocol ..., then scripts/pilot_test_eval.py
```
