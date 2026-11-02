# Indian pilot v2 — adding DCSASS: protocol (fixed before any DCSASS feature exists)

Date 2026-11-03. Authorised by the user to run end to end autonomously. Purpose: test
whether adding **matched** data improves violence detection, on a test set where the source
cannot give the answer away. The Indian pilot (v1) showed that Indian-only training learns
the source collection (see `REPORT.md`, `VALIDATION.md`).

## Data

- **Indian:** split v1 train (353) and val (79) clips with their source groups (freeze hash
  `ae9a0fab…`). The Indian v1 test split is spent and is **not** included.
- **DCSASS human-centric v1:** the frozen, source-safe split
  `data/manifests/dcsass_human_centric_v1_final.jsonl` (SHA256 `f3e7fc11…ee7258`; 203 UCF
  source recordings; 140 train / 32 val / 31 test sources; zero source overlap). Clips are on
  the Seagate drive under `professor_demo_bundle/data/raw/dcsass/DCSASS Dataset/DCSASS Dataset/`.
  Assignments are copied unchanged.
- **Label:** violence = DCSASS `binary_label` 1 in Abuse, Assault, Fighting or Robbery
  (physical violence and robbery/snatching, matching the Indian positives). The **Vandalism**
  category is excluded entirely (property damage is not an observable violent act against
  people). Normal (label 0) clips come from the same source recordings as the positives.
- DCSASS test after exclusion: 831 clips (382 violence, 449 normal) from 26 sources; 22 of
  them contain both labels. DCSASS is UCF-derived footage: **not Indian**.
- The UCF Sultani scorer excluded DCSASS test sources from its training, so its score is a
  fair zero-shot baseline here.
- The combined split (`manifests/v2-pilot/split_combined_v1/`) is hash-bound like split v1;
  `source_video_id` is prefixed by dataset (`indian:G001`, `dcsass:abuse001`).

## Features and models

- Same frozen C3D FC6 32-segment bags (checkpoint `beac4ff0…`), MPS, non-strict decoding.
- Same head, loss and optimiser as v1 (D15): one linear layer over the mean-pooled bag,
  class-balanced cross-entropy, Adam 1e-3, batch 16, fp32, CPU. **600 steps, patience 100**
  (v1's 300 steps would cover less than one pass over the ~5,000-clip training set).
  Checkpoint selected on combined validation PR-AUC (Indian val + DCSASS val).
- **Learning curve:** all Indian train groups plus nested subsets of DCSASS train sources at
  **0, 10, 25, 50, 100%**, stratified by DCSASS category, subset seed 7, the same subsets
  for seeds 7, 8 and 9. 0% is Indian-only training on the combined split (selection on the
  combined validation set), so every point uses the same selection data.
- **DCSASS-only** (100% DCSASS, no Indian clips), seeds 7/8/9: shows what the Indian clips
  add or cost.
- **Deployed model:** seed 7 at 100%. Threshold from combined validation only: max F1 subject
  to recall ≥ 0.8, candidates 0.05–0.95.

## One-time test evaluation (DCSASS test, 831 clips)

Test features are extracted only through the committed-protocol gate, after training and
calibration finish. Evaluated once by a script that refuses to rerun.

Metrics per model:
- clip-level PR-AUC and ROC-AUC;
- **within-source ROC-AUC**: the mean over test sources containing both labels (the source
  cannot help here);
- at the deployed threshold: precision, recall, F1, false-positive rate and confusion matrix;
- 95% CIs by **cluster bootstrap over test sources** (2000 resamples, seed 20261103).

Models scored: the curve (5 fractions × 3 seeds, mean ± std), DCSASS-only, the v1 Indian-only
head (`violence_f100_s7`), and zero-shot UCF Sultani.

## Pre-registered answer to "did the model improve?"

**Yes** only if both hold for the seed-mean of the 100% model versus the seed-mean of the 0%
model on DCSASS test:
1. clip-level PR-AUC is higher, with the paired source-bootstrap 95% CI of the difference
   excluding 0; and
2. within-source ROC-AUC is higher, with the same CI rule.

The learning curve's trend is reported as observed. Beating zero-shot Sultani is reported
separately with the same CI rule. Indian validation is reported as a sanity check only (it is
saturated): documented-Indian false-positive rate at the deployed threshold should not rise
above 10%.

## Afterwards

- The demo's "Indian pilot" score switches to the deployed v2 model only if the answer above
  is yes; otherwise it keeps the v1 head. The live/offline match check is repeated.
- When the team delivers the fresh 40–60 matched Indian clips, the deployed model and the
  zero-shot cascade are evaluated on them once, under a short protocol committed before
  scoring. Nothing is tuned on those clips.
