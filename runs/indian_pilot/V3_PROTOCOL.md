# Indian pilot v3 — UBI-Fights, XD-Violence and an IITB-Corridor test: protocol

Date 2026-11-03. Fixed before any v3 clip, feature or model exists. Purpose: test whether
adding more **matched** violence data (UBI-Fights, XD-Violence) improves the v2 head on fresh
Indian footage. Test set chosen by the user: about 50 clips from IITB-Corridor.

Known before writing this: dataset sizes, formats and frame counts (`DATASETS.md`), and the
IITB test class names and counts from `test.txt` (8 Fighting, 10 Normal, the rest non-violent
anomalies). No IITB per-frame annotation (`Test_Annotation/*.npy`) has been opened.

## Clips

Every new clip is a **2.0 s window** (`round(2 × fps)` frames, matching DCSASS's 2 s clips) on
a non-overlapping grid starting at frame 0, cut to its own file (H.264, CRF 18, no audio,
native resolution) with the ffmpeg bundled in `.venv` (`imageio_ffmpeg`). Window kinds:

- **violent:** every frame in the window is labelled violent;
- **matched normal:** a window in a violent video with no violent frame in it and none within
  1 s of either edge;
- **normal:** any window of a normal video.

From `n` candidates, `k` windows are taken evenly spaced: candidate `floor((i + 0.5) × n / k)`
for `i < k`, all of them if `n ≤ k`. No randomness.

## Test: IITB-Corridor (about 50 clips, scored once)

- **Violent videos:** the 8 `Fighting` videos only. Chasing, Protest, SuddenRunning and the
  other classes are not violence and are not used.
- Per Fighting video: **3 violent + 2 matched normal** windows. Per `Normal` video (10):
  **1 normal** window. At most 24 violent and 26 non-violent clips; shortfalls are recorded,
  not refilled.
- Label alignment: each IITB video has one more frame than `Count_Test_frames.csv` (the black
  first frame the readme drops), so label `i` belongs to video frame `i + 1`. If an annotation's
  length differs from the CSV count, the build stops and the fix is logged before any feature
  exists.
- IITB **train** (normal-only, the same camera as the test) is **not used anywhere**. Training
  on it would teach "this corridor is never violent", the v1 source shortcut in reverse.

## Training and validation data

| Data | Train | Validation |
|---|---|---|
| Indian | split v1 train (353) | split v1 val (79) |
| DCSASS (violent categories, as v2) | v2 train + the spent v2 test sources | v2 val |
| UBI-Fights | all videos not in `test_videos.csv` | the 67 `test_videos.csv` videos |
| XD-Violence test videos | 80% of sources | 20% of sources |

- **UBI-Fights:** per-frame labels. Per fight video up to **6 violent + 6 matched normal**
  windows; per normal video **3 normal** windows. One source per video.
- **XD-Violence:** violence = labels **B1 (fighting), B4 (riot), B5 (abuse)**, as v2's
  Abuse/Assault/Fighting/Robbery. A video carrying B2 (shooting), B6 (car accident) or G
  (explosion) is excluded, because its intervals do not say which type is where. `A` videos are
  normal. Windows as UBI. Source = the movie title or YouTube ID (before `__#`); sources split
  80/20 by seed 7, stratified by whether they contain violence. One XD test video is missing
  (`Before.Sunset…label_A`, normal; mirror error).
- **Overlap check before the split is frozen:** pHash fingerprints (D10 recipe: ~1 frame/s,
  near-blank skipped; a link is ≥ 2 frame pairs within Hamming 8) of every new source against
  every DCSASS clip, the Indian v1 clips and the IITB test videos. A source linked to DCSASS or
  Indian takes that source's split (same group); a source linked to IITB test is dropped from
  training. Links are logged.

## Features and models

- Same frozen C3D FC6 32-segment bags as v1/v2 (checkpoint `beac4ff0…`), MPS, non-strict
  decoding. Same head, loss, optimiser, batch and selection rule as v2.
- **Steps:** 2 passes over the training set, `steps = 100 × ceil(2 × N_train / 16 / 100)`,
  patience `steps / 6` rounded to 10 (v2: 600 steps ≈ 2.3 passes over 4,223).
- **Arms**, seeds 7, 8, 9, one shared validation set:
  - `v3_cctv`: Indian + DCSASS + UBI-Fights;
  - `v3_all`: `v3_cctv` + XD-Violence (mostly movies and broadcast footage; this arm shows
    whether it helps or hurts).
- **Deployed arm:** the one with the higher seed-mean validation PR-AUC on the non-XD
  validation clips (Indian + DCSASS + UBI), so XD's in-domain validation cannot pick XD. Seed 7
  of that arm. Threshold from all validation clips: max F1 subject to recall ≥ 0.8, candidates
  0.05–0.95. Validation sanity check: documented-Indian non-violence false-positive rate at
  that threshold ≤ 10%.

## One-time test evaluation

IITB test features are extracted only through the committed-protocol gate after training and
calibration, and scored once by a script that refuses to rerun. Models scored: both v3 arms
(seed mean ± std), v2 (`curve_d100_s7/8/9`, deployed s7 at 0.45), the v1 head
(`violence_f100_s7`, 0.40), zero-shot UCF Sultani.

Metrics: clip PR-AUC and ROC-AUC; **within-video ROC-AUC** (mean over the Fighting videos with
both labels); at each model's own threshold: precision, recall, F1, false-positive rate and
confusion matrix. 95% CIs by **cluster bootstrap over IITB test videos** (2000 resamples, seed
20261103).

## Pre-registered answer to "did v3 improve on v2?"

**Yes** only if the deployed v3 arm's seed mean beats v2's seed mean on IITB test on both:
1. clip PR-AUC, with the paired video-bootstrap 95% CI of the difference excluding 0; and
2. within-video ROC-AUC, same CI rule.

Otherwise the answer is "not shown", and the point estimates are reported as they are.
Expected power is low: all violent test clips come from 8 videos of one staged scene, so a real
gain can easily fail the CI rule. Beating zero-shot Sultani is reported separately with the same
rule.

## Afterwards

- The demo switches to the deployed v3 model only if the answer is yes; otherwise it keeps v2.
  The live/offline match check is repeated after any switch.
- The team's 40–60 fresh matched Indian clips (D22) remain a separate, later test.
