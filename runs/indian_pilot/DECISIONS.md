# Indian pilot — decision log

Context: `docs/handover/HANDOVER.md`. Newest entries at the bottom. Each entry: date,
decision, why, alternatives rejected.

## D1 — 2026-11-03 — Branch and code base

- **Decided:** new branch `pilot/indian-binary` from `472c7d8`; cherry-picked the
  Legion's Oct 25 commit `27d31f0` (lands as `aea9cd6`).
- **Why:** `27d31f0` is additive (68 new files; only `.gitignore`/`README.md` edited)
  and imports neither municipal platform version. Applied cleanly; full suite
  706 passed, 4 skipped.
- **Rejected:** merging `legion/municipal-oct25` (would pull in the Legion's
  conflicting platform version, deferred past Nov 10); rewriting the reconciliation,
  split and training code (exists and is tested).

## D2 — 2026-11-03 — Python environment

- **Decided:** keep the existing venv: Python 3.12.7, torch 2.14.1, torchvision 0.29.1
  (the pins in `requirements/municipal-macos-py312.txt`).
- **Why:** `scripts/demo_preflight.py` passes on it (5 checkpoints, 9 examples,
  frontend, MPS available). The handover says keep 3.12 if pins work.
- **Rejected:** rebuilding on 3.11 (no benefit; risk of drift from the pinned demo).

## D3 — 2026-11-03 — Indian data location and verification

- **Decided:** canonical copy at `/Volumes/Seagate_2TB/violence-datasets/`, extracted
  from the Drive transfer (`~/Documents/violence-pilot-transfer/2026-11/`).
- **Evidence:** all 4 zips pass CRC tests; zip SHA256s match the Legion's figures;
  566/566 `MANIFEST.csv` rows match size and SHA256, none missing, none extra.
  Reconciliation: 523 media on disk = 523 inventory rows; 521 annotation rows
  (the 2 gaps are the audio-only temple clips); labels 48 violence / 465
  non_violence / 7 not_sure + 1 blank; dispositions 504 candidates (39 / 465) and
  17 holds (16 annotation quarantine + 1 media hold). All counts match.
- **Rejected:** a second copy on the internal SSD (redundant; SHA256 binding in the
  split receipt makes location irrelevant to provenance). Risk accepted: unplugging
  the drive fails running jobs loudly rather than silently.

## D4 — 2026-11-03 — Demo smoke-test clip

- **Decided:** upload-mode smoke test used `Snatch Theft/6_0.mp4`, a `not_sure` hold.
- **Why:** the clip is excluded from the pilot, so viewing zero-shot cascade output
  on it cannot influence any split, model or threshold.
- **Result:** CPU, 3.5 s end to end, 32 C3D segments, Sultani peak 0.73, no people
  detected, alert `generic_anomaly` (review).

## D5 — 2026-11-03 — Clips whose headers overstate their frame count

- **Finding:** 13 clips decode cleanly from the start but stop 15–242 frames short of
  the header count (3 IPID, 10 Snatch Theft). 11 are pilot candidates, including
  **8 of the 39 violence candidates** (Snatch Theft 6, 8, 9, 10, 11, 12, 16, 17).
  The repo's `iter_clips` deliberately rejects such files.
- **Decided:** added an opt-in `strict=False` to `iter_clips` and
  `strict_frame_count=False` to `C3DExtractor.extract_video`; the pilot feature cache
  uses the frames that actually decode. Defaults stay strict, so the demo and all
  research paths are unchanged. Test added.
- **Why:** clip-level bag features need no exact frame-to-time mapping; dropping 8
  violence clips (20%) would cripple an already tiny positive class.
- **Rejected:** excluding them (cost above); re-encoding copies (changes pixels, adds
  a derived-media provenance layer, and ffmpeg is not installed).
- **Consequence:** the live demo still rejects these 11 files; demo clips will be
  chosen from consistently decoding test clips.

## D6 — 2026-11-03 — Feature extraction device

- **Decided:** extract pilot features on **MPS** (`PYTORCH_ENABLE_MPS_FALLBACK=1`);
  the five-model demo stays on CPU per its existing policy.
- **Evidence** (`runs/indian_pilot/mps_check/parity.json`, 5 excluded clips): C3D FC6
  max |CPU−MPS| ≤ 3.3e-7, I3D Mixed_4f ≤ 5.9e-6 (values up to ~2.4), cosine 1.000000
  on every bag/window; no MPS fallback warnings; C3D ~3.2× and I3D ~2× faster.
- **Rejected:** CPU extraction (slower, no accuracy benefit).

## D7 — 2026-11-03 — Collection tags

- **Decided:** two tags per clip: `geography` (documented_indian / unknown_geography)
  and `collection` (the source dataset name).
- **Why:** candidate counts show the confound is finer than geography:

  | dataset | violence | non_violence | geography |
  |---|---|---|---|
  | WalkIndia-50 | 0 | 335 | documented Indian |
  | IPID | 0 | 17 | documented Indian |
  | Snatch-1.0 | 27 | 7 | unknown |
  | HCI-User-Collection | 7 | 0 | unknown |
  | Unidentified-Refined | 5 | 2 | unknown |
  | Unidentified-Archive | 0 | 104 | unknown |

  Within unknown geography, HCI is all violence and Archive all non_violence, so a
  model could still separate them by source. Only Snatch-1.0 and Refined contain
  both classes. The probe and the within-subset reports will cover both levels.

## D8 — 2026-11-03 — What goes into git

- **Decided:** `.gitignore` now uses `runs/*` with an exception for
  `runs/indian_pilot/`, still excluding `features/`, weights (`*.pt`, `*.npy`
  globally), media, images and CSVs there. Split and source-group CSVs under
  `manifests/v2-pilot/` are committed as the handover instructs (clip IDs, groups,
  labels; no footage).

## D9 — 2026-11-03 — Pilot eligibility

- **Decided:** 504 eligible clips = every non-hold clip labelled violence/non_violence
  (39 / 465). All 504 decode. 19 excluded with written reasons: 17 quality holds (9
  violence, 7 not_sure, 1 blank) and 2 audio-only items.
- **Geography tag:** from the inventory's `geographic_origin`, not hard-coded:
  WalkIndia-50 + IPID = 352 documented-Indian (all non_violence); the rest unknown.

## D10 — 2026-11-03 — Source-group reconstruction

- **Decided:** union-find (reusing `indian_split.source_groups`) over: pHash links (≥ 2
  frame pairs within Hamming 8, ~1 frame/s, near-blank frames skipped); WalkIndia
  `{place}-Scene-NNN` and HCI `krishna-clip-*` filenames; HCI capture times ≤ 5 min apart;
  and explicit visual-review merges in `manifests/v2-pilot/review_decisions.json`.
- **Threshold evidence:** no cross-collection clip pair matches at ≤ 8; the first
  cross-collection matches appear at 10 (2 pairs) and 12 (15 pairs). Two-frame minimum
  guards against a single coincidental frame.
- **Rejected:** the Snatch `N` / `N_0` filename rule. Visual review showed different
  footage in all 5 pairs, and pHash agrees (no frame pair below 20 bits).
- **Rejected:** `imagehash` (new dependency); the same DCT recipe is ~10 lines with scipy.
- **Rejected:** weak hints (duration, resolution, file mtime) as automatic links. mtimes are
  transfer times, not recording times; identical resolution is shared by entire collections
  (all 1920×1080), so it would merge unrelated clips wholesale.
- **Result:** 118 groups; 32 contain violence (one of 5 clips, three of 2, 28 singletons);
  one mixed-label group (G096, Refined debris protest).

## D11 — 2026-11-03 — Visual review of violence clusters

- **Decided:** reviewed all 48 clips of the three collections containing violence (Snatch,
  HCI, Refined) on contact sheets, not only the violence groups, so cross-group same-camera
  pairs would be visible. Five merges recorded with evidence; details in
  `manifests/v2-pilot/violence_cluster_review.md`.
- **Label noise noted, labels unchanged:** Refined `Video4` (annotated non_violence) shows a
  shield line and thrown objects.
- **Geography note:** the 5 SHAWNRTV clips look non-Indian. "Unknown geography" must not be
  presented as Indian footage.

## D12 — 2026-11-03 — Split rules (fixed before viewing any assignment)

- **Decided:** seed 7; ratios 70/15/15; groups whole; strata label × collection; required in
  both val and test: ≥ 6 violence, ≥ 10 unknown-geography non_violence, ≥ 10
  documented-Indian non_violence. G096 (mixed labels, suspected label noise) forced to train.
- **Interpretation:** the handover's "non-violence from the same unknown-geography collection
  as the violence clips" is read at the geography level (its stated minimum tag). Within
  that level, the non-violence is mostly Unidentified-Archive; the only collections with
  both labels are Snatch (7 NV in 2 groups) and Refined (NV only inside G096). So a
  fully collection-matched within-collection test is not feasible with this data, and the
  report must say so.
- **Rejected:** forcing Snatch non-violence into val and test. With 2 groups (5 + 2 clips)
  that would leave train with no Snatch non-violence at all, and yield a 2-clip test
  subset that cannot carry a metric.

## D13 — 2026-11-03 — Split v1 frozen (test locked from here)

- **Result:** freeze hash `ae9a0fab918eb20992b3202093a0982af16f7e3bc28a57c9e28ff2598818b32f`;
  `split_v1.csv` SHA256 `cc58be5200003baebf538a930d44be7d735ddba36551e68a11e09e662f87a2c0`.
  First and only freeze attempt with the D12 rules; no seed or rule was changed after viewing.

  | split | clips | groups | violence | unknown-geo NV | documented-Indian NV |
  |---|---|---|---|---|---|
  | train | 353 | 77 | 27 | 80 | 246 |
  | val | 79 | 24 | 6 | 18 | 55 |
  | test | 72 | 17 | 6 | 15 | 51 |

- `verify_split` passes: 504 records, 0 groups crossing splits, all 504 media hashes bound.
- **Consequence for reporting:** 6 violence clips per evaluation split. Any recall or F1
  moves in steps of ~17 points per clip; CIs will be very wide.
- **Line endings:** CSVs are written with LF so committed bytes equal the hashed bytes
  (git normalises CRLF here, which would otherwise break `verify_split`).
