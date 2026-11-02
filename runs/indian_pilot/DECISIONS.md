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
