# End-to-end execution journal

The user's phase A–O specification is the design authority. Work proceeds in this
repository on `feat/end-to-end-surveillance-mvp`. Local milestone commits are
permitted; pushes, merges, history rewriting, test tuning, and replacement of the
two approved paper baselines are prohibited.

## 2026-10-06 — Initial repository and mandatory-asset gate

### Repository and preserved evidence

- Started on clean `feat/dcsass-surveillance-adaptation`, HEAD `82eede2`
  (`exp: analyze detected actor-set effects on Collective`). No user edits existed.
- Created `feat/end-to-end-surveillance-mvp` from that HEAD. The filesystem sandbox
  initially denied creation of the Git ref; the authorized local branch operation
  subsequently succeeded through sandbox escalation. No remote operations occurred.
- Reviewed package layout, dataset preparation/source recovery, manifest splitting,
  video decoding, anomaly inference, actor loss/trainer, experiment configs and
  previous Collective execution/diagnostic reports.
- Preserved Collective Protocol v1, modality results, automatic-detector settings,
  predefined late fusion and matched-only oracle diagnostic. No held-out model
  inference was rerun and no historical result was rewritten.
- Confirmed the existing actor trainer is Collective-specific: its joint loss
  computes actor cross entropy even with actor loss weight zero. DCSASS adaptation
  must introduce genuine group-only supervision, not fabricated actor targets or
  merely a zero multiplier on the current actor loss.
- Confirmed anomaly inference already exposes scores, timestamps, merged intervals
  and aggregate score. Its timestamp mapping explicitly uses approximate uniform
  duration segments; integration must validate the C3D-unit/frame mapping before
  claiming precise temporal localization.

### Environment and supplied assets

Verified Python 3.13.5, torch 2.13.0+cu126, torchvision 0.28.0+cu126,
CUDA 12.6 availability and NVIDIA GeForce RTX 4070 Laptop GPU. No packages changed.

SHA256 checks of existing local assets are recorded in
`runs/end-to-end-initial-audit/assets.json`:

| Asset | Repository-relative location | Expected SHA256 |
|---|---|---|
| Selected Collective RGB checkpoint | `runs/collective/rgb_gt_v1/rgb_gt/seed_0/best.pt` | `40a5fc3687c277914335916bd46dff72ccde5e4b6518c677f193178090de7fcd` |
| Validated I3D feature archive | `checkpoints/i3d_mixed4f_collective_rgb_v1.pt` | `fe7fc30ca6f26430232e2e0f6bdffd8514478a071db065e452bff46f60f4e0c4` |
| Original converted I3D weights | `checkpoints/external/pytorch-i3d/models/rgb_imagenet.pt` | `2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538` |
| Faster R-CNN COCO_V1 | `checkpoints/external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` | `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384` |

The saved RGB metrics independently confirm group accuracy 0.775483870967742 /
macro F1 0.7946864755459309 and actor accuracy 0.7821637426900585 /
macro F1 0.7922563592073513. These remain Collective measurements, not surveillance
behavior results.

### Phase A1 — blocked: DCSASS not installed in checked locations

`data/raw/` contains only Collective Activity. No DCSASS manifest, labels, archive,
or dataset directory was found in the repository or the filename searches of
`D:/Downloads`, `C:/Users/manan/Downloads`, `D:/Github Repos`, `D:/Videos` and
`D:/Documents`. One unrelated `airport-navigation/.pytest_cache` directory denied
enumeration; the search is evidence for the checked accessible locations, not a
claim to have searched every disk location.

The repository documents manual acquisition from Kaggle but provides no vetted
automatic acquisition script. Following A1/N1, dataset-dependent audit, split freeze,
detection, extraction and training stop at this gate. Actual dataset counts,
readability, categories, source recoverability and duplication remain **unmeasured**.
The required `runs/dcsass/audit.json` / `audit.md` record this blocked status using
null measurements, rather than inventing an empty dataset audit.

Acquisition and label requirements: [required assets](end-to-end-required-assets.md).
No dataset or pretrained model was downloaded.

### Phase F1 — blocked: real Sultani assets unavailable

No UCF-Crime tree, UCF split/temporal files, C3D feature bags, compatible pretrained
C3D FC6 checkpoint, or real Sultani run is installed in the repository. `runs/`
contains Collective experiments only. The Milestone 1 report establishes synthetic
software verification, not a real anomaly benchmark. A real Sultani checkpoint and
raw-video C3D inference remain mandatory for the final deliverable.

### Verification commands and initial fixture failures

- `git status --short`, `git branch --show-current`, `git log -8 --oneline`.
- `.\.venv\Scripts\python.exe -m ruff check .`: passed.
- `.\.venv\Scripts\python.exe -m ruff format --check .`: 132 files formatted.
- `git diff --check`: passed.
- Exact initial `.\.venv\Scripts\python.exe -m pytest -q`: 143 passed / 277 setup
  errors caused by access denial to the existing Windows pytest temporary root.
  A single-test reproduction traced this to pytest's numbered-directory scan,
  not project source. A fresh workspace temporary root passed the targeted test.
- The first full workaround omitted its new parent directory, producing 143 passed /
  277 setup errors (`FileNotFoundError`); corrected by explicitly creating
  `runs/end-to-end-initial-audit/` before invoking pytest. No source workaround or
  environment downgrade was necessary.
- Final baseline command (output in `runs/end-to-end-initial-audit/pytest-final.log`):

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q --basetemp runs/end-to-end-initial-audit/pytest-temp-final -o cache_dir=runs/end-to-end-initial-audit/pytest-cache
  ```

  Result: **420 passed, 325 warnings, 115.72 seconds, exit 0**. Warnings are
  retained in the full log. No source changes were needed to restore baseline
  verification. The workaround only changes test temporary/cache locations.

### Decisions and next action

1. Honor the user-provided architecture and autonomous execution method; skill
   review/approval prompts do not override the explicit instruction to continue
   ordinary engineering work without repeated approvals.
2. Preserve the existing repository and create the requested branch in place,
   because the clean working tree does not need a separate worktree.
3. Stop real-data execution at missing mandatory assets. Do not freeze a split
   before inspecting actual data or publish unmeasured MVP claims.
4. Before DCSASS training, recover common UCF source identities and determine the
   Sultani held-out population. Preserve official UCF test-source reservations if
   using its benchmark; otherwise freeze a documented shared research source split.
   Independently splitting the two related datasets would jeopardize unified
   evaluation even if each individual manifest passed its own leakage check.
5. Resume at A2 after local DCSASS videos and original annotations are available.
   Reuse existing binary preparation and source parser; add exhaustive taxonomy,
   excluded-category, decoding, duplicate and grouped-split tests before freezing
   Human-Centric v1. Then execute B–E, F, G–M in the supplied gated order. Optional
   seeds/comparisons follow evidence-based phase E only.

## 2026-10-06 — Supplied C3D and DCSASS; autonomous execution resumed

The user supplied OpenMMLab Sports-1M C3D weights and DCSASS, and instructed a
five-minute delay while the dataset copy completes. UCF-Crime is downloading;
UCF-dependent execution remains pending. The initial blocked receipts are history,
not the current asset status.

### C3D validation and compatibility conversion

- Inspected every original tensor, checked finite float32 shapes and verified the
  original SHA256 against its advertised filename prefix.
- Added strict `c3d_conversion.py`, `scripts/export_c3d.py`, eight conversion/
  preprocessing regression cases and the existing discovery regression.
- Observed the missing converter fail, implemented the key mapping, then observed
  wrong-preprocessing tests fail before adding export provenance enforcement.
- The existing adapter remains C3D FC6; no architecture or environment change.
- Converted the supplied asset to `checkpoints/c3d_fc6_openmmlab_v1.pt` and compared
  its post-ReLU FC6 to pinned upstream source on two real GPU fixed inputs.
  Maximum absolute error 0.0; repeats bit-identical; output `[1,4096]`, finite.
- Modern channel means and fixed repository resize are documented as a research
  baseline, not original Caffe preprocessing equivalence. Details and commands:
  [C3D validation](c3d-openmmlab-validation.md).
- Full suite after the first conversion increment: 426 passed / 325 warnings,
  147.43 seconds. Subsequent preprocessing and dataset tests have targeted checks;
  the expanded full suite runs before full dataset execution.

### DCSASS installation and label inspection

- Began video-layout inspection after five minutes. Observed two complete-looking
  nested installations with the same file-count/byte totals. Choose the explicit
  inner root `data/raw/dcsass/DCSASS Dataset/DCSASS Dataset`; do not combine copies.
- Headerless CSV columns are observed `clip_id,category,binary_label`; videos are
  nested in original-source directories whose names end in `.mp4`.
- Reproduced and fixed `discover_videos` treating `.mp4` directories as video files;
  its new regression test fails before the `is_file()` filter and passes after.
- Preliminary enumeration: 16,639 files and 16,631 raw CSV rows; these are not yet
  full-decode counts. Identical repeated rows, blank labels, absent videos and
  extra files are recorded explicitly. No label is guessed, typo corrected, or
  clip deleted. Unverified labels will be excluded with a documented reason.
- Added observed-format label parsing, taxonomy, deterministic category-aware
  original-source splitting and full video audit helpers. Targeted checks cover
  short videos, corrupt files, invalid labels, exact duplicates, source parsing,
  binary-normal overrides, excluded categories and UCF test reservations.

### Common-source protection before UCF finishes copying

Acquired only the small official author metadata files `Anomaly_Train.txt` and
`Temporal_Anomaly_Annotation.txt`, pinned to the author's repository revision
recorded in `data/splits/ucf_authors_provenance.json`. No UCF video download was
started by this agent. The 290 published test annotation rows establish original
source reservations before DCSASS optimization. SHA256s:

- Training list: `ea91e03de7581bfa6b139e663ebb7511ab1651c33427f85741d3f497a2027744`.
- Test annotations: `3b9542413f2ed9e94f73bf0488c151b3d7d595a8d2ad30f524e9243a2ae2a17c`.

Next: complete expanded software gates, bounded decode preflight, full settled
dataset audit, duplicate/source checks and Human-Centric v1 split freeze. Preserve
the old missing-asset receipt separately before publishing the current audit.

Expanded software gate: **450 tests passed / 325 warnings / 115.41 seconds**;
Ruff lint and formatting passed (140 files). Ten-clip real decode preflight passed:
10 readable, 0 unreadable, five Normal/five Abuse, 320x240, 30 FPS, 60 frames,
2 seconds per clip, stable before/after installation inventory. This bounded
preflight is not a dataset-wide audit or benchmark. Full audit launched with four
decoder workers; output `runs/dcsass/audit_v1.json`, log `runs/dcsass/full-audit.log`.

### Phase A audit and final protocol outcome

Full decode completed in 163.46 seconds: 16,639 readable / 0 unreadable,
520 original sources, all 320x240 at 30 FPS. Durations 1–29 seconds (median 2).
No exact duplicate-content groups were found. Every video and label CSV in the
outer copy is byte-identical to the inner installation (copy verification receipt).

49 installed clips have no accepted annotation and are quarantined. Three
identical annotation repeats are documented/collapsed; two blank labels remain
unknown. No filenames or labels were repaired. Human-Centric v1 selects 6,491
clips / 203 sources. The rejected category-only split accidentally lacked positive
Fighting validation clips. Before any model execution, a deterministic whole-source
swap (`fighting002` to val / `fighting005` to train) repairs support. The rejected
manifest/receipt remain preserved; test assignments are unchanged. The final
manifest is `dcsass_human_centric_v1_final.jsonl`, not the rejected candidate.

Final sources train/val/test: **140/32/31**; clips **4,478/1,022/991**.
All six classes have train/val/test support; positive Fighting val/test each has
only one source. Final source overlap and content overlap are zero, and all 14
installed selected UCF official-test sources are reserved. The current audit
entry points were updated after preserving the original missing-asset receipts.
Detailed table and limitations: [DCSASS protocol](dcsass-human-centric-protocol.md).

### Phase B/C tooling and transfer groundwork

Added deterministic clip-local ten-frame windows with short-video edge replication,
OpenCV RGB preprocessing matching the existing Collective path, group-only input
batching and coverage reporting without GT precision/recall. Detection and I3D
caches register video/split/model identities and validate every resumed clip.
The default confidence remains 0.7, NMS 0.5, max actors 20; empty scenes are retained.
No original-video timestamps are guessed from derived clip numbers.

Transfer preserves the RGB projection and encoder pathway and resets only the
six-class group classifier. The unused actor head is frozen, and group-only cross
entropy never consumes actor logits/targets. Regression checks verify gradients,
representation preservation, no fabricated actor targets, source/content isolation,
and clip-local cache identity.

Full suite following the final source protocol: **457 passed / 325 warnings /
310.50 seconds**; subsequent transfer/window tests pass in targeted checks.
The next full gate covers the complete cache/transfer increment before real
detector/I3D preflight and full cache execution.

The complete cache/transfer gate passed: **460 tests / 325 warnings / 110.85
seconds**, Ruff lint and formatting (148 files), diff check. Real training/
validation-only preflight detected/extracted 12 class-spanning clips using the
unchanged COCO_V1 and I3D archives. Detection took 3.54 seconds; I3D cache phase
took 2.22 seconds (including empty-scene bypass). Reports and independent tensors
are under `runs/dcsass/preflight_cache_v1/`. No test clip entered this preflight.
Prospective seed-0 optimization settings are now explicit in
`configs/experiments/dcsass_human_rgb_detected_v1.yaml`: Adam 1e-4, batch 8,
20,000 iterations, milestones 5,000/10,000, validation macro-F1 selection,
covered-training-only inverse-frequency CE weights, binary threshold 0.5.

Next bounded execution: cache all 6,491 selected clips under
`runs/dcsass/human_rgb_detected_v1_cache`, reporting coverage including no actors.
The cache registration freezes sampling/model/filter/manifest identities before
execution; resumptions reject changed identities and validate each artifact.

Correction to the preceding preflight detection timing: the saved report records
**3.7792 seconds**, not 3.54. Its population is 12 clips / 9 covered / 3 no-actor
clips; extraction is 2.2221 seconds. Three empty scenes remain explicit records.
The source/taxonomy/C3D/cache milestone was locally committed as `322ab2b`.

### Phase D real preflight

Added a validated, preloaded covered-feature dataset, group-only trainer, explicit
covered-training class weights, validation-only checkpoint selection and a separate
held-out evaluation command that refuses existing evaluation output. The trainer
never opens test features. Model assets and split receipts are checked before work.

New full verification: **463 tests passed / 325 warnings / 189.47 seconds**.
The subsequent real five-iteration preflight completed on the registered 12
training/validation clips: group loss 3.1129 -> 0.5764, gradients finite, checkpoint
saved and selected checkpoint reloaded with identical validation macro F1.
This uses unit class weights solely for bounded plumbing verification; the real
run derives balanced weights from all covered training examples. The tiny
preflight validation scores (selected macro F1 0.1111) are non-benchmark evidence.
Outputs: `runs/dcsass/behavior_preflight_v1/`; log:
`runs/dcsass/behavior-verification/real-preflight.log`.

Full fixed-threshold detection/extraction is running in a separate registered
cache. At roughly 0.27–0.29 seconds/clip, detection is a bounded half-hour job.
No threshold adjustment, additional seed, test behavior inference or model-selection
change has occurred. UCF's pending download ZIP was inspected read-only; it lacks
a readable completed ZIP directory and remains untouched.
