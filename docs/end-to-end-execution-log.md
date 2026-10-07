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

## Phase B/C/D outcomes and E-C control gate

Full fixed-threshold detection completed in **1,864.98 seconds**. Coverage is
4,589/6,491 clips (70.70%); 1,902 clips have no actors. Train/val/test coverage is
3,211/4,478 (71.71%), 692/1,022 (67.71%), 686/991 (69.22%). These are coverage
statistics, not detector precision/recall/IoU; no DCSASS GT boxes exist.
Frozen I3D extraction completed in **986.14 seconds**, with 4,589 validated covered
feature tensors and 1,902 explicit empty records. No model or filter changed.

Covered training counts: Normal 1,398; Abuse 306; Assault 103; Fighting 62;
Robbery 1,135; Vandalism 207. Covered validation counts: 358/56/48/**2**/194/34.
Fighting validation now has only two covered clips from one source. The detector
threshold remains 0.7; no test-guided coverage tuning occurred.

The pre-training full gate passed **472 tests / 325 warnings / 143.22 seconds**.
Transfer seed 0 completed 20,000 iterations in about 934 training-loop seconds.
Validation selected iteration **1,900**, macro F1 **0.3059814160**, accuracy
**0.5014450867**, binary AUC **0.6197688422**. Validation predicts five classes;
no Fighting prediction occurs in that tiny covered validation population.
Checkpoint reload passed; `selection.json` freezes its SHA256 before test inference.

The single held-out transfer pass measured, on **686 covered test clips**:

- Accuracy **52.0408%**, macro F1 **0.2464143424**, balanced accuracy **0.2444544986**.
- Binary accuracy **58.1633%**, precision **0.6034985423**, recall **0.5782122905**,
  F1 **0.5905848787**, ROC-AUC **0.6303055593**, fixed threshold 0.5.
- Correct 357, incorrect 329, uncovered/abstained 305: total 991. No actor accuracy
  was computed. Multiclass figures are conditional on actor coverage.
- Per-class recall: Normal .5915, Abuse .1321, Assault .0417, Fighting **0**,
  Robbery .6349, Vandalism .0667. This is weak behavior recognition, not a successful
  violence benchmark. Full errors/confusions and no-actor cases are preserved in
  `runs/dcsass/human_rgb_detected_v1/seed_0/held_out/`.

**E-C decision:** run exactly one predefined controlled random-initialization
comparison. Only the Actor-Transformer representation initialization changes;
the new group head uses the same seed and initialization, and all architecture,
data/cache/split, optimizer, class weights, schedule and seed remain fixed.
Selection between runs uses validation macro F1 only. The original transfer test
artifacts are immutable; no threshold/label/split change, seed 1/2 or architecture
replacement is authorized by this gate. Regression verifies unchanged head seed
and changed representation weights. The random control output is separate.

## Provisional Sultani DCSASS clip baseline while UCF remains unavailable

The user instructed continued use of the supplied weights and DCSASS during UCF
download. The existing Milestone 1 already supports DCSASS binary bags. A separate,
prospectively registered real Sultani baseline now uses every readable clip with
a valid binary annotation, including the actor-excluded categories and no-actor
clips. This is **not** UCF frame-level reproduction and does not replace the pending
UCF benchmark. Positive frame annotations are unknown; evaluation is clip/bag only.

Generic population: **16,590 clips / 519 sources** (49 annotation failures remain
quarantined). Train/val/test sources **333/81/105**; clips **10,642/2,590/3,358**.
Normal/anomaly counts: train 5,680/4,962; val 1,608/982; test 2,212/1,146.
All 203 actor-source memberships are preserved, and all installed published UCF
test sources stay held out. Test reservations increase the generic test fraction;
no source overlap is permitted. Protocol: `runs/dcsass/sultani_generic_v1/protocol.json`;
manifest: `data/manifests/dcsass_sultani_generic_v1.jsonl`.

Settings preserve 32 segments, C3D FC6, MIL ranking, sparsity/smoothness and Adagrad
1e-3 from the existing baseline; seed 0, 20 epochs and 30 positive/normal pairs
are explicit in `configs/experiments/dcsass_sultani_generic_v1.yaml`. No test clips
entered the eight-clip real C3D preflight. Full suite: **473 passed / 325 warnings /
135.63 seconds**. Real five-step MIL preflight produced finite gradients/losses
1.9752 -> 1.7670 and bit-identical checkpoint reload. Outputs are separate from
full execution. Full extraction must follow the latest control/software gate.

## Integration and interface groundwork (not measured end-to-end inference)

Added tested routing, normal bypass, model disagreement and anomaly-preserving
no-actor fallback. Raw C3D timestamp mapping now follows exact 16-frame unit
partitions and clamps padding; imported bags without unit provenance retain their
documented uniform approximation. Scores/loss/model architecture are unchanged.

Gradio 6.29.1 was installed as a pinned optional dependency from PyPI without a
torch/torchvision change. The local interface builds and launches at loopback only;
system fonts and disabled analytics support offline presentation. Browser review
found dark-mode contrast and width issues; theme corrections are in progress.
The interface explicitly reports absent trained assets and never substitutes
preflight/random weights. Cached examples and uploaded live inference are distinct.
No end-to-end accuracy, usable demo clip cache or latency benchmark is claimed yet.

The latest gate passed **474 tests / 325 warnings / 136.26 seconds**, with Ruff
and formatting clean. Random control launched with `--initialization random`,
separate output `runs/dcsass/human_rgb_detected_v1/random_init_seed_0/`; no other
training setting changed. Full generic C3D extraction launched after the successful
real five-step Sultani preflight. Log: `runs/dcsass/sultani-full-extraction.log`.
The source-safe generic test reservation is larger than 15% because installed
official UCF test sources cannot be moved into optimization.

## Controlled initialization outcome and independent deployment review

The predeclared E-C random control completed 20,000 iterations in 894.8 seconds.
Validation selection chose iteration 2,900, SHA256
`7870e512787e74cd76263df28e40c8bc39ecb8a3d0a5b2b8b6a7e72a218a8c69`.
Validation macro F1 = **0.4111765855**, accuracy = **0.4913294798**, compared with
transfer **0.3059814160 / 0.5014450867**. Select random initialization by validation
macro F1 only. The two covered Fighting validation clips make this difference
unstable: one correct Fighting prediction contributes F1 0.6667 for that class.
This does not establish strong aggressive-behavior recognition. No other settings
changed and no further tuning is planned. The original transfer held-out artifacts
remain intact. Selected-control test evaluation has not yet run at this entry.

Applied the requesting-code-review skill, which explicitly required a fresh
read-only reviewer; applied receiving-code-review to verify its findings before
implementation. The reviewer independently confirmed the manifest hash, six-class
mapping, zero actor source/content overlap, and identical split assignment across
all 203 actor/generic shared sources. It ran 41 CPU tests and no training/test jobs.

Four concrete deployment gaps were reproduced: live Sultani preflight acceptance,
truncated/inaccurate decode count versus exact frame mapping, unchecked live I3D/
detector hashes, and cached examples lacking model/policy provenance. Fixes reject
explicit preflight markers, require complete consistent decode counts, compare live
backbones with training hashes, and pin full checkpoint/preprocessing/policy identity
for live/cached results. Completed frozen checkpoint-selection receipts are required
for live learned models. Offline caches can validate pins without loading models.
Also tightened detection-cache filters from confidence-only to the entire frozen
DetectionConfig. This is a guard correction, not a change to actual cached actors.

Red regression cases reproduced these failures before fixes; 33 targeted tests
then passed. A broader suite exposed one old incomplete synthetic filter fixture;
it is being updated to the full detector metadata contract. Ruff and formatting
checks pass for 170 files before the later selection helper. Added separate stage
timing instrumentation without changing feature values or learning settings.

The Gradio preview was restarted after dark-mode/width fixes, but browser inspection
then failed because the Chrome debugger was unattached. This is a browser-review
limitation; application startup still reports its loopback URL. Final UI validation,
usable example caches, actual cascade inference and latency remain outstanding.

The revised full suite passed **494 tests / 325 warnings / 232.70 seconds**;
Ruff, formatting (173 files) and diff check passed. Subsequent targeted cascade
guards also passed eight tests, rejecting fractional/bool window budgets and any
invalid anomaly score even beyond the top-window budget. The reviewer followed up
with 42 CPU tests and found all five findings resolved, with no remaining findings
in its scope. Final asset pins, live cascade results and demo caches remain pending.

Frozen E-C selection was written with `scripts/select_dcsass_behavior.py` after
checking the two resolved configurations differ only in initialization. The helper
never opens held-out artifacts. Receipt: `initialization_selection.json`. Then
`scripts/evaluate_dcsass_behavior.py --run .../random_init_seed_0` completed one
held-out pass. Covered test accuracy **0.5102040816**, macro F1 **0.2140469448**,
balanced accuracy **0.2286685165**. Binary fixed-0.5 accuracy **0.5830903790**,
precision **0.5962566845**, recall **0.6229050279**, F1 **0.6092896175**,
ROC-AUC **0.6307228505**, false-positive rate **0.4603658537**. Population remains
991 total / 686 covered / 305 abstained, with 350 covered correct / 336 incorrect.

The selected control has zero correct held-out Abuse, Assault and Fighting
predictions; Normal and Robbery dominate outputs. Better validation macro F1 did
not generalize, consistent with its tiny covered Fighting support. Retain the
validation-selected control, preserve transfer results, and do not select using
test. These two experiments establish a weak surveillance adaptation rather than
reliable violence recognition; they do not prove fundamental non-learnability.
Continue the requested interpretable research demo with this limitation visible.
Complete metrics/confusions are documented in `docs/dcsass-surveillance-results-v1.md`.

UCF still has no installed raw tree; the visible archive remains incomplete at
1,423,573,551 bytes. Requested its final folder asynchronously while continuing
the supplied-data work. No downloader, partially copied data or archive was changed.

## Cascade software / presentation-example milestone

The expanded suite passed **506 tests / 325 warnings / 221.17 seconds**. Ruff and
formatting passed (180 files), and the independent deployment follow-up found no
remaining findings in its scope. These are software gates, not end-to-end metrics.
Real Sultani training still awaits completion of its 16,590-bag feature extraction.

Added a selection freezer that refuses incomplete/preflight Sultani runs, reloads
the best checkpoint and verifies validation bag ROC-AUC before writing its receipt.
`scripts/evaluate_registered_sultani.py` permits one frozen held-out bag pass and
saves all 32 scores per clip for later analysis without repeated model evaluation.
Existing Sultani objective/optimizer/selection behavior is reused unchanged.

Stage timings now distinguish decoding, input preprocessing, C3D, Sultani, person
detection, I3D, Actor-Transformer and other overhead. Lazy demo model-loading time
is recorded separately from pipeline inference. Live loading seeds deterministic
evaluation and configures the supported cuBLAS workspace. Actual latency is pending.

`scripts/prepare_demo_examples.py` copied seven real test clips into `data/examples/`,
leaving raw data intact. It verifies video/manifest/model identity and chooses the
first covered clip by ID for each label (at least two seconds), independently of
prediction correctness. A seventh post-hoc high-confidence mistake is included for
discussion. `configs/demo_examples.json` records labels, sources, baseline outputs,
hashes and this selection policy. These examples are not a new accuracy benchmark.

Examples: Normal `Abuse004_x264_0`; Abuse `Abuse004_x264_1`; Assault
`Assault011_x264_10`; Fighting `Fighting003_x264_20`; Robbery `Robbery018_x264_10`;
Vandalism `Vandalism015_x264_10`; limitation `Fighting003_x264_24`. Several chosen
representatives are actual baseline mistakes. No success-only presentation selection.
First/middle/last frames decoded successfully for all seven and were manually
inspected in `outputs/demo/example-contact-sheet.jpg`. Dataset clip labels do not
imply frame-level event annotations; the 17-second Abuse examples emphasize that
a middle ten-frame window may omit much of a labelled clip's interaction.

Gradio API checks against the running loopback app passed empty-input and missing
Sultani responses without crashing (`runs/dcsass/demo-api-smoke.json`). Browser
debugger attachment remains unavailable despite a fresh tab inventory; final visual
review is still pending. Example selection now previews its video, with explicitly
allowed local example paths. The demo config points to the separately identified
DCSASS Sultani population and selected random behavior checkpoint. Four model hashes
are pinned; the missing Sultani pin will be filled only after full validation selection.

README, measured behavior report, draft demo runbook and professor talk track now
state current results and pending assets. No claim of UCF frame ROC-AUC, reliable
six-class violence recognition, completed live cascade, or usable cached example
results has been made. Next: complete extraction, real Sultani train/freeze/test,
then pin deployment and measure actual cascades/caches before final readiness.

## Real DCSASS Sultani seed 0 completed

Full C3D extraction completed **16,590** finite bags with the approved backbone;
feature-manifest SHA256
`f7124bb4903285531e1bd4a4214514eb1e0d7705a3d59a06da5de58831820893`.
Confirmed 10,642 train / 2,590 validation / 3,358 test clips and 519 sources.
The pre-training gate passed **509 tests / 325 warnings / 162.26 seconds**, with
Ruff and formatting clean for 184 files.

Reused `scripts/train_sultani.py` with the frozen generic config and completed
20 seed-0 epochs on CUDA. PowerShell represented the first INFO stderr line as
`NativeCommandError`, so its process-wrapper exit was 1. This was a shell logging
artifact: the log records all 20 epochs and the final saved checkpoint, and the
subsequent completion/reload/validation gate passed. No Python training exception
or numerical failure was reported. Preserve that raw log rather than rewriting it.

`scripts/freeze_sultani.py` confirmed completed epoch 20 and selected **epoch 2**
by validation bag ROC-AUC **0.6705405001** (the earlier tail-only progress update
showed later epochs around 0.6691). Reloaded validation reproduces selection.
Selected SHA256: `cb51fdd02c3ba2e258c648a59673d9aa701e7f5132470338832096a3ac159872`.
Fixed threshold stays **0.5**, with no test/demo tuning. CPU validation/evaluation
used two OpenMP/MKL threads for bounded scoring; learned settings are unchanged.

One registered held-out bag pass completed on **3,358** clips: ROC-AUC
**0.6582629572**, precision **0.4569444444**, recall **0.5741710297**, F1
**0.5088940449**, false-positive rate **0.3535262206**. Confusion matrix
`[[1430,782],[488,658]]`. Per-clip 32-score predictions are saved for analysis.
These are real but weak DCSASS bag metrics, not UCF frame metrics or measured
temporal localization quality. See `docs/sultani-dcsass-results-v1.md`.

The user confirmed UCF will take roughly three more hours and named
`C:/Users/manan/Downloads` as its location. Use the explicit path directly, even
though the message described it as a D-drive Downloads folder. Do not require a
second repository copy. Continue the provided-data MVP while that asset is pending.

Media inspection found no external FFmpeg executable. Gradio's `include_audio=False`
preprocessor attempted mandatory re-encoding before the callback, changing frames
and failing when FFmpeg was absent. A red regression reproduced it; preserving the
original upload fixes it without installing binaries. Models consume RGB frames.
Also hash-check installed example videos before both cached and live processing.
The cache builder's unit tests verify genuine-call serialization and offline reuse;
actual cache generation remains pending the latest full gate.

All five demo model hashes are now pinned after both validation selections. Added
a checkpoint-verification CLI that loads the exact local model interfaces without
scoring test data. Next: verify loading, bounded actual cascade, cache seven clips,
measure timings/coverage and test the running demo's live/cached modes.

## Trained cascade and offline demonstration verified

All five installed deployment hashes and both completed selection receipts passed
actual model loading (`runs/dcsass/demo-asset-verification.json`). Genuine GPU
cascade inference completed on every original example. Six caches were generated
and the identical Robbery preflight cache was reused; original receipt:
`runs/dcsass/integration-demo-v1.json`. The initial seven cases retain a Normal
false alert, missed Assault/Vandalism anomalies, wrong behavior labels and an
anomaly/Normal disagreement. No test-derived parameter adjustments were made.

Added a disclosed supplemental Normal bypass example, `Abuse036_x264_10`, chosen
from saved held-out scores below the already frozen 0.5 threshold. All original
failures remain installed. Actual raw-video inference confirmed no anomaly at
0.4963226914; receipt `runs/dcsass/normal-bypass-receipt.json`. Eight examples now
have genuine caches and completed status. This post-hoc presentation choice is
not checkpoint selection, threshold selection or a representative benchmark.

There are 15 analyzed windows in the curated set, 14 with actors and one without.
The actual empty window retains a generic anomaly alert and null behavior outputs.
Case accounting is four true anomaly alerts, one false alert, two misses and one
correct Normal bypass; do not generalize these selected-case counts to population
accuracy. Localization metrics are unavailable without temporal GT.

Restarted the agent-owned loopback Gradio server after configuration/media fixes.
`scripts/validate_demo_api.py` passed five real requests: cached bypass, cached
Robbery, cached disagreement, live Robbery and an uploaded raw video. Timelines,
galleries, interval tables and JSON downloads render through the actual server API.
Preview and empty-input handling pass. Receipt: `runs/dcsass/demo-api-validation.json`.
The original video bytes are preserved; FFmpeg is not a mandatory upload dependency.

Actual fixed-input GPU repeat produced exactly equal segment scores, boxes,
detector confidences, behavior probabilities and final alert. Warm Robbery
inference: 1.0986263 s on NVIDIA GeForce RTX 4070 Laptop GPU. A real validation
no-actor middle window (`Abuse014_x264_0`) abstained and skipped I3D/Transformer.
The helper initially used the wrong split name `validation` instead of `val`;
corrected that verification-only script and reran. All five real model interfaces
also ran on CPU: 27.4000825 s and the expected Robbery review alert, with numerical
CPU/GPU equality not claimed. Details: `runs/dcsass/real-runtime-verification.json`.

The browser tab is visible in inventory, but Chrome attachment still times out
at focus setup. Final visual/playback inspection remains pending; API verification
is not presented as visual proof. The runbook includes a manual classroom check.
Updated README, asset inventory, execution gates, runbook, talk track and separate
Sultani/integration evidence with completed results and retained limitations.

Independent read-only review found no defects, verified all model/video/cache
identities and original-source test membership, loaded eight caches without model
construction, and passed 17 targeted CPU tests. The final whole suite passed
**510 tests / 325 warnings / 162.54 seconds**. Ruff and formatting are clean for
187 files; diff check passes. These software tests are not benchmark results.

The explicit Downloads root was checked read-only. `archive.zip` is the already
installed DCSASS archive (33,304 members), not UCF-Crime. The remaining
`Unconfirmed 235055.crdownload` is incomplete (193,867,723 bytes at inspection).
No extracted UCF tree is available. Do not train on partial data or redownload it.
The UCF frame-benchmark phase remains blocked on this mandatory external asset;
the provided-data surveillance demo is usable locally. Next: validate the final
UCF acquisition once complete, preserve shared held-out DCSASS sources, freeze
its compatible protocol before extraction/training, and complete frame evaluation.

## UCF-Crime external-drive acquisition available

User supplied `E:/anomaly-detection-dataset-UCF` with extracted videos and ZIPs.
Read-only inventory found 1,950 videos / 104,888,485,756 bytes, including 50 duplicate
filename event-recognition Normal videos. The anomaly-detection population has
exactly all 1,900 author-listed filenames: 1,610 train and 290 test, no missing,
extra or overlapping names. Event-recognition videos and ZIPs are not used.
The root train list and temporal annotations match pinned author bytes exactly;
the nested train file is empty, so use the verified root list.

Ruling: preserve shared DCSASS source assignments; exclude its held-out test
sources present in the authors' train list, while keeping the full original UCF
290-video test set. Remaining unshared training sources get a deterministic
category-aware 15% validation split. This prevents cascade source leakage and
changes the training population relative to the paper; disclose the deviation.
No UCF model/test scores have been inspected. Config remains the existing seed-0,
20-epoch Adagrad/MIL baseline; validation bag ROC-AUC and threshold 0.5 stay fixed.

Regression tests exposed original archive Normal-folder incompatibility and
cross-volume `os.path.relpath` failure. Added explicit archive-folder support,
event-recognition exclusion and absolute external-drive path fallback. Added exact
`c3d_units` frame projection matching the already validated 16-frame extraction
partitions; old uniform imported-feature modes remain available. Seven regression
tests passed after observed failures. A subsequent cache regression passed after
the missing-function failure, proving byte-bound reuse and stale/nonfinite rejection.

New audit/freeze CLI: `scripts/prepare_ucf_surveillance.py`. It checks author
provenance, shared-source isolation, metadata, first/last reported frames, content
hashes and duplicates. Full decode is verified at extraction before training, not
claimed from metadata probes. Resumable extraction CLI:
`scripts/extract_ucf_sultani.py`, with separate train/validation-only preflight.
Plan/settings are in `docs/ucf-sultani-execution.md`. Next: software gate, actual
audit/source freeze, bounded extraction/backprop/reload, then full extraction.

The first preparation run correctly rejected Arson011's endpoint: official end
1267 versus 1266 reported/decoded frames. A complete metadata check of the 290
annotated videos found exactly five overruns, all one or two frames. Arson011,
Arson016, Explosion033, Fighting003 and Shooting015 each match original ZIP CRC
and uncompressed byte size; full decode matches metadata. Evidence saved in
`runs/ucf-crime/annotation-boundary-diagnostic.json` and
`annotation-boundary-integrity.json`. No incomplete extracted file was found.

Ruling: explicitly intersect these verified-original annotation intervals with
available frames, record original/effective endpoints, and reject larger overruns
or intervals starting after the video. This changes no label on an existing
frame. Raw videos/annotations remain untouched; the generic API stays strict by
default. Regression observed missing-keyword failure, then passed the bounded
intersection/default-rejection tests. Nine new regressions now pass. Full gate
before this annotation change passed 518 tests / 325 warnings / 184.52 seconds;
the post-change full gate and read-only hash/boundary-frame audit are running.

Independent review confirmed source protection, external-drive paths and bounded
endpoint handling, and found two Important scientific guards to strengthen before
extraction/training: registered frame evaluation could default to uniform repeat
rather than the frozen C3D projection, and downstream feature loaders ignored
cache digests after extraction. Reproduced both missing guard behaviors in tests.
Added per-bag `feature_sha256` to new UCF manifests and byte verification in the
shared trainer/validation/evaluation loader; old manifests remain readable.
Completed selection receipts now pin parent protocol SHA plus prospective
evaluation mode/projection. Registered test rejects policy/protocol drift before
held-out directory creation and resolves omitted projection from the receipt.
Twenty-one targeted tests passed. The prior full gate passed 519 tests / 325
warnings / 192.05 seconds; the full post-review-fix gate is running. No held-out
model predictions or test-informed hyperparameter changes have occurred.

Full post-review gate passed **521 tests / 325 warnings / 215.84 seconds**, Ruff
and formatting clean (191 files), diff check passed. Independent reviewer verified
both fixes with 19 targeted tests. The actual audit completed 1,900 videos with no
file failures in 525.60 seconds, but correctly blocked protocol freeze on exact
duplicates. There are 21 groups: eight Normal train/test pairs, eleven Normal
train/train pairs, one Assault050/Robbery138 binary-positive train alias, and one
Normal test/test pair. All 42 duplicate-group files match archive CRCs and sizes.
Both anomaly aliases are absent from the earlier DCSASS manifests, so no frozen
DCSASS experiment is changed by this finding.

Ruling: remove exact duplicate *training entries* before splitting, prioritizing
all published test entries; retain the 936/937 test duplicate and disclose its
weighting. Keep raw data and binary labels unchanged. This resolves eight actual
content-leakage pairs and prevents train/validation aliases. A regression failed
for the missing helper, then all 12 UCF tests passed, including conflicting-label
rejection. `scripts/freeze_ucf_protocol.py` then froze the saved complete audit:
1,309 train / 254 validation / 290 test (289 unique test contents), binary counts
664/645 train, 117/137 validation, 150/140 test. Excluded 20 duplicate training
copies plus 27 shared DCSASS held-out sources. Explicit source/content overlap is
zero. No model scores have been used for these preparation decisions.

Raw manifest SHA256 `13a071d986ef4c4d62691a11b16b3052955b9b98bcbd2019b52f579437b62826`;
audit SHA256 `03aff4b4bfc8109affc98758429d0e726ce5ce9b723240102edbf718dfea9a77`.
Resolution is uniformly 320x240; actual FPS values 25/29.970029/30 are preserved.
Reported total 13,768,423 frames / 127.51 hours; maximum video is 32,550.1 seconds.
Streaming extraction is necessary. The new full gate/review is finishing before
bounded GPU extraction and backprop/reload.

Final preparation quality gate passed **522 tests / 325 warnings / 165.24 seconds**,
Ruff and formatting clean for 192 files, diff check passed. Independent review
passed 12 tests and verified exact accounting of every audited file, both excluded
populations, all receipt hashes, shared assignments and zero source/content leakage.
Existing deployed model identities also reload correctly. Started bounded
eight-video train/validation-only real C3D extraction with fixed batch 4, seed 0
and two OpenMP/MKL threads; no test scoring. Next: five real MIL iterations,
checkpoint reload, representative extraction timing, then full streaming extraction.

Eight real UCF train/validation videos extracted in 3.8975 seconds, with full
sequential decode/count checks, finite `[32,4096]` bags and registered feature
checksums. Five real MIL iterations passed: losses 1.97010, 1.86905, 1.78078,
1.74711, 1.67920; finite gradients and exact checkpoint reload confirmed. This
preflight checkpoint remains dry-run and cannot be served. No held-out scoring.
Receipts: `runs/ucf-crime/sultani_shared_safe_v1/preflight_cache/extraction.json`
and `preflight_training/report.json`. Representative C3D benchmark started on
training frame-count median (2,189 frames) and 95th percentile (24,320 frames),
using unchanged batch 4 and backbone preprocessing. Its valid caches can seed
the full run; no architecture or extraction precision changes are being made.

Benchmark completed: median-training sample 2,189 frames in 3.0172 seconds;
95th-percentile sample 24,320 frames in 33.2855 seconds. RTX 4070 Laptop peak
allocated CUDA memory 678,444,544 bytes, fixed batch 4. P95 stages: decode 8.5797 s,
preprocessing 10.5067 s, C3D 13.1275 s. Both cache entries are finite/byte-bound.
Started full unchanged streaming extraction at UTC 16:15:35, command
`python -u scripts/extract_ucf_sultani.py`, log `runs/ucf-crime/full-extraction.log`.
This is expected to take multiple hours; keep the external drive connected.
No UCF model or held-out score has been selected/evaluated yet. Next: inspect full
completion receipt, train seed 0, freeze validation choice and run registered frame
evaluation using its prospective exact C3D projection.

## Browser demo validation while UCF extraction runs

Chrome control recovered on a fresh local tab. The cached Normal example played
to its four-second end (readyState 4, ended true, no video error). Cached Robbery
and anomaly/Normal disagreement rendered correctly with timeline, actor galleries
and interval evidence. Visual inspection exposed low table-header contrast and
stale previous predictions when choosing another source. Regression reproduced
missing evidence-clear event; added source/upload/clear handling, preserving newly
uploaded bytes, and explicit uncalibrated-probability wording. Contrast now uses
RGB(228,237,242) background with RGB(33,60,78) text. Eight targeted tests passed.

Restarted only the agent-owned demo server through its known exec session; UCF
extraction continued independently. Browser inspection confirmed old evidence
clears, upload mode is selected on video clear, and disagreement remains visible.
Saved actual screenshot: `outputs/demo/visual-disagreement.jpg`. The automated
file chooser rejects selection because Chrome's ChatGPT extension lacks file-URL
access. Did not expand that permission; dismissed the intercepted chooser. This
is an automation restriction, not an application upload failure; five real cached,
live and upload API requests pass after the fixes. The runbook retains a manual
presentation check.

Full post-UI gate passed **523 tests / 325 warnings / 223.80 seconds**. No model,
cache values, scientific splits or thresholds changed. Full UCF extraction is
healthy and progressing; no held-out model evaluation has occurred.

## Prospective full cascade evaluation and asynchronous demo safeguards

Registered the full 991-clip / 31-source human test population prospectively in
`docs/cascade-evaluation-protocol-v1.md`. The new evaluator checks both model
selection receipts, human manifest, protected UCF source membership, raw clip
bytes and model identities. Canonical immutable registration rejects alternate
output paths, completed reruns and threshold/window/model drift; resume scores
only pending clips. Reports separate weak clip alerts, conditional six-class
behavior, coverage and stage latency. No interval or actor ground-truth claim.
Execution waits for the validation-selected real UCF checkpoint.

Independent review caught asynchronous UI callbacks that could restore old
evidence after the user changed source. Added per-session monotonically increasing
input revisions, checked before and after preview/inference (including errors).
Stale callbacks return skips; source previews use user input events so backend
upload updates do not erase the new upload. Threaded regression reproduces a
delayed inference followed by a new source and confirms no stale output. Five
real cached/live/upload requests passed against the restarted server, receipt
`runs/ucf-crime/demo-api-revision-guards.json`. Full quality gate/review running.
UCF extraction has reached 780/1853 retained videos without reported failures.

Follow-up review reproduced an additional source change during gallery rendering;
the callback now checks its revision again after constructing the full response.
The new threaded regression failed before this fix and passed after it. Also
added promised clip-alert accuracy using the all-clip confusion denominator.
Independent review found no remaining actionable issues. Final gate: **538 tests,
325 warnings, 180.65 seconds**, Ruff clean, 200 files formatted, diff check passed.
The restarted server passed five cached/live/upload API requests again, receipt
`runs/ucf-crime/demo-api-render-guard.json`. Actual browser checks confirmed Normal
bypass, Robbery output, clearing on input change and retained model disagreement;
screenshot `outputs/demo/visual-render-guard.jpg`.

Read-only cascade population verification: 991 clips / 31 sources; class counts
Normal 557, Abuse 65, Assault 32, Fighting 10, Robbery 275, Vandalism 52.
Fourteen sources are official UCF test sources and seventeen were explicitly
excluded from UCF optimization. No held-out inference occurred. Full extraction
now 886/1853, healthy. Next: wait for full extraction receipt, inspect all cache
identities/counts, run frozen seed-0 UCF training, freeze validation selection and
execute its registered frame test before versioned cascade evaluation.

Prepared an ignored local orchestration helper
`runs/ucf-crime/continue-training-after-extraction.py`, waiting in its own process.
It validates the full extraction completion receipt, all 1,853 feature hashes,
finite `[32,4096]` tensors, cache provenance and unchanged source/label metadata,
then invokes the existing frozen 20-epoch seed-0 training CLI. It stops after
training for validation inspection; it cannot score test data. Configuration and
protocol hashes are captured before waiting; an existing training run is preserved.
Launcher uses two OpenMP/MKL threads and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Fixed only the helper's UTF-16 PowerShell log decoding, restarting its waiting
process with a separate `training-after-extraction-v2.log`; extraction was untouched.

Generalized the API verification script with `--config`. It verifies server model
provenance and all 32 anomaly scores against genuine frozen example caches rather
than hardcoding the old scorer's classifications. Five actual cached/live/upload
requests passed (`runs/ucf-crime/demo-api-versioned-config.json`), Ruff/format/diff
clean. This is software consistency evidence, not a model accuracy measurement.

Completion audit added at `docs/end-to-end-completion-audit.md`, mapping all twenty
deliverables to inspected evidence and remaining UCF-specific verification.
Confirmed actual DCSASS held-out confusion/coverage and explicit absence of box
GT in the detection report. Both extraction (`44384`) and training launcher
(`38498`) remain live. D: has 327.73 GB free and E: has 1,628.23 GB free, adequate
for the remaining feature/checkpoint artifacts without copying original footage.

Before any UCF scoring, fixed fourteen timeline illustrations in
`runs/ucf-crime/sultani_shared_safe_v1/timeline_selection.json`: first author-held-out
video by ID per category, including Normal. No model output informed selection.
After the registered frame pass, render saved scores with the supplied temporal
annotations and exact C3D-unit boundaries. These illustrations are not another
test-selection/tuning pass. Extraction has progressed to 1183/1853 with no failures.

Long-recording extraction gates also completed without failure:
`Normal_Videos307_x264`, 628,020 frames / 20,934 seconds, in 772.99 seconds;
`Normal_Videos308_x264`, 976,503 frames / 32,550.1 seconds (the longest installed
video), in 1,195.06 seconds. The unchanged batch-4 path produced valid caches and
continued to 1350/1853. Live process polls and GPU activity confirmed these were
compute waits, not terminal failures; no restarts or scientific setting changes.
The full extraction completion receipt remains pending, so the launcher has not
started training or any held-out scoring.

Full UCF extraction completed with exit code 0: all 1,853 retained original videos
fully decoded with frame-count checks in 17,284.6454 seconds (4.80 hours), two valid
benchmark entries reused. Feature manifest SHA256
`2a888b0e0da7b0115efab7a8641f67490ef0a7d4cf9ef17238aac5ca9c9302e0`.
The launcher independently validated every feature's byte identity, finite
`[32,4096]` values, cache provenance and unchanged source/label metadata. Counts
remain 1309 train / 254 validation / 290 test. Validation receipt:
`runs/ucf-crime/sultani_shared_safe_v1/feature_validation_before_training.json`.
Protocol SHA256 `a163f05823825daf647622eb3f776ca3de4cefaf77c55ccf0aac6b2ecbf40c9d`;
training config SHA256
`3fe7c63a81e6b5bdd1ec121903c1a6d2f4c5bca31bfa0a86d6852614a09b4394`.

The fixed seed-0, 20-epoch CUDA training command started automatically only after
those gates. At epoch 8, loss decreased 1.754039 -> 1.160637 and validation bag
ROC-AUC improved 0.875413 -> 0.901803, without held-out scoring. Log:
`runs/ucf-crime/seed-0-training.log`. Next: verify completed training/reloaded
validation selection, freeze it, then perform the one registered frame pass.
## Cascade progress recovery and bounded atomic write retry

The first registered full cascade process stopped after saving clip
`Robbery054_x264_8`: Windows denied replacement of `progress.json.tmp` over
`progress.json` (WinError 5). The committed progress receipt contained 478 clips;
the complete staged receipt contained 479. No checkpoint, threshold or source
protocol changed. Both original progress versions were preserved under
`runs/integration/ucf_sultani_human_v1/recovery/` before recovery.

Every one of the 479 saved results passed file SHA256, clip/source/label,
original input SHA256, schema, five-model provenance and frozen routing-policy
validation. The only difference was the one completely serialized extra clip.
The recovery receipt declares no model scoring. An initial recovery-helper
assertion compared result policy to the smaller identity policy; inspection showed
the result's additional fixed confidence/fallback fields, which were then checked
explicitly. This failed assertion preceded all mutations.

`write_json` now retries only PermissionError during atomic replacement, at most
eight attempts with bounded backoff; a permanent denial preserves the old file
and the staged data. Regression tests first failed, then passed for transient
and permanent denial. Full gate: 540 passed, 325 warnings, 142.40 seconds;
Ruff, formatting (201 files), and diff check passed.

Recovery command: `python runs/ucf-crime/recover-cascade-progress.py`, receipt/log
`recover-cascade-progress-v2.log`. Resume command:
`python scripts/evaluate_surveillance_cascade.py --config configs/surveillance_demo_ucf_v1.yaml --resume`.
It uses the same canonical registration, verifies/reuses the 479 outputs, and
scores only the 512 remaining clips. Original failed execution log is retained;
resumed log is `runs/ucf-crime/full-cascade-evaluation-resume-v1.log`.

## UCF selection, one registered frame test and deployment

Fixed seed-0 training completed all 20 epochs, loss 1.754039 -> 0.894434.
Epoch 20 won validation bag ROC-AUC 0.910911; reloaded CPU validation matched
the training selection. Checkpoint SHA256
`7e5b548ea6260614209a727a947e6270a1a1af2185d29accf1778f91b4af7c66`.
Selection bound the prospective protocol, feature manifest, evaluation mode
`frame`, `c3d_units` projection and fixed 0.5 threshold before test scoring.

`python scripts/evaluate_registered_sultani.py --run runs/ucf-crime/sultani_shared_safe_v1/seed_0 --manifest data/manifests/ucf_sultani_shared_safe_features_v1.jsonl --mode frame`
completed once: 290 videos, 1,111,808 frames, ROC-AUC 0.7440528; precision
0.1911977, recall 0.5188646, F1 0.2794282, negative-frame false-positive rate
0.1801639. Confusion `[[842357,185113],[40578,43760]]`. No test-driven changes.
Fourteen predeclared timelines and the ROC plot read saved predictions only;
secondary bag ROC-AUC 0.8539048 is clearly separated from frame results.
Full report: `docs/sultani-ucf-results-v1.md`.

Verified all five CUDA deployment assets. Versioned UCF configuration and eight
genuine example caches preserve the same original clip choices and all failures.
The default config now uses the selected UCF scorer; the exact older DCSASS
configuration remains `configs/surveillance_demo_dcsass_v1.yaml`. All behavior,
I3D and detector settings remain unchanged. Validation-only live cascade preflight
on Abuse014_x264_22 and Robbery016_x264_27 exercised actual suspicious-window
behavior inference before the canonical held-out cascade.

Five actual cached/live/upload API requests passed against the new deployment
(`runs/ucf-crime/demo-api-ucf-v1.json`). The original Robbery example is now a
Sultani miss; the Normal example is a retained Robbery false alert. Browser
inspection confirmed the latter's genuine timeline, overlays, disagreement and
no-actor window (`outputs/demo/ucf_v1/visual-false-alert.jpg`). Demo outcomes are
curated illustrations, not benchmark metrics.

## Complete registered cascade, final runtime and quality gates

The resumed canonical cascade completed with exit 0 on all 991 clips / 31
protected sources. All result SHA256s, original input hashes, source/labels,
schema and five-asset provenance were independently verified; saved-score
summary recomputation exactly matches `metrics.json`. Its SHA256 is
`58eb76d828dbd1cd519437cb76f07a592d2b85fb8156d971b9be7f5d6f0e3a51`.
The completion receipt and original registration are preserved.

Clip alerts: accuracy 0.5418769, ROC-AUC 0.5117938, precision 0.4766355,
recall 0.4700461, F1 0.4733179, false-positive rate 0.4021544;
confusion `[[333,224],[230,204]]`. Routed 428, bypassed 563; 311 covered,
117 routed no-actor clips, 680 behavior abstentions; 1,020 analyzed windows,
339 no-actor windows. Conditional accuracy 0.4565916, macro F1 0.1887641.
No correct Abuse/Assault/Fighting classification. These weak-label cascade
results remain separate from standalone behavior and UCF frame benchmarks.
No parameters changed. Mean runtime 0.505093s, median 0.200839s, p95 1.346321s,
excluding model loading, serialization and UI.

For the professor presentation, added one explicitly disclosed post-hoc correct
Robbery case: first eligible clip by ID in saved registered results,
`Robbery020_x264_10`. Copied only this clip into `data/examples/`; its cache reads
the genuine registered inference without rescoring. All eight original cases,
including failures, remain. Receipt:
`runs/ucf-crime/demo-supplemental-correct-robbery-v1.json`. This is presentation
curation, not benchmark evidence or tuning.

The frozen four-second case completed exact repeated GPU inference (scores,
boxes, probabilities and alert), warm total 1.090948s; full CPU fallback completed
33.211586s, same alert with expected floating-point differences. Model loading
was 2.050689s GPU / 1.502774s CPU, separately. Receipt:
`runs/ucf-crime/deployment-runtime-v1.json`. Final server session 21323 uses the
default UCF config and nine entries. Six actual cached/live/upload API requests
passed plus the empty-input error, including a genuine uploaded Robbery actor
path: `runs/ucf-crime/demo-api-ucf-final-v2.json`.

Read-only milestone review caught the old API validator default output pointing
to historical DCSASS evidence. Fixed deployment-specific UUID receipt defaults,
explicit existing-path rejection and exclusive final creation. Two preservation
tests failed before the fix then passed. Reviewer confirmed the fix, recovery
hash accounting and UCF report; no remaining issues in that scope.
Final full suite: 542 passed / 325 expected upstream warnings / 145.48s;
Ruff passed, all 202 Python files formatted, diff check passed. New results,
README, asset instructions, runbook and talk track reflect the UCF deployment
and clearly retain its substantial failures.

Final browser inspection of the nine-example UCF deployment verified cached and
genuine on-demand Robbery, explicit no-actor and Normal disagreement windows,
anomaly timeline and detector overlays. Video is four seconds, browser readyState
4, no media error. Saved proof: `outputs/demo/ucf_v1/visual-live-cascade.jpg`;
retained-failure proofs: `visual-false-alert.jpg` and `visual-disagreement.jpg`.
The correct-case source menu discloses its curated selection, and on-demand mode
is visibly labelled. Final receipt/atomic regression recheck: four passed;
Ruff/format/diff remained clean after documentation and six-request API completion.

Final read-only review verified the completed metrics/provenance, all nine
video/cache identities, original eight selection policies and six-case API
receipt. Corrected three documentation points: original cases include a correct
normal bypass; standalone middle-frame coverage is explicitly distinct from
routed cascade coverage in the talk track; poor short-clip generalization is an
interpretation, not a measured causal domain-shift effect. No implementation or
scientific-integrity blockers remain. Final default asset verifier passed on
CUDA (`runs/ucf-crime/final-default-asset-verification.log`). Local milestone
checkpoint only; no push, merge, architecture replacement or data deletion.

## Final repository hygiene and merge/publication gate

The user subsequently authorized an explicit local `--no-ff` merge to main,
then authorized normal remote publication only after all local, remote-ancestry
and dry-run gates pass. No force push, rebase, remote change or automatic remote
divergence resolution is allowed. Initial inspection: clean feature branch
`0e3e82919e4dafedad671c91ae236362dd645e25`; local main
`a7f5d99dfa1c58d9b41a7288fc821263ee61986e` is an ancestor, sixteen commits behind.

Audited the tracked inventory individually by category/size and searched source,
scripts, configurations and tests for Windows drive paths, usernames and fixed
CUDA indices. Executable defaults use pathlib/configurable paths. The existing
drive-letter test is Windows-guarded. Historical scientific execution paths
remain intact; no architecture or benchmark settings changed.

Extended repository ignore rules for local environments/credentials/editor/OS
files, archives, external example frames, additional feature/checkpoint formats
and explicitly named local YAML overrides. Two regression tests check repository
rules independently of global Git excludes and verify source/protocol/templates
remain trackable. Null-delimited Git subprocess input avoids Windows CRLF
filename translation. Heavy-file protection failed before the rule additions,
then passed; no local files were deleted or untracked blindly.

Added `docs/macos-replication.md`: fresh Python 3.11+ environment (3.13 suggested),
dev/demo install, asset-free checks, CPU demo/MPS limitations, exact five-model
bundle and both validation-selection receipts, optional cached examples,
configurable external dataset roots and cache rebuild commands. Linked README,
runbook and asset instructions. Added byte-count/SHA256 metadata for the two
external author text inputs to the existing pinned source-provenance JSON;
the split files themselves remain external. No macOS hardware verification is
claimed, and Windows CUDA packages remain unchanged.

Final publication will require a clean main with a verified two-parent merge,
full tests/lint/format/diff and safe demo smoke, followed by fetch, remote-main
ancestry verification, dry-run and normal pushes. The actual commit/remote
hashes and verification results are reported in the final handoff rather than
modifying frozen experiment reports.

Pre-merge full validation completed: 544 tests passed, 325 upstream warnings,
185.21 seconds. Ruff passed; 203 Python files formatted; diff check clean.
The default demo help path passed, as did constructing the interface from a
tracked-source Git archive with no model/example assets installed. This smoke
loaded no model tensors and ran no inference. Tracked content signature scan
found no credential matches; all reachable historical blobs are below 1 MiB
(largest pre-finalization blob 63,114 bytes). Audit/validation logs remain ignored
under `runs/finalization/`. Legacy ignored pytest-directory permission warnings
did not affect tracked-file audits; no permission or original-data changes made.
