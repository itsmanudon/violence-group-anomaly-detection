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
