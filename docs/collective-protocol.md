# Collective benchmark protocol (Milestone 2C)

The implementation is verified with artificial fixtures. No real Collective
benchmark has been run here. The checked-in YAML is a **candidate configuration**;
the `freeze` stage creates the immutable receipt for a runnable experiment once
its real artifacts exist. A receipt does not certify paper reproduction.

## Fixed sources and supervision

`configs/experiments/collective_protocol.yaml` fixes the existing 32/12 source split:

- Test: **5,6,7,8,9,10,11,15,16,25,28,29**.
- Original training: all remaining IDs in **1..44**.
- Validation: **1,2,3**, subtracted from the original training sources.
- Effective optimization sources: **29**; validation sources: **3**; test sources: **12**.

The 32/12 IDs come from the related authors' [released configuration](https://github.com/wjchaoGit/Group-Activity-Recognition/blob/master/config.py).
Gavrilyuk et al. do not enumerate source IDs in the paper; exact agreement with
their experiment is unverified. The validation holdout is our implementation
choice. It is deterministic, source-based, and is not claimed to be class-balanced.
Inspect its class distributions before freezing; changing it requires a new
protocol version and new runs, never selection based on test scores.

Labels retain crossing, waiting, queueing, walking, talking. Raw action 1 is NA;
2..6 map to IDs 0..4. Group supervision is the majority valid actor action,
with the existing lowest-ID tie rule. Annotation centers are 1,11,21,...;
ten frames use offsets -5..4 with edge replication. These are the existing
released-data-loader conventions, not new inferred activity labels.

`validate_collective` decodes every installed frame, checks contiguous numbering,
constant valid dimensions within a source, annotation vocabulary and geometry,
and compares a supplied manifest to parsed annotations. Manifest checks detect
duplicates, temporal sampling errors, incorrect source splits, majority-label
errors and physical frame/feature leakage. Missing sources/frames, nonfinite or
out-of-bounds boxes and malformed annotations are structural failures. Boxes are
not silently corrected; repair requires documenting an explicit dataset revision.
Large actor counts generate warnings. A failure carries a JSON validation report.

## Feature contracts and cache provenance

Configure vetted local exports before extraction; see [actor-backbones.md](actor-backbones.md).
The benchmark uses HRNet-W32 pre-final maps flattened to **98304** features and
I3D Mixed_4f temporally averaged, resized to 90x160, and RoIAligned to 5x5
(**20800** features). Crop size is 256x192. Input images are RGB [0,1]; backbone
normalization comes from the archive metadata. No extra L2 normalization is
introduced. `normalization: backbone_only` makes this choice explicit.

Inspection uses a small **training-only** image subset and checks finite values,
dimensions, repeatability, reverse actor ordering and image-to-map coordinate
transforms. It saves JSON, not copies of dataset images. Frozen precomputed
features permit training without running backbones each epoch.

Extraction emits `.npy` arrays and **schema version 2** JSON sidecars:
dataset/scene/source IDs, ordered boxes, scene and source fingerprints, backbone
architecture/endpoint/metadata, checkpoint SHA256, extraction configuration/hash,
image size, tensor shape, feature-file SHA256 and ordered source-image-content
fingerprint. Source contents are checked before and after extraction and again
before benchmarking. Detected caches additionally
bind ordered detector boxes/scores and embed identical metadata in their JSONL.
Even zero-actor caches must have a valid empty array and provenance at the
benchmark boundary. The legacy trainer remains compatible with older features;
benchmark orchestration rejects them with instructions to re-extract.

Scene fingerprints bind annotation identities/labels/geometry. Source fingerprints
bind resolved frame paths/indices; the separate extraction-time content fingerprint
binds image bytes so replacement before freezing also invalidates the cache.
Real freeze receipts separately
hash referenced frame bytes, annotation files, arrays, sidecars and manifests.
Receipts also bind available local export/checkpoint bytes. Imported verified
caches may use a recorded checkpoint fingerprint without the export installed;
raw extraction and inspection still require that local export.

## Selection and freeze

All seeds **[0,1,2]** are declared before running. Checkpoints maximize validation
group accuracy, retaining the earliest checkpoint on a tie. Training defaults are
Adam, LR 1e-4, betas .9/.999, step milestones 5000/10000, gamma .1, horizon
20000, batch 16, gradient clip 1.0. Batch size, clipping, validation interval,
seeds and holdout are implementation choices. The paper-style architecture and
schedule are inherited from Milestone 2A; no new model mathematics is introduced.

Establish pose GT, then RGB GT, then late-fusion GT. Available completed modes
are ranked by **mean validation group macro F1**, then validation group accuracy,
then lexical experiment name. Candidates must use identical validation populations
and seed sets. Partial modality availability is disclosed. Detected pose/RGB/late
experiments are declared up front so choosing the validated winner does not
require changing the protocol after observing scores. A detected freeze rejects
an experiment whose GT reference is not the validation-selected available winner.

Detector tuning sweeps only confidence [.3,.4,.5,.6,.7], maximizing micro detector
F1 subject to recall >=.5. Ties prefer recall, then higher confidence. NMS=.5,
matching IoU=.5, max actors=20, min width/height=4 and area=0 remain fixed.
These thresholds are implementation defaults, not Gavrilyuk paper constants.
The initial config confidence is overridden by the validation selection receipt.
Matching IoU is **not** optimized to improve conditional actor accuracy.

Candidate collection reads validation center images only. Candidates disclose
confidence=0, NMS=1, zero minimum dimensions, cap=1000, and the underlying
RPN/backend proposal cap. Previously filtered/truncated artifacts are rejected.
Candidates must exactly cover validation scenes, with no training or test records.
An official test source relabeled `val` is rejected. Sweep receipts record source
IDs, boxes, trials, candidate SHA and pretrained detector SHA. The runner recomputes
the sweep from current validation candidates before accepting its receipt.

Local candidate/final detection records bind native reference-image SHA256; real
benchmark receipt checks reject changed images or absent content provenance.
Final detector inference uses the selected filters, records their hash, and never
optimizes them on test output. A detected freeze verifies that provenance and
the validation feature-mode decision. Detected evaluation reuses the **same GT
checkpoint for the same seed**, without retraining or changing Actor-Transformer.
Receipt/checkpoint/config hashes and identical scene identities are checked.

Receipts live under `runs/collective/protocols/`. They bind the resolved entire
protocol and artifact content. Existing receipts/runs cannot be overwritten.
Changed settings or data require a new protocol ID and output/artifact paths.
The software cannot prevent a researcher manually looking at test predictions;
its selection APIs consume validation values only, and experimental decisions
must follow this documented rule.

## Metrics and interpretation

Group and actor reports include accuracy, macro F1, per-class accuracy/F1,
support and confusion matrices. Macro F1 includes all five configured classes;
absent-class accuracy is null. Counts and class distributions accompany results.
Actor masks exclude padding. Unmatched detections have unknown labels and never
enter actor accuracy. Detection precision/recall/F1 are micro counts; matched
IoU, misses, extras, mean actors and empty-scene rate expose coverage.

GT metrics use all GT actors. Detected actor metrics use only matched actors.
Actor deltas compare exactly that same matched GT subset. Empty detector scenes
abstain; all-scene group accuracy counts abstention as incorrect. Supported-scene
group macro F1 and its GT comparison use the same nonempty subset. These conditional
metrics must be read alongside coverage; they are not directly interchangeable
with full-population paper results.

Aggregation rejects different protocol hashes, evidence types, dry-run flags,
scene/actor identities and duplicate seeds within an experiment. Matched-population
hashes include assignment identities and boxes. Scalars report individual seed
values, mean, and sample standard deviation (ddof=1). Undefined values remain null;
one-seed spread is unknown. Counts and matrices remain per-seed. Ablation rows stay
separate rather than averaging modalities or GT/detected populations.

Error reports identify correct/incorrect scenes, GT-success/detector-box-failure,
both-failure, no-actor scenes, high-confidence errors and low coverage. Records
retain IDs, true/predicted classes, confidence, counts, actor predictions and IoU.
Directed confusion summaries use actual labels; confidence summaries distinguish
correct and incorrect predictions. These show associations, not proven behavioral
causes. No imagery is copied or redistributed by reporting.

## Reproduction commands

From the repository root, activate the environment or substitute
`.venv/Scripts/python.exe` for `python`. Install Collective locally using
[data/README.md](../data/README.md). Set absolute or YAML-relative local paths in
`features.pose.checkpoint`, `features.rgb.checkpoint` and `detector.checkpoint`;
record SHA256 values if importing caches. All hyperparameters live in the YAML.
No script downloads weights or data.

```powershell
python scripts/run_collective_experiment.py --stage prepare
python scripts/run_collective_experiment.py --stage inspect --experiment pose_gt --max-scenes 10
# Extract only up to 10 training and 10 validation scenes, then train 5 steps:
python scripts/run_collective_experiment.py --stage preflight --experiment pose_gt --seed 0 --max-scenes 10 --max-iterations 5
python scripts/run_collective_experiment.py --stage extract --experiment pose_gt
# Bounded, validation-only training; all outputs marked NON-BENCHMARK:
python scripts/run_collective_experiment.py --stage run --experiment pose_gt --seed 0 --dry-run --max-scenes 10 --max-iterations 5
# Freeze actual artifacts, then run the three declared seeds:
python scripts/run_collective_experiment.py --stage freeze --experiment pose_gt
python scripts/run_collective_experiment.py --stage run --experiment pose_gt
```

`prepare` refuses an existing manifest. If one already exists, validate it with
`python scripts/validate_collective.py --root data/raw/collective --manifest data/manifests/collective.jsonl --report runs/collective/dataset_validation.json --val-sequences 1 2 3`.
Raw real data and required caches are validated before any run. A different protocol
uses `--protocol path/to/protocol.yaml` on every command.

For RGB and late fusion, repeat inspect/extract/freeze/run for `rgb_gt` and
`late_gt` when compatible exports are available. Then:

```powershell
python scripts/run_collective_experiment.py --stage collect-and-tune
# Or --stage tune when broad validation candidates are already cached.
python scripts/run_collective_experiment.py --stage detect
# Choose only the validation-selected available reference mode:
python scripts/run_collective_experiment.py --stage extract --experiment detected_pose
python scripts/run_collective_experiment.py --stage freeze --experiment detected_pose
python scripts/run_collective_experiment.py --stage run --experiment detected_pose
python scripts/aggregate_experiments.py --runs runs/collective/pose_gt/seed_0/metrics.json runs/collective/pose_gt/seed_1/metrics.json runs/collective/pose_gt/seed_2/metrics.json --output runs/collective/pose_summary
python scripts/analyze_collective_errors.py --gt runs/collective/detected_pose/seed_0/predictions.json --detected runs/collective/detected_pose/seed_0/detected_predictions.json --output runs/collective/error_summary
```

Use `detected_rgb` or `detected_late` instead if the corresponding GT mode wins
validation. Frozen detected evaluation does not train a new model. Repeat aggregation
per experiment, or pass explicit per-seed paths for several experiments to produce
separate ablation rows. Existing output directories are intentionally not overwritten.

Software-only smoke verification:

```powershell
python scripts/smoke_collective_experiment.py --output outputs/collective-smoke-new
python -m pytest --basetemp .pytest_cache/collective-verification-new
ruff check .
ruff format --check .
git diff --check
```

The smoke fixture contains artificial feature values, two seeds, variable actor
counts, missed/extra actors and an empty scene. Its metrics are never benchmark
results. Real HRNet/I3D/detector quality, CUDA operation and real-data accuracy
remain unverified until actual artifacts are installed and measured.

The `preflight` stage checks the full dataset's structural integrity but bounds
expensive extraction and training. It extracts actual local exports on training
and validation scenes only, writes strict provenance, and evaluates validation
only. It does not require a full feature cache or a freeze receipt. Existing
preflight artifact/run paths are rejected rather than overwritten. The simpler
`run --dry-run` stage uses already extracted features and also evaluates only val.
