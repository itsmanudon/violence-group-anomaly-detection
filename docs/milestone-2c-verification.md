# Milestone 2C verification record

## Initial repository state

Started clean on `feat/automatic-actor-detection`, HEAD `8f56b11`. The requested
`feat/collective-real-benchmark` branch did not exist. Read Milestone 2A/2B
verification, backbone/detector/data documentation, configs, Collective data,
masking, trainer, evaluation, inference and extraction implementations. Ran the
complete baseline **before edits: 234 tests passed**. Created and remained on
`feat/collective-real-benchmark`. No existing work was deleted or overwritten.

Followed [milestone-2c-plan.md](milestone-2c-plan.md): actual data validation;
resolved/frozen protocol; bounded feature inspection and provenance; validation-only
tuning; existing-component orchestration; seed/error reporting; offline verification.
Actor-Transformer and Sultani model/loss mathematics are unchanged. Trainer changes
only add observability and preserve checkpoint format, validation accuracy selection
and exact-resume behavior. No benchmark implementation bug was identified in the
previous milestones, and no prior synthetic result is retracted.

## Delivered protocol and evidence boundaries

See [collective-protocol.md](collective-protocol.md) for full definitions/commands.
The existing related-author-code 32/12 source list is preserved. Test IDs are
5,6,7,8,9,10,11,15,16,25,28,29. Training-only validation uses 1,2,3, leaving
29 optimization sources. Exact agreement with Gavrilyuk's unpublished source IDs
is unverified; the holdout and detector thresholds are our implementation choices.

The YAML declares seeds 0,1,2 and six GT/detected pose/RGB/late-fusion experiments.
It remains a candidate until actual input artifacts are validated and a `freeze`
receipt is produced. Test evaluation verifies immutable resolved settings and hashes
of caches, sidecars, local checkpoints, raw annotations and source frames.
Verification repeats after training and at evaluation boundaries. Detected receipts
also bind the original GT artifacts and selected per-seed checkpoint hashes.

Feature provenance version 2 binds scene labels/geometry, ordered source paths and
image bytes, checkpoint SHA, backbone endpoint, preprocessing configuration/hash,
image/clip dimensions, box source and tensor/content hashes. Cache validation rejects
stale preprocessing, boxes, labels, source images or arrays. GT filenames include
source identity/content as well as config to prevent unrelated-source collisions.
Raw preflight bounds extraction/training to training/validation scenes and is
explicitly non-benchmark. Structural validation still checks the installed dataset.

Confidence [.3,.4,.5,.6,.7] is selected using validation-only micro detector F1,
subject to recall >=.5. NMS, matching IoU, actor cap and geometry filters remain
fixed. Candidate provenance discloses the backend cap and rejects hidden prior
filtering. Official test sources cannot be relabeled for tuning. Real candidate
and final detection records bind reference-image bytes and detector weight SHA.
The selected settings must be present in detected cache provenance.

Feature mode selection reads validation macro F1 across identical seed/source
populations, not test metrics; ties use validation accuracy and experiment name.
Only available completed modes are ranked. Detected evaluation reuses the same
GT checkpoint; no separate detected model is trained. Matched-only actor metrics,
GT paired subsets, supported-scene group metrics, all-scene abstention-aware
accuracy and detector coverage are explicit. Aggregation refuses incompatible
populations and retains nulls, per-seed counts/matrices, mean and sample std.

Error/confusion/confidence reports include class labels, source IDs, predictions,
counts, coverage, IoU, actor predictions, GT-success/detected-failure, both-failure,
empty scenes, confident mistakes and low coverage. They make no unsupported claim
about why a person behaved a certain way. Reports do not copy dataset imagery.

## Verification actually run

- `.venv/Scripts/python.exe -m pytest --basetemp .pytest_cache/m2c-full-final-2 -q`
  — **347 passed**, **89 upstream TorchScript FutureWarnings**, 83.78 seconds.
  All prior 234 tests pass; **113 added tests** cover validation, protocol/freeze,
  provenance, selection, reporting, observability and orchestration.
- `ruff check .` — passed.
- `ruff format --check .` — passed, 124 Python files formatted.
- `git diff --check` — passed. Git emits ordinary Windows LF-to-CRLF notices;
  they are not whitespace errors.
- `python scripts/smoke_collective_experiment.py --output outputs/milestone2c-verified`
  — passed, two seeds (0,1), three GT training iterations per seed, checkpoint
  reload, same-checkpoint detected comparison, separate aggregates and error reports.
  Fixtures use actor counts 1/3 in training and 2/3 in test. Detected test data has
  one empty scene, one matched actor, four missed GT actors, and one unmatched
  detection. These are deliberately constructed software cases, **not benchmark
  measurements**. Outputs are gitignored.
- The bounded raw preflight test exercised actual HRNet adapter crop/resize and
  RoIAlign using a local **synthetic TorchScript archive**, strict feature sidecars,
  one training scene, one validation scene and one optimization step. Pose/RGB
  inspection tests use distinct left/right synthetic pixels, local archives,
  repeated/reordered features and coordinate checks. No pretrained model was used.
- CLI help succeeded. Missing-data validation exited 1 and saved
  `outputs/milestone2c-missing-data.json` with missing-sequence diagnostics, as intended.
- Reviewed tracked diff, all new files, git status and branch before finishing.

Regression guards cover source image replacement before freeze, stale arrays,
GT artifacts changed after detected freeze, different detector weight fingerprints,
test artifacts changed during training, official test sources relabeled validation,
run overwrite prevention and validation-only bounded runs. These corrected review
findings in newly written Milestone 2C tooling; they do not change previous model math.

## Local environment and remaining dependencies

Python **3.13.5**, torch **2.14.1+cpu**, torchvision **0.29.1+cpu**, Windows.
No Python 3.13 incompatibility was encountered. The existing feature export contract
uses TorchScript, now deprecated upstream; its warnings are preserved, and no
architecture or exporter migration was introduced in this protocol milestone.

**No real Collective tree, real HRNet export, real I3D export or pretrained detector
checkpoint was available in the repository.** There was no real-data dry run,
GT benchmark, detector benchmark or measured ablation table. No downloads occurred.
CUDA is unavailable in the installed CPU build; CUDA remains **unverified**.
Synthetic checkpoints/arrays under ignored outputs are not real pretrained assets.

For the first real experiment, install all 44 Collective sequences, supply a vetted
HRNet-W32 feature export (and RGB/Faster R-CNN weights for later phases), and configure
their paths/fingerprints in the protocol. Run prepare -> inspect -> bounded preflight
-> extract -> freeze -> three-seed GT training. Then validate RGB/fusion as available,
collect-and-tune validation detector candidates, detect with frozen filters, extract
the validation-selected detected mode, freeze, evaluate the same GT checkpoints,
aggregate and inspect errors. Exact commands are in the protocol document and README.

Recommended next milestone: execute this frozen real-data protocol and establish
measured GT/detected evidence before surveillance adaptation. Assess actor coverage,
class imbalance, confusion and backbone compatibility before choosing a surveillance
label/annotation protocol. The Sultani cascade, violence head and UI remain deferred.

## Exact file inventory

Modified:

```text
README.md
data/README.md
scripts/_actor_features.py
src/surveillance/training/actor_transformer_trainer.py
```

Added:

```text
configs/experiments/collective_protocol.yaml
docs/collective-protocol.md
docs/milestone-2c-plan.md
docs/milestone-2c-verification.md
scripts/aggregate_experiments.py
scripts/analyze_collective_errors.py
scripts/inspect_actor_features.py
scripts/run_collective_experiment.py
scripts/smoke_collective_experiment.py
scripts/tune_collective_detector.py
scripts/validate_collective.py
src/surveillance/datasets/collective_validation.py
src/surveillance/experiments/__init__.py
src/surveillance/experiments/environment.py
src/surveillance/experiments/preparation.py
src/surveillance/experiments/protocol.py
src/surveillance/experiments/reporting.py
src/surveillance/experiments/runner.py
src/surveillance/experiments/synthetic.py
src/surveillance/experiments/thresholds.py
src/surveillance/features/provenance.py
tests/test_collective_orchestration.py
tests/test_collective_protocol.py
tests/test_collective_validation.py
tests/test_experiment_reporting.py
tests/test_feature_provenance.py
tests/test_training_observability.py
tests/test_validation_thresholds.py
```

32 authored files: four modified, 28 added. Generated fixtures, metrics, logs,
checkpoints and pytest artifacts remain ignored. Working tree changes are
uncommitted on `feat/collective-real-benchmark`; nothing was staged, committed,
pushed, merged, or published. Remotes/history were unchanged.
