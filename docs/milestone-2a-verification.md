# Milestone 2A implementation and verification

## Initial state and plan

Started from clean `main` at `a7f5d99`. Inspected README, packaging/configs,
dataset abstractions, model/trainer/inference code, tests, and ignore rules.
All **28 Milestone 1 tests passed before changes**. Created
`feat/actor-transformer-baseline`; no commits, pushes, merges or remote changes.

Followed [the implementation plan](milestone-2a-plan.md): verified Collective
annotation/split sources; added actor records/batches; implemented positional
encoding, transformer and loss; added local backbone adapters and extraction;
implemented training, metrics and inference; documented and verified the workflow.
Milestone 1 source and test files are unchanged. Shared seed/device helpers are reused.

## Delivered architecture and interfaces

Annotated ten-frame Collective scenes or precomputed features feed pose/RGB
projections, 2D position, a post-norm transformer, an individual-action head, and
a masked-max group head. Pose, RGB, early concat fusion and weighted late
probability fusion work independently of Sultani. No person detector, surveillance
cascade, DCSASS actor adaptation or demo UI was added.

- Features `[B,N,D]`, boxes `[B,N,4]` normalized xyxy, Boolean valid mask `[B,N]`.
- Actor logits `[B,N,C_actor]`, group logits `[B,C_group]`.
- Attention per branch `[B,layers,heads,N,N]`; inference removes padded axes.
- Padding is cleared before projection, masked as attention keys, zeroed after
  encoder blocks, excluded from max pooling and actor loss/metrics. Empty scenes fail.

The [README](../README.md#fidelity-and-implementation-choices) distinguishes paper
settings from explicit choices: center-frame training pose, positional coordinate
scale, crop/RoI conventions, local export contract, frozen default, joint late
fusion training, validation holdout and optimizer epsilon. The exact paper split
IDs are not published in the paper; the documented 32/12 IDs come from a related
method's released loader. No exact reproduction or benchmark accuracy is claimed.

## Verification results

Environment: Windows, Python **3.13.5**, torch **2.14.1+cpu**, torchvision
**0.29.1+cpu**. No Python 3.13 compatibility failures occurred. GPU execution was
not tested. Modern MultiheadAttention uses explicit Boolean key masking and
per-head weights. TorchScript is a documented interim compatibility boundary;
73 upstream deprecation warnings were emitted by synthetic archive tests.

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -q --basetemp .pytest_cache/m2-final` | **123 passed** in 22.95s; original 28 plus 95 new cases |
| `ruff check .` | Pass |
| `ruff format --check .` | Pass, 77 Python files formatted |
| `git diff --check` | Pass; only Git's normal LF/CRLF notices |

Tests cover real-format annotation fixtures and source/physical-path leakage;
variable actor batches; independent xy position; all four feature modes; single
actors; padded NaNs and permutation invariance; masked max/attention/loss/metrics;
gradient propagation; local archive compatibility, geometry and raw/precomputed
parity; trainable/frozen backbones; checkpoints and exact deterministic resume;
changed preprocessing rejection; config errors; CLI path rebasing, including
Windows drive boundaries; serializable inference and attention plotting.

Each command below completed three training iterations, saved and reloaded
checkpoints, evaluated and inferred without network/data/model downloads:

```powershell
python scripts/smoke_actor_transformer.py --output outputs/m2_final_pose --mode pose_only --iterations 3
python scripts/smoke_actor_transformer.py --output outputs/m2_final_rgb --mode rgb_only --iterations 3
python scripts/smoke_actor_transformer.py --output outputs/m2_final_late --mode pose_rgb_late_fusion --iterations 3
```

Training includes scenes with 1/2/3 actors; test inference yields two scenes with
2 and 3 actors (five valid actor predictions). Probabilities and attention row
sums were checked. Standalone evaluation and inference CLIs also passed on the
late-fusion checkpoint. An example attention image is in the ignored
`outputs/m2_final_late/actor_attention.png`. These are software checks, not useful
accuracy estimates. Outputs/runs and synthetic arrays are ignored by Git.

## Exact file inventory

Modified:

```text
README.md
data/README.md
pyproject.toml
```

Added:

```text
configs/actor_transformer.yaml
configs/actor_transformer_pose_only.yaml
docs/actor-backbones.md
docs/collective-format.md
docs/milestone-2a-plan.md
docs/milestone-2a-verification.md
scripts/_actor_features.py
scripts/evaluate_actor_transformer.py
scripts/extract_i3d_features.py
scripts/extract_pose_features.py
scripts/infer_group_activity.py
scripts/prepare_collective.py
scripts/smoke_actor_transformer.py
scripts/train_actor_transformer.py
src/surveillance/actor_config.py
src/surveillance/datasets/actor_batch.py
src/surveillance/datasets/collective.py
src/surveillance/evaluation/group_activity_metrics.py
src/surveillance/features/actor_backbones.py
src/surveillance/features/geometry.py
src/surveillance/features/hrnet_pose.py
src/surveillance/features/i3d.py
src/surveillance/inference/group_activity_pipeline.py
src/surveillance/models/actor_transformer/__init__.py
src/surveillance/models/actor_transformer/actor_transformer.py
src/surveillance/models/actor_transformer/encoder.py
src/surveillance/models/actor_transformer/fusion.py
src/surveillance/models/actor_transformer/loss.py
src/surveillance/models/actor_transformer/positional_encoding.py
src/surveillance/training/actor_transformer_trainer.py
src/surveillance/visualization/actor_attention.py
tests/test_actor_attention.py
tests/test_actor_backbones.py
tests/test_actor_batch.py
tests/test_actor_config.py
tests/test_actor_extraction_cli.py
tests/test_actor_masking.py
tests/test_actor_transformer.py
tests/test_actor_transformer_loss.py
tests/test_collective_dataset.py
tests/test_group_activity_inference.py
```

## Remaining external dependencies and next steps

Real experiments require Collective frames/annotations and compatible pretrained
HRNet-W32 COCO/I3D feature exports. The adapters validate declared metadata and
runtime outputs, not architecture identity or pretrained provenance. Raw checkpoint
reload still needs the original architecture export. Archive export must be
validated against the actual research backbone before claiming paper fidelity.
Resume fingerprints manifest bytes, not the underlying image/feature contents.

Exact prepare/extract/train/evaluate/infer commands are in
[README](../README.md#collective-setup-and-precomputed-workflow), with the expected
layout in [data/README](../data/README.md#collective-activity-milestone-2a) and export
instructions in [actor-backbones](actor-backbones.md).

Milestone 2B: first measure the annotated-box baseline on real Collective data;
then introduce detected boxes and measure the gap, handle actor alignment/missed
detections, curate annotated surveillance behavior clips with source-aware splits,
and finally connect Sultani windows. Keep behavior labels distinct from identity
or character judgments. A Gradio interface follows component validation.

Final working branch remains `feat/actor-transformer-baseline`: three modified
tracked files and 41 new files, all unstaged. No commit or push was performed.
