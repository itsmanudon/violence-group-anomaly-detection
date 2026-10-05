# Collective RGB-only asset audit — 2026-10-04

**Status: blocked at the I3D asset gate. No real RGB features or benchmark
metrics were produced.** Intended experiment: **Collective / RGB / GT boxes /
seed 0 / our frozen protocol**. This is not an exact Actor-Transformer reproduction.

The required pretrained I3D checkpoint and a vetted feature export are absent
from the inspected local locations. The asset policy requires preparing an exact
acquisition command rather than silently downloading weights. Acquisition and
validation are required before continuing. No replacement backbone, random I3D,
Charades weights, fusion, commit, or push was used.

## Initial state and preserved evidence

- Branch: `feat/collective-pose-real-baseline`.
- HEAD: `527a88f4aea3e3fbc0f4c1733afa04f26ac7706c`.
- Latest commits: `527a88f` first real pose baseline; `319b7bb` Collective
  benchmark protocol; `8f56b11` automatic actor detection and box robustness.
- Initial modifications: `README.md`, `docs/collective-protocol.md`, and
  `docs/milestone-2c-real-pose.md`. These are pre-existing documentation changes;
  there were no uncommitted implementation changes at inspection.
- Read all six requested documents and inspected the seed-0 metrics, errors,
  training/validation curves, and confusion-matrix image.

Ignored evidence is saved in `runs/collective/rgb_gt_v1_audit/`: initial branch,
HEAD, log, status, binary-capable worktree patch, SHA256 values for all 154 initial
tracked files, and SHA256 values for 16 seed-0 artifacts plus the pose freeze
receipt. The pose cache and receipt were not edited or regenerated. This audit
document is the only newly authored tracked-path file.

## Pose seed-0 verification

Values below were read from the actual `metrics.json` and independently
recomputed from its saved confusion matrices; no test inference was repeated.
The displayed confusion matrix agrees with those integer counts. `errors.md`
reports 527 correct, 248 incorrect, and 137 high-confidence incorrect scenes.
The curves show falling training loss without a corresponding validation gain.

| Metric | Verified held-out pose seed 0 |
| --- | ---: |
| Group accuracy | 0.6800000000000000 (68.00%) |
| Group macro F1 | 0.6548178080805587 |
| Actor accuracy | 0.5900584795321637 (59.01%) |
| Actor macro F1 | 0.5697935287481597 |
| Scenes / annotated actors | 775 / 3,420 |
| Invalid / abstained scenes | 0 / 0 |

| Class | Pose group recall | Pose actor recall |
| --- | ---: | ---: |
| crossing | 62.5850% | 54.9133% |
| waiting | 41.4815% | 32.6693% |
| queueing | 58.0645% | 46.3617% |
| walking | 68.8073% | 66.1475% |
| talking | 96.1538% | 78.5166% |

Observed group confusions: crossing → walking **55**, walking → crossing **51**,
waiting → crossing **40**. Actor counts for the same directed pairs are
296, 232, and 115. RGB improvements and metric deltas are **unavailable**.

The installed `data/raw/collective/ActivityDataset` has all 44 source directories.
All 25,196 unique frame paths referenced by `data/manifests/collective.jsonl`
exist. Manifest counts are train **1,687 / 9,027**, validation **85 / 427**, test
**775 / 3,420** scenes/actors: total **2,547 / 12,874**. This is a fresh presence
and manifest-count check, not a new full image-decoding validation.

CUDA verification: Python **3.13.5**, torch **2.13.0+cu126**, torchvision
**0.28.0+cu126**, CUDA **12.6**, CUDA available, **NVIDIA GeForce RTX 4070 Laptop
GPU**, 8,585,216,000 reported device-memory bytes. No packages were changed.

## Existing RGB infrastructure

| Question | Verified implementation |
| --- | --- |
| Existing I3D implementation | `features/i3d.py` is an actor-feature adapter, not a complete pretrained I3D network. There is no I3D exporter analogous to `hrnet_export.py`. |
| Accepted backbone artifact | Feature-only **TorchScript** with embedded `actor_backbone.json`; arbitrary classifier state dictionaries are rejected. Precomputed finite `.npy` actor tensors are also supported. |
| Endpoint | Metadata and runtime require I3D `Mixed_4f`, `[B,832,t,h,w]`. |
| Frame selection | `temporal_frame_indices`: offsets -5 through +4, edge replication within the source; reference index 5. |
| Frame decode | `ActorFeatureDataset`: OpenCV decode, resize to H=480/W=720, BGR→RGB, float32 `/255`, yielding `[B,10,3,480,720]`. |
| Backbone preprocessing | Adapter bilinear resize to explicit archive input size, `align_corners=False`; declared channel mean/std; permutation to `[B,3,10,H,W]`. No implicit normalization. |
| Temporal pooling | `features.mean(dim=2)`. |
| Spatial resize | Bilinear to `[B,832,90,160]`, `align_corners=False`. |
| GT geometry | Normalized middle-frame xyxy edges multiply `[160,90,160,90]`; equivalent to separately scaling 720-wide x and 480-high y. |
| RoIAlign | `spatial_scale=1`, `sampling_ratio=2`, `aligned=True`, output `[sum(valid),832,5,5]`; flatten to 20,800, restore manifest actor order, zero padded rows. |
| Actor-Transformer | Existing `rgb_only` mode has `Linear(20800,128)`, positional encoding, one layer/head, FF=256, dropout=.1, actor and group heads. |
| Extraction | `scripts/extract_i3d_features.py` calls the shared `_actor_features.py`; currently one scene/clip per extraction call, effectively extraction batch size 1. |
| Inspection/preflight | Existing orchestration supports training-only inspection and bounded train/validation preflight, plus checkpoint reload and validation-only selection. |
| Provenance | Schema v2 binds boxes/order/labels, source paths/indices and bytes, checkpoint hash, metadata, preprocessing, temporal mean, resize, RoI settings, and feature content. Stale caches are rejected. |

Existing offline tests cover RGB shapes/gradients, temporal mean, ten-frame
validation, padded actor order, coordinate scaling, numeric RoIAlign, frozen eval,
synthetic extraction/inspection, provenance, and stale-cache rejection. They
validate infrastructure, not the identity or numerical correctness of a real I3D.

The standalone RGB config already uses training batch **8**. The shared
Collective candidate has batch **16** across experiments. A separate RGB protocol
must explicitly set **8** without modifying the pose protocol. Keep Adam 1e-4,
betas .9/.999, current epsilon 1e-8, LR ×.1 at 5,000/10,000, 20,000 iterations,
gradient clipping 1, validation/checkpoint interval 100, and earliest maximum
validation group accuracy. These retain the existing protocol choices.

The [paper's implementation section](https://arxiv.org/html/2003.12737#S4.SS2)
supports the requested ten frames, Collective resolution, endpoint/pooling,
90×160 map, 5×5 RoIs, projection, dynamic batch 8, and schedule. Full-frame
480×720 extraction differs from I3D's original 224×224 classification crop;
retain the requested actor-localization geometry and document that adaptation.
The existing validation split stays unchanged: group support [24,0,0,60,1],
walking majority 70.59%, no waiting/queueing group scenes.

## Asset search and chosen candidate

Searched accessible files under `D:/Github Repos/`, including ignored checkpoint
locations, project `checkpoints/` and `checkpoints/external/`, and the documented
external HRNet checkout. No I3D checkout, converted pretrained I3D checkpoint,
or real I3D feature archive was found. Existing RGB smoke outputs are synthetic
Actor-Transformer runs, not pretrained I3D assets. Search logs preserve access
errors in unrelated private/test-cache directories; this is not a claim to have
searched every inaccessible file on the computer.

Canonical model source is
[DeepMind Kinetics-I3D](https://github.com/google-deepmind/kinetics-i3d), using
TensorFlow/Sonnet. Its RGB preprocessing maps pixels to [-1,1]. The compatible
candidate is [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d), whose
maintainer describes `rgb_imagenet.pt` as converted DeepMind ImageNet+Kinetics
weights and claims identical outputs. That claim has **not** been independently
validated here. This is **converted DeepMind I3D weights**, not official PyTorch
I3D weights. The repository includes an
[Apache-2.0 license](https://github.com/piergiaj/pytorch-i3d/blob/master/LICENSE.txt).

Public GitHub API metadata, retrieved without fetching checkpoint bytes:

| Field | Candidate |
| --- | --- |
| Pinned source commit | `05783d11f9632b25fe3d50395a9c9bb51f848d6d` |
| Commit date | 2018-06-28 |
| Checkpoint | `models/rgb_imagenet.pt` |
| Exact size | 50,883,138 bytes (48.53 MiB) |
| Git blob SHA1 | `73a4fd1d87eb3dbd97f33a9779fa1caa3d28111a` |
| SHA256 | Unavailable before acquisition; compute and record locally |
| Expected local repository files | `checkpoints/external/pytorch-i3d/` |
| Expected local checkpoint | `checkpoints/external/pytorch-i3d/models/rgb_imagenet.pt` |
| Proposed feature export | `checkpoints/i3d_mixed4f_collective_rgb_v1.pt` (does not exist) |

The Git blob hash is **not** a raw-file SHA1 or SHA256. Verify it with
`git hash-object --no-filters`, then record SHA256 separately.
`upstream_asset_metadata.json` and `upstream_source_metadata.json` retain the
pinned URLs, byte counts, and fingerprints.

## Compatibility work required after acquisition

Do not install the old runtime or create another I3D architecture. Import the
pinned upstream model, construct the full 400-class RGB model, strictly validate
the full converted state dictionary, then wrap its registered blocks through
`Mixed_4f`. The upstream constructor returns before `build()` for intermediate
endpoints, and its default `extract_features()` proceeds beyond Mixed_4f; neither
is directly the required export API.

A minimal repository-owned feature/export wrapper will need current-PyTorch
tests for strict loading (including legacy BatchNorm bookkeeping), endpoint
equality, eval determinism, finite values, ten-frame input, and CPU/CUDA tolerance.
The upstream Python/NumPy padding code may need a carefully bounded trace or
script-compatible wrapper; this has not been executed. Any inference trace must
declare its input contract, preserve padding and full-precision convolution
semantics, and receive cold/hot CPU/CUDA checks informed by the HRNet experience.
No compatibility implementation has been added without the asset.

**Derived, not measured:** with upstream SAME-padding strides, RGB input
`[B,3,10,480,720]` should produce Mixed_4f `[B,832,3,30,45]`. After temporal mean:
`[B,832,30,45]`; after resize: `[B,832,90,160]`; actor RoIs:
`[sum(valid),832,5,5]`; padded flattened actor features: `[B,N,20800]`.
The archive should declare input size [480,720], incoming range [0,1], RGB,
and mean/std [0.5,0.5,0.5], applying normalization exactly once.

## Exact acquisition command — prepared, not executed

From the project root, after choosing to acquire the pinned public asset:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\runs\collective\rgb_gt_v1_audit\acquire_i3d_assets.ps1
```

The saved script downloads only `pytorch_i3d.py`, `README.md`, `LICENSE.txt`, and
`models/rgb_imagenet.pt` from the pinned revision, checks every Git blob hash
and byte count, records SHA256, and refuses an existing destination. It does not
install dependencies, import upstream code, export a model, or start experiments.
Total requested asset size is recorded in the script. This command is provided
for user action under the requested download policy; it has **not been run**.

## Requested result fields and remaining gates

| Requested field | Status |
| --- | --- |
| 1–3: repository, pose artifacts, CUDA | Verified above; initial state preserved. |
| 4: existing I3D infrastructure | Adapter/geometry/cache/model path audited; existing synthetic tests pass. |
| 5–6: source/checkpoint | Pinned conversion candidate identified; not acquired or accepted as validated. |
| 7: compatibility work | Required export wrapper identified; not implemented. |
| 8: Mixed_4f contract | Existing generic contract verified; full-resolution shape derived, unmeasured. |
| 9: real ten-scene RGB inspection | Not run: no real I3D asset. |
| 10: GPU feasibility benchmark | Not run; no throughput, peak memory, or runtime estimate claimed. |
| 11: real five-step RGB preflight | Not run. |
| 12: full RGB extraction | 0 scenes / 0 actors extracted; target 2,547 / 12,874. |
| 13: RGB freeze receipt | Not created; proposed ID `collective_rgb_gt_v1`; pose receipt preserved. |
| 14: selected RGB checkpoint iteration | Unavailable. |
| 15–16: RGB validation group/actor accuracy and F1 | Unavailable. |
| 17–18: RGB held-out group/actor accuracy and F1 | Unavailable; no RGB test inference. |
| 19: RGB per-class recalls | Unavailable; verified pose recalls above. |
| 20: RGB−Pose deltas | Unavailable for all four metrics. |
| 21: major confusion-pair improvement | Unknown until RGB completes. |
| 22: errors/warnings | Missing I3D asset; temp-directory permissions; upstream TorchScript deprecations, Python launcher warning, and a nonterminating Windows exception diagnostic during final pytest. |
| 23: final pytest | **392 passed**, 253 warnings, 272.92 seconds, exit 0. |
| 24: Ruff | Lint passes; formatting passes, 129 Python files. `git diff --check` passes. |
| 25: Git | Same branch/HEAD and three original modifications; new untracked `docs/collective-rgb-asset-audit.md`; no staging, commit, or push. |
| 26: next experiment | Complete RGB-only seed 0 first; evidence does not yet support recommending fusion. |

Initial exact pytest invocation ended with **143 passed / 249 setup errors**,
because its sandbox temporary root could not create numbered directories.
The existing `.pytest_cache` was also inaccessible. A fresh ignored run directory
resolved fixture setup without source or environment changes: **392 passed,
253 upstream warnings, 287.68 seconds**. Lint and formatting initially passed
(129 Python files); `git diff --check` passed with ordinary LF/CRLF notices.

The final full suite uses a distinct temporary directory under this audit output,
with `-o cache_dir=.../pytest_cache`; its log is `pytest_final.log`.
It completed **392 passed / 253 warnings**, exit 0. During the run, Python printed
`Windows fatal exception: code 0xc0000008` while executing a dataset-validation
test, then continued normally. This additional diagnostic remains in the log;
an isolated validation-suite recheck reproduced it with **18 passed**, exit 0.
The stack points to `cv2.imread` in the deliberately missing-image test. Running
that case with `--capture=sys` passed (**1 passed**, exit 0), emitting only the
expected OpenCV missing-file warning and no Windows exception diagnostic.
This localizes the observed behavior to the missing-file/capture interaction;
the underlying native cause is not established. No code or test was changed.
`preservation_verification.json` confirms that all **154 original tracked files**
and **17 pose artifacts/receipt files** are byte-for-byte unchanged.
No new tests were added because no compatibility/export implementation changed.
Once assets are supplied, resume with strict source/checkpoint validation, then
synthetic backbone checks, a tiny real sample, structured ten-scene inspection,
GPU benchmark, **REAL RGB PREFLIGHT — NOT BENCHMARK**, full extraction, separate
freeze, validation-selected 20k-step seed 0, and one held-out evaluation. Stop
after the pose/RGB comparison; fusion remains outside this experiment.

## Acquired-assets resume (2026-10-04)

The missing-asset blocker above is historical and is now resolved. The supplied
model snapshot/checkpoint passed size, pinned-source/Git-blob fingerprint and
strict architecture validation. See [the execution record](collective-rgb-execution.md)
and [the I3D export boundary](i3d-export.md) for the actual asset SHA256,
compatibility changes and new offline regression coverage.

The exported model passed full-resolution CPU/CUDA and native-endpoint checks,
ten real training scenes, a bounded GPU benchmark and a five-iteration real
preflight. Full RGB extraction completed with **2,547 scenes / 12,874 actors**;
all 2,547 caches passed content/provenance validation with no errors/warnings.
This resume keeps the original audit and pose artifacts, rather than rewriting
the earlier blocker as if assets had been present at that time.
