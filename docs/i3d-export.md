# Local RGB I3D Mixed_4f export

Use the supplied `piergiaj/pytorch-i3d` implementation at revision
`05783d11f9632b25fe3d50395a9c9bb51f848d6d` and
`models/rgb_imagenet.pt`. These are **converted DeepMind ImageNet+Kinetics I3D
RGB weights**, not official PyTorch I3D weights or Charades fine-tuned weights.
TensorFlow-to-PyTorch equivalence has not been independently established here.
No source, dependencies or checkpoint is downloaded by the exporter.

The installed four-file source snapshot was verified against the pinned Git
blob fingerprints. It is not a full Git checkout. Model-source SHA256 is
`c322c3ca015dab5ee43e4d15707149d510b98931406c4a5d58b27934ed6777ed`;
the 50,883,138-byte checkpoint has SHA256
`2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538`.

## Conversion boundary

The repository imports the user-supplied upstream file without editing it.
It constructs the complete 400-class, three-channel model and validates its
complete checkpoint before selecting registered native blocks through
`Mixed_4f`. Upstream intermediate-endpoint construction returns before
`build()` registers modules; constructing the full model avoids that trap.
Neither the upstream final `forward()` nor `extract_features()` is the desired
Mixed_4f export endpoint.

Checkpoint keys, shapes, dtypes and finite values are checked. The sole legacy
shim initializes absent `num_batches_tracked` integer buffers for actual
BatchNorm modules, recording every addition. The acquired checkpoint needs 57
such counters; no learned tensor or running statistic is omitted. This
bookkeeping is unused in eval inference. Full state loading is then strict.

The native eval prefix is traced for a fixed geometry and wrapped with scripted
input guards. Old Python/NumPy SAME-padding branches are specialized for that
geometry; a runtime guard rejects different channels, frame count or resolution.
Batch size remains variable. The archive is inference-only and cannot be used
for backbone fine-tuning. Trace specialization/deprecation warnings are retained.

CPU tracing binds full float32 convolution semantics with TF32 disabled while
preserving the caller's cuDNN enabled/benchmark/deterministic flags. The RGB
adapter uses unoptimized JIT execution for inference-only archives, matching
the established HRNet policy and avoiding cold/hot profiling rewrites. This
policy is bound into extraction provenance. Other archives retain their
existing execution policy. No Actor-Transformer mathematics changes.

## Input and output

| Boundary | Contract |
| --- | --- |
| Dataset/adapter input | float32 RGB `[B,10,3,480,720]` in `[0,1]` |
| Normalization, once in adapter | mean/std `[0.5,0.5,0.5]`, mapping to `[-1,1]` |
| Archive input | normalized RGB `[B,3,10,480,720]` |
| Mixed_4f | `[B,832,3,30,45]` |
| Temporal mean | `[B,832,30,45]` |
| Bilinear resize | `[B,832,90,160]`, `align_corners=False` |
| RoIAlign | `[sum(valid),832,5,5]`, `aligned=True`, sampling ratio 2 |
| Actor features | `[B,N,20800]`, padded rows zero |

The full-frame 480x720 localization input differs from the original I3D
224x224 classification crop. It is an explicit Actor-Transformer pipeline
choice, not an 8-GB-memory fallback. Ten frames and all feature/RoI dimensions
remain fixed. Alternate resolutions in the export API exist for explicit
offline fixtures; do not use them to silently change this benchmark.

## Export and verify

From the repository root, using its virtual environment:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe scripts/export_i3d_features.py --i3d-repo checkpoints/external/pytorch-i3d --checkpoint checkpoints/external/pytorch-i3d/models/rgb_imagenet.pt --checkpoint-sha256 2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538 --output checkpoints/i3d_mixed4f_collective_rgb_v1.pt
```

The archive and `.json` report must not already exist. The default source
revision/hash select the pinned implementation. Different trusted source
copies require explicit expected hashes and a declared revision; hashing alone
does not authenticate a revision. Source/checkpoint changes during export fail.

Embedded `actor_backbone.json` records architecture, endpoint, preprocessing,
source revision/hash/path, raw checkpoint hash/path, legacy counters, exporter
version and PyTorch version. Existing archive/report files are refused, not
reused or overwritten. Benchmark cache/receipt validation binds the archive
SHA256 and extraction configuration, rejecting incompatible/stale artifacts.

Native/export comparison and adapter acceptance are part of export validation.
Before real extraction, also check CPU/CUDA agreement, cold/repeated eval,
temporal pooling, resizing, RoI coordinates, actor ordering and Actor-Transformer
input acceptance. The executed synthetic validation report is under ignored
`runs/collective/rgb_gt_v1_execution/backbone_validation.json`.

Current full-resolution synthetic validation: native/export CUDA difference
zero; CPU/CUDA maximum absolute difference `1.52587890625e-5`, passing
`rtol=1e-3, atol=1e-4`. These are numerical contract checks, not real benchmark
performance. No tolerance was loosened to mask the earlier TF32 discrepancy.

The offline tests use a tiny explicitly non-I3D external fixture, not random
weights presented as a pretrained I3D. They require no internet or real assets:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_i3d_export.py -q
```
