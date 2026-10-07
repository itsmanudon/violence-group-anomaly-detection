# Official local HRNet-W32 feature export

Milestone 2C-R1 targets the official COCO keypoint model
`pose_hrnet_w32_256x192.pth` from
[deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
Supply a vetted local checkout or validated copy and the official checkpoint.
Nothing downloads either asset. Importing the supplied model executes that local
Python file; use the reviewed upstream code. The checkpoint filename alone does
not establish authenticity. An optional expected SHA256 binds an independently
verified copy, not an invented canonical hash.

## Required files

```text
<local HRNet checkout>/
  lib/models/pose_hrnet.py
  experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
checkpoints/
  pose_hrnet_w32_256x192.pth
```

The exporter imports that model file directly through `importlib`, avoiding the
old upstream training/COCO toolchain and `models/__init__.py`. The official model
factory accepts the parsed YAML mapping, so no upstream YACS configuration loader
is required. Our wrapper stays in this repository; upstream code is neither copied
nor edited, including bytecode caches. Existing PyTorch and PyYAML dependencies
suffice for the inspected
upstream implementation. A different upstream revision must pass the same checks.

From the repository root, using the installed project environment:

```powershell
python scripts/export_hrnet_features.py --hrnet-repo "D:/path/to/deep-high-resolution-net.pytorch" --checkpoint checkpoints/pose_hrnet_w32_256x192.pth --output checkpoints/hrnet_w32_features.pt
# Optional: append --checkpoint-sha256 <trusted-original-checkpoint-SHA256>
```

The default JSON report is `checkpoints/hrnet_w32_features.json`; `--report` can
choose another destination. Existing archive/report paths are refused. Missing
assets or incompatible weights produce a nonzero exit and an actionable message.
The example checkout path is a placeholder to replace with your supplied location.

## Endpoint and validation

The upstream [forward implementation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/lib/models/pose_hrnet.py)
passes the high-resolution stage-4 branch to `final_layer`. Its
[W32 configuration](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml)
uses image size **[192,256] in width/height order**. Our adapter's size convention
is **[256,192] in height/width order**.

Conversion performs these checks before publishing an archive:

1. Validate the W32/COCO configuration, stage widths/blocks and input/heatmap sizes.
2. Construct upstream `get_pose_net(cfg, is_train=False)`. Upstream partial
   pretrained initialization is bypassed; no ImageNet weight path is loaded.
3. Load all original weights, including the heatmap head, with exact key,
   shape and dtype matching and finite tensors. Bare state dictionaries,
   `{"state_dict": ...}` and uniformly `module.`-prefixed keys are supported.
   Mixed prefixes, missing keys and classifier removal before loading are rejected.
   Loading uses `weights_only=True`; obsolete serialized Python model objects
   need a separately vetted conversion to a tensor state dictionary.
4. Capture the native head input and verify finite **[N,32,64,48]** features
   and **[N,17,64,48]** joint heatmaps on synthetic 256x192 crops.
5. Replace only `final_layer` with `Identity` and verify exact equality with
   the captured pre-head tensor. No flattening/projection/normalization changes
   are used to hide an incompatible representation.
6. Trace native eval forward; add a scripted input guard requiring nonempty
   **[N,3,256,192]** crops. Check batch sizes 1,2,3, native/export agreement,
   finite output and repeated eval determinism.
7. Save the existing TorchScript archive contract with `actor_backbone.json`;
   reload it through **HRNetPoseExtractor**, run a synthetic actor crop and
   verify **[1,1,98304]** finite, repeatable features.
8. Verify source/model/checkpoint hashes did not change during conversion.

These checks validate the supplied implementation and weights' compatibility;
they cannot authenticate a malicious or unrelated source claiming the same shape.
The automated external-repository fixture is deliberately a tiny non-HRNet network.
It verifies exporter mechanics and strict failures, never pretrained model quality.

The archive accepts **already normalized RGB crops**. The existing adapter takes
RGB [0,1], performs RoIAlign actor cropping, and applies the official evaluation
mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]` once. These values are
used by the upstream [pose evaluation tool](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/tools/test.py).
The pre-head map is flattened to **32 x 64 x 48 = 98,304** per actor by the
adapter, then projected by the unchanged Actor-Transformer to 128 dimensions.

## Provenance and supported use

Embedded metadata and the export report record architecture, feature endpoint and
shape, input resolution, original checkpoint SHA256/path/format, source repository
identifier/path, Git commit where available, model-source/config SHA256 and exporter
version/method. The report additionally hashes the final archive. COCO training
provenance is a user-supplied declaration, not a certification by the exporter.

There are **two different checkpoint identities**: the original official `.pth`
and the converted feature `.pt`. Set `features.pose.checkpoint_sha256` in the
experiment protocol to the **feature archive** SHA256 from the report. Its embedded
metadata retains the original weight hash. Extraction sidecars already bind both
through the full archive metadata, annotation/order/source-image fingerprints and
feature-file hashes. Old synthetic or stale caches must fail benchmark validation.

Exports are **inference-only**, intended for frozen feature extraction. The adapter
rejects `frozen=False` for these eval-traced archives. Other vetted scripted archives
retain the previous fine-tuning contract. TorchScript is deprecated in current
PyTorch; it is retained here to preserve the versioned local archive interface.
No model architecture migration is part of this task.

Real CUDA execution exposed TorchScript profiling optimization changing the
first versus subsequent forward's floating-point output (maximum difference
`3.51e-4` on a fixed synthetic crop under torch 2.13.0/cu126). Subsequent outputs
were bitwise stable. The pose adapter disables JIT execution optimization only
for inference-only archives, preserving one graph from the cold call onward.
Weights, endpoint and preprocessing are unchanged; other scripted backbones and
fine-tuning retain their prior execution policy. Feature sidecars bind
`jit_execution_policy: unoptimized_inference_only` to reject caches made under
the previous policy. CPU-only synthetic verification did not expose this GPU
startup behavior; no earlier real benchmark results existed.

Exporter version 2 explicitly traces with `cudnn_allow_tf32=False`. The first
real CPU/CUDA comparison exposed version 1 embedding `allow_tf32=True` in all
292 convolution nodes; runtime backend settings could not override those
serialized constants. Version 2 retains native FP32 arithmetic when moving
the archive to CUDA. Its metadata records the policy, and its different archive
fingerprint prevents stale cache reuse. This fixes numerical portability without
changing weights, tensor dimensions, crops, or network architecture. Retain a
version-1 archive/report as diagnostic evidence; export version 2 to a new path
such as `checkpoints/hrnet_w32_features_fp32.pt` rather than overwriting it.
The trace context explicitly preserves the caller's cuDNN enabled, benchmark
and deterministic flags; its defaults must not accidentally disable cuDNN.

The existing protocol stretches actor boxes via RoIAlign; it does not adopt the
upstream pose estimator's aspect-ratio expansion, affine augmentation or pose
heatmap decoding. This preprocessing distinction and the unverified historical
split equivalence preclude an exact paper-reproduction claim. The conversion
does not change the validated Actor-Transformer mathematics.

Continue with the bounded inspection and frozen run in
[milestone-2c-real-pose.md](milestone-2c-real-pose.md). No official assets were
available during the original implementation verification. The subsequent real
execution validated the supplied upstream W32 configuration and complete COCO
checkpoint, exported `checkpoints/hrnet_w32_features_fp32_cudnn.pt`, and verified
the endpoint on CPU and the RTX 4070 Laptop GPU. Its JSON export report retains
the original checkpoint/source/config hashes and archive identity. Read the real
execution record in that document before treating older diagnostic archives or
preflight results as benchmark evidence.
