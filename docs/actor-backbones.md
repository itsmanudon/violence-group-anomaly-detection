# Local actor backbone exports

The raw actor pipeline requires locally supplied, vetted **feature-only TorchScript
archives**. It never downloads weights, silently initializes random backbones, or
substitutes another architecture. No research repository is vendored. Precomputed
actor features remain usable without either archive.

## Python API

```python
from pathlib import Path
from surveillance.features.hrnet_pose import HRNetPoseExtractor
from surveillance.features.i3d import I3DActorExtractor

pose = HRNetPoseExtractor(Path("hrnet_features.pt"), frozen=True)
rgb = I3DActorExtractor(Path("i3d_mixed4f.pt"), frozen=True)
# frames: float RGB [B,10,3,H,W] in [0,1]
# boxes: float normalized xyxy [B,N,4]; valid: boolean [B,N]
pose_features = pose(frames[:, 5], boxes, valid)  # [B,N,98304]
rgb_features = rgb(frames, boxes, valid)         # [B,N,20800]
```

Both are `nn.Module`s with `feature_dim`, `.backbone`, `.metadata`, `.to(device)`,
and normal `state_dict`/`load_state_dict` behavior. Inputs and masks must share
the module's device. `frozen=True` disables parameter gradients, keeps the
backbone in evaluation mode even after a parent `.train()`, and runs extraction
without recording gradients. Construct with `frozen=False` to fine-tune through
the backbone, resizing, and RoIAlign. Use a scripted export that preserves
training/evaluation behavior and its parameters for this; an inference-optimized
or frozen archive, or a trace with evaluation branches baked in, cannot support
reliable fine-tuning. Reload the original feature archive when changing modes.

## Declared export contract

Save UTF-8 JSON as the embedded extra file `actor_backbone.json`. For HRNet:

```json
{
  "schema_version": 1,
  "architecture": "pose_hrnet_w32",
  "endpoint": "pre_final_layer",
  "output_channels": 32,
  "feature_only": true,
  "provenance": "Record source repository + revision, weight source + SHA256, training dataset, and export procedure here",
  "preprocessing": {
    "color_order": "rgb",
    "input_range": [0, 1],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "input_size": [256, 192]
  }
}
```

Mean/std above illustrate ImageNet normalization; **use the values required by
your actual pretrained weights**. There is no assumed normalization default.
The archive accepts already normalized inputs and must not normalize them again.
`input_range` describes the adapter's incoming RGB frames before normalization.

For I3D use `architecture: "i3d"`, `endpoint: "Mixed_4f"`,
`output_channels: 832`, and add `input_frames: 10`. Supply explicit positive
`preprocessing.input_size: [height,width]` from the checkpoint's training/export
protocol. The adapter bilinearly resizes frames to that size without a center
crop, then applies the declared mean/std. For weights expecting [-1,1], declare
mean `[0.5,0.5,0.5]`, std `[0.5,0.5,0.5]`, with incoming `input_range: [0,1]`.

An archive is produced in the vetted model's own environment after loading the
original state dictionary strictly and constructing a feature-only wrapper:

```python
import json
import torch

# feature_model must be the reviewed HRNet/I3D wrapper with pretrained weights.
# metadata must contain the exact contract above and real provenance details.
scripted = torch.jit.script(feature_model)
torch.jit.save(scripted, "features.pt", _extra_files={
    "actor_backbone.json": json.dumps(metadata),
})
```

Compare the exported wrapper numerically against the original model on known
inputs, including train/eval behavior if fine-tuning is intended. For HRNet-W32,
return the high-resolution stage-4 branch `y_list[0]` immediately before
`final_layer`; do not return the final joint heatmaps. This boundary is visible
in the [original HRNet pose implementation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/lib/models/pose_hrnet.py).
For I3D, execute the stem and inception blocks through `Mixed_4f`, return that
tensor, and omit all later pooling/classification operations; the
[I3D PyTorch implementation](https://github.com/piergiaj/pytorch-i3d/blob/master/pytorch_i3d.py)
defines this endpoint with 832 output channels. Source implementations may need
a small script-compatible wrapper in their own environment; a classifier's
default `forward` is not a feature export.

The loader checks required metadata fields/types and preprocessing values, then
checks runtime tensor dimensions and finite outputs. These checks establish the
**declared interface**, not model identity or weights provenance: arbitrary code
can claim the right architecture and emit the right shapes. Only use archives
whose origin and export process you have independently vetted. No real
pretrained weights have been loaded or real-data accuracy measured by the
synthetic tests. TorchScript is deprecated in the installed PyTorch version;
this explicit archive contract is an interim local compatibility boundary, and
future `torch.export` support will require its own versioned schema.

## Coordinates and features

HRNet crops each valid actor to `[3,256,192]` using RoIAlign. The export must return
`[sum(valid),32,64,48]`, flattened to 98,304 values per actor. Crops stretch the
box directly; no unrecorded aspect-ratio expansion or affine augmentation is
applied. The caller selects the center frame (index 5 for the ten-frame protocol).

I3D receives `[B,3,10,H,W]` after adapter preprocessing and must return
`[B,832,t,h,w]`. The adapter averages the temporal axis, bilinearly resizes the
feature grid to 90x160 with `align_corners=False`, and RoIAligns each actor to
5x5, yielding 20,800 values. These feature dimensions follow the
[Actor-Transformers paper's feature extraction protocol](https://arxiv.org/html/2003.12737).
Center-frame boxes are used for this pooled clip representation; this does not
track moving actors across the clip.

Geometry functions use xyxy **edge coordinates**; size tuples are `(height,width)`.
Normalized x multiplies width, normalized y multiplies height. The image-to-map
transform scales x and y independently; for a 200x100 image and 160x90 map,
`[50,10,150,90]` becomes `[40,9,120,81]`. RoIAlign uses `spatial_scale=1`,
`sampling_ratio=2`, and `aligned=True` (half-pixel convention).

Valid boxes must be finite with positive width/height and lie in [0,1]. Explicit
`clip_outside=True` allows partial boxes to be clipped; inverted or completely
clipped-away boxes fail. Padding boxes are never validated or sampled and may
contain NaNs. Outputs restore `[B,N,D]` with exactly zero padded rows. Standalone
geometry and adapters support empty selections; dataset/training scene validation
must reject all-empty scenes before classification.

## Verification

`tests/test_actor_backbones.py` uses deliberately synthetic scripted networks,
not HRNet/I3D substitutes intended for research. Tests check coordinate scaling,
RoIAlign's numeric output and gradients, padding, temporal mean, output shapes,
frozen/trainable modes, and rejection of invalid metadata or geometry. These
require PyTorch and torchvision with compatible compiled RoIAlign operators and
run on CPU without weights or network access.
