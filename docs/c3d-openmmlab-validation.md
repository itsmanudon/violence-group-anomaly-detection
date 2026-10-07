# Supplied OpenMMLab C3D asset

The supplied `checkpoints/external/c3d_sports1m_pretrain_20201016-dcc47ddc.pth`
is 311,985,866 bytes, SHA256
`dcc47ddcdc7dd62eabc24bc70491b47c82b3b1045b864c52f6bdb669243dfcff`.
It contains eight native PyTorch convolutions, FC6 and FC7. Every tensor is
finite float32 with the expected shape. The trusted acquisition URL is referenced
by [OpenMMLab's configuration](https://github.com/open-mmlab/mmaction2/blob/a5a167dff2399e2d182a60332325f9c0d4663517/configs/_base_/models/c3d_sports1m_pretrained.py).

The converter explicitly renames `conv1a.conv` to `conv1`, `conv2a.conv` to
`conv2`, and the remaining `.conv` modules to existing local names. FC6 is preserved
without transposition. FC7 is validated, then omitted. Unknown, absent, incorrectly
shaped, nonfinite or non-float32 tensors fail. The exporter requires the vetted
original checkpoint hash and refuses to overwrite an export or its receipt.

```powershell
.\.venv\Scripts\python.exe scripts/export_c3d.py --checkpoint checkpoints/external/c3d_sports1m_pretrain_20201016-dcc47ddc.pth --output checkpoints/c3d_fc6_openmmlab_v1.pt
```

The resulting export SHA256 is
`beac4ff065de3663bf5c6bab36aedca4fe56ec59b231c66bf5374c23eda14831`.
Its JSON sidecar records the original hash, key conversion and preprocessing.
The existing extraction adapter now rejects mean/channel settings that disagree
with a supplied export's provenance. Legacy checkpoints without metadata retain
their explicitly configured preprocessing contract.

## Endpoint verification

Pinned upstream revision: `a5a167dff2399e2d182a60332325f9c0d4663517`.
Upstream `c3d.py` SHA256:
`d5553c1fd1947042e9a87ae9ce8b8a4627fcb00f9004d5ff9cba92d87c86f668`.
The snapshot, configuration, license and acquisition record are under the ignored
`checkpoints/external/openmmlab-c3d-source/` directory.

The bounded GPU check executes the pinned upstream C3D class with a transparent
`ConvModule` compatibility shim using native `Conv3d` + `ReLU`; it does not install
MMCV/MMEngine or change PyTorch. A hook captures post-ReLU FC6 before FC7.
Two independently seeded `[1,3,16,112,112]` inputs match local FC6 exactly:
**maximum absolute error 0.0**, finite `[1,4096]` outputs, and bit-identical repeats.
Peak allocated CUDA memory was 680,573,952 bytes for the comparison.
The receipt is `runs/c3d-conversion-validation/endpoint_equivalence.json`.
This establishes implementation equivalence, not anomaly-detection accuracy.

## Preprocessing and reproduction limits

Use **RGB**, raw 0–255 values, channel mean `[104,117,128]`, std `[1,1,1]`,
16-frame clips. The repository uses fixed resize to 128x171 followed by its
existing 112x112 center crop. These settings follow the modern upstream RGB
configuration for means and the repository's existing spatial contract.

The original Caffe volume-mean pipeline and modern channel means are not identical;
the modern upstream evaluation also uses aspect-ratio-preserving resize. This is a
documented C3D/Sultani research baseline, **not a paper-exact reproduction**.
The Actor-Transformer I3D preprocessing and backbone remain untouched.

UCF-Crime copying is still in progress. No real Sultani optimization, held-out AUC,
or calibrated anomaly threshold is claimed by this feature-extractor check.
