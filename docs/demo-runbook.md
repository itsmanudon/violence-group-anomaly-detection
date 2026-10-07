# Local professor demo runbook

## Readiness status

The offline interface, all five pinned model assets and nine genuine cascade
example caches are installed. The default Sultani scorer is trained on protected
UCF sources: held-out frame ROC-AUC 0.7441. The DCSASS six-class behavior model has
51.02% standalone conditional accuracy, 0.2140 macro F1 and 69.22% actor coverage.
The separate complete cascade has clip-alert F1 0.4733 and 31.38% overall behavior
coverage. These results show substantial weaknesses; this is a research demo.
See [UCF evidence](sultani-ucf-results-v1.md) and
[cascade evidence](cascade-results-ucf-v1.md).

## Environment and launch

For another machine, including macOS, use [the replication handoff](macos-replication.md).
Datasets, pretrained/trained checkpoints, selection receipts and result caches
are intentionally external to Git. That guide distinguishes the minimal live
inference asset bundle from the larger research dataset/cache installation.

Open PowerShell in the repository root. The validated environment is Python 3.13.5,
torch 2.13.0+cu126 and torchvision 0.28.0+cu126. Preserve these matched packages.

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe scripts/verify_demo_assets.py
.\.venv\Scripts\python.exe -m surveillance.demo
```

Open `http://127.0.0.1:7860`. One process serves the interface and inference.
There is no public share link. Optional UI dependencies are installed with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[demo]"
```

All installed checkpoints, video examples and result caches are local. Do package
installation and acquisition before the classroom session; installed presentation
execution requires no Internet access. System fonts and disabled Gradio analytics
avoid an external font/telemetry dependency.

## Required assets and verification

`configs/surveillance_demo.yaml` names five local assets: a completed Sultani scorer,
the converted C3D FC6 export, a completed six-class behavior checkpoint, COCO_V1
Faster R-CNN, and the validated I3D Mixed_4f export. Final deployment pins must
contain all five SHA256 values and identify each subsystem's training population.

Both trained checkpoints require a completed `selection.json` in their run folder,
with `selection_frozen=true` and a matching checkpoint hash. Preflight checkpoints
cannot be served. Live behavior also verifies detector and I3D hashes against its
training configuration. A changed model requires a deliberately updated deployment
receipt and recomputed example results; an old result must not silently appear as
the new model's output.

Dataset downloads, model checkpoints and runs are not stored in Git. Keep their
existing local paths or install them using [the asset instructions](end-to-end-required-assets.md).
The C3D conversion and preprocessing caveat is documented in
[the backbone validation](c3d-openmmlab-validation.md).

## Expected interface

Select an installed example or upload a readable prerecorded video, then press
**Analyze video**. The interface displays the video, final review alert, separate
behavior probabilities, anomaly timeline, highlighted analyzed intervals, actor
boxes on reference frames, detector confidence, interval evidence and result JSON.

An uploaded video always uses genuine on-demand inference. Installed examples can
use precomputed results and are explicitly marked **Cached example**. Uncheck the
cache option to process the installed clip live. Caches bind video bytes, trained
model hashes, preprocessing and routing policy. Cached playback can work without
loading the checkpoint tensors when the configured identities match.

No suspicious segment means actor analysis is skipped. If Sultani detects an
anomaly but the behavior model predicts Normal, both outputs and disagreement are
shown. If no people are detected, the generic anomaly alert remains and behavior
abstains. Boxes describe only analyzed reference frames; they are not person tracks.

## Presentation procedure

Before class, start the demo, verify a normal example and an anomaly example,
check video playback and confirm the cache/live mode label. Keep a known limitation
example visible for discussion. Model selection and thresholds are frozen before
choosing presentation clips. Do not adjust settings to improve the displayed clips.

Verify the running interface with `python scripts/validate_demo_api.py`. Each run
writes a unique receipt under the configured deployment's `api-validation/`
output directory; an explicit existing receipt path is rejected. The actual
curated four-second Robbery case takes 1.09 seconds on warm GPU and 33.21 seconds
on CPU, excluding model loading. Cache mode shows recorded inference time, not
current playback time. The full 991-clip cascade averages 0.505 seconds (p95 1.346),
including bypasses; this cannot establish full-video real-time FPS. See
[measured cascade evidence](cascade-results-ucf-v1.md) for timing scope and failures.
The [historical deployment report](integration-demo-v1.md) preserves older timings.
Switching source clears prior evidence; uploading or
clearing a video selects upload mode. Perform a manual upload/playback check before
class; automated file selection is blocked by the Chrome extension's file-access
permission, while the real live-upload server API test passes.

Nine real held-out clips are installed in `data/examples/` and recorded in
`configs/demo_examples_ucf_v1.json`. Show **Normal activity (previous bypass example)**,
then **Robbery: correct cascade example (curated)**, then
**Limitation: high-confidence mistake** for anomaly/Normal disagreement.
The correct Robbery case is explicitly post-hoc: first correct Robbery alert by
clip ID in saved registered results. It illustrates the interface and is not an
accuracy benchmark. All eight original cases remain, including failures.
**Normal example** is a false Robbery alert; the original **Robbery example** and
**Assault example** are missed anomalies. **Fighting example**, **Abuse example**
and **Vandalism example** alert with the wrong Robbery behavior. The previous
Normal bypass case still bypasses under the new model. Every entry records source,
dataset label and selection policy. Models and threshold remain frozen.
Never substitute synthetic test videos as research examples.

The default config is `configs/surveillance_demo.yaml`, identical to the versioned
UCF deployment. To inspect the preserved older DCSASS anomaly deployment, launch
with `--config configs/surveillance_demo_dcsass_v1.yaml`; its original caches and
measurements remain separate. Prerecorded uploads and examples are supported;
webcam capture is not part of this tested MVP.

Keep the repository, virtual environment, checkpoints, selected run/selection
receipts, `data/examples/` and `outputs/demo/` on the Windows machine. Raw datasets
may remain on E: for research; installed cached examples and uploaded-video
inference do not require the entire UCF archive. Do not copy the Windows virtual
environment to a Mac as a portable executable environment: recreate dependencies
for that platform. The validated presentation hardware is this Windows GPU system.

## Troubleshooting

- **Missing checkpoint:** the interface lists absent assets and live analysis
  returns a useful error. Install the completed local checkpoint and receipt.
- **Unreadable/truncated video:** use a complete file with consistent frame metadata
  and a supported codec. Decode-count mismatch is rejected to protect timestamps.
  Uploads preserve original video bytes; no audio analysis or mandatory FFmpeg
  re-encoding is performed. Browser playback still depends on codec support.
- **Zero actors:** this is an abstention, not a Normal classification. Review the
  anomaly timeline and the detector's coverage limitation.
- **CUDA unavailable:** automatic selection falls back to CPU. Processing may be
  slow. Explicit CPU launch is `python -m surveillance.demo --device cpu`.
- **Port occupied:** launch with `--port 7861` and open the matching local address.
- **Cache provenance error:** restore its pinned models/policy/video or recompute
  using the frozen trained deployment. Do not edit result values by hand.
- **Offline presentation:** use installed caches; retain the live upload option
  for a deliberate demonstration of genuine inference when time permits.

Predictions are a research prototype, may be wrong and require human review.
Explain the weak minority-class behavior results and the lack of temporal DCSASS
ground truth before presenting timeline localization as evidence.
