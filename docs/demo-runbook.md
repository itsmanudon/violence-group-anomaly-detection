# Local professor demo runbook

## Readiness status

The interface, real trained DCSASS models and eight genuine cascade example caches
are installed. Sultani held-out bag ROC-AUC is 0.6583; the six-class behavior model
has 51.02% conditional accuracy, 0.2140 macro F1 and 69.22% coverage. These results
show substantial weaknesses. UCF-Crime remains a separate pending frame benchmark.

## Environment and launch

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

Verify the running interface with `python scripts/validate_demo_api.py`. The warm
GPU two-second example takes about 1.10 seconds, while the full CPU fallback was
27.4 seconds. Cache mode shows recorded inference time, not current playback time.
See [measured integration evidence](integration-demo-v1.md) for timing scope and
the retained failures. Final browser playback/visual inspection remains a manual
pre-presentation check because browser automation could not attach.

Eight real held-out clips are installed in `data/examples/` and recorded in
`configs/demo_examples.json`: one per dataset label and a separate high-confidence
behavior mistake. They were selected transparently, include several failures, and
decode successfully. All have measured cascade caches. A supplemental
**Normal: no-alert example** was selected after evaluation to illustrate bypass;
its post-hoc selection is disclosed and all original failures remain. Show that
example, **Robbery example**, then **Limitation: high-confidence mistake** for
anomaly/Normal disagreement. **Normal example** is a retained false alert;
**Assault example** and **Vandalism example** are retained missed anomalies.
Every entry records source, dataset label and selection policy.
Never substitute synthetic test videos as research examples.

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
