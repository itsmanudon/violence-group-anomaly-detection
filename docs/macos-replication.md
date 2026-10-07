# Clone and external-asset handoff

Git contains the package, scripts, scientific configurations, tests, source
provenance and research reports. It intentionally excludes footage, models,
features, detections, generated manifests/splits, run receipts and result caches.
A fresh clone can install, run software tests and open the interface; trained
inference needs the separate assets below. Missing assets return useful errors.
No random/preflight model is substituted and no model is automatically downloaded.

A local merge becomes available through GitHub only after its owner publishes
it. Clone/pull the published main, or transfer the Git repository separately;
check its commit against the final handoff. A Windows virtual environment cannot
be reused on a Mac.

## macOS environment

The package requires Python 3.11+. The validated Windows research environment
uses Python 3.13.5, torch 2.13.0+cu126 and torchvision 0.28.0+cu126. Start with
Python 3.13 on the Mac and install wheels compatible with its architecture;
do not copy Windows packages or use a CUDA wheel index on macOS. Matched torch
and torchvision are required, including their compiled NMS/RoIAlign operators.
Consult [official PyTorch installation instructions](https://pytorch.org/get-started/locally/)
for supported platform builds if dependency installation fails.

From the cloned repository root, in Terminal:

```bash
git clone https://github.com/itsmanudon/violence-group-anomaly-detection.git
cd violence-group-anomaly-detection
git switch main
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,demo]"
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m surveillance.demo --help
python -m surveillance.demo --device cpu
```

Use your installed Python 3.11+ command if it is named differently. Open
`http://127.0.0.1:7860`. All checks should run without research datasets or
checkpoints; one drive-letter test is intentionally skipped outside Windows.
Tests require Git as well as the dev/demo Python extras.

The deployed CLI supports `auto`, `cpu` and `cuda`. `auto` selects CPU on a Mac;
CUDA acceleration is optional and platform-dependent, not available on macOS.
PyTorch has an [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html),
but this cascade's TorchScript, deterministic operations and detector/RoIAlign
path have not been validated on MPS. No MPS speed or compatibility claim is made.
CPU inference may be slow; the measured Windows CPU times are not Mac benchmarks.
Mac hardware was not available for this audit: verification was performed on the
existing Windows environment, with platform-neutral paths and asset-free tests.

## Smallest existing-demo asset handoff

For genuine uploaded-video inference, transfer the exact five byte-pinned files
named by `configs/surveillance_demo.yaml`, preserving these relative destinations:

| Asset | Destination |
|---|---|
| Selected UCF Sultani scorer | `runs/ucf-crime/sultani_shared_safe_v1/seed_0/best.pt` |
| Selected six-class RGB Actor-Transformer | `runs/dcsass/human_rgb_detected_v1/random_init_seed_0/best.pt` |
| Converted C3D FC6 | `checkpoints/c3d_fc6_openmmlab_v1.pt` |
| Converted I3D Mixed_4f export | `checkpoints/i3d_mixed4f_collective_rgb_v1.pt` |
| Faster R-CNN COCO_V1 | `checkpoints/external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` |

Also copy each trained model's sibling `selection.json`. They must declare frozen,
non-preflight selection and match its checkpoint hash. Preserve export sidecars
(`.json`) and original source/acquisition receipts for provenance. These receipts
are intentionally external run artifacts. Uploaded inference does not require
all raw UCF/DCSASS/Collective data or their feature caches.

For the nine classroom examples, additionally transfer `data/examples/` and
`outputs/demo/ucf_v1/examples/`. The tracked example index pins the video bytes
and refers to relative files. Cached service relocates playback to the installed
video; overlays decode that local video, rather than opening historical Windows
paths embedded in saved metadata. Do not edit model scores or identity pins.
Copying historical metadata unchanged preserves provenance; it does not make old
absolute paths valid for new training/evaluation jobs.

```bash
python scripts/verify_demo_assets.py
python scripts/cache_demo_examples.py --output runs/demo-generation-mac.json
python -m surveillance.demo --device cpu
```

The cache command reuses valid installed results and generates only missing ones.
Use a new generation receipt path. Installed cached playback requires the byte-
pinned examples/results, but can work without loading model tensors. Installation
and asset acquisition need to happen before an offline presentation.

Keep default relative paths, or copy the deployment YAML to
`configs/surveillance_demo.local.yaml` (ignored), set absolute **Mac** paths such
as `/Volumes/ResearchDrive/models/...`, and pass `--config` to the verifier,
cache generator and demo. All model hashes/populations/preprocessing remain
unchanged when only relocating exact assets. Configuration paths are resolved
against the repository root; run these commands from that root.

## Research datasets and rebuilding derived artifacts

Dataset acquisition/usage terms remain external; retain original annotations,
source identity and version. See [required assets](end-to-end-required-assets.md),
[data formats](../data/README.md), [Collective protocol](collective-protocol.md),
[DCSASS protocol](dcsass-human-centric-protocol.md) and [UCF protocol](ucf-sultani-execution.md).

| External input | Placement/configuration | Derivation entry point |
|---|---|---|
| DCSASS videos and actual binary annotations | `data/raw/dcsass/` or audit `--root /Volumes/...` | `audit_dcsass.py`, `freeze_dcsass.py`, `cache_dcsass_actors.py` |
| UCF-Crime originals, official splits/temporal annotations and original category ZIPs for integrity checks | `data/raw/ucf_crime/` or preparation `--root /Volumes/...` | `prepare_ucf_surveillance.py`, `freeze_ucf_protocol.py`, `extract_ucf_sultani.py` |
| Collective seqNN frames/annotations | `data/raw/collective/` or `prepare_collective.py --root` | `prepare_collective.py`, `run_collective_experiment.py`, pose/RGB extraction CLIs |
| Sports1M C3D source weights | `checkpoints/external/` or export `--checkpoint` | `export_c3d.py`; [conversion contract](c3d-openmmlab-validation.md) |
| Pinned converted DeepMind RGB I3D weights and upstream source | Local checkout/snapshot supplied with `--i3d-repo` and `--checkpoint` | `export_i3d_features.py`; [export contract](i3d-export.md) |
| HRNet-W32 source/checkpoint (Collective pose comparisons only) | Local reviewed checkout with `--hrnet-repo`, checkpoint with `--checkpoint` | `export_hrnet_features.py`, `extract_pose_features.py`; [contract](hrnet-export.md) |
| COCO_V1 person detector | Default checkpoint path above, or explicit detector config/CLI path | `detect_people.py`, `cache_dcsass_actors.py`; [detector contract](person-detection.md) |

HRNet and Collective data are unnecessary for the selected RGB uploaded-video
demo once its five assets are installed. Rebuilding pretrained exports on another
software/platform version may produce different file hashes; install the exact
frozen exports to use the published deployment pins. A new training/export run
needs a separately named configuration and new validated selection/provenance;
never silently change the existing protocol or measured reports.

The specialized source-safe UCF workflow also expects two external author text
files in `data/splits/`: `Anomaly_Train.txt` and `Temporal_Anomaly_Annotation.txt`.
Obtain them from the author repository at the revision recorded in
`data/splits/ucf_authors_provenance.json`, or copy the already verified files.
That tracked JSON records byte counts and SHA256 values; compare before use.
The supplied UCF train/annotation files must match these pins.

Example rebuild sequence, only on a fresh working population/output location:

```bash
export DCSASS_ROOT="/Volumes/ResearchDrive/DCSASS Dataset/DCSASS Dataset"
export UCF_ROOT="/Volumes/ResearchDrive/anomaly-detection-dataset-UCF"
python scripts/audit_dcsass.py --root "$DCSASS_ROOT" --output runs/dcsass/audit_v1.json
python scripts/freeze_dcsass.py --audit runs/dcsass/audit_v1.json --ucf-test-annotations data/splits/Temporal_Anomaly_Annotation.txt --manifest data/manifests/dcsass_human_centric_v1_final.jsonl --receipt runs/dcsass/human_centric_v1/final_split_receipt.json
python scripts/cache_dcsass_actors.py --stage detect --output runs/dcsass/human_rgb_detected_v1_cache --device cpu
python scripts/cache_dcsass_actors.py --stage extract --output runs/dcsass/human_rgb_detected_v1_cache --device cpu
python scripts/run_sultani_dcsass.py --stage prepare
python scripts/prepare_ucf_surveillance.py --root "$UCF_ROOT"
# Inspect duplicate findings in audit.json before the explicit freeze:
python scripts/freeze_ucf_protocol.py
python scripts/extract_ucf_sultani.py --preflight
python scripts/extract_ucf_sultani.py
python scripts/train_sultani.py --config configs/experiments/ucf_sultani_shared_safe_v1.yaml --manifest data/manifests/ucf_sultani_shared_safe_features_v1.jsonl --output runs/ucf-crime/sultani_shared_safe_v1/seed_0
python scripts/freeze_sultani.py --run runs/ucf-crime/sultani_shared_safe_v1/seed_0 --manifest data/manifests/ucf_sultani_shared_safe_features_v1.jsonl --population ucf_sultani_shared_safe_v1
```

The duplicate audit deliberately exits before freezing when diagnosis is needed;
this is a scientific guard, not permission to bypass it. Rebuilt manifests contain
local path/inventory identities and differ from the original Windows byte hashes.
Do not transplant them into old registered evaluations. For Actor-Transformer
training, use the tracked human config with `train_dcsass_behavior.py`. Both
initializations require the pinned Collective checkpoint to establish the same
architecture/provenance; random initialization does not reuse its learned weights.
The controlled random run uses `--initialization random` and selection remains validation-only.
See the [execution journal](end-to-end-execution-log.md) for original commands.

`.gitignore` protects these generated assets. Keep small intentional source
configurations, templates and protocol metadata in Git. Keep local absolute-path
overrides, credentials, archives and OS/editor files out of Git.
