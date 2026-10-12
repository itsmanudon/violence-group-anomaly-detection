# macOS demo assets

Generated from the actual deployment config and example index. Full UCF-Crime, DCSASS and Collective datasets are **not required** to launch or infer uploads. No asset is downloaded implicitly.

## Required frozen models

| Component | Exact path | SHA256 |
|---|---|---|
| Sultani scorer | `runs/ucf-crime/sultani_shared_safe_v1/seed_0/best.pt` | `7e5b548ea6260614209a727a947e6270a1a1af2185d29accf1778f91b4af7c66` |
| C3D FC6 | `checkpoints/c3d_fc6_openmmlab_v1.pt` | `beac4ff065de3663bf5c6bab36aedca4fe56ec59b231c66bf5374c23eda14831` |
| Actor-Transformer behavior | `runs/dcsass/human_rgb_detected_v1/random_init_seed_0/best.pt` | `7870e512787e74cd76263df28e40c8bc39ecb8a3d0a5b2b8b6a7e72a218a8c69` |
| Faster R-CNN | `checkpoints/external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` | `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384` |
| I3D Mixed_4f | `checkpoints/i3d_mixed4f_collective_rgb_v1.pt` | `fe7fc30ca6f26430232e2e0f6bdffd8514478a071db065e452bff46f60f4e0c4` |

The Sultani and behavior checkpoints each require sibling `selection.json`: `selection_frozen: true`, not dry-run, matching checkpoint SHA256. Preserve C3D/I3D export sidecars and source receipts. Model archive loading is checked only after all byte pins pass.

Copy these exact bytes from the completed research machine or SSD. The learned scorers and frozen exports have no equivalent public download. The detector originated from TorchVision COCO_V1; copying the pinned file avoids network dependence. Do not replace any checkpoint or rebuild an export merely to satisfy its filename.

## Rehearsed presentation assets

Copy `data/examples/` and `outputs/demo/ucf_v1/examples/` from the same frozen bundle. The committed `configs/demo_examples_ucf_v1.json` names all nine videos, content hashes, caches and selection provenance. Cache acceptance requires exact video bytes and complete scientific model/preprocessing/policy identity; arbitrary uploads always run live. Known mistakes remain included.

See [machine-readable inventory](../configs/demo/mac_assets.example.yaml) for every filename and hash. Cache files have no independently published byte pins; their internal provenance is checked. They are qualitative examples, not new benchmark results.

## Machine-local relocation

Copy `configs/demo/local_assets.example.yaml` to `configs/demo/local_assets.yaml` and set Mac/SSD paths. `~` and environment variables are expanded. Only checkpoint, example-index and output paths can be overridden. Paths in the example JSON resolve from the repository root; copy the nine selected videos/caches into their default folders, or supply a relocated index with the same identities.

```bash
source .venv/bin/activate
python scripts/demo_preflight.py
python scripts/demo_preflight.py --json
python -m surveillance.demo --profile professor
```

No full dataset, HRNet, upstream I3D checkout, training feature cache or Collective transfer initialization is needed for this selected frozen demo.

## Exact Legion transfer tree

Copy this small subset of the completed Legion repository onto the SSD, preserving
relative directories. Place it under a named folder such as `professor_demo_bundle/`
and supply that folder path. Trained-model `selection.json` receipts are mandatory.

```text
checkpoints/
  c3d_fc6_openmmlab_v1.pt
  i3d_mixed4f_collective_rgb_v1.pt
  external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
runs/
  ucf-crime/sultani_shared_safe_v1/seed_0/
    best.pt
    selection.json
  dcsass/human_rgb_detected_v1/random_init_seed_0/
    best.pt
    selection.json
data/examples/                         # nine indexed demonstration videos
outputs/demo/ucf_v1/examples/           # their nine genuine JSON result caches
```

Include existing export JSON sidecars alongside the C3D/I3D archives where available.
The full UCF folder already mounted on the Mac contains footage/annotations and
has no model checkpoints. Copying more full datasets will not supply these learned
assets. Independent qualitative rehearsal clips can be supplied separately with
source/permission evidence and expectations recorded before their first inference.
