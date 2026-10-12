# Project overview

## What this project does

This is a research prototype for reviewing surveillance video. It combines weakly supervised anomaly localization with person detection and group behavior recognition. It reports suspicious intervals and observable behavior for human review; it does not infer intent, guilt, or a person's character. Predictions can be wrong.

## How the demo works

1. **C3D FC6** converts a video into temporal feature bags.
2. **Sultani MIL scorer** assigns anomaly scores across 32 temporal segments and proposes intervals for inspection.
3. **Faster R-CNN** detects people in selected windows.
4. **I3D Mixed_4f** extracts RGB features for actor regions.
5. **Actor-Transformer** estimates six group behavior classes: Normal, Abuse, Assault, Fighting, Robbery, and Vandalism.
6. The console shows the anomaly timeline, actor evidence, behavior probabilities, disagreement, and cases where the system abstains because no people were detected.

The React console is served by the local Python API at `http://127.0.0.1:7860`. It offers verified cached examples, uploaded-video live inference, and known limitation cases. Cached examples are clearly identified; uploads always run live inference. The older Gradio interface is available with `--ui gradio`.

## Research status and limitations

- The default demo combines a validation-selected UCF-Crime anomaly scorer with a DCSASS behavior model and pinned C3D, I3D, and detector weights.
- The UCF scorer measured 0.7441 held-out frame ROC-AUC at the frozen 0.5 threshold; it also has a substantial false-positive rate. This is a research result, not a deployment guarantee.
- The behavior and cascade results are weak, especially for minority classes. DCSASS has weak clip labels and no temporal ground truth; its scores do not establish temporal localization accuracy.
- Probabilities are uncalibrated. Domain shift, detector errors, source/context bias, limited seeds, and absent actor-level supervision remain material limitations.
- Full research datasets are not needed to run this demo. Training and evaluation workflows require separately acquired datasets, manifests, and feature/checkpoint assets.

## Repository map

- `src/surveillance/models/`, `features/`, `detection/`, `inference/`: model components and prediction pipelines.
- `src/surveillance/training/`, `datasets/`, `evaluation/`, `experiments/`: data preparation, training, evaluation, and experiment registration.
- `src/surveillance/demo*.py`: frozen demo service, API, rendering, and readiness checks.
- `frontend/`: React presentation console.
- `configs/`: model, dataset, experiment, and frozen demo policies.
- `scripts/`: command-line preparation, training, evaluation, inference, and demo utilities.
- `docs/`: protocols, measured results, runbooks, and machine-specific asset records.
- `data/`, `checkpoints/`, `runs/`, `outputs/`: local datasets, weights, run artifacts, and demo caches; most are intentionally ignored by Git.

## Run the local demo

From the repository root:

```bash
source .venv/bin/activate
python scripts/demo_preflight.py
python -m surveillance.demo --profile professor --device cpu
```

Then open `http://127.0.0.1:7860`. Rebuild the web UI after frontend source changes with `npm run build --prefix frontend`. The preflight checks pinned weights, selection receipts, model loading, cached examples, and the built UI bundle.

For detailed presentation guidance, see `docs/mac-professor-demo-runbook.md`; exact frozen asset pins are in `docs/mac-demo-assets.md` and `configs/demo/mac_assets.example.yaml`.
