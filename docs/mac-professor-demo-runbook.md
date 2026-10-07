# Mac professor presentation

The measured Windows/RTX research results remain unchanged. Deployment and
qualitative rehearsal are not training or new benchmark evaluation.

## Installation and fresh-shell rehearsal

From the repository root (Python 3.11+; this Mac uses the existing 3.12 environment):

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,demo]'
npm ci --prefix frontend
npm run build --prefix frontend
python scripts/demo_preflight.py
python -m surveillance.demo --profile professor --device cpu
```

Open `http://127.0.0.1:7860`. The React UI is served locally by the Python process, has no external fonts,
CDNs or analytics, and uses system fonts. Node is needed to build once; it is
not needed to launch the built console. Do not train, edit thresholds, or substitute missing models.
Professor mode binds the original UCF scientific identity. Local config accepts
paths only. CPU is the reliable Mac default; CUDA is unavailable on macOS.
MPS availability alone does not validate the frozen C3D/I3D/operator path.

Copy the exact assets in [the inventory](mac-demo-assets.md) before class.
For reproducibility, preserve `runs/demo_mac/environment.json`,
`runs/demo_mac/requirements.txt`, asset pins and validation receipts.
The editable dependency ranges are broader than this installed snapshot.

Close the old UI process with Ctrl-C, open a new Terminal, change to the repository,
activate `.venv`, run preflight, then launch with the same commands above. Select a
cached example, then upload a short local clip with the upload source selected.
Record the fresh-shell result; a documented command sequence alone is not proof
of reboot survival. An actual reboot is a manual presentation-day check.

## Presentation sequence

1. Run preflight; proceed with live inference only after all models/receipts pass.
2. Start the local UI and show a verified cached normal-label example. If the
   frozen model alerts, explain the false alert; do not promise a correct Normal
   prediction from the existing example names.
3. Show the curated correct Robbery example, disclosing its post-hoc selection.
4. Explain the fixed Sultani threshold (0.5) and 32-segment timeline.
5. Explain detector boxes, frozen I3D actor features and Actor-Transformer group reasoning.
6. Upload one short external qualitative clip through **Live model inference**.
7. Show the result, probabilities, disagreement/abstention and latency without edits.
8. Show **Known limitation example**, retaining its wrong prediction.
9. Explain the measured weak DCSASS/cascade results and need for human review.
10. Discuss independent surveillance validation, detector errors and domain shift.

The left mode selectors separate live upload, verified indexed examples and the named
limitation example. Cached output is explicitly labeled and never presented as
new inference. Uploaded bytes do not use example caches. The downloaded JSON
records content/profile identity, devices and timing. Historic cache timing is the
original model execution time, not Mac cache latency (`request_seconds` records that).

## External qualitative rehearsal

Store authorized short footage in ignored `demo_external/`; fill the example
manifest BEFORE first inference. Aim for 5–8 cases: normal activity, non-graphic
aggressive interaction, normal crowd, running, rough play, occlusion, shake/low
quality, and optionally a no-person anomaly. Document license/permission and
attest that none were used in training, validation, test or tuning.

```bash
cp demo_external/manifest.example.yaml demo_external/manifest.yaml
# Fill source, rights, expected behavior and filenames before inference.
python scripts/rehearse_demo.py
python scripts/demo_mac_smoke.py --clip demo_external/clip01.mp4 --cached-example 'Robbery: correct cascade example (curated)' --probe-mps
```

Rehearsal saves predeclarations, clip/model identity and raw results beneath
`runs/demo_rehearsal/<UTC timestamp>/`. It never changes a model or threshold.
Use the selected results to update [external clip notes](external-demo-clips.md).
Do not report selected-set accuracy as generalization performance. Previously
rehearsed footage is an **external qualitative example**; footage never inferred
before class is a **live unseen example**. Do not preview results and then claim unseen.

## Offline and fallback checks

After installing dependencies and copying all assets, disable Wi-Fi manually
(or use a permitted network-denied process), then launch and exercise cached,
live upload and limitation paths. Preflight and model loaders never download.
Rendering/playback still needs the actual local clips. Check browser video
playback as well as API output; keep the SSD mounted if paths point to it.

Primary: Mac live upload with a short clip; full CPU inference may be slow.
Fallback 1: exact byte/profile-verified caches in the same local UI.
Fallback 2: saved timeline PNG, actor overlays and browser screenshots/video captured
from genuine results. Prepare them before class; do not fabricate output or depend
on Internet. The smoke tool writes timeline PNG plus raw JSON. Record live and
cached request latency separately, including cold model load.

The previous Gradio interface remains an explicit fallback:

```bash
python -m surveillance.demo --ui gradio --profile professor --device cpu
```

## Quality gates

```bash
npm run build --prefix frontend
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
git diff --check
python scripts/demo_preflight.py
```

Do not push while asset, live, cached or offline Mac validation is incomplete.
