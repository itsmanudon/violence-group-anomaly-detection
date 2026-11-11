# Run the demo model on your own laptop

This guide is for teammates who want to score video clips with the frozen v3 demo model. You
need this repository, a ~520 MB **model bundle** (shared privately, never through GitHub), and
about 15 minutes. No training data is needed.

How well the model performs: [docs/results/MODEL_PERFORMANCE.md](results/MODEL_PERFORMANCE.md).

## 1. Requirements

- Python **3.11 or 3.12** (3.12 is what the demo machine uses).
- About 3 GB free disk (Python packages ~2 GB plus the 520 MB bundle).
- Any recent macOS, Windows or Linux laptop. A GPU is **not** needed; everything runs on CPU.
- Node.js 18+ **only** if you want the web UI (step 5). The batch scorer (step 4) does not need it.

## 2. Get the code and install

macOS / Linux:

```bash
git clone https://github.com/itsmanudon/violence-group-anomaly-detection.git
cd violence-group-anomaly-detection
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[demo]'
```

Windows (PowerShell):

```powershell
git clone https://github.com/itsmanudon/violence-group-anomaly-detection.git
cd violence-group-anomaly-detection
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[demo]"
```

## 3. Add the model bundle

Download `violence-demo-bundle.zip` from the team's shared folder. Unzip it **at the repository
root** (the folder containing `pyproject.toml`), so its `checkpoints/`, `runs/`, `data/` and
`outputs/` folders merge into the repo.

```bash
unzip violence-demo-bundle.zip          # Windows: Expand-Archive violence-demo-bundle.zip -DestinationPath .
shasum -a 256 -c MANIFEST.sha256        # optional; every line should end in OK
python scripts/demo_preflight.py
```

The preflight should print `PASS` for every model and example. It will show
`FAIL: frontend_bundle` until you build the web UI; that is fine if you only use the batch
scorer. The models also re-check their own SHA256 hashes every time they load, so a corrupted
or wrong file is refused rather than giving silently wrong scores.

## 4. Score a folder of clips (recommended for the 60 clips)

Put the clips in one folder (sub-folders are fine), then:

```bash
python scripts/score_pilot_clips.py --clips ~/demo_clips --output results/scores.csv
```

Windows: `python scripts\score_pilot_clips.py --clips C:\Users\you\demo_clips --output results\scores.csv`

You get one CSV row per clip:

| Column | Meaning |
|---|---|
| `pilot_violence_score` | the v3 model's violence score, 0–1 |
| `pilot_flagged` | `1` if the score is ≥ 0.40 (the frozen threshold), else `0` |
| `cascade_alert_state`, `cascade_behavior`, `sultani_max_score` | the older five-stage cascade's output, shown in the demo next to the pilot score |
| `video_sha256` | fingerprint of the exact file scored |
| `error` | filled only if that clip could not be read |

Add `--json-dir results/json` to also save the full result for each clip. The script never
overwrites an existing CSV. A clip takes a few seconds to about half a minute on a laptop CPU,
depending on its length.

Scores are identical to the demo's **Live model inference** upload: both use the same code path.

## 5. Optional: the web UI (one clip at a time)

```bash
npm ci --prefix frontend
npm run build --prefix frontend
python -m surveillance.demo --profile professor --device cpu \
  --indian-pilot runs/indian_pilot/demo/indian_pilot_spec_v3.json
```

Open <http://127.0.0.1:7860>, choose **Live model inference** and upload a clip. On Windows,
put the last command on one line without the `\`.

## 6. Optional: the walkthrough notebook

`notebooks/professor_walkthrough.ipynb` runs one clip through every backend stage, shows what each
model outputs, and recomputes the test metrics from the saved scores.

```bash
python -m pip install jupyterlab
jupyter lab notebooks/professor_walkthrough.ipynb
```

Start it from the repository root and run all cells; the first run takes about 10 seconds to load the models.

## 7. Clip guidelines for the demo set

- Short clips work best: 2–10 seconds, one event per clip. The test set used 2-second clips.
- Common formats are fine (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`).
- Never commit clips to git. `*.mp4` and similar are already ignored, and the repository is public.
- **If the 60 clips are going to be reported as a test result**, write each clip's label
  (violent / non-violent) and the scoring rules into a committed file **before** anyone scores
  them. Score them once and keep the 0.40 threshold. Scoring first and labelling or picking clips
  afterwards turns the numbers into a demo, not a test. The earlier tests followed this rule; see
  `runs/indian_pilot/V3_PROTOCOL.md` for an example.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Live inference requires local trained assets: ...` | The bundle is not unzipped at the repo root. Check that `checkpoints/c3d_fc6_openmmlab_v1.pt` exists. |
| `Indian pilot checkpoint differs from its pinned SHA256` | The download is corrupted or incomplete; download the bundle again. |
| `React UI is not built` | Run the two `npm` commands in step 5, or use the batch scorer instead. |
| `python3.12: command not found` | Install Python 3.12 from python.org (or `brew install python@3.12`). |
| Very slow scoring | Expected on CPU for long videos. Trim clips to a few seconds. |

## For the maintainer: making and sharing the bundle

On the machine that has the models (this Mac):

```bash
python scripts/pack_demo_bundle.py --output ~/Desktop/violence-demo-bundle.zip
```

It collects exactly the 28 files the demo needs (five cascade models with their receipts, the
v3 pilot head, nine demo examples and their caches), checks the pilot head against its pinned
hash, and writes a `MANIFEST.sha256` into the zip. It does **not** include training data or
feature caches.

Upload the zip to the team's Google Drive folder and share it with your teammates by email
("Restricted", not "Anyone with the link"). Do not attach it to a GitHub release: the
repository is public, and the weights were trained on research datasets (UCF-Crime, DCSASS,
UBI-Fights, XD-Violence) whose licences restrict redistribution.
