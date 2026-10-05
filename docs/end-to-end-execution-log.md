# End-to-end execution journal

The user's phase A–O specification is the design authority. Work proceeds in this
repository on `feat/end-to-end-surveillance-mvp`. Local milestone commits are
permitted; pushes, merges, history rewriting, test tuning, and replacement of the
two approved paper baselines are prohibited.

## 2026-10-06 — Initial repository and mandatory-asset gate

### Repository and preserved evidence

- Started on clean `feat/dcsass-surveillance-adaptation`, HEAD `82eede2`
  (`exp: analyze detected actor-set effects on Collective`). No user edits existed.
- Created `feat/end-to-end-surveillance-mvp` from that HEAD. The filesystem sandbox
  initially denied creation of the Git ref; the authorized local branch operation
  subsequently succeeded through sandbox escalation. No remote operations occurred.
- Reviewed package layout, dataset preparation/source recovery, manifest splitting,
  video decoding, anomaly inference, actor loss/trainer, experiment configs and
  previous Collective execution/diagnostic reports.
- Preserved Collective Protocol v1, modality results, automatic-detector settings,
  predefined late fusion and matched-only oracle diagnostic. No held-out model
  inference was rerun and no historical result was rewritten.
- Confirmed the existing actor trainer is Collective-specific: its joint loss
  computes actor cross entropy even with actor loss weight zero. DCSASS adaptation
  must introduce genuine group-only supervision, not fabricated actor targets or
  merely a zero multiplier on the current actor loss.
- Confirmed anomaly inference already exposes scores, timestamps, merged intervals
  and aggregate score. Its timestamp mapping explicitly uses approximate uniform
  duration segments; integration must validate the C3D-unit/frame mapping before
  claiming precise temporal localization.

### Environment and supplied assets

Verified Python 3.13.5, torch 2.13.0+cu126, torchvision 0.28.0+cu126,
CUDA 12.6 availability and NVIDIA GeForce RTX 4070 Laptop GPU. No packages changed.

SHA256 checks of existing local assets are recorded in
`runs/end-to-end-initial-audit/assets.json`:

| Asset | Repository-relative location | Expected SHA256 |
|---|---|---|
| Selected Collective RGB checkpoint | `runs/collective/rgb_gt_v1/rgb_gt/seed_0/best.pt` | `40a5fc3687c277914335916bd46dff72ccde5e4b6518c677f193178090de7fcd` |
| Validated I3D feature archive | `checkpoints/i3d_mixed4f_collective_rgb_v1.pt` | `fe7fc30ca6f26430232e2e0f6bdffd8514478a071db065e452bff46f60f4e0c4` |
| Original converted I3D weights | `checkpoints/external/pytorch-i3d/models/rgb_imagenet.pt` | `2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538` |
| Faster R-CNN COCO_V1 | `checkpoints/external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` | `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384` |

The saved RGB metrics independently confirm group accuracy 0.775483870967742 /
macro F1 0.7946864755459309 and actor accuracy 0.7821637426900585 /
macro F1 0.7922563592073513. These remain Collective measurements, not surveillance
behavior results.

### Phase A1 — blocked: DCSASS not installed in checked locations

`data/raw/` contains only Collective Activity. No DCSASS manifest, labels, archive,
or dataset directory was found in the repository or the filename searches of
`D:/Downloads`, `C:/Users/manan/Downloads`, `D:/Github Repos`, `D:/Videos` and
`D:/Documents`. One unrelated `airport-navigation/.pytest_cache` directory denied
enumeration; the search is evidence for the checked accessible locations, not a
claim to have searched every disk location.

The repository documents manual acquisition from Kaggle but provides no vetted
automatic acquisition script. Following A1/N1, dataset-dependent audit, split freeze,
detection, extraction and training stop at this gate. Actual dataset counts,
readability, categories, source recoverability and duplication remain **unmeasured**.
The required `runs/dcsass/audit.json` / `audit.md` record this blocked status using
null measurements, rather than inventing an empty dataset audit.

Acquisition and label requirements: [required assets](end-to-end-required-assets.md).
No dataset or pretrained model was downloaded.

### Phase F1 — blocked: real Sultani assets unavailable

No UCF-Crime tree, UCF split/temporal files, C3D feature bags, compatible pretrained
C3D FC6 checkpoint, or real Sultani run is installed in the repository. `runs/`
contains Collective experiments only. The Milestone 1 report establishes synthetic
software verification, not a real anomaly benchmark. A real Sultani checkpoint and
raw-video C3D inference remain mandatory for the final deliverable.

### Verification commands and initial fixture failures

- `git status --short`, `git branch --show-current`, `git log -8 --oneline`.
- `.\.venv\Scripts\python.exe -m ruff check .`: passed.
- `.\.venv\Scripts\python.exe -m ruff format --check .`: 132 files formatted.
- `git diff --check`: passed.
- Exact initial `.\.venv\Scripts\python.exe -m pytest -q`: 143 passed / 277 setup
  errors caused by access denial to the existing Windows pytest temporary root.
  A single-test reproduction traced this to pytest's numbered-directory scan,
  not project source. A fresh workspace temporary root passed the targeted test.
- The first full workaround omitted its new parent directory, producing 143 passed /
  277 setup errors (`FileNotFoundError`); corrected by explicitly creating
  `runs/end-to-end-initial-audit/` before invoking pytest. No source workaround or
  environment downgrade was necessary.
- Final baseline command (output in `runs/end-to-end-initial-audit/pytest-final.log`):

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q --basetemp runs/end-to-end-initial-audit/pytest-temp-final -o cache_dir=runs/end-to-end-initial-audit/pytest-cache
  ```

  Result: **420 passed, 325 warnings, 115.72 seconds, exit 0**. Warnings are
  retained in the full log. No source changes were needed to restore baseline
  verification. The workaround only changes test temporary/cache locations.

### Decisions and next action

1. Honor the user-provided architecture and autonomous execution method; skill
   review/approval prompts do not override the explicit instruction to continue
   ordinary engineering work without repeated approvals.
2. Preserve the existing repository and create the requested branch in place,
   because the clean working tree does not need a separate worktree.
3. Stop real-data execution at missing mandatory assets. Do not freeze a split
   before inspecting actual data or publish unmeasured MVP claims.
4. Before DCSASS training, recover common UCF source identities and determine the
   Sultani held-out population. Preserve official UCF test-source reservations if
   using its benchmark; otherwise freeze a documented shared research source split.
   Independently splitting the two related datasets would jeopardize unified
   evaluation even if each individual manifest passed its own leakage check.
5. Resume at A2 after local DCSASS videos and original annotations are available.
   Reuse existing binary preparation and source parser; add exhaustive taxonomy,
   excluded-category, decoding, duplicate and grouped-split tests before freezing
   Human-Centric v1. Then execute B–E, F, G–M in the supplied gated order. Optional
   seeds/comparisons follow evidence-based phase E only.
