# Preparation verification — 23 October 2026

Baseline commit: `4d7c27ad0d7412752307a37eff273db58f93386b`. Changes are
local and uncommitted. No push, merge, deployment or remote resource mutation.
Existing research receipts and checkpoint files were not modified.

## Environment and assets actually checked

- Existing `.venv/Scripts/python.exe`: Python 3.13.5; torch 2.13.0+cu126;
  torchvision 0.28.0+cu126; OpenCV 5.0.0; scikit-learn 1.9.1.
- CUDA build 12.6, `torch.cuda.is_available()` True;
  **NVIDIA GeForce RTX 4070 Laptop GPU**.
- SHA256 computed for all five deployment assets; exact values and paths are
  recorded in [the audit](repository-audit.md). All matched configured pins.
- `.venv/Scripts/python.exe scripts/verify_demo_assets.py --config configs/surveillance_demo_ucf_v1.yaml`
  **passed**, output `verified: true`, device `cuda`. This loaded the existing
  deployed C3D, Sultani, detector, I3D and six-class behavior assets and checked
  receipts; it did not score any new or protected footage.

## Stage P1/P2 checks

First focused run:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_indian_domain.py tests/test_indian_split.py tests/test_manifests.py -p no:cacheprovider --basetemp runs/indian-preparation-tests-20261023-v1
```

**39 passed in 17.26s**. Fixture videos are generated 32×24 MJPG clips with random
pixels. They are software fixtures, not Indian footage, and their counts/splits
must not appear in research tables. Covered label/schema/quarantine, media paths,
readability, hashes, temporal semantics, provenance, duplicate groups, deterministic
splitting, immutable membership and changed media rejection.

Expanded regression run in `runs/indian-preparation-regression-20261023-v1`:
**99 passed in 46.01s**, 14 existing TorchScript deprecation warnings. Included
existing manifest, Sultani model, actor model/transfer, DCSASS behavior training,
six-class RGB loading, evaluation, feature provenance, cascade/registration and
training-observability tests. No tests were deleted or weakened.

Initial lint identified formatting, unused imports and import-order issues in new
files. These were corrected; subsequent checks passed. No failed software test
was hidden or converted to a skip to make the preparation pass.

## Binary/evaluation foundations

First binary run:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_indian_binary.py -p no:cacheprovider --basetemp runs/indian-binary-smoke-20261023-v1
```

**7 passed in 15.71s**, including both CPU and CUDA parameterizations (neither
skipped). This performed one-batch forward/backward, three-step synthetic-feature
training, save/reload and interrupted/resumed training. Final weights and complete
metric/loss histories matched uninterrupted training in the same environment.
This is binary-head software validation, **not** real frozen-C3D extraction or
Indian-domain fine-tuning validation. The fixture registration explicitly records
`synthetic_software_fixture_only`.

Actor tests verify binary `[B,2]` shape, frozen representation gradients, selective
projection gradients and unchanged source six-class `[B,6]` behavior. Existing
six-class loader rejects the isolated binary checkpoint. Evaluation tests check
confusion/FPR/F1/ROC calculations, recall-constrained validation calibration,
causal temporal persistence/hysteresis/smoothing, gaps, and invalid inputs.
Later tests add missing-person/confidence abstention denominators, malformed CSV
row widths, explicit provenance conflicts and train/validation-only feature
caching. Extraction caching is tested using a substituted fixture extractor;
that does not validate real C3D model output.

Combined targeted run in `runs/indian-final-targeted-20261023-v1`:
**110 passed in 45.83s**, 14 existing TorchScript deprecation warnings.
Subsequent final changes added the three-mixed-incident allocation boundary test,
recorded requested training horizons, clarified cache evidence type and linked
the new workflow from README. A complete suite run is recorded below.

## Final verification

Full suite command (new temporary directory, preserving earlier receipts):

```powershell
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp runs/indian-full-regression-20261023-v1
```

**Incomplete: interrupted with Ctrl-C (exit 1) after several minutes without
progress.** Output reached 34% plus further dots, with no assertion failure
reported. The latest fixture directory was
`test_actual_callback_suppresses_delayed_result_after_new_source` from existing
`tests/test_demo_revisions.py`; it constructs the Gradio app. This identifies the
observed stall location, not a proven root cause. No existing UI test was changed
or weakened. Do not interpret this run as a full-suite pass.

Afterward, tightened placeholder-source rejection and required complete production
C3D cache provenance while preserving explicit synthetic-fixture labeling. Final
relevant regression rerun used `runs/indian-final-targeted-20261023-v2`:
**112 passed in 50.42s**, with 14 existing TorchScript deprecation warnings and
no skips. This final run includes both CPU/CUDA smoke/resume checks, production
cache-provenance rejection and the three-mixed-incident boundary case, plus all
previously listed relevant regressions. All eleven new Python files passed targeted
`ruff check` and `ruff format --check`; `git diff --check` passed. All three new
CLI `--help` commands ran successfully. Do not reuse an existing basetemp
directory when preserving test receipts; use a new version suffix.

## Unrun or blocked

| Check/experiment | Status and reason |
| --- | --- |
| Real CSV header/labels/annotation review | Blocked: CSV not available to this session |
| Real Indian media decoding, counts, duplicate screening, source/incident grouping | Blocked: media and provenance not supplied |
| Historical inventory completeness and transformed-footage overlap review | Blocked: requires actual incoming footage plus curator scope/evidence; no complete inventory attestation made |
| Actual 70/15/15 split counts and test freeze | Blocked by preceding data/provenance review; no real split selected |
| Real C3D feature cache / binary training | Unrun: awaiting frozen independent dataset; only synthetic-head smoke executed |
| Indian actor cache and full binary Actor-Transformer training orchestration | Outstanding implementation, guided by real coverage/temporal failure analysis |
| A/B/C controlled comparison, hard negatives and threshold/temporal ablations | Unrun; primitives are tested, no research outcomes claimed |
| H100 credentials, driver/environment, receipt relocation and GPU smoke | Unverified/unrun; no institute access assumed |
| Final test, inference latency and temporal detection latency | Unrun; no final model/config/test population frozen |
| False alarms per camera-hour | Unavailable: no representative continuous footage or complete event annotations |
| Architecture experiments | Deferred until measured baseline/failure analysis; no novelty claims |
| Professor/municipal readiness outcome | Runbook prepared; Indian validation and operational readiness not established |

The original UCF and DCSASS published measurements remain documentary baselines
with different populations. No Indian-domain metrics or deployment guarantees
are claimed by these tests.
