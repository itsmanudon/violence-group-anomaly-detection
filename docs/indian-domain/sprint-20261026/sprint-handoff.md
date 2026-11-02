# Consolidated sprint handoff — 26 October 2026

Executed assignments 1–4 sequentially in this checkout/session. Deadline remains
10 November 2026. Branch **feat/municipal-evaluation-platform**, HEAD
**4d7c27ad0d7412752307a37eff273db58f93386b**. All work remains uncommitted.
No commit, push, merge, deploy, publication, footage upload or Indian-domain
supervised training/evaluation occurred. No protected split was created.

## Preservation and actual changes

Startup snapshot and preservation receipt:
`runs/indian-domain/sprint-20261026-v1/{startup,preservation}.json`.
Of 32 pre-sprint changed files, **31 are byte-identical**. The one intentional
extension is `src/surveillance/datasets/indian_readiness.py`: parse duration with
the same MM:SS/seconds parser as endpoints; quarantine invalid duration and keep
raw values. Existing CSV integration, training/resume/calibration and six-class
implementations were reused, not replaced. All five original annotation CSVs and
their five workspace copies retain their registered hashes. All five pretrained
checkpoints retain their startup hashes. Historical metrics and both earlier
readiness outputs remain unchanged. No old cached feature or experiment output
was written by these new tools.

The existing tracked modifications `.gitignore` and `README.md` remain. Existing
untracked Indian-domain modules/docs/tests/configs and sprint instructions remain.
Exact final status: `runs/indian-domain/sprint-20261026-v1/working-tree.txt`.
Final code/config snapshots and hashes are in that run's `code-snapshot/` and
`completion.json`; these also capture the untracked new code beyond HEAD.

New code files:

- `src/surveillance/evaluation/cascade_failures.py`, `scripts/analyze_cascade_failures.py`:
  hash-verified read-only historical attribution.
- `src/surveillance/inference/cascade_review.py`: optional legacy/coverage/
  coverage_consistency assessment wrapper, no raw anomaly alert suppression.
- `src/surveillance/evaluation/cascade_review.py`, `scripts/compare_cascade_review.py`:
  matched six-class validation comparison, exact membership/source/model checks.
- `src/surveillance/datasets/indian_review.py`, `scripts/screen_indian_review.py`:
  deduplicated review/provenance/fingerprint/timing observations, no approvals.
- `src/surveillance/evaluation/indian_frozen.py`, `scripts/evaluate_indian_frozen.py`:
  future source-approved validation-only baseline and exact cached review replay.
- `scripts/profile_surveillance_sprint.py`: bounded real RTX inference diagnostics.
- Tests `test_cascade_failures.py`, `test_cascade_review.py`, `test_indian_review.py`,
  `test_indian_frozen.py`; prospective `configs/indian-domain/sprint-20261026.json`.

Documentation created in this folder: failure-analysis, architecture-research,
model-improvement-decision, model-implementation, model-comparison, gpu-benchmarks,
dataset-quality-report, source-overlap-report, source-readiness,
experiment-readiness, phase-status and this handoff.

## Phase outcomes and research findings

**1 — failure analysis/research complete.** Verified all 991 registered DCSASS
prediction hashes and exactly reproduced saved metrics. **224/557** normal clips
alerted; **230/434** abnormal clips bypassed; **117/428** routed clips had no
actors; **169/311** covered clips had wrong six-class predictions. Additional
descriptive counts: 56 mixed-coverage clips and 83 cross-window class-disagreement
clips among 311 covered (not mutually exclusive). Wrong chosen behavior predictions
had median maximum softmax **0.949687**; confidence is not reliability. Actor
ground truth and interval truth are unavailable. These describe historical
held-out errors only; no settings were tuned against them.

DCSASS clip F1 **0.473318**, ROC-AUC **0.511794**, FPR **0.402154**, confusion
`[[333,224],[230,204]]` remain distinct from **UCF frame ROC-AUC 0.7441**. Neither
is an Indian-domain metric. Five primary references were verified to the scope
recorded in [architecture research](architecture-research.md), including Sultani
2018, Actor-Transformers 2020, SelectiveNet 2019, RTFM 2021 and VideoMAE V2 2023.
Two direct CVF HTML requests returned 403; accessible primary search metadata/PDF
abstracts were used and full-text review/licensing is not invented. No download.

**2 — selected bounded implementation complete; recognition comparison blocked.**
Opt-in coverage+agreement assessment abstains from a single behavior suggestion
when any routed window lacks actors or covered classes disagree. Legacy remains
default and returns the original result unchanged. Fallback/ablation checks only
coverage. Every raw alert, six-class output and original model identity remains
available. This improves the explicit review-evidence contract; **it does not
reduce binary alert counts or establish accuracy improvement**. A0/A1/A2 matched
validation recognition utility is NOT MEASURED: complete joint cascade validation
predictions were unavailable. A behavior validation summary alone is insufficient.

**3 — automated dataset screening complete; human evidence outstanding.**
Rechecked current video bytes, original annotations/catalog and registered inputs.
Prior hash-bound decoding/fingerprints were reused; no new full decode claimed.
All **521 videos plus two audio-only exclusions** exist and match catalog hashes.
Missing catalogued video annotations: **zero**; the retired 536 estimate does not
prove 15 missing videos.

| Population | Actual count |
| --- | ---: |
| Original violence / nonviolence / not_sure / blank | 48 / 465 / 7 / 1 |
| Strict-loader qualified on unchanged media | 520 |
| Quality candidates | **504 = 39 violence + 465 nonviolence** |
| Quality quarantine | **17** |
| Source-approved training / evaluation | **0 / 0** |
| Unassigned clips | **521** |

All 48 positives occur in Snatch/HCI/Refined (30/11/7); negative counts across
Snatch/HCI/Refined/Archive/IPID/WalkIndia are 7/0/2/104/17/335. All 352 clips with
catalogued Indian geography are negative; all 48 positives have unknown geography.
Class imbalance remedies cannot fix this source/geography confounding. Recording
and incident identities remain unknown for all 521. Fourteen HCI municipal
reservations are retained.

Exact duplicate groups across 521 SHA hashes: **zero**. One Archive visual pair
remains ambiguous: only one ordered sampled-frame match, not proven shared
footage/incident. Ten filename families are suggestions, not verified sources.
No full transformed-footage/historical-source clearance. See [source report](source-readiness.md).

**4 — integration complete; prospective experiment gates retained.**
Added a thin baseline CLI because the existing historical cascade evaluator is
bound to the original test protocol. It refuses absent/mutated approved splits,
infers validation only, records fixed anomaly-vs-violence proxy semantics and
per-source binary metrics, and never fabricates six-class truth. A cached replay
compares assessment modes on exactly the same saved results. E0/E1/E3 command
templates are implemented and verified by help/tests. **E2 hard-negative training
and E4 actor fine-tuning are conditional and not launchable**: missing prediction/
failure evidence plus specific sampler/actor orchestration are listed in
[experiment readiness](experiment-readiness.md). No fabricated CLI flags or outcomes.

## Actual RTX 4070 engineering measurements

Real five-model inference on existing pinned four-second demonstration clip
`Robbery020_x264_10`: 120 frames, 30 FPS, 320×240; C3D batch four, original
preprocessing/window policy. One warmup, five repeated uncached runs:

| Measurement | Actual result |
| --- | ---: |
| Checkpoint/model startup, one observation | **2.470153 s** |
| First complete inference, one observation | **3.020937 s** |
| Warm complete inference median / p95 | **1.298047 / 1.321201 s** |
| Peak PyTorch allocated / reserved | **972.81 / 1,170 MiB** |
| Total VRAM / free before load | **8,585,216,000 / 7,411,335,168 bytes** |
| Cached Sultani-only median | **0.621 ms** |
| Cached last covered actor-head-only median | **3.942 ms** |
| New review assessment CPU median | **0.207 ms** |

Python 3.13.5, torch 2.13.0+cu126, CUDA 12.6, RTX 4070 Laptop. Five warm outputs
exactly matched on scores, boxes, probabilities and alert. Detailed actual C3D,
detector, I3D, transformer, decode and preprocessing median/p95 values and hashes
are in [GPU benchmarks](gpu-benchmarks.md) and the immutable profile JSON.
Cached head measurements exclude backbones/transfers and are not a cached whole
pipeline. No Indian throughput/accuracy, universal real-time FPS, temporal event
latency, camera-hour burden or new-backbone backward memory was measured. Earlier
10.93 ms synthetic binary training is a different workload, not video latency.

## Reproducible verification

All runs used fresh output directories. No tests were removed/weakened.

| Run under `runs/indian-domain/sprint-20261026-v1/` | Result |
| --- | --- |
| phase1-tests | **9 passed in 8.14 s** |
| phase2-tests | **41 passed in 13.59 s** |
| phase3-tests | **63 passed in 14.15 s** |
| phase4-adapter-tests | **27 passed in 19.99 s** |
| integration-tests + integration-tests.xml | **135 passed in 45.02 s** |
| timing-tests + timing-tests.xml, after supplement | **28 passed in 0.90 s** |

Overlapping runs are not added into a fictitious unique-test total. No test failures
or skips in these runs. Initial style issues were fixed; final **15 touched Python
files pass lint and format**, `git diff --check` passes, all **ten CLI help checks
pass** (`cli-help.json`). No full Gradio suite, new research training or protected
test pass. CPU/CUDA binary fixture smoke/resume/calibration/cache compatibility and
six-class regressions are included in the integrated run; these are software tests.

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_cascade_failures.py tests/test_cascade_review.py tests/test_indian_frozen.py tests/test_indian_review.py tests/test_indian_readiness.py tests/test_indian_domain.py tests/test_indian_split.py tests/test_indian_binary.py tests/test_cascade_evaluation.py tests/test_surveillance_pipeline.py tests/test_rgb_behavior_loading.py tests/test_demo_loading.py tests/test_anomaly_frame_mapping.py tests/test_deployment_provenance.py -p no:cacheprovider --basetemp runs/indian-domain/sprint-integration-new-version --junitxml=runs/indian-domain/sprint-integration-new-version.xml
```

The final code adds one timing test after the 135-test run; reproducing that full
command now includes it. The actual final affected timing/readiness run is the
separate 28-test result above, not a claim of a new full 136-test execution.

## Teammate queue and exact next action

Private artifact root: `runs/indian-domain/sprint-20261026-v1/data-review/`.
`review-queue.csv` has 521 unique rows; **first-priority.csv has 58**:
39 candidate positives + 17 holds + two ambiguous visual-pair records. Human
disposition/reviewer/date/source fields remain blank. Work those rows first and
provide recording/incident/rights/overlap evidence in the existing source queue.

Timing supplement: **341** presentation-vs-frame/FPS differences exceed one
frame+1 ms (WalkIndia 335, IPID three, Snatch two, HCI one); FPS disagreement count
is zero. `timing-review-queue.csv` and `timing-registration.json` retain exact
observations. This is a separate overlapping **temporal-use diagnostic**, not
341 added quality quarantines or corrupt clips. Resolve clocks before localization/
latency; do not silently shift annotations. Binary quality candidates remain 504.

Safe command runnable now, without new inference/training:

```powershell
Import-Csv -LiteralPath 'runs/indian-domain/sprint-20261026-v1/data-review/first-priority.csv' | Select-Object clip_id,collection,original_label,quality_status,flags | Format-Table -Wrap
```

After real source-safe approval only, execute E0 then E1 smoke/cache/training and
validation calibration, and paired E3 replay using the exact commands in
[experiment readiness](experiment-readiness.md). Those commands target prospective
paths; they have not been run on Indian media. Do not approve rows solely to
unlock the CLI. The default existing demonstration remains the original pipeline.

## October 28–November 10 priorities and blockers

- **Ready now:** software contracts, real RTX diagnostic, private review queues,
  source-safe verification, future E0/E1/E3 commands. Review 58 priority records
  and source evidence in batches. Do not repeat unchanged GPU benchmarks.
- **Needs human evidence, Oct 28–30:** adjudicate holds and positive content;
  document true original recording/incident/rights/geography and historical overlap;
  clear the visual pair. Recount independent positive/negative sources before a
  reproducible grouped proposal. No approved source or frozen test today.
- **Conditional, Oct 30–Nov 4:** if source independence/class support passes, freeze
  once, run frozen validation baseline, real cache smoke and E1, then calibration
  and E3. Preserve all failures and source-wise recall/coverage. Otherwise report
  a data-limited methodology study without invented Indian performance.
- **Optional, Nov 4–6:** E2 or E4 only if validated failures justify their specific
  missing orchestration and memory smoke. Cut these first if data review is late.
- **Conditional final test, Nov 7–8:** freeze selected model/preprocessing/policy/
  thresholds and hashes; verify untouched test independence; register one test pass.
- **Nov 9–10:** finalize research report, limitations and professor human-review
  demonstration. Municipal deployment readiness is not established.

Remaining blockers are concrete: no verified original source/incident/rights and
historical-overlap clearance; 17 unresolved quality holds; no representative
independent positive Indian sources established; no complete eligible joint cascade
validation predictions for six-class A0/A1/A2 utility; timing mappings unverified
for localization; no continuous representative event-labeled camera-hours. E2/E4
also lack the explicitly listed specialized orchestration and empirical rationale.
No automatic enforcement, intent/guilt inference or deployment guarantee is added.
