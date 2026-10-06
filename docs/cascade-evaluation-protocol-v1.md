# Prospective UCF-to-human-behavior cascade evaluation

This protocol is fixed before the new UCF Sultani model produces held-out scores.
It evaluates all 991 clips from the 31 frozen DCSASS Human-Centric v1 test sources
(557 Normal, 434 abnormal). It does not use the eight curated demonstration clips
as a benchmark. The human manifest is pinned by SHA256
`f3e7fc11159fce191f61b8ecf1d0d667cb68c9f949afa3ab84758c0532ee7258`.

## Models and routing

Use the validation-selected seed-0 UCF Sultani checkpoint from
`ucf_sultani_shared_safe_v1`, after its selection receipt is frozen. Its exact
checkpoint hash will be registered before the cascade pass. Use the already
selected DCSASS RGB Actor-Transformer checkpoint, SHA256
`7870e512787e74cd76263df28e40c8bc39ecb8a3d0a5b2b8b6a7e72a218a8c69`.
Keep the validated C3D, Faster R-CNN COCO_V1 and I3D assets and preprocessing.
The anomaly threshold is 0.5; analyze at most three distinct suspicious windows.
Do not adjust settings after inspecting cascade test outputs.

Before inference, verify every human test source is either an official UCF test
source or explicitly excluded from UCF optimization because it belongs to the
DCSASS held-out population. Verify the behavior checkpoint's training manifest,
both validation-selection receipts, raw clip hashes and all five model identities.

## Reporting

Report binary clip-alert accuracy, precision, recall, F1, ROC-AUC and confusion
matrix. These labels describe whole clips; they do not validate the timestamps
of individual suspicious windows. Preserve the Sultani alert when behavior is
Normal or unavailable.

Report six-class behavior accuracy, macro F1 and confusion matrix only on routed
clips with at least one covered window. Choose the final-alert covered window;
if that alert has no covered window, use the first covered window, matching the
demo display. Report the population denominator, source count, routed/bypassed
clips, covered clips, no-actor routed clips, analyzed windows and coverage both
overall and among routed clips. Bypassed clips have unexamined actors, not zero
actors. Do not fabricate actor labels or actor accuracy.

Report mean, median and p95 stage latency from genuine on-demand inference.
Keep these coarse cascade measurements separate from UCF frame-level anomaly
evaluation, DCSASS standalone behavior results and Collective results.

## Execution and recovery

After the UCF feature/training/evaluation gates and versioned deployment config:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_surveillance_cascade.py --config <versioned-UCF-demo-config>
```

The only result directory is `runs/integration/ucf_sultani_human_v1`. An immutable
registration binds the model pair, source protocol, manifest and routing policy.
A second completed pass or alternate output directory is rejected. `--resume`
only scores pending clips under the same registration, verifies saved result
hashes, and preserves partial outputs for diagnosis if an interrupted write is
not registered. `completion.json` is written after all clips and metrics exist.

The protocol above was fixed before scoring. The registered pass is now complete
on all 991 clips; see [results and limitations](cascade-results-ucf-v1.md).
The Windows progress-write interruption was recovered from verified serialized
outputs, preserving the same registration and scoring only pending clips.
