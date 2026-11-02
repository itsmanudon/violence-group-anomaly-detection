# Continuation verification — 25 October 2026

Baseline commit remains `4d7c27ad0d7412752307a37eff273db58f93386b`.
Initial dirty state matched the prior preparation work: README modified, Indian
modules/configs/docs/scripts/tests untracked. No reset, clean, commit, push,
merge, publication, original annotation edit or checkpoint write was performed.
The 23 October audit, plan, runbook and 112-test verification were inspected and
reused. No unrelated full-suite or model-asset audit was repeated.

## Changes and acceptance criteria

- `datasets/indian_domain.py`: explicit decimal/MM:SS[.fraction] parser, also
  used for mapped `_sec` columns. Accept only finite nonnegative values; retain
  raw strings. No clipping, label inference or annotation-status changes.
- New `datasets/indian_readiness.py`: dry-run reconciliation of the actual CSV,
  priority file, handover catalog, reviewer CSVs and hash-bound media/timing
  evidence. Validate ID/path consistency, inspect physical inventory scopes,
  quarantine priority records and preserve conservative guards/reservations.
  Emit no trainable manifest, split approval or fabricated independent source.
- `scripts/prepare_indian_domain.py`: `--dry-run` branch and explicit local
  catalog/evidence inputs. Refuse existing output directories.
- `configs/indian-domain/columns.combined-v1.json`: explicit mapping after
  inspecting the real header. This mapping does not repair missing provenance
  or make absent package paths available.
- New `tests/test_indian_readiness.py`: timestamp format/invalid-input cases,
  preservation of uncertain intervals, unapplied rounding proposals, filename
  hints versus source identities, direct importer round-trip, catalog/priority
  consistency, unchanged originals, decode reuse and mutation rejection.

## Tests actually executed

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_indian_readiness.py tests/test_indian_domain.py tests/test_indian_split.py -p no:cacheprovider --basetemp runs/indian-readiness-tests-20261025-v1
.venv/Scripts/python.exe -m pytest -q tests/test_indian_readiness.py tests/test_indian_domain.py tests/test_indian_split.py -p no:cacheprovider --basetemp runs/indian-readiness-tests-20261025-v2
```

First run: **56 passed in 27.45s**. After adding physical inventory and literal
package-path reporting: **56 passed in 14.71s**. Tests use generated fixture
videos; their counts and metrics are not research results. No failed tests or
skips. Initial lint found long lines/import formatting in new code; corrected.
Final targeted `ruff check`, `ruff format --check` (four affected Python files)
and `git diff --check` passed. Existing tests were not deleted or weakened.

## Actual dry-run execution

Command and results are recorded in [the data-readiness report](data-readiness-20261025.md).
Version directory: `runs/indian-domain/readiness-20261025-v1`.
Registration binds supplied CSVs, original reviewer exports, historical manifests,
local catalog/source groups and previous decode/timing evidence. Input SHA256 is
checked before/after the run. Every original relative annotation path and resolved
source-local path is retained separately.

Prior sequential/full-decode evidence is reused only after current video SHA256
matches. Current eight-position visual samples are decoded, and files without
qualifying evidence are fully decoded. Reuse is explicitly distinguished from
fresh complete decoding. Existing media statuses and temporal text stay unchanged.
No transcoding, normalization, copying, new human approval or original-source
identity assertion is inferred from successful local playback.

Observed outcome: **521/521 original local paths exist and match catalog SHA256;
520 qualify, one fails exact decoded/reported frame-count agreement**. Reused
294 matching decode receipts, completed 226 additional qualifying full decodes,
and retained the extra 599-versus-615 Snatch `1.mp4` case on hold. Visual screening
covered the 520 qualified clips and found one Archive candidate pair, not a
confirmed duplicate. Exact hash comparison over all 521 found no duplicate group.
No historical exact hit among 6,491 available hashes clears original-source or
transformed-footage overlap.

The 16 priority annotation holds plus this one separate media hold yield **17
unique quality holds**, leaving **504 candidates (39 positive, 465 negative)**.
All remain source-held: training/protected-evaluation eligible **0/0**. The
separate derived-disposition directory binds the original dry-run records/report
hashes and preserves all 521 raw annotation rows; no original receipt was replaced.
All output and input hashes were verified again at handoff.

## Unrun / blocked

- No source-safe split was approved or frozen. No protected membership changed.
- No frozen-cascade baseline, feature extraction, binary optimization, threshold
  selection or architecture experiment ran on these annotations.
- No new RTX/H100 smoke was needed for CPU-only reconciliation; reuse the prior
  dated RTX validation rather than claim new GPU testing. H100 remains unverified.
- No camera-hour rate or detection latency was estimated from this clip corpus.
- Human adjudication, original recording/incident reconstruction and historical
  transformed-footage overlap review remain pending; see exact blockers in the
  readiness report and [controlled matrix](training-matrix-20261025.md).
