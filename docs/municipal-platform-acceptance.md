# Phase 1 acceptance evidence

Validation date: 2026-10-17, Mac arm64, Python 3.12.7. Evidence describes local
software verification, not benchmark performance or Windows hardware certification.
The starting worktree was clean `main` at `01d1ef1`. Work is on the requested
`feat/municipal-evaluation-platform` branch; no remote operations.

## Automated gates

Baseline before changes: **561 passed, 2 skipped**, 61.15 seconds. Existing warnings
are TorchScript deprecation warnings. The upload-service unit test now supplies a
real tiny MJPEG fixture rather than invalid bytes, because ingestion decoding is
intentionally stricter; cache-isolation/failure assertions are preserved.

The focused catalog/queue/API checkpoint passed **28 tests** (including 22 new
municipal cases and six existing API cases). Final full-suite and gate results
are recorded in [completion](municipal-platform-completion.md).

| Acceptance | Evidence |
| --- | --- |
| Existing suite | Full pytest baseline and final full regression run |
| Safe import/dedup | Whole-manifest rollback, duplicate identity/membership checks, SHA256/media checks |
| 100 clips | 100 real synthetic MJPEG files, dry run leaves zero rows, commit/reimport/reopen/backup counts |
| Batch success/failure/cancel | Mixed completed/failed/cancelled items; missing media and retry subset |
| Restart/resume | Claim interrupted item, reopen catalog, recover/run; profile drift fails; completed stays complete |
| GPU serialization | Three worker threads, controlled active clip + cancellation; max concurrent inference one |
| No actors/disagreement | Raw no-actor abstention and disagreement persisted; existing pipeline regression tests |
| Cache/fresh | Queue rejects cached results; original service cache identity tests retained; browser actual inference |
| Corrupt/missing input | Full-decode failures, missing file before worker, changed hash, empty bytes |
| Upload/path boundary | Absolute/traversal/backslash/symlink denial, ignored filename, extension, size limit, cleanup, origin/host denial |
| Review/reference | Premature reveal rejected, version conflict, immutable reference/prediction/membership triggers, interval bounds, append-only review |
| Metric integrity | Synthetic/demo/upload exclusion, latest-run dedup, coverage/abstention, conditional class metrics, temporal false-alert overlap |
| Exports | CSV/JSON, formula escaping, reference opt-in, rights filtering, local path omission; JSON includes aggregate metrics |
| Frontend | Vite static production build; Node tests; manual browser dashboard/library/queue/analysis/review/results walkthrough |
| Launch/preflight | Actual loopback Uvicorn process, health + static page; all five asset hashes/receipts and all model loads pass |
| Frozen real smoke | Actual unchanged curated cascade, Python socket connect denied, 32 segments and 3 actor windows |

Schema migration v1→v2 is exercised with existing data preserved. SQLite online
backup row counts are verified. Managed upload deletion refuses queued/running
clips; import originals are never deleted. Missing/corrupt asset preflight checks
return explicit failure without checkpoint substitution.

## Real/offline smoke

Commands executed:

```bash
.venv/bin/python -m surveillance.municipal preflight
.venv/bin/python scripts/municipal_fixtures.py --output outputs/municipal-fixtures/phase1 --count 100
.venv/bin/python -m surveillance.municipal --data-dir outputs/municipal-fixtures/catalog --import-root outputs/municipal-fixtures/phase1 import outputs/municipal-fixtures/phase1/manifest.json
# Same command with --commit, then local serve --port 7861
.venv/bin/python scripts/municipal_smoke.py --example 'Robbery: correct cascade example (curated)' --output outputs/municipal/frozen-offline-smoke.json
```

Preflight: **PASS**, model_readiness_verified=true, all components CPU. Ignored
local receipts: `outputs/municipal/preflight.json`, real smoke raw/summary JSON and
separate fixture catalog. Real smoke: on_demand, profile
`fbf78341e078a3293a2ff3958e76f211f1b2a18cc1c6a8ad007ca2be71f4fc65`,
32 segments, 3 routed actor windows, 4.325s inference, 0.742s loading, 5.073s total.
Those single-clip times are diagnostics, not expected pilot throughput.

The browser selected synthetic-00000, submitted a genuine local queue job,
opened the 32-segment timeline and unexamined-behavior abstention, saved an
uncertain review and displayed it on Results. Reference reveal was disabled
before the review. Synthetic population produced no benchmark metrics. No
Indian footage was added. The console uses local static assets/system fonts;
Node is unnecessary at presentation time.

## Limits of this evidence

Python socket connects were denied during real model load/inference. The host's
physical network was not disabled and a full OS reboot was not performed. The
Windows launcher/OS-lock code is supplied but Windows/NVIDIA hardware execution
cannot be claimed from this Mac run. Do the target-hardware, codec/playback and
physical offline rehearsal in the runbook before the pilot. MJPEG fixture playback
is not a browser H.264 compatibility guarantee. Browser print/save-PDF is the
human-readable report path; no server PDF dependency was added. No TypeScript
configuration/typecheck existed; Vite parses the JSX, and `npm test` is now present.

Final restart verification: stopped the local server cleanly, relaunched against
the same fixture catalog, reloaded the browser and verified all 100 clips, the
completed actual inference and its independent review remained present. Schema
was v2. Final focused rerun remained 28 passed; final preflight remained PASS.
