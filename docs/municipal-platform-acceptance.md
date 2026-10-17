# Phase 1 acceptance evidence — 18 October 2026 (Asia/Kolkata)

Base: clean local main `415bbd0`. Branch: feat/municipal-evaluation-platform.
Windows Python 3.13.5; torch 2.13.0+cu126; torchvision 0.28.0+cu126;
Gradio 6.29.1. Existing suite: **544 passed** (231.91 seconds). Initial sandboxed
execution stalled at Windows asyncio socketpair creation; the unchanged suite
passed in the authorized local process. This was an execution permission issue,
not a failing test or an application host/origin bypass.

New acceptance tests live in test_municipal_platform.py and test_municipal_api.py.
The final combined run passed **576 tests** (544 existing + 32 municipal), with
325 existing TorchScript deprecation warnings, in 323.24 seconds. The local JUnit
receipt is `outputs/municipal-acceptance.xml`. A separate latest-code municipal run
covers catalog snapshot/rights/status refinements and frontend callback/schema
migration coverage: **32 passed**, followed by a focused frontend check of the
final timing table's current Gradio column API. Synthetic fixtures test
software only and are never reported as benchmark performance.

| Acceptance requirement | Evidence |
| --- | --- |
| Existing suite green | 544 existing tests passed without modifying existing tests or scientific implementations |
| Clip import/dedup | owned copy, hash pin, inspected metadata, duplicate bytes and split/source rejection |
| 100-clip library | 100 distinct synthetic AVI clips imported in integration test; actual CLI fixture dry run count 100 and real catalog count 0 |
| Multi-clip queue states | successful, failed and cancelled items; cancellation during running item preserves its success and cancels remaining work |
| Restart/resume | OS lease conflict, recovered running item, preserved completed work and changed-profile rejection |
| No actors/disagreement | actual SurveillancePipeline orchestration in queued fixture tests; generic anomaly retained, no invented Normal label |
| Cache/fresh distinction | mismatch rejected; cache and actual inference separated in metric selection; production runner restricts cache to installed pinned examples |
| Missing/corrupt media | missing file, invalid decode, changed hash and upload decoding validation |
| Upload/path safety | traversal, Windows drive/backslash, unsupported extension, origin/host, custom header, size limit and invalid request fields |
| Review/reference integrity | immutable table triggers, stale reference version rejection, separate reviewer opinions, hidden reference default, audited reveal and unblinded later reviews |
| Metric populations | references only, no unknown-label metrics, separate cache/upload/cohort, abstentions counted, covered metrics separate, temporal interval accounting |
| CSV/JSON/report | HTTP downloads, reference opt-in, formula-safe cells, escaped printable HTML without raw media, per-item filtering |
| Offline frontend | installed Gradio built JS/CSS served locally; all six tabs; disabled analytics and no remote Google fonts; browser screenshot inspection |
| Local launch/preflight | localhost server started with active worker and empty library; health response; missing-asset unit preflight and installed-asset verification |
| Real frozen-model smoke | two real local clips, including full routed detector/I3D/Actor-Transformer execution; exact hashes verified |

## Real-model smoke scope

No benchmark was rerun and no references were inferred. The scripts explicitly
selected existing local demonstration videos, copied them only to isolated smoke
catalogs and retained their source/rights uncertainty. The pilot library remains
empty. Smoke output is ignored by Git.

* `outputs/municipal-real-smoke-v1/receipt.json`: genuine on_demand inference,
  32 anomaly scores, behavior bypass, ~0.998 s pipeline / ~3.879 s request including
  ~2.859 s model loading. This does not exercise behavior on that bypassed clip.
* `outputs/municipal-real-routed-smoke-v1/receipt.json`: genuine on_demand inference,
  32 scores, three routed windows with actor counts [1, 0, 2], real actor outputs,
  preserved no-actor abstention, ~3.783 s pipeline / ~7.320 s
  request including ~3.514 s model loading. Detector ~1.850 s, I3D ~0.329 s,
  Actor-Transformer ~0.084 s. These are single engineering-run times, not throughput
  or cross-platform performance claims.

Frozen profile hash:
`becf960d56e6e50f83ae1142caeb9aa55f022a1dbde6903ec456262e40853ff9`.
Canonical full configuration SHA256:
`1776b504a348c20849a8fceaa6ec82aae5a2a17c5537bf4a7c309916e8e24785`.
The deployed five model hashes remain exactly those in surveillance_demo.yaml.

## Frontend and portability limits

There is no package.json or standalone React source in this checkout. Therefore npm
build/typecheck/unit commands are inapplicable. Instead, the supplied pinned Gradio
wheel's existing static build is mounted and requested in tests, and frontend
callback tests exercise evidence inspection and editable timed review submission.
Browser inspection caught and repaired a dark-theme contrast issue; the app remains
responsive, with an overflow tab menu at narrow widths.

Mac CPU is supported by the same platform-neutral entry point and a documented
direct dependency profile, but no Mac hardware or independently provisioned offline
wheelhouse was available. Full disconnected-machine installation and all possible
browser codecs are not claimed. The runtime requires no internet once compatible
dependencies and exact local assets are installed. Local-host browser access was
verified; no cloud, tracking or external enforcement connection was introduced.

## Scientific and dataset limits

Historical UCF ROC-AUC .7441, behavior macro F1 .2140 and cascade alert F1 .4733
remain unchanged. No thresholds, models or populations were retuned. Source grouping
and byte hashes prevent known overlaps, not unknown re-encodings or misleading source
IDs. Curators must independently audit actual dataset overlap and reference validity.
The library is not automatically an unbiased benchmark. Indian acquisition,
geographic evidence, licensing/redistribution permission, original recording identity
and verified labels remain dataset-dependent follow-up work.
