# Phase 3: current annotation and media screening

Freshly checked on 26 October: authoritative catalog exists; all 521 referenced
video files have current matching byte hashes and unchanged raw annotations.
Both separately catalogued audio-only files also exist and match their hashes.
Registered parent inputs/output receipts were verified before reuse. Prior
complete-decode and eight-sample visual evidence was reused, **not freshly
decoded or described as new decoder validation**. No frames exported/uploaded.

Artifacts: `runs/indian-domain/sprint-20261026-v1/data-review/`:
`quality.json`, `registration.json`, `receipt-recheck/`, `review-queue.csv`,
`first-priority.csv`, `duplicate-candidates.json`. All private/local/ignored.
The raw five CSV snapshots remain byte-identical to original Downloads files.

| Gate | Reconfirmed count |
| --- | ---: |
| Catalog media / videos / audio exclusions | 523 / 521 / 2 |
| Missing annotations for catalogued videos | 0 |
| Raw violence / nonviolence / uncertain / blank | 48 / 465 / 7 / 1 |
| Prior strict-loader qualified, current bytes unchanged | 520 |
| Annotation/media candidates | 504: 39 violence, 465 nonviolence |
| Quality holds | 17: 16 annotation, one additional loader timing hold |
| Source-approved training / evaluation | 0 / 0 |
| Assigned/frozen partitions | None |

Eight original temporal ends overshoot by 0.100–0.360 seconds. Seven `not_sure`
records remain quarantined, including two with temporal intervals. One blank
label/cannot_play record remains held. No temporal value was clipped or used to
infer whole-video violence. All 278 `needs_review` media values and 243 `verified`
values are preserved; media status alone does not negate a violence annotation.
Missing source IDs, URLs and reviewed_at remain 521 each.

One concrete parser defect fixed: `assess_annotation` parsed MM:SS endpoints but
converted duration with `float` when deriving rounding proposals. Duration now
uses the same strict parser, reports invalid/nonpositive duration as a quarantine
issue even without endpoints, and preserves all original text. Actual numeric
CSV values and existing 17-hold membership did not change.

Snatch `1.mp4` remains a strict-loader hold: prior 615 encoded vs 599 decoded
frames; separate edit-timeline receipt accounts for 599 presented samples but
does not prove frame identity mapping or change the existing loader. No silent
compatibility override or new complete-decode claim.

The deduplicated queue has **521 unique rows**, not one row per flag. Its first
priority subset is **58 = 39 candidate-positive second looks + 17 quality holds
+ two records in the visual candidate pair**. Sets are disjoint on these inputs.
All rows require source evidence; 14 HCI municipal reservations remain excluded
from optimization. Blank human-disposition/evidence/reviewer/date fields are
intentional, and `training_approved=False` everywhere. No fabricated hard-negative
queue: approved Indian prediction scores are unavailable.

Regenerate into a new directory:

```powershell
.venv/Scripts/python.exe scripts/screen_indian_review.py --audit runs/indian-domain/readiness-20261025-v1 --csv data/indian-domain/annotations-20261025-v1/indian_annotations_combined_unverified.csv --catalog 'D:/violence-datasets/_manifests/v1-20261021/master-clip-inventory.csv' --output runs/indian-domain/screen-review-new-version
```

Tests: **63 passed in 14.15 s**, covering new review automation and existing
ingestion/readiness/source-split paths, no failures/skips. Includes uncertainty,
MM:SS, provenance conflicts, exact/near duplicates, ambiguous static matches,
deduplication, no mutation and no automatic approval. No full suite run.

## Integration timing supplement

`timing-observations.json`, `timing-registration.json` and
`timing-review-queue.csv` were added from the same current-hash-verified prior
metadata, with no new decoding. For all **521** records, presentation duration
was compared separately with decoded frame count/FPS. **341** differ by more
than one decoder-frame duration plus 1 ms: WalkIndia 335, IPID three, Snatch two,
HCI one. Catalog/decoder FPS disagreements over 1e-6: **zero**. The 341-row queue
is a temporal-clock diagnostic and overlaps the 521-row provenance queue; do not
add it to the 17 quality holds or interpret it as 341 corrupt/unlabeled videos.
The binary quality-candidate counts remain unchanged. Existing edit/presentation
metadata may explain differences; no explanation or repaired mapping is assumed.
Resolve clocks before event localization/latency, not by replacing annotations
with frame-count/FPS endpoints. New timing helper and affected readiness tests:
**28 passed in 0.90 s** after the 135-test integrated run.
