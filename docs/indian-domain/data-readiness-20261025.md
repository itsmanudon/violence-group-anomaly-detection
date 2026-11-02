# Indian-domain data readiness — 25 October 2026

**Decision: HOLD.** No split approved/frozen, no Indian baseline scoring, and no
training. Deadline remains 10 November. Existing local work, historical receipts,
checkpoints and six-class pipeline were preserved. The existing preparation code
was reused; only affected local tests ran.

**Latest continuation:** see [the implementation handoff](implementation-handoff-20261025.md).
The user has confirmed 521 videos as authoritative and retired the 536 estimate.
The receipt-based continuation also reports duration and provides workspace review
queues. Historical dry-run receipts are retained unchanged.

## Exact counts

| Gate | Count |
| --- | ---: |
| Supplied annotation rows / unique clip IDs | 521 / 521 |
| Labels: violence / non_violence / not_sure / missing | 48 / 465 / 7 / 1 |
| Annotation-clean binary candidates before media qualification | 505 = 40 positive + 465 negative |
| Priority annotation quarantine | 16 = 8 interval overshoots + 7 uncertain + 1 missing/cannot_play |
| Additional media qualification hold | 1 positive clip |
| Annotation-clean **and** media-qualified candidates | **504 = 39 positive + 465 negative** |
| Combined quality quarantine, unique clips | **17** |
| Original-source/incident independence holds | **All 521** |
| Training eligible / protected-evaluation eligible | **0 / 0** |
| Approved train / validation / test / unassigned | **0 / 0 / 0 / 521** |

The 504 candidates are **not approved training data**. Near-duplicate/source
reviews can change the eventual independent population. All 521 `reviewed_at`,
`source_video_id` and `video_url` fields are blank. Reviewers contributed 174, 178
and 169 rows. Each combined row exactly matches the common fields of one original
reviewer CSV row; paths, reviewer identity, raw notes/statuses and CSV row numbers
are retained. No timestamp, label or original status was filled in.

## Inventory and media reconciliation

The actual handover catalog contains **523 media entries: 521 video-bearing files
plus two audio-only exclusions**. All 521 video IDs are annotated. A physical scan
of the six catalogued collection folders found those same 523 media files, with
no uncatalogued or missing catalogued files. Thus **zero identifiable video
annotations are missing from this catalog**. The user has confirmed that 536 was
an earlier verbal estimate, not a separate inventory. **There are no 15 missing
videos to report.** The operative baseline is the 521 catalogued videos.

The two legitimate exclusions, not missing violence annotations, are WalkIndia
`temple-Scene-001-scene006.mp4` (`80ec9cc209a4da372c8d82eb`) and
`temple-Scene-004-scene001.mp4` (`b73c70cbe78ab5b9986c3d8b`), both audio-only.

All **521 catalog-resolved original local files exist and match catalog SHA256**.
The literal `01_Videos/...` package paths in the supplied CSV are absent locally
(0/521); they were explicitly joined by clip ID **and exact relative path** to
`original_local_path`, not rewritten in the original CSV. Records preserve both
paths. Do not point the ordinary importer at the empty local package tree.

**520 passed strict media qualification**: 294 reused hash-bound complete/
sequential-decode receipts (278 Phase 2A timing cases plus 16 catalog receipts),
and 226 passed fresh full decoding. Current eight-position visual samples were
decoded for these 520. One additional fresh sequential decode produced a count
mismatch, not proof of corruption:

- Snatch `1.mp4`, ID `f1b5b1596421b4283627ff3e`: **599 decoded versus 615 reported
  frames**, unchanged SHA256. The original catalog records an edit-list start of
  15116/30000 seconds. A targeted reuse of the existing timing parser now yields
  **599 presented samples**, exactly matching the sequential decode count.
  This explains the encoded/presented count difference, while not proving decoded
  frame identity. The current exact-count loader still rejects it. Keep it on hold until
  presentation-time/frame mapping and loader compatibility are explicitly
  qualified; do not loosen the existing decoder or silently normalize the video.

The 278 original `media_status=needs_review` values remain unchanged. Prior
edit-list qualification is separate evidence, not a replacement annotation status.

## Priority dispositions

All 16 supplied priority IDs exactly match the issues independently derived from
the combined CSV. Decimal seconds and **MM:SS[.fraction] in `_sec` columns** are
now tested. Parsing produces derived seconds while retaining original strings.

- **Seven `not_sure`: quarantined**, never a third class or default negative.
  Snatch `13_0.mp4` and `16_0.mp4` retain their supplied temporal ranges, which do
  not override the uncertain labels.
- **One blank-label/cannot_play: quarantined** (`f74dc1433140f8b84451eb09`, Snatch
  `23_0.mp4`). The original local file qualified for decoding, but that does not
  resolve the reviewer's playback experience or supply a human label.
- **Eight end-time overshoots: quarantined pending explicit adjudication.** Excess
  ranges from 0.100 to 0.360 seconds. Each record has an **unapplied** proposal to
  end at the recorded presentation duration; originals and derived input intervals
  remain unchanged. No tolerance silently accepts or truncates these intervals.

| File | Excess seconds |
| --- | ---: |
| HCI `krishna-clip-3.mp4` | 0.233 |
| HCI `Screen_Recording_20261020_182710_YouTube.mp4` | 0.157 |
| Refined `Video7.mp4` | 0.233 |
| HCI `krishna-clip-1.mp4` | 0.300 |
| HCI `krishna-clip-2.mp4` | 0.100 |
| Snatch `2_0.mp4` | 0.360 |
| Snatch `30_0.mp4` | 0.360 |
| Refined `Video1.mp4` | 0.266 |

Prioritize recovering these eight positive examples through documented interval
review rather than permanently excluding difficult/short positives. Missing
intervals remain unknown, not whole-video violence annotations.

## Source/label confounding and split proposal

These are **collection counts**, not verified independent recording counts.

| Collection | Violence | Nonviolence | Uncertain | Missing | Quality candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Snatch-1.0 | 30 | 7 | 4 | 1 | 34 |
| HCI-User-Collection | 11 | 0 | 3 | 0 | 7 |
| Unidentified-Refined | 7 | 2 | 0 | 0 | 7 |
| Unidentified-Archive | 0 | 104 | 0 | 0 | 104 |
| IPID | 0 | 17 | 0 | 0 | 17 |
| WalkIndia-50 | 0 | 335 | 0 | 0 | 335 |
| **Total** | **48** | **465** | **7** | **1** | **504** |

All 48 positives come from three collections; 456 negatives come from collections
with no positives. Snatch is 30/37 positive among binary labels, HCI 11/11, Refined
7/9; the other collections have zero positive support. This corpus is not balanced.
A random clip split could reward camera/encoding/location/duration cues rather
than violence. Class weights alone do not remove this confounding. Seek genuinely
annotated within-source hard negatives and independent positive recordings, and
report source-wise support and errors. Confirm a shared observable-violence
definition for snatching/theft rather than silently remapping those labels.

Catalog geography is **unknown for all 48 positives**; the 352 documented-India
WalkIndia/IPID videos are all labeled nonviolence. Indian-positive generalization
is therefore unestablished even if a future clip score is high.

**Verified original recording/incident identities: zero.** Retain null identities;
numeric Snatch/`_0` families and HCI `krishna-clip-*` are only relationship hints.
Conservative guards hold each of Snatch (42), HCI (14), Refined (9), Archive (104)
and IPID (17) together. WalkIndia uses the documented whole-tour guards: hilltown
114, metro 112, temple 109. Preserve all **14 HCI municipal reservations** outside
optimization. With that reservation, only two nonreserved positive-bearing
collection guards remain; they cannot establish three independent positive
partitions, and even those guards are not verified original recordings.

Exact SHA256 screening over all 521 found **zero exact duplicate groups**. The
new eight-sample screening over 520 qualified clips found **one unresolved visual
candidate pair**, Archive `video77.MOV` / `video78.MOV` (IDs
`1a858677680083cbc1b8b2fa` / `75bebe30b777f2e0fa9fd006`). Keep them in the same
conservative guard and review overlap; a visual match is not proof of duplication
or independence. `1.mp4` lacks qualified current visual screening. Zero matches
against the **6,491 available historical clip hashes** is not UCF/DCSASS/Collective
source clearance; original-source and transformed-footage overlap remain unresolved.

**Proposal:** recover/verify original recording and incident mappings, connect
source/incident/content overlaps, preserve reservations, then propose seeded
70/15/15 whole-group partitions using the existing splitter (seed 7). Require
both classes in every partition and report actual source/clip counts. All 521
remain unassigned now. No allocation or split approval has been manufactured.

## Next actions and experiment commands

1. Adjudicate the 16 priority records and the additional `1.mp4` timing case,
   recording reviewer/date and any approved derived boundary adjustment.
2. Supply original recording URLs/IDs, incident and rights evidence; review the
   visual pair and benchmark overlap. The 521-video inventory is reconciled.
   Do not replace
   missing identities with clip IDs or collection names.
3. Approve a source-safe split before inference/training. Frozen existing-model
   baseline and precision/recall/F1/PR-AUC/FPR/confusion/source-wise performance
   are **unrun**, not zero. No camera-hour false-alarm rate is estimable here.

The [controlled A–D matrix](training-matrix-20261025.md) contains exact local RTX
cache, three-step training and resume commands, with explicit entry gates and
pending actor/temporal orchestration. They cannot run legitimately against the
current unassigned readiness records. The existing frozen cascade remains intact.

Reproduce the dry run only when inputs change, choosing a **new** output version:

```powershell
$taskPacket = 'D:/violence-datasets/_reports/v1-20261021/drive-handover-20261021/private-upload-v2/Violence-Surveillance-Dataset'
$taskEvidence = 'D:/violence-datasets/_manifests/v1-20261021'
.venv/Scripts/python.exe scripts/prepare_indian_domain.py --csv 'C:/Users/manan/Downloads/indian_annotations_combined_unverified.csv' --dry-run --priority 'C:/Users/manan/Downloads/indian_annotations_priority_review.csv' --catalog "$taskPacket/03_Manifests/master_clip_inventory.csv" --source-groups "$taskPacket/03_Manifests/source_groups.csv" --evidence-catalog "$taskEvidence/master-clip-inventory.jsonl" --timing-evidence "$taskEvidence/phase2a/phase2a-timing-cases.jsonl" --historical-manifests data/manifests/dcsass_human_centric_v1_final.jsonl data/manifests/dcsass_sultani_generic_v1.jsonl data/manifests/ucf_sultani_shared_safe_v1.jsonl data/manifests/collective.jsonl --reviewer-files 'C:/Users/manan/Downloads/reviewer_1.csv' 'C:/Users/manan/Downloads/reviewer_2.csv' 'C:/Users/manan/Downloads/reviewer_3.csv' --output runs/indian-domain/readiness-20261025-v2 --workers 2
```

## Receipts and verification

- [Dry-run report](../../runs/indian-domain/readiness-20261025-v1/readiness.json),
  [all raw/resolved records](../../runs/indian-domain/readiness-20261025-v1/records.json),
  [input/output hashes](../../runs/indian-domain/readiness-20261025-v1/registration.json).
- [Exact quality counts](../../runs/indian-domain/readiness-dispositions-20261025-v1/counts.json),
  [17 quality holds](../../runs/indian-domain/readiness-dispositions-20261025-v1/quality-holds.json),
  [521 row dispositions](../../runs/indian-domain/readiness-dispositions-20261025-v1/dispositions.csv).
  Derived hold = annotation issue **or** media qualification failure; this gate
  never grants source independence or split approval.
- **56 affected tests passed**, lint/format/whitespace checks passed; see the
  [dated verification log](readiness-verification-20261025.md). Prior 112-test and
  RTX receipts were reused, not relabeled as fresh validation. No full suite,
  model training, inference experiment, push, merge or publication was run.
