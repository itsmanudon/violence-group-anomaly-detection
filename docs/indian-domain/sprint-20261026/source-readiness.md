# Phase 3 source/label confounding and smallest useful review batches

Counts recomputed from unchanged current annotation rows, not inferred from folders:

| Collection | Violence | Nonviolence | Not sure | Blank |
| --- | ---: | ---: | ---: | ---: |
| Snatch-1.0 | 30 | 7 | 4 | 1 |
| HCI-User-Collection | 11 | 0 | 3 | 0 |
| Unidentified-Refined | 7 | 2 | 0 | 0 |
| Unidentified-Archive | 0 | 104 | 0 | 0 |
| IPID | 0 | 17 | 0 | 0 |
| WalkIndia-50 | 0 | 335 | 0 | 0 |

All 48 original positives occur in three collections; HCI contains no binary
negative, while three large collections contain no positive. Random clip splits
could learn collection/encoding/background signals. Holding a collection out can
also leave a one-class evaluation or remove most positives. Class weighting or
balanced sampling cannot establish causal violence features or fix provenance.

Catalog geography: 352 India-documented clips, all nonviolence. The remaining
169 have unknown geography, including **all 48 original positives**. Neither the
dataset name nor general scene appearance verifies Indian positive generalization.
Unknown original source fraction is 521/521 (100%); verified incident count is 0.
There is no source-safe 70/15/15 proposal that can be approved today. Source-group
ratios must not be reported as clip ratios; actual train/val/test counts remain
unassigned, not fabricated 365/78/78.

Smallest useful human work:

1. Work the **58-row first-priority queue**: 39 quality-candidate positives for
   a second look, 17 annotation/media holds and the two ambiguous Archive clips.
   A completed content review does not alone approve source independence.
2. Supply evidence in batches for the ten filename families and eight conservative
   collection/tour guards, mapping their clips to actual original recordings and
   shared incidents (or document unresolved identities). Preserve municipal holds.
   A shared incident across collections must connect groups before allocation.
3. Provide source URLs/catalog evidence, research-use rights and historical overlap
   dispositions. Resolve the visual pair without guessing an incident identity.
4. Count independent cleared positive and negative groups. At least three
   independent class-supporting groups are needed even to attempt three-way
   splitting; adequacy/uncertainty may require more. If evidence cannot supply
   representative independent positive sources, collect permitted material rather
   than force a split. No new footage acquisition is performed in this sprint.

All 521 records stay source-held, 504 remain quality candidates and 17 remain
quarantined. Candidate positives are not currently eligible training positives.
No source approval, split freeze, Indian inference baseline or supervised training.
