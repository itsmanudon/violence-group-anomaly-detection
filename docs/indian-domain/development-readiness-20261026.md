# Updated Indian exploratory-development readiness — 26 October 2026

This updates assumptions after the user's human-annotator report and subsequent
explicit permission instruction. Earlier dated audit/receipt findings are
preserved; their blanket unresolved-positive-incident blocker is superseded for
**provisional exploratory grouping**, not for independently verified metadata.

## Attestation and unchanged annotations

Recorded evidence: user reporting human annotators on 26 October 2026,
Asia/Calcutta: **“All labeled violence clips represent distinct incidents.”**
Evidence type `annotator_attestation`, independent source verification **false**.
The statement covers 48 original violence labels, including held records; it
does not repair temporal/media issues, confirm geography, establish original
recording URLs, certify research rights or override automatic duplicate evidence.
No annotator name/signature or per-record review timestamp is invented.

Original violence/nonviolence labels are reused. Routine relabeling and recovery
of every original source URL are **not prerequisites for exploratory training**.
Seven `not_sure` and the blank/cannot_play record stay quarantined, as do all
other original holds. The **17 quality holds are unchanged**, and quality
candidates remain **504 = 39 positive + 465 negative**. All source CSVs, notes,
timestamps and media paths remain unchanged.

Negative chunks may be consecutive/overlapping. The new grouping focuses their
continuity on existing collection/tour evidence, cached fingerprints and filename
families; unresolved negative recording cohorts are kept intact. Candidate visual
or filename links connect components rather than being declared independent.
Positive incident independence is attested, while filename/content links may
still conservatively group distinct incidents into a recording cohort.
No original source ID is fabricated from a unique clip ID.

Automatic checks remain global: all 521 current video hashes against 6,491
available historical clip hashes (zero hits); existing near-duplicate screening
on all 520 strict-readable clips, including positives (one ambiguous Archive pair,
kept within one negative cohort). These are limited screens, not exhaustive
transformed-footage clearance. No new full decode, face matching or footage upload.

## Provisional proposal, not approved training

Artifacts: `runs/indian-domain/development-proposal-20261026-v1/`:
`proposal.json`, `proposed-membership.csv`, `collection-partition-counts.csv`,
`permission-evidence-queue.csv`, current hash recheck and registration.
Seed **7**, target group ratios 70/15/15, existing seeded allocator reused.
Groups are recording/content **cohorts**, not independently verified source IDs.

Of 504 candidates, seven existing HCI municipal reservations remain excluded
from optimization. Nine negative candidates (Snatch seven, Refined two) are
separately excluded from this initial proposal because their recording contexts
within positive collections are unresolved. These are **development exclusions,
not added quality holds or relabeling**. The remaining **488 = 32 positive +
456 negative** are assigned provisionally:

| Proposed partition | Clips | Violence | Nonviolence | Provisional cohorts |
| --- | ---: | ---: | ---: | ---: |
| Train | 357 | 22 | 335 | 22 |
| Validation | 22 | 5 | 17 | 5 |
| Development reserve, stored as test | 109 | 5 | 104 | 5 |

Cohort ratios are 22/5/5 out of 32 (68.75/15.625/15.625%); actual clip ratios
are **73.16/4.51/22.34%**, not 70/15/15. All connected members stay together.
The proposed reserve is **not a frozen final independent-domain test** and has
not been evaluated. No training manifest, source-approved split receipt or model
cache was created. Membership may be reviewed before a development registration;
no existing protected test membership is modified.

## Explicit confounding and estimation limits

Train negatives are all 335 WalkIndia clips; validation negatives are all 17
IPID clips; reserve negatives are all 104 Archive clips. Positives are Snatch
19/4/4 and Refined 3/1/1 across the three partitions. The conservative grouping
reduces continuity leakage but creates strong collection differences. Within
this proposal, collection remains perfectly associated with label. Distinct
positive incidents can share camera/processing/context cues. Excluding the nine
mixed-collection negatives worsens label confounding; recover their recording
context before preferring this design over alternatives.

All 48 original positives still have unknown catalogued geography; all 352
India-documented clips remain negative. This is not verified Indian-positive
generalization. Five validation positives and one negative cohort cannot yield
stable independent-domain accuracy. Any future metrics must be exploratory,
source/collection-specific, and accompanied by recall, PR-AUC, confusion and
coverage. A defensible held-out evaluation remains a separate future requirement.

## Permissions and execution gate

The user explicitly confirmed: **keep all six collections permission-unverified,
promote none of the 504 candidates, and do not start Indian training/evaluation
until permission evidence is recorded and checked**. Accordingly:

- Research permission remains `unverified` for all six collections.
- **Training approved: 0; evaluation approved: 0.**
- No Indian feature extraction, training, calibration or model evaluation occurred.
- Labels/incident attestation do not imply footage usage rights. Software licenses
  or dataset README assertions alone were not promoted into footage permissions.

The six-row permission queue has blank evidence/authorized-attestor/check-date
fields. Record actual local-research usage scope and the supporting evidence;
do not fill approval merely to unlock training. Once checked, integrate an
explicitly exploratory group/attestation/permissions receipt into the existing
cache/training verifier and run the RTX extraction/one-batch/save-resume smoke
before a bounded frozen-C3D binary-head run. Original URLs are optional evidence,
not a universal gate. Original six-class checkpoints remain untouched.

## Implementation and tests

New `datasets/indian_development.py` and `scripts/propose_indian_development.py`
produce reports only; they do not relax the existing trainer's approval guard.
Synthetic tests cover positive attestation, negative connected components,
cross-collection exact duplicates, ambiguous links, preserved holds/reservations,
missing identities, and absence of training manifests/permission promotion.
Final focused result: **23 passed in 7.68 s** (development, existing split/review
tests). The earlier five-test addition plus regressions passed 22 tests; a newly
added integration fixture initially failed because nine pure-class groups could
not satisfy the existing three-way group-ratio constraint. It was expanded to
adequate synthetic groups; no production split guard/test was weakened.

Safe regeneration into a new directory:

```powershell
.venv/Scripts/python.exe scripts/propose_indian_development.py --audit runs/indian-domain/readiness-20261025-v1 --csv data/indian-domain/annotations-20261025-v1/indian_annotations_combined_unverified.csv --catalog 'D:/violence-datasets/_manifests/v1-20261021/master-clip-inventory.csv' --output runs/indian-domain/development-proposal-new-version --seed 7
```

This command is preparation, not permission approval or training. The earlier
controlled DCSASS validation study is separate and uses no Indian footage.

Final combined affected verification: **64 passed in 22.04 s**, including the
proposal tests, existing grouped split/review and six-class/cascade compatibility.
All six new Python files pass lint/format and prior uncommitted files remain
unchanged. No permissive trainer fallback was added: this report-only proposal
cannot be passed off as an approved split by the existing binary trainer.
