# DCSASS Human-Centric v1: audited population and source split

The selected installation is
`data/raw/dcsass/DCSASS Dataset/DCSASS Dataset/`. Its outer sibling contains a
byte-identical copy of all 16,639 videos and 13 label CSVs; that duplicate
installation is not counted twice and neither copy was deleted.

The full audit decoded every frame of every video, checked stable geometry and
reported/decoded frame counts, and SHA256-hashed every clip. The before/after
file inventory was unchanged. Results: **16,639 readable clips, 0 unreadable,
520 original sources, no exact duplicate-content groups** within this installation.
All videos are 320x240 at 30 FPS. Duration min/25th/median/75th/max is
1/1/2/3/29 seconds. This is the actual local export, not an expected dataset count.

## Annotations and intended population

The observed category CSV schema is headerless `clip_id,category,binary_label`.
Binary label 0 means Normal, including clips inside anomaly-category folders.
Binary label 1 maps to its observable category. The model has six group classes:
Normal, Abuse, Assault, Fighting, Robbery, Vandalism. No actor action labels or
person identities are available or inferred.

There are 16,631 raw CSV rows, including three repeated identical rows and two
blank binary labels. Identical duplicate rows are collapsed with their locations
recorded. Missing labels, missing-video references and apparent filename typos
are recorded without correction. **49 installed clips have no accepted annotation**;
they remain on disk and are excluded from optimization/evaluation, not relabeled.
Of these, 34 are Abuse clips (all 32 Abuse042 clips plus Abuse048 clips 0/1), and
two are Robbery clips. The rest are in an excluded category.

The actor task additionally excludes 10,099 valid-labeled clips from Arrest,
Arson, RoadAccidents, Burglary, Explosion, Shooting, Stealing and Shoplifting.
They remain available for later generic anomaly research. The selected population
is **6,491 labeled clips from 203 original sources**. This is human-centric
observable behavior classification, not inference of criminal character.

## Split construction and freeze

Source IDs come from the common UCF basename: e.g. `Fighting002_x264_31.mp4`
and its parent `Fighting002_x264.mp4` both recover `fighting002`. This agreement
was checked for every clip. Representative filenames, parent identities and
category CSV rows were inspected, and parser regressions cover the observed form.

Seed 0 assigns original sources in category strata targeting 70/15/15. Published
UCF test annotations reserve 14 installed selected sources for test. No reserved
source may enter training or validation. The freeze checks supervision, unchanged
installation inventory, zero source overlap and zero duplicate-content overlap.

An initial category-only candidate accidentally assigned the normal-only source
`fighting005` to validation. Before any detector run, feature extraction,
optimization or test predictions, a deterministic positive-label support check
exchanged whole sources `fighting002` (train to validation) and `fighting005`
(validation to train). Source counts and all test assignments stayed fixed.
The rejected candidate manifest/receipt remain preserved. **The final manifest is
`data/manifests/dcsass_human_centric_v1_final.jsonl`**, SHA256
`f3e7fc11159fce191f61b8ecf1d0d667cb68c9f949afa3ab84758c0532ee7258`.

| Population | Train | Validation | Test |
|---|---:|---:|---:|
| Original sources | 140 | 32 | 31 |
| Clips | 4,478 | 1,022 | 991 |
| Normal | 2,116 | 591 | 557 |
| Abuse | 435 | 68 | 65 |
| Assault | 164 | 82 | 32 |
| Fighting | 67 | 9 | 10 |
| Robbery | 1,384 | 232 | 275 |
| Vandalism | 312 | 40 | 52 |

**Fighting validation/test positives each come from one source**. Their small,
correlated clip populations limit class-specific conclusions and checkpoint
selection stability. This limitation is established before training and must not
be repaired after examining held-out predictions. Class weights will use covered
training clips only. No-actor clips will be excluded from actor optimization,
reported as uncovered in evaluation, and never forced to Normal.

## Related UCF population

The pinned author repository revision is
`8aa957cdaa7eac821e07115fe63920c3b07da59d`. Its 290 test annotation entries and
1,610 training-list entries were acquired as small text files while the user's
UCF video download continues. Source reservations are independent of download
completion. DCSASS and UCF are not independent datasets.

When defining Sultani optimization, also reserve the additional DCSASS test
sources if evaluating the full DCSASS cascade test population. Otherwise use the
official-UCF-test intersection for an end-to-end demonstration evaluation and
state its distinct population. Any supplemental holdout from official training
must be described as a research protocol, not an unchanged paper benchmark.

## Artifacts and commands

- Full audit: `runs/dcsass/audit_v1.json`, per-clip hashes/metadata and exclusions.
- Current audit entry points: `runs/dcsass/audit.json` / `audit.md`; earlier
  missing-asset receipts are retained separately.
- Duplicate-installation verification: `runs/dcsass/duplicate_installation.json`.
- Final split receipt: `runs/dcsass/human_centric_v1/final_split_receipt.json`.
- Rejected category-only receipt: `runs/dcsass/human_centric_v1/split_receipt.json`.
- Author metadata provenance: `data/splits/ucf_authors_provenance.json`.

```powershell
.\.venv\Scripts\python.exe scripts/audit_dcsass.py --root "data/raw/dcsass/DCSASS Dataset/DCSASS Dataset" --output runs/dcsass/audit_v1.json --workers 4
.\.venv\Scripts\python.exe scripts/freeze_dcsass.py --audit runs/dcsass/audit_v1.json --ucf-test-annotations data/splits/Temporal_Anomaly_Annotation.txt --manifest data/manifests/dcsass_human_centric_v1_final.jsonl --receipt runs/dcsass/human_centric_v1/final_split_receipt.json
```

Existing audit/freeze outputs are protected from overwrite. These artifacts contain
no model accuracy or anomaly AUC; they define the population before training.
