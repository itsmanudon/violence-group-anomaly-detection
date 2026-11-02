# Indian annotation preparation contract

**25 October update:** actual combined CSV inspection and read-only catalog
reconciliation are documented in [data readiness](data-readiness-20261025.md).
`parse_annotation_seconds` now accepts explicit decimal seconds or MM:SS[.fraction],
including values in `_sec` columns. Raw strings remain unchanged. The original
23 October preparation status below is retained as the initial contract context.
Use the new dry-run command before attempting a trainable import of the supplied
combined file; package-relative paths need explicit catalog resolution.

Real CSV, videos and provenance have **not been supplied or ingested**. The
configuration example matches the proposed header only; it is not a verified
export. These commands operate only on local files and never contact Drive.

From the repository root, use the existing environment:

```powershell
.venv/Scripts/python.exe scripts/prepare_indian_domain.py --csv PATH/annotations.csv --inspect-header
.venv/Scripts/python.exe scripts/prepare_indian_domain.py --csv PATH/annotations.csv --media-root PATH/media --columns configs/indian-domain/columns.example.json --provenance PATH/provenance.json --output data/indian-domain/import-v1
```

Inspect the first command's actual header before configuring canonical→actual
column names. Required canonical fields: `clip_id`, `file_path`, `label`. Optional
fields: reviewer, timestamp, playback_status, violence_start, violence_end, notes,
source_video_id, incident_id, source, review_status, rights. Unmapped raw columns
are retained. One CSV row is one clip annotation; repeated clip IDs quarantine
**all** repeated rows for adjudication, including repeated identical annotations.

Labels normalize case, whitespace and hyphen/space separators only:
`violence`→1; `non_violence`→0; `not_sure`→quarantine. Other vocabularies (including
Normal, Abuse, Assault, 0/1) require explicit annotation-definition review and are
rejected. Paths resolve under the explicit media root; escapes/symlink escapes
fail. No folder-based labels. Full OpenCV decode, frame-count agreement and SHA256
are required. Metadata duration assumes constant FPS; investigate VFR separately.
No transcoding or silent repair is performed.

Temporal boundaries are seconds `[start,end)`. Both must be present, finite,
ordered, within duration, and paired with violence. Both missing means unknown
for **either** label; it does not create whole-video violence or complete negative
temporal truth. One interval is supported by this CSV; multiple intervals need an
explicit schema extension, not an invented parsing convention.

The provenance file is a JSON object keyed by exact clip ID, for example this
**synthetic schema illustration**, not a real dataset record:

```json
{
  "example-only": {
    "source_video_id": "collection:original-recording-id",
    "incident_id": "collection:shared-incident-id",
    "source": "Original URL/acquisition record or local provenance receipt",
    "rights": "Document actual terms; unknown until reviewed",
    "review_status": "Document actual review status"
  }
}
```

Source IDs represent original recordings; incident IDs join distinct recordings
of the same event. When an incident is independently known to correspond to one
recording, curators may explicitly assign its own incident ID with evidence.
Never generate a unique incident for every clip merely to make splitting pass.
IDs are preserved verbatim; grouping compares stripped/casefolded identities.
No provenance is inferred from appearance, filenames or folders. Conflicting CSV
and supplied provenance values quarantine the row. Reviewer/playback strings are
preserved and tabulated, not interpreted as verified approval.

Importer outputs in an exclusive new directory:

- `import.json`: unsplit accepted binary records, raw annotations, provenance,
  temporal metadata, full-decode results, SHA256 and visual samples.
- `quarantine.json`: raw uncertain/invalid/conflicting rows with reasons; never
  read for training. Resolve by producing a new annotation/import version.
- `quality-report.json`: observed header/mapping, actual counts, label/review
  distributions, unresolved provenance, candidate duplicates and limitations.
- `registration.json`: hashes binding the three artifacts.

No common `Record` JSONL is emitted at import, because its schema permits only
assigned train/val/test records. Missing identities may be staged for repair, but
cannot be frozen. No automatically approved review file is generated.

## Historical benchmark exclusion and review

Create an explicit inventory of **all previously used** benchmark manifests,
including original UCF/DCSASS sources and Collective actor sources. The example
below lists existing relevant files; a curator must verify completeness, add
any historical aliases/other training sources, and retain acquisition evidence.

```powershell
.venv/Scripts/python.exe scripts/freeze_indian_split.py inventory --manifests data/manifests/ucf_sultani_shared_safe_v1.jsonl data/manifests/dcsass_human_centric_v1_final.jsonl data/manifests/collective.jsonl --evidence "Describe reviewed historical dataset scope" --output PATH/benchmark-inventory-v1.json
```

The inventory reads explicit source IDs and recorded video SHA256; where available
it hashes local video bytes. Missing video hashes are reported (frame-based actor
manifests may have none). This command can take time over large local videos;
it does not decode historical footage. Source/hash screening cannot establish
independence for renamed, cropped or re-encoded recordings. Curators must check
origin aliases and historical content, record evidence, and exclude overlap from
**all** Indian partitions. Absence of an automatic hit is not approval.

Review every exact and visual candidate pair in `import.json`. Visual screening
uses eight sampled 64-bit dHashes with Hamming distance ≤4, requiring at least two
matched samples per side; temporal alignment is not required. Uniform/low-detail
scenes can overmatch; partial reuse, crops and overlays can escape. Inspect the
originals and source evidence, record `same_footage` or `independent`; unresolved
candidates block freezing. Curators may add missed duplicate pairs. Exact copies
cannot be declared independent. Same-footage edges join connected groups; exact
copies with conflicting labels require adjudication/exclusion first.

Create `review.json` using actual evidence, bound to SHA256 of the imported
`import.json` and benchmark inventory. Required structure:

```json
{
  "import_sha256": "REPLACE_WITH_ACTUAL_SHA256",
  "benchmark_inventory_sha256": "REPLACE_WITH_ACTUAL_SHA256",
  "benchmark_scope_complete": true,
  "benchmark_scope_evidence": "Who checked which sources/aliases/content, and where evidence is saved",
  "excluded_clips": {"excluded-example": "Actual exclusion reason"},
  "clip_reviews": {
    "retained-example": {
      "annotation_approved": true,
      "provenance_approved": true,
      "benchmark_independent": true,
      "evidence": "Actual reviewer, date and annotation/provenance/overlap evidence"
    }
  },
  "candidate_decisions": [{
    "pair": ["retained-example", "another-retained-example"],
    "relation": "same_footage",
    "evidence": "Actual comparison evidence"
  }]
}
```

Example IDs must be replaced; clip_reviews must cover exactly retained records.
No example asserts an actual approval. Unknown provenance, unchecked annotation,
and missing overlap evidence fail closed. Rights/access review belongs in the
provenance evidence; software does not grant training or redistribution rights.

```powershell
.venv/Scripts/python.exe scripts/freeze_indian_split.py freeze --import-path data/indian-domain/import-v1/import.json --inventory PATH/benchmark-inventory-v1.json --review PATH/review.json --output data/indian-domain/split-v1 --seed 7
```

Splitting joins source, incident, exact-content and reviewed same-footage edges;
then searches 2,048 seeded group assignments near 70/15/15, minimizing class clip
ratio error and requiring both classes in every partition. At least three
independent groups per class are necessary. Group ratios can differ substantially
from clip ratios. Inspect actual clip/group/source/class counts in the immutable
`split-receipt.json`; no real counts are known yet. Mixed-class sources are valid.

Outputs are compatible `manifest.jsonl` records and a content-addressed receipt
binding original import/inventory/review, media hashes and protected test IDs.
`verify_split` rechecks every downstream run; changed inputs/media fail. Existing
output directories are refused. Do not create another split after inspecting
test results or alter frozen membership. Review source distributions and sampling
adequacy before starting model selection. Preserve all referenced files when
moving artifacts (paths currently reflect the local installation). Current
research execution is local RTX 4070 only; no transfer is required.
