# Dataset adapter contract, version 1

This contract is dataset-independent. Do not acquire/download footage through the
app. Indian datasets must be reviewed separately for source identity, rights,
geography, reference annotations and protected evaluation membership before import.
An adapter transforms supplied records into this manifest; it does not select models
or thresholds, silently merge populations, infer labels from names or retune models.

## Clip manifest

UTF-8 JSON, top-level `schema_version: 1` and `clips: []`. A minimal synthetic example:

```json
{
  "schema_version": 1,
  "clips": [{
    "clip_id": "fixture-001",
    "source_video_id": "synthetic-recording-001",
    "file_path": "fixture-001.avi",
    "source": "Synthetic generator; software tests only",
    "license": "Synthetic fixture",
    "external_handover_permission": "no",
    "training_permission": "no",
    "assignment": "test",
    "cohort": "synthetic_fixture",
    "selection_method": "Deterministic software fixture; not a benchmark",
    "geography": {"location": null, "confidence": "unknown"},
    "notes": "No real persons or geographic claim"
  }]
}
```

| Field | Contract |
| --- | --- |
| clip_id | optional generated ID, or 1–128 ASCII letters/numbers/underscore/hyphen; unique per content |
| source_video_id | required canonical original recording identity, shared across all derived clips; namespace IDs by dataset/acquisition |
| file_path | file under the explicit --media-root; CLI resolves relative paths against that root; traversal, symlink escapes and out-of-root absolute paths fail |
| sha256 | optional expected SHA256; if supplied it must match; actual hash always computed and stored |
| source | required provenance/citation/acquisition record; use unknown when absent |
| geography | location plus confidence unknown/inferred/reported/verified; reported/verified require evidence; inferred appearance is not actual location proof |
| license | required original terms or unknown; retain associated rights/acquisition receipts separately |
| external_handover_permission | required yes/no/unknown; never equate training permission with municipal redistribution permission |
| training_permission | required yes/no/unknown; does not initiate any training |
| assignment | train/validation/test/demo/upload |
| cohort | curated_demo/untouched_evaluation/arbitrary_upload/synthetic_fixture |
| selection_method | required documented sampling/curation procedure, including any prediction-assisted selection |
| created_at | optional original creation timestamp string; import time separately generated in UTC |
| notes | optional untrusted plain text, bounded to 4000 characters |

The catalog inspects actual duration, fps, frame count, decoded resolution, codec
and first-frame readability. These inspected values override manifest claims.
File size defaults to 512 MiB per clip; configure explicitly for the pilot. Supported
container extensions: mp4, avi, mov, mkv, webm, m4v. Decoding is mandatory, and browser
playback is separately qualified; no silent transcoding or substitution occurs.

Untouched_evaluation requires test assignment; curated_demo requires demo;
arbitrary_upload requires upload. Canonical source IDs and duplicate SHA256 content
cannot cross assignment/cohort boundaries. Identical bytes deduplicate into one
owned clip; alternate source IDs are retained in clip_sources and protected against
later leakage. No algorithm can discover all semantically identical/re-encoded
recordings: adapters must supply canonical recording IDs and independently audit
overlap with historical training/evaluation datasets before claiming untouched data.
Source IDs are case-sensitive; normalize them consistently in the adapter.

The app starts empty and never auto-imports example datasets. Uploads remain
unlabeled. Protected membership has no mutable public endpoint. Original test sources
used for post-hoc demonstration must be cataloged as curated_demo, with original
scientific split retained in source metadata; they must not masquerade as new
untouched evaluation data. The selected frozen models and historical protocols
are unaffected by catalog assignments.

## Verified references are a separate authority

Do not embed label/reference/event_intervals in clip imports. Add legitimate
references explicitly through `POST /api/references/{clip_id}`:

```json
{
  "payload": {
    "verified": true,
    "evidence": "Fixture annotation receipt; replace with actual annotation authority",
    "behavior": "Fighting",
    "violence": true,
    "temporal_complete": true,
    "event_intervals": [[0.2, 0.5]]
  },
  "author": "curator-identifier",
  "reason": "Initial verified reference import",
  "expected_version": 0
}
```

Behavior can be null or one of the exact frozen classes Normal, Abuse, Assault,
Fighting, Robbery, Vandalism. Violence is an independently supplied boolean or
null, never manufactured from an anomaly score. Optional temporal event intervals
are [start,end] seconds with 0 <= start < end <= decoded duration. To use false-alert
metrics they must be explicitly marked temporal_complete=true. An empty interval
list under that complete contract is an actual supplied no-event annotation; missing
intervals are unknown. Retain original frame-based annotations and conversion
receipts outside the application when converting to seconds.

The initial expected_version is 0. Revisions require the latest current version,
curator identity and reason; every version is preserved and audited. Ground-truth
reference reveals are explicit; reviews stay separate and cannot edit references.
Unknown/unverified annotations should not be submitted as verified. Physical
database access is trusted administration; this local pilot has no identity provider.

## Dry run and fixture adapter

Generate a fixture-only library (no actual surveillance footage):

```powershell
python scripts/municipal_fixture_library.py --output outputs/new-fixtures --count 100
python -m surveillance.municipal --manifest outputs/new-fixtures/manifest.json --media-root outputs/new-fixtures --data-dir outputs/fixture-catalog
```

The default is dry-run. It validates decoding, hashes and constraints against a
disposable copy of the current catalog, including conflicts between entries in the
same manifest. It leaves the real catalog/media unchanged. It does temporarily
copy decoded/inspected videos into the disposable staging library for validation;
staging is removed on completion. A manifest can contain at most 10000 entries.

After inspecting the dry-run receipt, an operator can explicitly append
`--commit-import` to import. Committed import transactions are per clip; if a later
entry fails, earlier clips remain committed. Reruns deduplicate them. Original files
are never moved, modified or deleted. Copying requires sufficient local disk space.
Do not interpret synthetic inference results or fixture import counts as benchmark
performance. No asset or dataset is downloaded by these commands.

Duplicate byte imports must retain the same license/training/handover permissions;
conflicting rights are rejected for curator resolution rather than silently relaxed.
Imported snapshots are size-bounded during copying and decoded again after copying.
