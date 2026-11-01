# Dataset adapter and manifest contract v1

No dataset/video is downloaded or imported automatically. Adapters produce JSON
for `Manifest` / `ClipSpec` in `municipal/catalog.py`; they do not call training
code or alter frozen models. Unknown fields fail validation. Existing library
identity and membership cannot be edited by reimporting a manifest.

Place already-authorized media inside the configured import root. The manifest
contains relative paths beneath that root; no absolute paths, `..`, Windows
backslash paths or escaping symlinks. Forward slashes support nested folders on
Windows/Mac. Supported containers: MP4, MOV, M4V, AVI, MKV, WebM. OpenCV must decode
every reported frame; fps, duration, resolution and frame count are measured at
import. A supported extension alone is insufficient. Browser codec support varies;
H.264 MP4 is preferred for playback, but no silent transcoding changes the bytes.
A codec failure returns a useful validation error rather than a prediction.

```json
{
  "schema_version": 1,
  "clips": [{
    "clip_id": "municipal-source01-clip01",
    "source_video_id": "original-recording-01",
    "file_path": "authorized/source01-clip01.mp4",
    "sha256": "REPLACE_WITH_64_LOWERCASE_HEX_DIGITS",
    "source": "Publisher, recording identity and acquisition evidence",
    "location": null,
    "location_confidence": "unknown",
    "geographic_evidence": "",
    "license": "Record actual terms and evidence; unknown is not permission",
    "training_permission": false,
    "municipal_redistribution_permission": false,
    "assignment": "unassigned",
    "population": "curated_demo",
    "selection_method": "Record who selected clips and how, before model viewing",
    "training_overlap": "unknown",
    "evaluation_eligibility": "",
    "annotation_status": "unlabeled",
    "notes": "No verified event labels supplied"
  }]
}
```

The placeholder SHA deliberately fails validation; compute it from the actual file
with `Get-FileHash -Algorithm SHA256` or `shasum -a 256`. Do not guess hashes.

Required: clip_id (1–100 ASCII letters/numbers/underscore/hyphen), source_video_id,
file_path, SHA256, source, license, selection_method. Other fields have conservative
unknown/false defaults. Persisted metadata adds duration seconds, fps, resolution
[width,height], frame count, import time and availability/deletion status.

`source_video_id` means the original recording, not each derived clip or filename.
All clips from one recording must retain the same assignment; evaluation and
non-evaluation membership cannot mix within it. Exact bytes deduplicate by SHA256;
conflicting clip identity, recording, assignment or population is rejected. Near
duplicates/alternate encodes still need a dataset curator's source/content audit;
SHA256 alone cannot discover these. Provenance changes require an explicitly
reviewed future migration; reimport does not mutate an existing record.

Assignments: train/validation/test/demo/unassigned. Populations:
evaluation/curated_demo/upload/synthetic. Protected evaluation requires `test`,
`training_overlap="excluded"`, and nonempty `evaluation_eligibility` documenting
checks against training/validation/tuning and related source/duplicate recordings.
Known overlap or unknown overlap is not eligible. Keep selection and membership
fixed before any model inference; no test-driven tuning or post-hoc benchmark
selection. A curator declaration is evidence to audit, not an automated guarantee.
The platform never trains even when training_permission is true.

Geography: unknown, inferred_appearance, documented. Appearance is never proof of
Indian location. Documented requires geographic_evidence (recording record,
publisher statement, source metadata, etc.). Keep the actual location unknown
unless legitimately established. Record rights for training separately from
municipal redistribution. External exports exclude unknown/false redistribution
permission, including cached/failure/review records for that clip. Review notes
can still contain private information; inspect an export before handover.

Annotation status is unlabeled/pending at manifest ingestion. `verified` is rejected
there: references enter separately through an audited local command. References
are optional; unlabeled does not mean Normal or nonviolent. To install one:

```json
{
  "class_label": "Fighting",
  "violence": true,
  "events": [[1.2, 2.4]],
  "temporal_complete": true
}
```

Each reference field may be omitted/unknown except temporal_complete defaults
false. Class labels use the approved six-class vocabulary. `violence` must be an
independently supplied boolean; do not derive it from unknown footage or a review.
`events` are verified observable event intervals in seconds within duration.
`temporal_complete=true` attests that the entire clip was annotated for those
anomaly-worthy events; an empty list then legitimately denotes no such events.
Without that completeness attestation no temporal false-alert metrics are computed.
Do not encode criminal intent, guilt or person identities.

```bash
python -m surveillance.municipal --import-root /authorized/local/root import manifest.json
python -m surveillance.municipal --import-root /authorized/local/root import manifest.json --commit
python -m surveillance.municipal reference municipal-source01-clip01 reference.json --author curator --reason "Independent annotation verified" --expected-version 0
```

Dry run is the default and rolls back all catalog row changes; the app may create
its empty local database/directories. Commit is transactional across the manifest.
A repeated identical import reports duplicates. Reference v2 requires expected
version 1 and a new author/reason; prior payloads cannot be overwritten. Local
operators can review predictions but cannot edit references through the UI.

Synthetic-only readiness exercise, separate from real clips:

```bash
python scripts/municipal_fixtures.py --output outputs/import-fixtures-v1 --count 100
python -m surveillance.municipal --data-dir outputs/fixture-catalog --import-root outputs/import-fixtures-v1 import outputs/import-fixtures-v1/manifest.json
python -m surveillance.municipal --data-dir outputs/fixture-catalog --import-root outputs/import-fixtures-v1 import outputs/import-fixtures-v1/manifest.json --commit
```

These deterministic images contain no real people and are marked synthetic/demo.
Do not change their population to evaluation or describe fixture results as model
performance. When Indian datasets arrive: audit rights/geography and original
recording groups; establish protected membership independently; annotate/verify;
produce this manifest; dry-run and review discrepancies; commit explicitly. Do not
silently merge with old benchmarks or overwrite any historical run.
