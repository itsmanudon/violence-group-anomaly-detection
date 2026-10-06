# External-drive UCF-Crime baseline

Installation root: `E:\anomaly-detection-dataset-UCF`. The extracted original
videos are used directly; ZIP files remain untouched. Inventory found all 1,900
author-listed anomaly-detection videos and 50 additional event-recognition Normal
videos, which are excluded from this anomaly baseline. No extra repository copy
is needed. Cross-volume manifests use absolute video paths; changing the drive
letter requires an explicit path update and cache identity verification.

The root `Anomaly_Train.txt` has 1,610 entries and exactly matches the pinned author
file. The nested training list is empty and is not used. The nested
`Anomaly_Test.txt` has 290 entries; its names are present. The supplied temporal
annotations exactly match the pinned author file used to reserve DCSASS sources.

## Prospective scientific settings

The approved C3D FC6 export, RGB scalar means `[104,117,128]`, 16-frame units,
32 normalized segments and existing Sultani scorer/MIL objective remain fixed.
This is a modern compatible baseline, not an exact original Caffe reproduction.
`configs/experiments/ucf_sultani_shared_safe_v1.yaml` fixes seed 0 and the existing
20-epoch Adagrad configuration. Validation bag ROC-AUC selects the checkpoint;
threshold remains 0.5. Test predictions do not select or modify either setting.

All existing shared DCSASS source memberships are protected. Sources already
assigned to DCSASS validation remain validation. DCSASS test sources found in
the authors' training list are excluded from UCF optimization and remain on disk.
Unshared author-training sources get a deterministic category-aware 85/15
train/validation split. The original 290 author-test videos stay test. This
training-population deviation must be stated alongside any UCF metric.

The audit probes metadata, decodes the first and last reported frames and hashes
every included video. Full sequential decode and frame-count consistency are
verified by C3D extraction before training. Metadata sampling alone is not called
full decode verification. Exact content duplicates or unreadable files block
protocol freezing until diagnosed.

Frame evaluation uses `c3d_units` projection, matching the actual 16-frame unit
partition boundaries rather than assuming all 32 bins have exactly equal duration.
For short videos, repeated scores over the same unit are averaged. Annotation
indices retain the existing one-based-inclusive to zero-based-half-open conversion.

Initial annotation validation found five end overruns: Arson011/Arson016/Fighting003
by one frame and Explosion033/Shooting015 by two. All five videos fully decode to
their reported counts and match original archive sizes/CRCs. The new protocol
explicitly intersects these verified-original intervals with available frames and
records original/effective endpoints. No existing-frame label changes; raw files
and annotation text remain untouched. Larger overruns or starts beyond the video
remain errors. The existing generic annotation API stays strict by default.
Evidence: `runs/ucf-crime/annotation-boundary-diagnostic.json` and
`annotation-boundary-integrity.json`, with adjustments included in audit/protocol.

## Execution gates

1. Full software gates, then read-only external dataset audit and source freeze.
2. Bounded real train/validation C3D extraction; five MIL steps and checkpoint reload.
3. Bounded representative GPU timing, then resumable full extraction with identities.
4. Seed-0 training, completed validation-only checkpoint freeze, one held-out frame pass.
5. Separate UCF evidence, cascade checks and deliberately versioned deployment receipt.

Actual counts, measurements, limitations and commands will be appended after
each gate. Existing DCSASS reports/checkpoints/caches are preserved.

UCF feature manifests embed each NPY's SHA256. The shared training/validation/test
loader verifies those bytes, so a finite changed bag cannot pass on shape alone.
Selection receipts bind the prospective protocol and its frame projection;
registered evaluation resolves that frozen projection and rejects drift before
creating any held-out output. Legacy DCSASS manifests remain readable unchanged.

## Completed audit and frozen population

All 1,900 author-listed anomaly-detection videos passed metadata and first/last
frame checks, with complete byte fingerprints. Audit took 525.60 seconds. Actual
resolutions are 320x240; FPS values are 25, 29.970029 and 30, preserved as supplied.
The population contains 13,768,423 reported frames / 127.51 hours. Duration
minimum/median/95th percentile/maximum: 3.47 / 71.07 / 712.60 / 32,550.10 seconds.
Full sequential decode remains a required extraction gate.

Hashing found 21 exact-content duplicate groups. Eight Normal train/test pairs
would leak test content; eleven Normal train/train pairs and the binary-positive
Assault050/Robbery138 alias are also redundant. All 42 duplicate-group videos match
their original ZIP CRCs/sizes. These are supplied duplicates, not extraction damage.
Both anomaly aliases are absent from the frozen DCSASS population.

Excluded 20 redundant author-training copies and 27 sources held out in DCSASS.
No raw file was removed and no binary label changed. All 290 original test entries
remain, including the Normal_Videos_936/937 duplicate pair: 289 unique contents.
The prospective split has zero source or exact-content overlap across partitions.

| Split | Video/source IDs | Unique contents | Normal | Anomaly |
|---|---:|---:|---:|---:|
| Train | 1,309 | 1,309 | 664 | 645 |
| Validation | 254 | 254 | 117 | 137 |
| Test | 290 | 289 | 150 | 140 |

Manifest SHA256: `13a071d986ef4c4d62691a11b16b3052955b9b98bcbd2019b52f579437b62826`.
Audit SHA256: `03aff4b4bfc8109affc98758429d0e726ce5ce9b723240102edbf718dfea9a77`.
Receipt: `runs/ucf-crime/sultani_shared_safe_v1/protocol.json`. The 290-entry frame
evaluation will explicitly disclose duplicate weighting and the changed training
population; it is not an independent 290-source or paper-exact reproduction claim.

The first audit deliberately blocked freezing upon discovering duplicates. After
diagnosis, `scripts/freeze_ucf_protocol.py` completed the explicit exclusions from
that saved audit, preserving its historical evidence without rehashing all videos.

## Real bounded preflight and extraction benchmark

Eight short train/validation videos fully decoded and extracted in 3.8975 seconds.
Five real MIL steps produced finite losses 1.97010 / 1.86905 / 1.78078 / 1.74711 /
1.67920, finite gradients and an exactly equal checkpoint reload. No test scoring
occurred; the resulting checkpoint remains explicitly dry-run.

RTX 4070 Laptop, batch 4, unchanged float32 backbone: the training frame-count
median sample (2,189 frames) took 3.0172 seconds; the 95th-percentile sample (24,320
frames) took 33.2855 seconds. Peak allocated CUDA memory was 678,444,544 bytes.
The latter comprised 8.5797 seconds decoding, 10.5067 preprocessing and 13.1275
C3D. Both valid bags seed the main cache. This supports a bounded streaming full
run on the available hardware; total extraction remains a multi-hour job.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_ucf_surveillance.py --root E:\anomaly-detection-dataset-UCF
# A supplied duplicate finding blocks automatic freeze; inspect audit then:
.\.venv\Scripts\python.exe scripts/freeze_ucf_protocol.py
.\.venv\Scripts\python.exe scripts/extract_ucf_sultani.py --preflight
.\.venv\Scripts\python.exe scripts/preflight_sultani.py --config configs/experiments/ucf_sultani_shared_safe_v1.yaml --manifest runs/ucf-crime/sultani_shared_safe_v1/preflight_cache/features.jsonl --output runs/ucf-crime/sultani_shared_safe_v1/preflight_training
.\.venv\Scripts\python.exe scripts/extract_ucf_sultani.py
```

Use existing frozen receipts rather than overwriting them. The full cache and
feature manifest are in the repository's ignored run/data directories on D:;
original videos stay on E:. The extraction process revalidates completed entries
when resumed, rejects stale bytes/provenance and verifies full decoded frame counts.
