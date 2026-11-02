# Repository audit — 23 October 2026

Read-only inspection completed before implementation. Baseline commit:
`4d7c27ad0d7412752307a37eff273db58f93386b`. Initial `git status --short`
was empty. No remote mutations, footage downloads, model training, or Indian-data
evaluation were performed for this audit. An initial unrestricted `rg` encountered
an unreadable historical `.pytest_cache`; subsequent inventories excluded it.

## Grounded inventory

| Area | Existing implementation and evidence | Classification / action |
| --- | --- | --- |
| Layout | `src/surveillance/{datasets,features,models,training,experiments,evaluation,inference,detection,municipal,video}`; `scripts`, `configs`, `tests`, `docs`, installed local `checkpoints`, `runs`, `data/manifests` | Already implemented; extend in place |
| C3D | `features/c3d.py`: 8-convolution FC6, 4,096 dimensions; strict local state loading, 16-frame clips, 32-segment aggregation; `c3d_conversion.py`, `video/transforms.py` | Already implemented; preserve pinned RGB mean [104,117,128] |
| Sultani | `models/sultani/model.py`: 4096→512→32→1, sigmoid; `loss.py`: MIL ranking, sparsity and smoothness; `training/sultani_trainer.py` supports deterministic resume | Reusable; anomaly is not a violence annotation |
| Person detector | `detection/torchvision_detector.py`: local COCO Faster R-CNN ResNet50 FPN, strict versioned FPN/RPN key migrations and FrozenBatchNorm; filtering and matching in adjacent modules | Already implemented; coverage must be measured |
| I3D | `features/i3d.py`, `actor_backbones.py`: local TorchScript Mixed_4f, 832 channels, ten RGB frames, temporal mean, resize 90×160, 5×5 RoIs → 20,800 features/actor | Reusable frozen; inference-only exports may prohibit backbone tuning |
| Actor-Transformer | `models/actor_transformer/actor_transformer.py`: projections, spatial position, masked encoder, masked max, actor/group heads; four feature modes | Reusable with modification; binary group-only head must be separate |
| Six-class transfer | `surveillance_transfer.py` and `training/dcsass_behavior.py`: fixed six-class loss, strict `dcsass_group_behavior` loader, covered-only training and explicit abstention | Preserve; do not feed binary labels to fixed six-class loss |
| Training CLIs | `scripts/train_sultani.py`, `train_actor_transformer.py`, `train_dcsass_behavior.py`; orchestration `run_collective_experiment.py`, `run_sultani_dcsass.py`; smoke CLIs `smoke_actor_transformer.py`, `smoke_collective_experiment.py`, `smoke_detection.py` | Already implemented; Indian frozen-feature training needs its own guarded entry point |
| Manifests | `datasets/common.py` `Record`: dataset/video/source/path/split/binary label, metadata, nullable temporal intervals, feature path/hash. `read_manifest` rejects duplicate IDs. `check_leakage` checks sources across datasets. `split_records` is deterministic but unstratified | Reuse final JSONL schema; unsplit annotation staging needs a sidecar because `Record` only permits train/val/test |
| Existing datasets | Local UCF binary manifests; DCSASS audited custom JSONL with SHA256/source/binary/group labels; Collective actor manifests; UCF feature manifests | Do not merge populations. Build an explicit benchmark exclusion inventory from these heterogeneous schemas |
| Source/content protocols | `datasets/dcsass_protocol.py`, `experiments/sultani_ucf.py`: source reservation, duplicate-content exclusion; `features/provenance.py` binds actor extraction/cache settings | Reusable patterns; no general Indian provenance or near-duplicate review workflow exists |
| Evaluation | `evaluation/anomaly_metrics.py`: ROC, precision/recall/F1, confusion/FPR, frame mapping; `group_activity_metrics.py`, `cascade.py`, detection metrics; `scripts/evaluate_{sultani,registered_sultani,actor_transformer,dcsass_behavior,surveillance_cascade,detector}.py` | Reusable; binary violence PR-AUC, validation recall-constrained calibration and temporal ablations need explicit population contracts |
| Registration | `experiments/protocol.py`: canonical/file hashes and immutable freeze receipts; `environment.py`: allowlisted software/GPU/git metadata; `reporting.py`, `cascade_registration.py`, `behavior_selection.py`, `sultani_selection.py` | Reuse hashing/environment; current protocols are dataset-specific |
| UI | `inference/surveillance_pipeline.py` routes top anomaly windows, preserves Normal disagreement/no-actor generic alerts; `municipal` supports human review and protected references | Already implemented; new binary research path must not silently change deployment |
| Tests | 60+ modules, including manifests, MIL, actor shapes/masking/resume, feature provenance, six-class loading/selection, source/content protocols, metrics, cascade, municipal/API | Already implemented; add focused tests and run relevant regression suite |
| Execution | `pyproject.toml`: Python ≥3.11, torch ≥2.2, torchvision ≥0.17, NumPy/OpenCV/sklearn/YAML/TensorBoard; optional dev/demo/municipal extras; existing `.venv`; Windows/macOS lock lists in `requirements` | GPU/environment validation pending at audit time; avoid reinstalling known working CUDA stack |

## Available deployment assets

These files exist and SHA256 was **computed from local bytes during this audit**,
matching `configs/surveillance_demo_ucf_v1.yaml`. Hash agreement is not inference
or accuracy validation.

| Asset | Path | SHA256 |
| --- | --- | --- |
| C3D | `checkpoints/c3d_fc6_openmmlab_v1.pt` | `beac4ff065de3663bf5c6bab36aedca4fe56ec59b231c66bf5374c23eda14831` |
| Sultani | `runs/ucf-crime/sultani_shared_safe_v1/seed_0/best.pt` | `7e5b548ea6260614209a727a947e6270a1a1af2185d29accf1778f91b4af7c66` |
| Behavior | `runs/dcsass/human_rgb_detected_v1/random_init_seed_0/best.pt` | `7870e512787e74cd76263df28e40c8bc39ecb8a3d0a5b2b8b6a7e72a218a8c69` |
| Faster R-CNN | `checkpoints/external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` | `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384` |
| I3D | `checkpoints/i3d_mixed4f_collective_rgb_v1.pt` | `fe7fc30ca6f26430232e2e0f6bdffd8514478a071db065e452bff46f60f4e0c4` |

Other installed assets: three HRNet-W32 feature exports, original Sports-1M C3D,
original HRNet, upstream I3D RGB ImageNet weights/source/acquisition and license
records, and Collective training runs. Their presence does not establish Indian
domain suitability. Inspect export metadata before selective backbone fine-tuning.

## Missing, blocked, and research-dependent work

**Missing:** header-inspecting CSV importer, uncertain quarantine, explicit
source/incident joins, near-duplicate candidate review, Indian benchmark-overlap
review, protected Indian split receipt, binary training registration and handoff.

**Blocked by unavailable data:** real CSV header/label review, media readability,
actual clip counts (536 is supplied approximate context), source completeness,
rights/provenance verification, genuine split counts, training, failure analysis,
Indian evaluation, continuous camera-hour and temporal-latency metrics. No Drive
access is assumed. The proposed CSV lacks original source/incident identities;
these must arrive as mapped columns or a separately curated provenance mapping.

**Requiring additional research:** binary definition versus anomaly/behavior labels;
representative Indian sampling and independence; sufficient source support per
class; near-duplicate review beyond exact hashes; acceptable validation recall
floor; optional interaction/aggregation experiment justified by observed errors.

## Compatibility and reproducibility risks

- Six-class loaders/loss/vocabulary are intentionally strict. Add a distinct binary
  checkpoint type; never overwrite or reinterpret the historical heads.
- Original `extract_c3d.py` can overwrite outputs and does not itself bind every
  feature's bytes. New work needs exclusive output directories, hashes and receipts.
- Generic `Record` has no unassigned split or provenance fields. Stage raw import
  separately and emit compatible `Record` JSONL only after split approval checks.
- `probe_video` checks metadata, not full readability; reuse `audit_video` full
  decoding, disclose variable-frame-rate/frame-count limitations.
- Source IDs alone cannot detect renamed/re-encoded footage. Hashes, visual
  candidates and curator evidence complement source/incident grouping.
- OpenCV codecs, torch/torchvision operator compatibility, TorchScript export
  restrictions and CUDA capability must be checked on each machine. H100 access
  and environment are unverified. Frozen cached features minimize memory cost.
- `environment_metadata` records commit but uncommitted code also needs a diff
  hash when registering preparation/training work.

Historical results are documentary context, not rerun here: UCF frame ROC-AUC
0.7441; separate DCSASS cascade clip ROC-AUC 0.5118, F1 0.4733, normal FPR 40.22%,
covered routed behavior macro F1 0.1888. Their denominators must stay separate.
Neither is an Indian-domain measurement or a municipal deployment guarantee.
