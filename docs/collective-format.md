# Collective actor scenes

## Verified sources and protocol

The [Actor-Transformers paper, sections 4.1–4.2](https://arxiv.org/html/2003.12737)
describes 44 videos, a 32/12 video split, five activities, majority actor activity
as the group target, and ten frames at offsets -5 through +4 around the annotation.
This implementation uses those ten offsets, edge replication, and 480 × 720 RGB
frames. All clips from a source stay in one split.

Raw layout was checked against the related method authors'
[Collective loader](https://github.com/wjchaoGit/Group-Activity-Recognition/blob/master/collective.py).
It reads `seqNN/annotations.txt`, using the first six whitespace columns as
`frame x y width height action`; trailing columns are unused metadata. Images are
`seqNN/frameNNNN.jpg`. Frame IDs start at 1. The loader selects IDs 1, 11, 21, ….
Raw action 1 is NA; 2–6 are crossing, waiting, queueing, walking, talking.
Coordinates use the author's direct `x/W`, `y/H`, `(x+w)/W`, `(y+h)/H` conversion;
no origin subtraction is added. Our manifest orders coordinates **xyxy**, while
the reference loader internally orders **yxyx**. Actual decoded source dimensions
are used, including sequences whose frames differ from 480 × 720.

The [authors' configuration](https://github.com/wjchaoGit/Group-Activity-Recognition/blob/master/config.py)
specifies test IDs **5, 6, 7, 8, 9, 10, 11, 15, 16, 25, 28, 29**; training IDs are
the remainder of 1–44. This verified related-method 32/12 split is our default.
The Actor-Transformers paper does not enumerate IDs, so exact agreement with its
unreleased split is not asserted. Explicit sequence lists can override the default;
32/12 counts and disjointness remain required unless `--allow-subset` is supplied.

Implementation decisions: NA actors are excluded from the five-class task; an
all-NA scene fails preparation. A majority tie chooses the smallest class ID.
Boxes must have positive area inside the source image; malformed boxes fail rather
than being silently clipped. The annotation center is used for static pose at
train and test time. Center boxes also identify RGB RoIs. No actor tracking or
box interpolation is claimed. Unlike the related loader's forward-only clips,
our temporal windows are centered and clamped to the first/last existing frame.
These choices prevent a claim of exact reproduction before real-data validation.

## Prepare installed data

```powershell
python scripts/prepare_collective.py --root data/raw/collective --output data/manifests/collective.jsonl
# Optional explicit validation sources, held out only from the training list:
python scripts/prepare_collective.py --root data/raw/collective --output data/manifests/collective.jsonl --val-sequences 1 2 3
```

No downloads occur. The source tree must contain consecutive frames beginning at
1. All frame paths written by preparation are absolute. Manual manifests may use
paths relative to their JSONL file. Validation changes the training pool from 32
to 32 minus the holdout count; it is an explicit development protocol, not the
paper's complete training split. Never select checkpoints using the test split.

## ActorRecord JSONL

One line describes one center-annotated scene. This is separate from Milestone 1's
binary anomaly `Record` schema. Required fields:

| Field | Meaning |
|---|---|
| `dataset`, `video_id`, `source_video_id`, `clip_id` | Nonempty identities; `source_video_id` links all clips from one source |
| `split` | `train`, `val`, or `test` |
| `frame_paths` | Ten image paths in temporal order |
| `frame_indices` | Ten nonnegative source indices; repeated edge indices allowed; index 5 is the center |
| `actor_boxes` | N nonempty, finite normalized `[x1,y1,x2,y2]` boxes |
| `actor_labels` | N nonnegative integer class IDs |
| `group_label` | Nonnegative integer class ID |
| `pose_feature_path`, `rgb_feature_path` | Optional `.npy` files, default null |

Collective targets map crossing/waiting/queueing/walking/talking to 0/1/2/3/4.
The generic record and batch accept other nonnegative IDs for future datasets;
the model/trainer must validate IDs against configured class counts. Readers and
writers reject duplicate clip IDs, duplicate source/center pairs, source/video
identities spanning splits, and reused physical frame/feature paths spanning splits.
Overlapping temporal windows inside one split are expected. Renaming files or
source identities cannot be reliably detected as copied content; provenance still
depends on honest source IDs.

`ActorFeatureDataset(manifest, split, mode='pose_only', pose_feature_dim=98304,
rgb_feature_dim=20800, input_mode='precomputed', image_size=(480,720))` loads
numeric floating-point `.npy` arrays with exact `[N,D]` shapes and finite values.
Rows must follow the manifest's actor order; no automatic feature extraction,
reshaping, or actor matching occurs. Modes are `pose_only`, `rgb_only`,
`pose_rgb_early_fusion`, `pose_rgb_late_fusion`. `split=None` selects all scenes
after validating the complete manifest. `input_mode='raw'` instead loads float RGB
frames `[10,3,H,W]` in [0,1]; local backbone preprocessing happens downstream.

`collate_actors` produces `[B,Nmax,4]` boxes, `[B,Nmax]` valid masks (True means
valid), actor labels padded with -100, group labels `[B]`, optional features
`[B,Nmax,D]`, optional frames `[B,10,3,H,W]`, and a metadata list. Feature and box
padding is zero. Empty scenes and mismatched actor counts/modalities fail.
