"""Collective five-class scenes; original raw columns and protocol in docs/collective-format.md."""

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from surveillance.datasets.common import resolve_path

CLASSES = ("crossing", "waiting", "queueing", "walking", "talking")
# Published author code: wjchaoGit/Group-Activity-Recognition, config.py.
TEST_SEQUENCES = (5, 6, 7, 8, 9, 10, 11, 15, 16, 25, 28, 29)
TRAIN_SEQUENCES = tuple(s for s in range(1, 45) if s not in TEST_SEQUENCES)


@dataclass(frozen=True)
class ActorRecord:
    """A center-annotated scene; paths resolve relative to the JSONL file."""

    dataset: str
    video_id: str
    source_video_id: str
    clip_id: str
    split: str
    frame_paths: list[str]
    frame_indices: list[int]
    actor_boxes: list[list[float]]
    actor_labels: list[int]
    group_label: int
    pose_feature_path: str | None = None
    rgb_feature_path: str | None = None

    def __post_init__(self) -> None:
        for value in (self.dataset, self.video_id, self.source_video_id, self.clip_id):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("dataset/video/source/clip identifiers must be nonempty strings")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        if len(self.frame_paths) != 10 or len(self.frame_indices) != 10:
            raise ValueError("A scene requires ten frame paths and frame indices")
        if not all(isinstance(p, str) and p for p in self.frame_paths):
            raise ValueError("frame paths must be nonempty strings")
        if not all(type(i) is int and i >= 0 for i in self.frame_indices):
            raise ValueError("frame indices must be nonnegative integers")
        if self.frame_indices != sorted(self.frame_indices):
            raise ValueError("frame indices must be ordered")
        if not self.actor_boxes or len(self.actor_boxes) != len(self.actor_labels):
            raise ValueError("A scene needs matching nonempty actor boxes and labels")
        for box in self.actor_boxes:
            if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
                raise ValueError("Actor boxes must be finite normalized xyxy")
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("Actor boxes must have positive width and height")
        for label in [*self.actor_labels, self.group_label]:
            if type(label) is not int or label < 0:
                raise ValueError("Actor and group labels must be nonnegative integers")
        for value in (self.pose_feature_path, self.rgb_feature_path):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("Feature paths must be nonempty strings or null")


def _validate_manifest(records: list[ActorRecord], path: Path) -> None:
    if not records:
        raise ValueError("Actor manifest is empty")
    clips, centers = set(), set()
    sources, videos, physical = {}, {}, {}
    for row in records:
        key = (row.dataset, row.clip_id)
        center = (row.source_video_id, row.frame_indices[5])
        if key in clips:
            raise ValueError(f"duplicate clip: {key}")
        clips.add(key)
        for mapping, identity in (
            (sources, row.source_video_id),
            (videos, (row.dataset, row.video_id)),
        ):
            if identity in mapping and mapping[identity] != row.split:
                raise ValueError(f"Source-video leakage across splits: {identity}")
            mapping[identity] = row.split
        if center in centers:
            raise ValueError(f"duplicate source/center frame: {center}")
        centers.add(center)
        for value in [*row.frame_paths, row.pose_feature_path, row.rgb_feature_path]:
            if value is None:
                continue
            identity = str(resolve_path(value, path).resolve()).casefold()
            if identity in physical and physical[identity] != row.split:
                raise ValueError(f"Physical-path leakage across splits: {value}")
            physical[identity] = row.split


def read_actor_manifest(path: Path) -> list[ActorRecord]:
    """Read and validate all splits before a consumer filters its requested split."""
    path = Path(path)
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(ActorRecord(**json.loads(line)))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path}: invalid actor manifest line {number}: {error}") from error
    _validate_manifest(rows, path)
    return rows


def write_actor_manifest(records: list[ActorRecord], path: Path) -> None:
    path = Path(path)
    _validate_manifest(records, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(r), allow_nan=False) + "\n" for r in records), encoding="utf-8"
    )


def temporal_frame_indices(center: int, num_frames: int) -> list[int]:
    """One-based source frames, edge replication at either end, center at position 5."""
    if not 1 <= center <= num_frames:
        raise ValueError("Center must lie within the source video")
    return [max(1, min(num_frames, center + offset)) for offset in range(-5, 5)]


def parse_collective_annotations(path: Path, image_size: tuple[int, int]) -> dict[int, dict]:
    """Read frame,x,y,w,h,action[,unused metadata]; raw 1=NA, raw 2..6 become 0..4.

    Select source frames 1,11,21,... as the published related author loader does.
    Drop NA actors; fail if a selected scene has no supervised actors. Ties choose
    the smallest class ID deterministically. No tracking/interpolation is claimed.
    """
    height, width = image_size
    if min(height, width) <= 0:
        raise ValueError("Image dimensions must be positive")
    scenes = {}
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = line.split()
            if len(values) < 6:
                raise ValueError("expected at least six columns: frame x y width height action")
            frame, x, y, w, h, action = (int(v) for v in values[:6])
            if frame < 1 or action not in range(1, 7):
                raise ValueError("frame must be positive and raw action in 1..6")
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                raise ValueError("box must have positive area and lie inside the source image")
            identity = (frame, x, y, w, h)
            if identity in seen:
                raise ValueError("duplicate actor box in a frame")
            seen.add(identity)
            if frame % 10 != 1:
                continue
            scene = scenes.setdefault(frame, {"actor_boxes": [], "actor_labels": []})
            if action == 1:
                continue
            scene["actor_boxes"].append([x / width, y / height, (x + w) / width, (y + h) / height])
            scene["actor_labels"].append(action - 2)
        except ValueError as error:
            raise ValueError(f"{path}: annotation line {number}: {error}") from error
    if not scenes:
        raise ValueError(f"{path}: no annotated scenes")
    for frame, scene in scenes.items():
        counts = Counter(scene["actor_labels"])
        if not counts:
            raise ValueError(f"{path}: frame {frame} has no actors with five-class labels")
        scene["group_label"] = min(counts, key=lambda label: (-counts[label], label))
    return scenes


def prepare_collective(
    root: Path,
    train_sequences: list[int] | tuple[int, ...] = TRAIN_SEQUENCES,
    test_sequences: list[int] | tuple[int, ...] = TEST_SEQUENCES,
    require_full_split: bool = True,
) -> list[ActorRecord]:
    """Prepare locally installed seqNN/annotations.txt and frameNNNN.jpg files."""
    train, test = list(train_sequences), list(test_sequences)
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise ValueError("Sequence splits contain duplicates or source leakage")
    if (
        not train
        or not test
        or any(type(s) is not int or s not in range(1, 45) for s in train + test)
    ):
        raise ValueError("Both splits must contain sequence IDs in 1..44")
    if require_full_split and (len(train), len(test)) != (32, 12):
        raise ValueError("Collective protocol requires 32/12 train/test source videos")
    root = Path(root).resolve()
    records = []
    for sid in sorted(train + test):
        folder = root / f"seq{sid:02d}"
        frames = {}
        for path in folder.glob("frame*.jpg"):
            suffix = path.stem.removeprefix("frame")
            if suffix.isdigit():
                index = int(suffix)
                if index in frames:
                    raise ValueError(f"Duplicate frame index {index} in {folder}")
                frames[index] = path
        if not frames:
            raise FileNotFoundError(f"No frameNNNN.jpg files in {folder}")
        if sorted(frames) != list(range(1, max(frames) + 1)):
            raise ValueError(f"Missing frames: {folder} must contain consecutive frames from 1")
        image = cv2.imread(str(frames[1]))
        if image is None:
            raise ValueError(f"Cannot decode {frames[1]}")
        scenes = parse_collective_annotations(folder / "annotations.txt", image.shape[:2])
        for center, scene in sorted(scenes.items()):
            indices = temporal_frame_indices(center, len(frames))
            records.append(
                ActorRecord(
                    dataset="collective",
                    video_id=folder.name,
                    source_video_id=f"collective:{folder.name}",
                    clip_id=f"{folder.name}:{center:04d}",
                    split="train" if sid in train else "test",
                    frame_paths=[str(frames[i]) for i in indices],
                    frame_indices=indices,
                    **scene,
                )
            )
    _validate_manifest(records, root / "manifest.jsonl")
    return records


class ActorFeatureDataset(Dataset):
    """Read validated actor .npy features or ten resized RGB frames per scene."""

    def __init__(
        self,
        manifest: Path,
        split: str | None,
        mode: str = "pose_only",
        pose_feature_dim: int = 98304,
        rgb_feature_dim: int = 20800,
        input_mode: str = "precomputed",
        image_size: tuple[int, int] = (480, 720),
    ):
        if mode not in {"pose_only", "rgb_only", "pose_rgb_early_fusion", "pose_rgb_late_fusion"}:
            raise ValueError(f"Unknown actor mode: {mode}")
        if input_mode not in {"precomputed", "raw"}:
            raise ValueError("input_mode must be precomputed or raw")
        if min(pose_feature_dim, rgb_feature_dim, *image_size) <= 0 or len(image_size) != 2:
            raise ValueError("Feature and image dimensions must be positive")
        self.manifest = Path(manifest)
        self.records = [
            r for r in read_actor_manifest(self.manifest) if split is None or r.split == split
        ]
        if not self.records:
            raise ValueError(f"No actor scenes in split {split}")
        self.mode, self.input_mode, self.image_size = mode, input_mode, image_size
        self.pose_feature_dim, self.rgb_feature_dim = pose_feature_dim, rgb_feature_dim

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        sample = {
            "actor_boxes": torch.tensor(row.actor_boxes, dtype=torch.float32),
            "actor_labels": torch.tensor(row.actor_labels, dtype=torch.long),
            "group_label": row.group_label,
            "metadata": {
                "dataset": row.dataset,
                "video_id": row.video_id,
                "source_video_id": row.source_video_id,
                "clip_id": row.clip_id,
                "split": row.split,
                "frame_indices": row.frame_indices,
            },
        }
        if self.input_mode == "raw":
            frames = []
            height, width = self.image_size
            for value in row.frame_paths:
                path = resolve_path(value, self.manifest)
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Missing or undecodable frame: {path}")
                image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(image).permute(2, 0, 1).float() / 255)
            sample["frames"] = torch.stack(frames)
        else:
            modalities = []
            if self.mode != "rgb_only":
                modalities.append(("pose", self.pose_feature_dim))
            if self.mode != "pose_only":
                modalities.append(("rgb", self.rgb_feature_dim))
            for modality, dim in modalities:
                value = getattr(row, f"{modality}_feature_path")
                if value is None:
                    raise ValueError(f"{row.clip_id}: {modality}_feature_path is required")
                path = resolve_path(value, self.manifest)
                if path.suffix.lower() != ".npy":
                    raise ValueError("Actor features must be numeric .npy arrays")
                array = np.load(path, allow_pickle=False)
                if (
                    array.shape != (len(row.actor_labels), dim)
                    or not np.issubdtype(array.dtype, np.floating)
                    or not np.isfinite(array).all()
                ):
                    raise ValueError(
                        f"{path}: expected finite float [{len(row.actor_labels)},{dim}]"
                    )
                features = torch.from_numpy(array.astype(np.float32))
                if not torch.isfinite(features).all():
                    raise ValueError(f"{path}: features overflow float32")
                sample[f"{modality}_features"] = features
        return sample
