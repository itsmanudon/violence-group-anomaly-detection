"""Indian pilot: eligibility, collection tags and reconstructed source groups.

Team labels are unadjudicated and provenance is unresolved. Everything here runs
under the explicit PILOT_OVERRIDE and is pilot evidence, never a source-approved
split. Groups are reconstructed heuristically: when unsure, clips are merged,
because over-grouping only costs split flexibility while under-grouping leaks.
"""

import csv
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.fft import dct

from surveillance.datasets.indian_split import source_groups

PILOT_OVERRIDE = "team_labels_unadjudicated"
DISPOSITIONS = (
    "violence-group-anomaly-detection/runs/indian-domain/"
    "readiness-dispositions-20261025-v1/dispositions.csv"
)
INVENTORY = "_manifests/v1-20261021/master-clip-inventory.csv"
LABELS = {"non_violence": 0, "violence": 1}


def _rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def local_path(root: Path, windows_path: str) -> Path:
    """Map a Legion D:\\violence-datasets\\... path onto the local dataset root."""
    relative = windows_path.replace("\\", "/").split("violence-datasets/", 1)
    if len(relative) != 2 or ".." in relative[1].split("/"):
        raise ValueError(f"Unexpected dataset path: {windows_path}")
    return Path(root) / relative[1]


def pilot_candidates(root: Path) -> tuple[list[dict], dict[str, str]]:
    """Clips labelled violence/non_violence that are not quality holds.

    Returns (candidates, excluded) where excluded maps clip_id to a written reason.
    Readability is checked later by decoding; holds keep their recorded blockers.
    """
    root = Path(root)
    inventory = {row["clip_id"]: row for row in _rows(root / INVENTORY)}
    candidates, excluded = [], {}
    for row in _rows(root / DISPOSITIONS):
        key = row["clip_id"]
        if row["quality_disposition"] == "hold":
            excluded[key] = f"quality hold (label={row['label'] or 'blank'})"
            continue
        if row["label"] not in LABELS:
            excluded[key] = f"label not binary: {row['label'] or 'blank'}"
            continue
        item = inventory[key]
        candidates.append(
            {
                "clip_id": key,
                "path": local_path(root, row["resolved_local_path"]),
                "sha256": item["sha256"],
                "label": LABELS[row["label"]],
                "collection": row["dataset_name"],
                "geography": "documented_indian"
                if item["geographic_origin"] == "India"
                else "unknown_geography",
                "duration_seconds": float(item["duration_seconds"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
            }
        )
    for key, row in inventory.items():
        if key not in excluded and key not in {c["clip_id"] for c in candidates}:
            excluded[key] = "not annotated (audio-only item without a video stream)"
    return sorted(candidates, key=lambda c: c["clip_id"]), excluded


def phash(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash (the standard imagehash.phash recipe)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    low = dct(dct(small, axis=0), axis=1)[:8, :8]
    bits = (low > np.median(low)).ravel()
    return int(np.packbits(bits).view(">u8")[0])


def fingerprint(path: Path, per_second: float = 1.0, min_std: float = 8.0, thumbs: int = 3):
    """Decode every frame once; hash ~per_second frames and keep evenly spaced thumbnails.

    Near-uniform frames (black/blank, gray std < min_std) are not hashed: they match
    across unrelated clips and would create false links.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        step = max(1, round(fps / per_second)) if fps > 0 else 30
        thumb_at = {int(reported * (i + 0.5) / thumbs) for i in range(thumbs)} if reported else {0}
        hashes, times, images, index = [], [], [], 0
        while capture.grab():
            want_hash, want_thumb = index % step == 0, index in thumb_at
            if want_hash or want_thumb:
                ok, frame = capture.retrieve()
                if ok:
                    if want_thumb:
                        scale = 240 / frame.shape[0]
                        images.append(cv2.resize(frame, None, fx=scale, fy=scale))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if want_hash and gray.std() >= min_std:
                        hashes.append(phash(gray))
                        times.append(index / fps if fps > 0 else float(index))
            index += 1
    finally:
        capture.release()
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")
    return {
        "decoded_frames": index,
        "reported_frames": reported,
        "fps": fps,
        "codec": codec,
        "hashes": np.array(hashes, dtype=np.uint64),
        "times": times,
        "thumbnails": images,
    }


def frame_matches(fingerprints: dict[str, dict], threshold: int) -> dict[tuple, list]:
    """All cross-clip frame pairs with Hamming distance <= threshold, keyed by clip pair."""
    owners, hashes = [], []
    for key, item in fingerprints.items():
        owners.extend([key] * len(item["hashes"]))
        hashes.extend(item["hashes"].tolist())
    owners, hashes = np.array(owners), np.array(hashes, dtype=np.uint64)
    pairs: dict[tuple, list] = {}
    for start in range(0, len(hashes), 1024):
        block = np.bitwise_count(hashes[start : start + 1024, None] ^ hashes[None, :])
        rows, cols = np.nonzero(block <= threshold)
        for row, col in zip(rows + start, cols, strict=True):
            a, b = owners[row], owners[col]
            if a < b:
                pairs.setdefault((a, b), []).append(int(block[row - start, col]))
    return pairs


_STAMP = re.compile(r"(20\d{2})-?(\d{2})-?(\d{2})[ _](\d{2})(\d{2})(\d{2})")


def filename_key(candidate: dict) -> str | None:
    """Shared-recording key from naming conventions observed in each collection.

    Snatch Theft N.mp4 / N_0.mp4 share numbers but visual review showed different
    footage in all five pairs, so numbering is deliberately not treated as a link.
    """
    name = candidate["path"].stem
    if candidate["collection"] == "WalkIndia-50":
        match = re.match(r"(.+-Scene-\d+)-", name)  # chunks/scenes of one walk recording
        return f"walkindia-{match.group(1)}" if match else None
    if name.startswith("krishna-clip-"):
        return "hci-krishna"
    return None


def capture_time(candidate: dict) -> datetime | None:
    match = _STAMP.search(candidate["path"].name)
    return datetime(*map(int, match.groups())) if match else None


def reconstruct_groups(
    candidates: list[dict],
    fingerprints: dict[str, dict],
    *,
    hamming_threshold: int,
    min_frame_matches: int,
    capture_window_seconds: float = 300.0,
    review_merges: Sequence[dict] = (),
):
    """Union-find over content, filename, capture-time and human-review links.

    review_merges are {"name", "clips", "evidence"} entries from visual review.
    Returns (groups, reasons, links): groups as sorted clip-id lists (via
    indian_split's union-find, which also joins exact SHA256 duplicates); reasons
    maps each clip_id to its sorted link reasons.
    """
    by_id = {c["clip_id"]: c for c in candidates}
    links: dict[tuple, set] = {}

    def link(a, b, reason):
        if a != b:
            links.setdefault(tuple(sorted((a, b))), set()).add(reason)

    for (a, b), distances in frame_matches(fingerprints, hamming_threshold).items():
        if len(distances) >= min_frame_matches:
            link(a, b, f"phash<={hamming_threshold}x{len(distances)}")
    keys: dict[str, list] = {}
    for candidate in candidates:
        key = filename_key(candidate)
        if key:
            keys.setdefault(key, []).append(candidate["clip_id"])
    for key, members in keys.items():
        for other in members[1:]:
            link(members[0], other, f"filename:{key}")
    stamped = sorted(
        (capture_time(c), c["clip_id"]) for c in candidates if capture_time(c) is not None
    )
    for (t1, a), (t2, b) in zip(stamped, stamped[1:]):
        same = by_id[a]["collection"] == by_id[b]["collection"]
        if same and (t2 - t1).total_seconds() <= capture_window_seconds:
            link(a, b, "capture_time<=5min")
    for merge in review_merges:
        if not merge.get("evidence", "").strip() or set(merge["clips"]) - by_id.keys():
            raise ValueError(f"Review merge needs evidence and known clips: {merge.get('name')}")
        for other in merge["clips"][1:]:
            link(merge["clips"][0], other, f"review:{merge['name']}")
    rows = [{**c, "source_video_id": c["clip_id"], "incident_id": c["clip_id"]} for c in candidates]
    groups = source_groups(rows, [list(pair) for pair in links])
    reasons = {c["clip_id"]: set() for c in candidates}
    for (a, b), why in links.items():
        reasons[a].update(why)
        reasons[b].update(why)
    return groups, {key: sorted(value) for key, value in reasons.items()}, links
