"""Indian pilot v3 (V3_PROTOCOL.md): plan, cut and fingerprint 2 s clips.

plan        windows from UBI-Fights, XD-Violence and the IITB-Corridor test videos
cut         one H.264 file per window (bundled ffmpeg), then probe frames and hash
fingerprint pHash of every new source, DCSASS clip and IITB test video
links       new sources that share frames with DCSASS, Indian v1 or IITB test

Everything is written under /Volumes/Seagate_2TB/violence-training/v3 (never git).
"""

import argparse
import json
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.v3_clips import (
    evenly,
    grid_windows,
    intervals_to_labels,
    safe_name,
    window_frames,
    xd_kind,
    xd_source,
)

ROOT = Path("/Volumes/Seagate_2TB/violence-training")
UBI = ROOT / "raw/ubi-fights/extracted/UBI_FIGHTS"
XD = ROOT / "raw/xd-violence/data"
IITB = ROOT / "raw/iitb-corridor"
IITB_META = IITB / "train/Train_IITB_Corridor"
IITB_TEST = IITB / "test/Test_IITB-Corridor"
OUT = ROOT / "v3"
COMBINED_V2 = Path("manifests/v2-pilot/split_combined_v1/manifest.jsonl")
INDIAN_PRINTS = Path("runs/indian_pilot/features/fingerprints/fingerprints.json")
TRAIN_CAPS = {"violent": 6, "matched_normal": 6, "normal": 3}
TEST_CAPS = {"violent": 3, "matched_normal": 2, "normal": 1}
HAMMING, MIN_MATCHES = 8, 2  # D10


def probe(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        return capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().split("\n") if line.strip()]


def plan_video(dataset, name, path, labels, fps, caps, pool, source, offset=0):
    """Clips for one video. `labels` is None for a normal video; offset maps label to frame."""
    width = window_frames(fps)
    if labels is None:
        starts = list(range(0, frames_for(path, offset) - width + 1, width))
        chosen = [("normal", s, 0) for s in evenly(starts, caps["normal"])]
    else:
        violent, normal = grid_windows(labels, width, round(fps))
        chosen = [("violent", s, 1) for s in evenly(violent, caps["violent"])]
        chosen += [("matched_normal", s, 0) for s in evenly(normal, caps["matched_normal"])]
    return [
        {
            "clip_id": f"{dataset}_{safe_name(name)}_{start + offset:06d}",
            "dataset": dataset,
            "source": source,
            "video": name,
            "src_path": str(path),
            "start_frame": start + offset,
            "frames": width,
            "fps": fps,
            "label": label,
            "kind": kind,
            "pool": pool,
        }
        for kind, start, label in sorted(chosen, key=lambda c: c[1])
    ]


def frames_for(path, offset):
    return probe(path)[1] - offset


def plan_ubi():
    test = set(lines(UBI / "test_videos.csv"))
    clips = []
    for kind in ("fight", "normal"):
        videos = (UBI / "videos" / kind).glob("*.mp4")
        for path in sorted(p for p in videos if not p.name.startswith("._")):
            fps, frames = probe(path)
            text = (UBI / "annotation" / f"{path.stem}.csv").read_text().split("\n")
            labels = np.array([int(x) for x in text if x.strip()], dtype=bool)
            if len(labels) != frames:
                raise ValueError(f"UBI {path.stem}: {len(labels)} labels for {frames} frames")
            pool = "val" if path.stem in test else "train"
            clips += plan_video(
                "ubi",
                path.stem,
                path,
                labels if kind == "fight" else None,
                fps,
                TRAIN_CAPS,
                pool,
                path.stem,
            )
    return clips


def plan_xd():
    annotations = {}
    for line in (XD / "test_annotations.txt").read_text().split("\n"):
        parts = line.split()
        if parts:
            annotations[parts[0].removesuffix(".mp4")] = [int(x) for x in parts[1:]]
    clips, kinds = [], Counter()
    for line in (XD / "test_list.txt").read_text().split("\n"):
        name = line.strip().removeprefix("test_videos/")
        if not name:
            continue
        path = XD / "video/test_videos" / f"{name}.mp4"
        kind = xd_kind(name)
        kinds[kind] += 1
        if kind == "excluded":
            continue
        if not path.exists():
            kinds[f"{kind}_missing"] += 1
            continue
        fps, frames = probe(path)
        labels = None
        if kind == "violent":
            labels = intervals_to_labels(frames, annotations[name])
        clips += plan_video("xd", name, path, labels, fps, TRAIN_CAPS, None, xd_source(name))
    print("xd videos", dict(kinds))
    return clips


def plan_iitb_test():
    counts = dict(line.split() for line in lines(IITB_META / "Count_Test_frames.csv"))
    classes = {}
    for line in lines(IITB_META / "test.txt"):
        label, video_id = line.split()
        classes[video_id] = label
    clips = []
    for video_id, label in sorted(classes.items()):
        is_fight, is_normal = label.startswith("Fighting_"), label == "Normal"
        if not (is_fight or is_normal):
            continue
        path = IITB_TEST / "Test" / video_id / f"{video_id}.avi"
        fps, frames = probe(path)
        if frames != int(counts[video_id]) + 1:
            raise ValueError(f"IITB {video_id}: {frames} frames, CSV {counts[video_id]} + 1")
        labels = None
        if is_fight:
            labels = np.load(next((IITB_TEST / "Test_Annotation").rglob(f"{video_id}.npy")))
            labels = np.asarray(labels).reshape(-1).astype(bool)
            if len(labels) != int(counts[video_id]):
                raise ValueError(f"IITB {video_id}: {len(labels)} labels, CSV {counts[video_id]}")
        # Label i is frame i + 1 (the black first frame is not labelled).
        clips += plan_video(
            "iitb", video_id, path, labels, fps, TEST_CAPS, "test", video_id, offset=1
        )
    return clips


def cmd_plan(args):
    target = OUT / "plan.json"
    if target.exists():
        raise FileExistsError(f"Refuse to overwrite {target}")
    clips = plan_ubi() + plan_xd() + plan_iitb_test()
    if len({c["clip_id"] for c in clips}) != len(clips):
        raise ValueError("Duplicate clip IDs")
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(clips, indent=0))
    summary = Counter((c["dataset"], c["pool"], c["kind"]) for c in clips)
    for key in sorted(summary, key=str):
        print(key, summary[key])
    iitb = [c for c in clips if c["dataset"] == "iitb"]
    short = Counter(c["video"] for c in iitb if c["kind"] != "normal")
    print(
        "iitb fighting videos with fewer than 5 clips:", {v: n for v, n in short.items() if n < 5}
    )


def cut_one(clip):
    target = OUT / "clips" / clip["dataset"] / f"{clip['clip_id']}.mp4"
    if not target.exists():
        import imageio_ffmpeg

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial.mp4")
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-y",
                "-ss",
                f"{clip['start_frame'] / clip['fps']:.6f}",
                "-i",
                clip["src_path"],
                "-frames:v",
                str(clip["frames"]),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(partial),
            ],
            check=True,
            timeout=600,
        )
        partial.rename(target)
    fps, frames = probe(target)
    return {**clip, "path": str(target), "fps": fps, "num_frames": frames, "sha256": sha256(target)}


def cmd_cut(args):
    target = OUT / "clips.json"
    if target.exists():
        raise FileExistsError(f"Refuse to overwrite {target}")
    clips = json.loads((OUT / "plan.json").read_text())
    done = []
    with ThreadPoolExecutor(args.workers) as pool:
        for number, clip in enumerate(pool.map(cut_one, clips), 1):
            done.append(clip)
            if number % 500 == 0:
                print(f"{number}/{len(clips)}", flush=True)
    short = [c["clip_id"] for c in done if c["num_frames"] != c["frames"]]
    target.write_text(json.dumps(done, indent=0))
    print("clips", len(done), "frame count differs from plan:", len(short), short[:5])


def print_one(item):
    from surveillance.datasets.indian_pilot import fingerprint

    key, path = item
    result = fingerprint(Path(path), thumbs=0)
    return key, [format(int(h), "016x") for h in result["hashes"]]


def cmd_fingerprint(args):
    target = OUT / "fingerprints.json"
    if target.exists():
        raise FileExistsError(f"Refuse to overwrite {target}")
    plan = json.loads((OUT / "plan.json").read_text())
    items = {f"{c['dataset']}:{c['source']}|{c['video']}": c["src_path"] for c in plan}
    for line in COMBINED_V2.read_text().splitlines():
        row = json.loads(line)
        if row["dataset"] == "dcsass":
            source = row["source_video_id"].split(":", 1)[1]
            items[f"dcsass:{source}|{row['video_id']}"] = row["path"]
    for video in sorted((IITB_TEST / "Test").iterdir()):
        if video.is_dir() and not video.name.startswith("._"):
            items[f"iitb:{video.name}|{video.name}"] = str(video / f"{video.name}.avi")
    prints = {}
    with ProcessPoolExecutor(args.workers) as pool:
        results = pool.map(print_one, sorted(items.items()), chunksize=4)
        for number, (key, hashes) in enumerate(results, 1):
            prints[key] = hashes
            if number % 500 == 0:
                print(f"{number}/{len(items)}", flush=True)
    target.write_text(json.dumps(prints))
    print("fingerprinted", len(prints), "hashes", sum(map(len, prints.values())))


def cmd_links(args):
    prints = json.loads((OUT / "fingerprints.json").read_text())
    indian = json.loads(INDIAN_PRINTS.read_text())
    reference, new = {}, {}
    for key, hashes in prints.items():
        source = key.split("|", 1)[0]
        bucket = new if source.split(":", 1)[0] in {"ubi", "xd"} else reference
        bucket.setdefault(source, []).extend(int(h, 16) for h in hashes)
    for clip_id, item in indian.items():
        reference.setdefault(f"indian:{clip_id}", []).extend(int(h, 16) for h in item["hashes"])
    ref_owner = np.array([s for s, hs in reference.items() for _ in hs])
    ref_hash = np.array([h for hs in reference.values() for h in hs], dtype=np.uint64)
    links = {}
    for source, hashes in new.items():
        if not hashes:
            continue
        counts = Counter()
        hashes = np.array(hashes, dtype=np.uint64)
        for start in range(0, len(hashes), 256):
            distance = np.bitwise_count(hashes[start : start + 256, None] ^ ref_hash[None, :])
            counts.update(ref_owner[np.nonzero(distance <= HAMMING)[1]].tolist())
        matched = {ref: n for ref, n in counts.items() if n >= MIN_MATCHES}
        if matched:
            links[source] = matched
    (OUT / "links.json").write_text(json.dumps(links, indent=1, sort_keys=True))
    kinds = Counter((s.split(":")[0], r.split(":")[0]) for s, refs in links.items() for r in refs)
    print("linked new sources", len(links), dict(kinds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    cut = commands.add_parser("cut")
    cut.add_argument("--workers", type=int, default=6)
    prints = commands.add_parser("fingerprint")
    prints.add_argument("--workers", type=int, default=6)
    commands.add_parser("links")
    args = parser.parse_args()
    {"plan": cmd_plan, "cut": cmd_cut, "fingerprint": cmd_fingerprint, "links": cmd_links}[
        args.command
    ](args)


if __name__ == "__main__":
    run_cli(main)
