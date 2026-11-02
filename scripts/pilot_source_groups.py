"""Indian pilot Phase B: fingerprint candidates, tune the pHash threshold, write groups."""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from _common import run_cli

from surveillance.datasets.indian_pilot import (
    PILOT_OVERRIDE,
    filename_key,
    fingerprint,
    frame_matches,
    pilot_candidates,
    reconstruct_groups,
)

ROOT = Path("/Volumes/Seagate_2TB/violence-datasets")
CACHE = Path("runs/indian_pilot/features/fingerprints")


def load_cache(cache: Path) -> dict:
    data = json.loads((cache / "fingerprints.json").read_text())
    for item in data.values():
        item["hashes"] = np.array([int(h, 16) for h in item["hashes"]], dtype=np.uint64)
    return data


def cmd_fingerprint(args):
    candidates, excluded = pilot_candidates(args.root)
    args.cache.mkdir(parents=True, exist_ok=True)
    (args.cache / "thumbs").mkdir(exist_ok=True)
    out, started = {}, time.perf_counter()
    for number, candidate in enumerate(candidates, 1):
        item = fingerprint(candidate["path"])
        for index, image in enumerate(item.pop("thumbnails")):
            cv2.imwrite(str(args.cache / "thumbs" / f"{candidate['clip_id']}-{index}.jpg"), image)
        item["hashes"] = [f"{h:016x}" for h in item["hashes"].tolist()]
        out[candidate["clip_id"]] = item
        if number % 50 == 0:
            print(f"{number}/{len(candidates)} {time.perf_counter() - started:.0f}s", flush=True)
    (args.cache / "fingerprints.json").write_text(json.dumps(out))
    unreadable = [k for k, v in out.items() if v["decoded_frames"] == 0]
    print(f"done {len(out)} clips, unreadable {unreadable}, excluded {len(excluded)}")


def cmd_tune(args):
    """Distance evidence: filename-known same-incident pairs vs cross-collection pairs."""
    candidates, _ = pilot_candidates(args.root)
    prints = load_cache(args.cache)
    by_id = {c["clip_id"]: c for c in candidates}
    matches = frame_matches(prints, 20)
    keys = {}
    for c in candidates:
        if filename_key(c) and filename_key(c).startswith("snatch-theft"):
            keys.setdefault(filename_key(c), []).append(c["clip_id"])
    positives = [tuple(sorted(v)) for v in keys.values() if len(v) == 2]
    print("known same-incident pairs (Snatch N / N_0):")
    for pair in positives:
        d = sorted(matches.get(pair, []))
        names = [by_id[k]["path"].name for k in pair]
        print(f"  {names} n_frames={[len(prints[k]['hashes']) for k in pair]} best={d[:8]}")
    cross = Counter()
    for (a, b), d in matches.items():
        if by_id[a]["collection"] != by_id[b]["collection"]:
            for value in d:
                cross[value] += 1
    print("cross-collection frame-pair distances <=20 (should be noise):", sorted(cross.items()))


def cmd_group(args):
    candidates, excluded = pilot_candidates(args.root)
    prints = load_cache(args.cache)
    unreadable = {k for k, v in prints.items() if v["decoded_frames"] == 0}
    for key in unreadable:
        excluded[key] = "no decodable video frames"
    candidates = [c for c in candidates if c["clip_id"] not in unreadable]
    by_path = {str(c["path"].relative_to(args.root)): c["clip_id"] for c in candidates}
    merges = []
    if args.review:
        review = json.loads(args.review.read_text())
        for merge in review["merges"]:
            missing = [p for p in merge["paths"] if p not in by_path]
            if missing:
                raise ValueError(f"Review merge {merge['name']} names unknown clips: {missing}")
            clips = [by_path[p] for p in merge["paths"]]
            merges.append({"name": merge["name"], "clips": clips, "evidence": merge["evidence"]})
    groups, reasons, links = reconstruct_groups(
        candidates,
        {c["clip_id"]: prints[c["clip_id"]] for c in candidates},
        hamming_threshold=args.threshold,
        min_frame_matches=args.min_matches,
        review_merges=merges,
    )
    by_id = {c["clip_id"]: c for c in candidates}
    ordered = sorted(groups, key=lambda g: (by_id[g[0]]["collection"], g[0]))
    rows = []
    for number, group in enumerate(ordered, 1):
        for key in group:
            c = by_id[key]
            rows.append(
                {
                    "clip": key,
                    "group_id": f"G{number:03d}",
                    "collection": c["collection"],
                    "link_reasons": ";".join(reasons[key]) or "singleton",
                    "label": c["label"],
                    "geography": c["geography"],
                    "relative_path": str(c["path"].relative_to(args.root)),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "source_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "pilot_override": PILOT_OVERRIDE,
        "eligible_clips": len(rows),
        "excluded": dict(sorted(excluded.items())),
        "hamming_threshold": args.threshold,
        "min_frame_matches": args.min_matches,
        "links": len(links),
        "link_kinds": dict(
            Counter(reason.split(":")[0].split("<")[0] for why in links.values() for reason in why)
        ),
        "groups": len(ordered),
        "groups_with_violence": sum(any(by_id[k]["label"] for k in g) for g in ordered),
        "mixed_label_groups": sum(len({by_id[k]["label"] for k in g}) > 1 for g in ordered),
        "largest_groups": sorted((len(g) for g in ordered), reverse=True)[:10],
    }
    (args.output / "source_groups_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "excluded"}, indent=2))


def _tile(path: Path, width=200, height=150) -> np.ndarray:
    canvas = np.full((height, width, 3), 40, np.uint8)
    image = cv2.imread(str(path)) if path.is_file() else None
    if image is not None:
        scale = min(width / image.shape[1], height / image.shape[0])
        image = cv2.resize(image, None, fx=scale, fy=scale)
        y, x = (height - image.shape[0]) // 2, (width - image.shape[1]) // 2
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
    return canvas


def cmd_sheets(args):
    """Contact sheets of every group containing violence, one sheet per collection."""
    with (args.groups).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    violent = {r["group_id"] for r in rows if r["label"] == "1"}
    chosen = [r for r in rows if r["group_id"] in violent]
    if args.collection:
        chosen = [r for r in rows if r["collection"] == args.collection]
    args.output.mkdir(parents=True, exist_ok=True)
    for collection in sorted({r["collection"] for r in chosen}):
        members = sorted(
            (r for r in chosen if r["collection"] == collection),
            key=lambda r: (r["group_id"], r["relative_path"]),
        )
        for page in range(0, len(members), args.per_page):
            lines = []
            for r in members[page : page + args.per_page]:
                text = np.full((150, 420, 3), 255, np.uint8)
                tag = "VIOLENCE" if r["label"] == "1" else "non-violence"
                for y, value in (
                    (30, f"{r['group_id']}  {tag}"),
                    (65, Path(r["relative_path"]).name[:40]),
                    (100, r["clip"][:12]),
                    (135, r["link_reasons"][:44]),
                ):
                    cv2.putText(text, value, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                tiles = [_tile(args.cache / "thumbs" / f"{r['clip']}-{i}.jpg") for i in range(3)]
                lines.append(np.hstack([text, *tiles]))
                lines.append(np.full((4, lines[-1].shape[1], 3), 200, np.uint8))
            name = f"{collection}-{page // args.per_page + 1:02d}.png".replace(" ", "_")
            cv2.imwrite(str(args.output / name), np.vstack(lines))
            print(args.output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fingerprint")
    commands.add_parser("tune")
    group = commands.add_parser("group")
    group.add_argument("--threshold", type=int, required=True)
    group.add_argument("--min-matches", type=int, required=True)
    group.add_argument("--output", type=Path, default=Path("manifests/v2-pilot"))
    group.add_argument("--review", type=Path, help="review_decisions.json with merges")
    sheets = commands.add_parser("sheets")
    sheets.add_argument("--groups", type=Path, default=Path("manifests/v2-pilot/source_groups.csv"))
    sheets.add_argument("--output", type=Path, default=CACHE / "review_sheets")
    sheets.add_argument("--collection", help="every clip of one collection, not only violence")
    sheets.add_argument("--per-page", type=int, default=12)
    args = parser.parse_args()
    commands = {
        "fingerprint": cmd_fingerprint,
        "tune": cmd_tune,
        "group": cmd_group,
        "sheets": cmd_sheets,
    }
    commands[args.command](args)


if __name__ == "__main__":
    run_cli(main)
