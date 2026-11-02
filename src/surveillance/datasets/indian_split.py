"""Conservative source/incident/content grouping and immutable Indian split receipts."""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from surveillance.datasets.common import Record, check_leakage, read_manifest, write_manifest
from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import write_new_json
from surveillance.experiments.protocol import content_hash

SPLITS = ("train", "val", "test")


def _identity(value: str) -> str:
    if not isinstance(value, str) or value.strip().casefold() in {"", "unknown", "none", "n/a"}:
        raise ValueError("Unresolved source/incident provenance")
    return value.strip().casefold()


def source_groups(rows: list[dict], same_footage: list[list[str]]) -> list[list[str]]:
    """Connected components preserve all shared sources, incidents and exact content."""
    parents = {row["clip_id"]: row["clip_id"] for row in rows}
    if len(parents) != len(rows):
        raise ValueError("Duplicate clip IDs in staged import")

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(a, b):
        if a not in parents or b not in parents:
            raise ValueError("Duplicate review references unknown or excluded clip")
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)

    seen = {}
    hash_labels = {}
    for row in rows:
        if row["sha256"] in hash_labels and hash_labels[row["sha256"]] != row["label"]:
            raise ValueError("Exact duplicate content has conflicting labels")
        hash_labels[row["sha256"]] = row["label"]
        for kind in ("source_video_id", "incident_id", "sha256"):
            identity = (kind, _identity(row.get(kind, "")))
            if identity in seen:
                join(row["clip_id"], seen[identity])
            seen[identity] = row["clip_id"]
    for a, b in same_footage:
        join(a, b)
    groups = defaultdict(list)
    for key in sorted(parents):
        groups[root(key)].append(key)
    return sorted(groups.values())


def assign_groups(
    rows: list[dict], groups: list[list[str]], seed: int, ratios: tuple[float, float, float]
) -> dict[str, str]:
    """Seeded group allocation, stratifying class counts when independent support permits.

    Search assignments with fixed approximate group ratios. Minimize per-class clip
    deviation; never claim that source-group ratios imply exact clip ratios.
    """
    if (
        type(seed) is not int
        or seed < 0
        or len(ratios) != 3
        or any(not math.isfinite(r) or r <= 0 for r in ratios)
        or abs(sum(ratios) - 1) > 1e-8
    ):
        raise ValueError("Need nonnegative integer seed and three positive ratios summing to 1")
    groups = sorted(sorted(group) for group in groups)
    labels = {row["clip_id"]: row["label"] for row in rows}
    if set(labels.values()) != {0, 1}:
        raise ValueError("Both binary classes are required")
    vectors = [Counter(labels[key] for key in group) for group in groups]
    if any(sum(v[label] > 0 for v in vectors) < 3 for label in (0, 1)):
        raise ValueError("At least three independent source groups per class are required")
    targets = [len(groups) * ratio for ratio in ratios]
    counts = [max(1, math.floor(target)) for target in targets]
    while sum(counts) > len(groups):
        eligible = [i for i in range(3) if counts[i] > 1]
        counts[max(eligible, key=lambda i: counts[i] - targets[i])] -= 1
    while sum(counts) < len(groups):
        counts[max(range(3), key=lambda i: targets[i] - counts[i])] += 1
    totals = Counter(labels.values())
    rng, best = random.Random(seed), None
    for _ in range(2048):
        order = list(range(len(groups)))
        rng.shuffle(order)
        assignments, histograms = {}, [Counter() for _ in SPLITS]
        offset = 0
        for split_index, count in enumerate(counts):
            for group_index in order[offset : offset + count]:
                histograms[split_index].update(vectors[group_index])
                assignments.update(dict.fromkeys(groups[group_index], SPLITS[split_index]))
            offset += count
        if any(hist[label] == 0 for hist in histograms for label in (0, 1)):
            continue
        cost = sum(
            (hist[label] / totals[label] - ratios[i]) ** 2
            for i, hist in enumerate(histograms)
            for label in (0, 1)
        )
        if best is None or cost < best[0]:
            best = (cost, assignments)
    if best is None:
        raise ValueError("Cannot obtain class support in every split; collect/review more sources")
    return best[1]


def freeze_split(
    import_path: Path,
    inventory_path: Path,
    review_path: Path,
    output: Path,
    *,
    seed: int = 7,
    ratios=(0.7, 0.15, 0.15),
) -> dict:
    """Freeze once after explicit provenance, benchmark and candidate review.

    Reviews bind exact import/inventory bytes. Exclusions require evidence; no
    unresolved or historical benchmark source can enter any Indian partition.
    """
    if Path(output).exists():
        raise FileExistsError(f"Refuse to overwrite frozen split: {output}")
    stage = json.loads(Path(import_path).read_text(encoding="utf-8"))
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if stage.get("schema_version") != 1 or stage.get("state") != "unsplit":
        raise ValueError("Expected version-1 unsplit Indian import")
    if review.get("import_sha256") != sha256(import_path) or review.get(
        "benchmark_inventory_sha256"
    ) != sha256(inventory_path):
        raise ValueError("Review is stale: import/inventory hashes differ")
    if (
        review.get("benchmark_scope_complete") is not True
        or not review.get("benchmark_scope_evidence", "").strip()
        or not inventory.get("artifacts")
        or not inventory.get("source_video_ids")
    ):
        raise ValueError("Complete historical benchmark scope review is required")
    for artifact in inventory["artifacts"]:
        if sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("Historical manifest changed since benchmark inventory")
    exclusions = review.get("excluded_clips", {})
    all_ids = {row["clip_id"] for row in stage["records"]}
    if (
        not isinstance(exclusions, dict)
        or set(exclusions) - all_ids
        or any(not isinstance(value, str) or not value.strip() for value in exclusions.values())
    ):
        raise ValueError("Exclusions need known clip IDs and written reasons")
    rows = [row for row in stage["records"] if row["clip_id"] not in exclusions]
    clip_reviews = review.get("clip_reviews", {})
    if set(clip_reviews) != {row["clip_id"] for row in rows}:
        raise ValueError("Review must cover exactly every retained clip")
    historical_sources = {_identity(source) for source in inventory["source_video_ids"]}
    historical_hashes = set(inventory.get("video_sha256", []))
    for row in rows:
        key = row["clip_id"]
        _identity(row.get("source", ""))
        approved = clip_reviews[key]
        if (
            approved.get("annotation_approved") is not True
            or approved.get("provenance_approved") is not True
            or approved.get("benchmark_independent") is not True
            or not approved.get("evidence", "").strip()
        ):
            raise ValueError(f"Unresolved annotation/provenance/overlap review: {key}")
        if _identity(row.get("source_video_id", "")) in historical_sources:
            raise ValueError(f"Historical benchmark source overlap: {key}")
        if row["sha256"] in historical_hashes:
            raise ValueError(f"Historical benchmark content overlap: {key}")
        if sha256(Path(row["file_path"])) != row["sha256"]:
            raise ValueError(f"Video changed after import: {key}")
    required_pairs = {
        tuple(c["pair"]): c["kind"]
        for c in stage["duplicate_candidates"]
        if not set(c["pair"]) & exclusions.keys()
    }
    decisions, same = {}, []
    retained = {row["clip_id"] for row in rows}
    for decision in review.get("candidate_decisions", []):
        pair = tuple(sorted(decision["pair"]))
        if (
            len(pair) != 2
            or pair[0] == pair[1]
            or set(pair) - retained
            or pair in decisions
            or decision.get("relation") not in {"same_footage", "independent"}
            or not decision.get("evidence", "").strip()
        ):
            raise ValueError("Invalid or duplicate near-duplicate review")
        if required_pairs.get(pair) == "exact" and decision["relation"] != "same_footage":
            raise ValueError("Exact duplicate content cannot be declared independent")
        decisions[pair] = decision
        if decision["relation"] == "same_footage":
            same.append(list(pair))
    if set(required_pairs) - decisions.keys():
        raise ValueError("Unresolved duplicate/near-duplicate candidates")
    groups = source_groups(rows, same)
    assignments = assign_groups(rows, groups, seed, tuple(ratios))
    records = [
        Record(
            dataset="indian_domain",
            video_id=row["clip_id"],
            source_video_id=row["source_video_id"],
            path=row["file_path"],
            split=assignments[row["clip_id"]],
            label=row["label"],
            duration_sec=row["duration_sec"],
            fps=row["fps"],
            num_frames=row["num_frames"],
            temporal_annotations=row["temporal_annotations"],
        )
        for row in rows
    ]
    check_leakage(records)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_manifest(records, output / "manifest.jsonl")
    payload = {
        "schema_version": 1,
        "dataset": "indian_domain",
        "seed": seed,
        "ratios": list(ratios),
        "split_unit": "connected_source_incident_content_group",
        "method": "2048 seeded group permutations; minimum class-ratio squared error",
        "groups": groups,
        "assignments": assignments,
        "excluded_clips": exclusions,
        "clip_counts": dict(Counter(assignments.values())),
        "group_counts": dict(Counter(assignments[group[0]] for group in groups)),
        "source_counts": {
            split: len({r.source_video_id for r in records if r.split == split}) for split in SPLITS
        },
        "class_counts": {
            split: {
                str(label): sum(r.split == split and r.label == label for r in records)
                for label in (0, 1)
            }
            for split in SPLITS
        },
        "test_clip_ids": sorted(r.video_id for r in records if r.split == "test"),
        "video_sha256": {row["clip_id"]: row["sha256"] for row in rows},
        "manifest_sha256": sha256(output / "manifest.jsonl"),
        "artifacts": {
            str(Path(path).resolve()): sha256(path)
            for path in (import_path, inventory_path, review_path)
        },
        "limitations": "Visual screening is not exhaustive; curator independence evidence required",
    }
    receipt = {**payload, "freeze_hash": content_hash(payload)}
    write_new_json(output / "split-receipt.json", receipt)
    return receipt


def verify_split(directory: Path, *, verify_media: bool = True) -> tuple[list[Record], dict]:
    """Reject altered membership/inputs/media before every downstream training run."""
    directory = Path(directory)
    receipt = json.loads((directory / "split-receipt.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in receipt.items() if key != "freeze_hash"}
    if receipt.get("freeze_hash") != content_hash(payload):
        raise ValueError("Split freeze receipt changed")
    if sha256(directory / "manifest.jsonl") != receipt["manifest_sha256"]:
        raise ValueError("Protected split manifest changed")
    for path, digest in receipt["artifacts"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    rows = read_manifest(directory / "manifest.jsonl")
    check_leakage(rows)
    by_id = {row.video_id: row for row in rows}
    if {key: row.split for key, row in by_id.items()} != receipt["assignments"]:
        raise ValueError("Protected assignments differ")
    for group in receipt["groups"]:
        if len({by_id[key].split for key in group}) != 1:
            raise ValueError("Source/incident/content group leakage")
    if verify_media:
        for row in rows:
            if sha256(Path(row.path)) != receipt["video_sha256"][row.video_id]:
                raise ValueError(f"Frozen video changed: {row.video_id}")
    return rows, receipt
