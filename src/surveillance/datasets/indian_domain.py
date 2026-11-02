"""Auditable unsplit Indian annotations; no folder labels or invented provenance."""

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2

from surveillance.datasets.dcsass_audit import audit_video, sha256

REQUIRED = {"clip_id", "file_path", "label"}
OPTIONAL = {
    "reviewer",
    "timestamp",
    "playback_status",
    "violence_start",
    "violence_end",
    "notes",
    "source_video_id",
    "incident_id",
    "source",
    "review_status",
    "rights",
}
LABELS = {"violence": 1, "non_violence": 0, "not_sure": None}


def parse_annotation_seconds(value: str) -> float:
    """Parse explicit decimal seconds or MM:SS[.fraction], preserving raw text upstream.

    A column ending in _sec does not guarantee that annotators used decimal
    seconds. No rounding, clipping, wall-clock interpretation or status change.
    """
    value = value.strip()
    if re.fullmatch(r"\d+:[0-5]\d(?:\.\d+)?", value):
        minutes, seconds = value.split(":")
        result = 60 * int(minutes) + float(seconds)
    elif re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", value):
        result = float(value)
    else:
        raise ValueError(f"Invalid annotation time: {value!r}; use seconds or MM:SS")
    if not math.isfinite(result) or result < 0:
        raise ValueError("Annotation time must be finite and nonnegative")
    return result


def write_new_json(path: Path, value: dict | list) -> None:
    """Exclusive creation: existing research artifacts are never overwritten."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inspect_header(path: Path) -> list[str]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
        raise ValueError("CSV requires a nonempty header with unique, nonempty columns")
    return header


def normalize_label(value: str) -> str:
    """Only whitespace/case/separator normalization, never semantic synonyms."""
    normalized = re.sub(r"[\s-]+", "_", value.strip().casefold())
    if normalized not in LABELS:
        raise ValueError(f"Ambiguous or unsupported label: {value!r}")
    return normalized


def visual_signature(path: Path, num_frames: int) -> list[str]:
    """Eight sampled 64-bit dHashes, screening only (not duplicate ground truth)."""
    cap = cv2.VideoCapture(str(path))
    signatures = []
    try:
        for i in range(8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(i * (num_frames - 1) / 7))
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Cannot decode visual fingerprint sample")
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8))
            bits = (gray[:, 1:] > gray[:, :-1]).ravel()
            value = sum(int(bit) << j for j, bit in enumerate(bits))
            signatures.append(f"{value:016x}")
    finally:
        cap.release()
    return signatures


def near_duplicate(a: list[str], b: list[str], distance: int = 4) -> bool:
    """Two distinct samples on each side must match, allowing temporal offsets.

    Uniform scenes can overmatch; crops, overlays and partial footage can escape.
    Human/source review remains mandatory even when this returns False.
    """
    edges = [
        (i, j)
        for i, x in enumerate(a)
        for j, y in enumerate(b)
        if (int(x, 16) ^ int(y, 16)).bit_count() <= distance
    ]
    return len({i for i, _ in edges}) >= 2 and len({j for _, j in edges}) >= 2


def duplicate_candidates(rows: list[dict]) -> list[dict]:
    candidates = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            exact = a["sha256"] == b["sha256"]
            if exact or near_duplicate(a["visual_signature"], b["visual_signature"]):
                candidates.append(
                    {
                        "pair": sorted([a["clip_id"], b["clip_id"]]),
                        "kind": "exact" if exact else "visual_candidate",
                    }
                )
    return sorted(candidates, key=lambda row: row["pair"])


def import_annotations(
    annotations: Path,
    media_root: Path,
    output: Path,
    *,
    columns: dict[str, str] | None = None,
    provenance: dict[str, dict] | None = None,
) -> dict:
    """Stage valid binary clips and quarantine every invalid/uncertain CSV row.

    Provenance is explicitly keyed by clip_id; original annotation values survive.
    No train/val/test assignment is manufactured during import.
    """
    annotations, media_root, output = Path(annotations), Path(media_root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite import directory: {output}")
    csv_hash = sha256(annotations)
    header = inspect_header(annotations)  # inspect before resolving any mapping
    mapping = (
        columns
        if columns is not None
        else {key: key for key in REQUIRED | OPTIONAL if key in header}
    )
    if (
        not isinstance(mapping, dict)
        or not REQUIRED <= mapping.keys()
        or set(mapping) - (REQUIRED | OPTIONAL)
    ):
        raise ValueError("Column mapping requires clip_id,file_path,label; unknown canonical keys")
    if any(not isinstance(v, str) or v not in header for v in mapping.values()):
        raise ValueError(f"Mapped columns absent from inspected header: {header}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Column mappings must be one-to-one")
    provenance = provenance or {}
    if not isinstance(provenance, dict) or any(
        not isinstance(v, dict) for v in provenance.values()
    ):
        raise ValueError("Provenance must map clip IDs to objects")
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    id_counts = Counter((row.get(mapping["clip_id"]) or "").strip() for row in raw_rows)
    rows, quarantine, unresolved = [], [], []
    for index, raw in enumerate(raw_rows, 2):
        reasons = []
        row = {key: (raw.get(actual) or "").strip() for key, actual in mapping.items()}
        clip_id = row["clip_id"]
        if None in raw or any(raw.get(name) is None for name in header):
            reasons.append("CSV row width differs from header")
        if not clip_id:
            reasons.append("missing_clip_id")
        elif id_counts[clip_id] > 1:
            reasons.append("duplicate_clip_id")
        try:
            label = normalize_label(row["label"])
            if label == "not_sure":
                reasons.append("uncertain_label")
        except ValueError as error:
            reasons.append(str(error))
            label = None
        supplied = provenance.get(clip_id, {})
        for key in OPTIONAL - {"violence_start", "violence_end"}:
            if key in supplied:
                if not isinstance(supplied[key], str):
                    reasons.append(f"non-string provenance: {key}")
                elif row.get(key) and row[key] != supplied[key].strip():
                    reasons.append(f"conflicting provenance: {key}")
                else:
                    row[key] = supplied[key].strip()
        details = {}
        if not reasons:
            try:
                if not row["file_path"]:
                    raise ValueError("missing_file_path")
                path = (media_root / row["file_path"]).resolve()
                if not path.is_relative_to(media_root):
                    raise ValueError("Video path escapes media root")
                if not path.is_file():
                    raise ValueError("missing_video")
                details = audit_video(path)
                if not details["readable"]:
                    raise ValueError(details.get("error", "unreadable_video"))
                signature = visual_signature(path, details["num_frames"])
                if sha256(path) != details["sha256"]:
                    raise ValueError("Video changed during audit")
                start, end = row.get("violence_start", ""), row.get("violence_end", "")
                intervals = None
                if start or end:
                    if not start or not end:
                        raise ValueError("Temporal annotation requires both start and end")
                    start_value = parse_annotation_seconds(start)
                    end_value = parse_annotation_seconds(end)
                    if not 0 <= start_value < end_value <= details["duration_sec"]:
                        raise ValueError("Invalid temporal interval in seconds")
                    if label != "violence":
                        raise ValueError("Temporal violence interval conflicts with non_violence")
                    intervals = [[start_value, end_value]]
                accepted = {
                    **row,
                    **details,
                    "file_path": str(path),
                    "label": LABELS[label],
                    "normalized_label": label,
                    "csv_row": index,
                    "raw": raw,
                    "supplied_provenance": supplied,
                    "temporal_annotations": intervals,
                    "visual_signature": signature,
                }
                rows.append(accepted)
                missing = [
                    key
                    for key in ("source_video_id", "incident_id", "source")
                    if row.get(key, "").casefold() in {"", "unknown", "none", "n/a"}
                ]
                if missing:
                    unresolved.append({"clip_id": clip_id, "missing": missing})
            except (ValueError, OSError, cv2.error) as error:
                reasons.append(str(error))
        if reasons:
            quarantine.append(
                {
                    "csv_row": index,
                    "clip_id": clip_id,
                    "reasons": reasons,
                    "raw": {k if k is not None else "__extra_fields__": v for k, v in raw.items()},
                    "media_audit": details,
                }
            )
    rows.sort(key=lambda row: row["clip_id"])
    candidates = duplicate_candidates(rows)
    if sha256(annotations) != csv_hash:
        raise ValueError("Annotation CSV changed during import")
    stage = {
        "schema_version": 1,
        "dataset": "indian_domain",
        "state": "unsplit",
        "csv_sha256": sha256(annotations),
        "csv_path": str(annotations.resolve()),
        "header": header,
        "columns": mapping,
        "records": rows,
        "duplicate_candidates": candidates,
    }
    report = {
        "schema_version": 1,
        "header": header,
        "columns": mapping,
        "csv_sha256": stage["csv_sha256"],
        "input_rows": len(raw_rows),
        "accepted_binary_clips": len(rows),
        "quarantined_rows": len(quarantine),
        "label_counts": dict(Counter(row["normalized_label"] for row in rows)),
        "quarantine_reasons": dict(Counter(reason for q in quarantine for reason in q["reasons"])),
        "unresolved_provenance": unresolved,
        "duplicate_candidates": candidates,
        "unused_provenance_ids": sorted(set(provenance) - set(id_counts)),
        "review_values": {
            key: dict(Counter(row.get(key, "") for row in rows))
            for key in ("reviewer", "playback_status", "review_status")
        },
        "split_approval": "blocked_pending_provenance_and_duplicate_overlap_review",
        "readability": "full decode with exact reported frame-count agreement",
        "visual_screening": "eight dHash samples; candidates require review; not exhaustive",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "import.json", stage)
    write_new_json(output / "quarantine.json", quarantine)
    write_new_json(output / "quality-report.json", report)
    write_new_json(
        output / "registration.json",
        {
            "import_sha256": sha256(output / "import.json"),
            "quality_sha256": sha256(output / "quality-report.json"),
            "quarantine_sha256": sha256(output / "quarantine.json"),
        },
    )
    return report


def benchmark_inventory(manifests: list[Path], evidence: str) -> dict:
    """Read explicitly selected historical JSONL schemas; hash media if not recorded.

    This is an inventory, not a claim that all historical datasets were supplied.
    Curators must attest completeness and independently review transformed footage.
    """
    if not evidence.strip() or not manifests:
        raise ValueError("Benchmark inventory needs manifests and scope evidence")
    sources, hashes, artifacts, missing = set(), set(), [], []
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        artifacts.append({"path": str(manifest), "sha256": sha256(manifest)})
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source_video_id")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Historical manifest row lacks source_video_id")
            sources.add(source.strip())
            digest = row.get("sha256")
            if digest is None and row.get("path"):
                path = (manifest.parent / row["path"]).resolve()
                if path.is_file():
                    digest = sha256(path)
            if digest is not None:
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Invalid historical video SHA256")
                hashes.add(digest)
            else:
                missing.append(
                    {"manifest": str(manifest), "id": row.get("video_id", row.get("clip_id"))}
                )
    return {
        "schema_version": 1,
        "evidence": evidence,
        "artifacts": artifacts,
        "source_video_ids": sorted(sources),
        "video_sha256": sorted(hashes),
        "missing_video_hashes": missing,
    }
