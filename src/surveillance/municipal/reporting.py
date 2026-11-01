"""Explicit metric populations, blinded records and privacy-aware local exports."""

import csv
import io
import json
from collections import Counter

from surveillance.datasets.dcsass_protocol import CLASSES
from surveillance.municipal.catalog import now


def records(catalog, *, references=False):
    with catalog.connect() as db:
        items = db.execute(
            "SELECT items.*,jobs.profile,jobs.created_at AS submitted_at "
            "FROM items JOIN jobs USING(job_id) ORDER BY jobs.created_at,items.rowid"
        ).fetchall()
        rows = []
        for item in items:
            clip = catalog.clip(item["clip_id"])
            reviews = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM reviews WHERE item_id=? ORDER BY rowid",
                    (item["item_id"],),
                )
            ]
            result = json.loads(item["result"]) if item["result"] else None
            row = {
                **dict(item),
                "result": result,
                "profile": json.loads(item["profile"]),
                "clip": clip,
                "reviews": reviews,
            }
            if references:
                reference = db.execute(
                    "SELECT * FROM reference_versions WHERE clip_id=? "
                    "ORDER BY version DESC LIMIT 1",
                    (item["clip_id"],),
                ).fetchone()
                row["reference"] = (
                    {**dict(reference), "payload": json.loads(reference["payload"])}
                    if reference
                    else None
                )
            rows.append(row)
    return rows


def predicted_class(result):
    # Generic/no-actor/no-route outcomes do not imply a Normal behavior classification.
    return result.get("final_alert", {}).get("behavior") if result else None


def class_metrics(pairs, labels):
    per_class = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in pairs)
        fp = sum(a != label and b == label for a, b in pairs)
        fn = sum(a == label and b != label for a, b in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[str(label)] = dict(
            precision=precision,
            recall=recall,
            f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            support=sum(a == label for a, _ in pairs),
        )
    return dict(
        n=len(pairs),
        accuracy=sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
        macro_f1=sum(v["f1"] for v in per_class.values()) / len(labels) if pairs else None,
        per_class=per_class,
    )


def merged(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(end, out[-1][1])
        else:
            out.append([start, end])
    return out


def temporal_false_alert(result, reference, duration):
    events = merged(reference.get("events", []))
    anomaly = result["anomaly"]
    alerts = merged(
        [
            pair
            for score, pair in zip(anomaly["segment_scores"], anomaly["timestamps"])
            if score >= result["policy"]["anomaly_threshold"]
        ]
    )
    overlap = sum(max(0, min(b, d) - max(a, c)) for a, b in alerts for c, d in events)
    alerted = sum(b - a for a, b in alerts)
    negative = duration - sum(b - a for a, b in events)
    return alerted - overlap, max(0, negative)


def evaluation(catalog, profile_sha256, allowed_clip_ids=None):
    all_rows = records(catalog, references=True)
    # One latest completed actual inference per clip, same frozen profile. Never pool retries.
    latest = {}
    excluded = Counter()
    for row in all_rows:
        clip, result = row["clip"], row["result"]
        if allowed_clip_ids is not None and row["clip_id"] not in allowed_clip_ids:
            continue
        if (
            clip["population"] != "evaluation"
            or clip["assignment"] != "test"
            or row["status"] != "completed"
            or not result
            or result.get("execution_mode") != "on_demand"
            or result.get("profile_sha256") != profile_sha256
        ):
            excluded[clip["population"] + ":" + row["status"]] += 1
            continue
        latest[row["clip_id"]] = row
    pairs, binary_pairs, alert_pairs = [], [], []
    labeled = binary_labeled = temporal_n = 0
    false_seconds = negative_seconds = 0.0
    reference_versions = {}
    violence_map = {"Normal": False, "Assault": True, "Fighting": True}
    for row in latest.values():
        reference = row.get("reference")
        if not reference:
            continue
        reference_versions[row["clip_id"]] = reference["version"]
        truth = reference["payload"]
        predicted = predicted_class(row["result"])
        if truth.get("class_label") is not None:
            labeled += 1
            if predicted in CLASSES:
                pairs.append((truth["class_label"], predicted))
        if truth.get("violence") is not None:
            binary_labeled += 1
            if predicted in violence_map:
                binary_pairs.append((truth["violence"], violence_map[predicted]))
        if truth.get("temporal_complete") is True:
            temporal_n += 1
            alert_pairs.append(
                (
                    bool(truth.get("events", [])),
                    row["result"]["final_alert"]["state"] != "no_anomaly",
                )
            )
            false, negative = temporal_false_alert(
                row["result"],
                truth,
                row["clip"]["metadata"]["duration"],
            )
            false_seconds += false
            negative_seconds += negative
    clips = [
        c for c in catalog.clips() if allowed_clip_ids is None or c["clip_id"] in allowed_clip_ids
    ]
    return dict(
        generated_at=now(),
        profile_sha256=profile_sha256,
        warning="Curated libraries are not unbiased benchmarks. "
        "Synthetic tests measure software only.",
        definition="Latest completed actual inference per protected test clip and frozen profile. "
        "Behavior metrics are conditional on class coverage; macro F1 uses all six classes. "
        "Violence mapping: Assault/Fighting=true, Normal=false; other outputs abstain.",
        eligible_inferred_clips=len(latest),
        reference_versions=reference_versions,
        class_labeled=labeled,
        class_abstentions=labeled - len(pairs),
        class_coverage=len(pairs) / labeled if labeled else None,
        behavior=class_metrics(pairs, CLASSES),
        violence_labeled=binary_labeled,
        violence_abstentions=binary_labeled - len(binary_pairs),
        violence_coverage=len(binary_pairs) / binary_labeled if binary_labeled else None,
        violence=class_metrics(binary_pairs, [False, True]),
        alert=class_metrics(alert_pairs, [False, True]),
        temporal=dict(
            n=temporal_n,
            false_alert_seconds=false_seconds,
            reference_negative_seconds=negative_seconds,
            false_alert_fraction=false_seconds / negative_seconds if negative_seconds else None,
        ),
        excluded_job_items=dict(excluded),
        composition=dict(
            populations=dict(Counter(c["population"] for c in clips)),
            assignments=dict(Counter(c["assignment"] for c in clips)),
            source_recordings=len({c["source_video_id"] for c in clips}),
            selection_methods=dict(Counter(c["spec"]["selection_method"] for c in clips)),
            geography=dict(Counter(c["spec"]["location_confidence"] for c in clips)),
        ),
    )


def export_rows(catalog, *, references=False, external=False):
    rows, excluded = [], 0
    for row in records(catalog, references=references):
        clip, result = row["clip"], row["result"] or {}
        if external and not clip["spec"]["municipal_redistribution_permission"]:
            excluded += 1
            continue
        # No local paths, raw video, actor imagery or facial crops are embedded.
        result = {k: v for k, v in result.items() if k not in {"video_path", "video_metadata"}}
        if "anomaly" in result:
            result["anomaly"] = {k: v for k, v in result["anomaly"].items() if k != "metadata"}
        latest_review = row["reviews"][-1] if row["reviews"] else {}
        entry = dict(
            clip_id=clip["clip_id"],
            source_video_id=clip["source_video_id"],
            source=clip["spec"]["source"],
            license=clip["spec"]["license"],
            assignment=clip["assignment"],
            population=clip["population"],
            selection_method=clip["spec"]["selection_method"],
            clip_sha256=clip["sha256"],
            item_id=row["item_id"],
            job_id=row["job_id"],
            status=row["status"],
            predicted_class=predicted_class(result),
            execution_mode=result.get("execution_mode"),
            anomaly_scores=result.get("anomaly", {}).get("segment_scores"),
            review_decision=latest_review.get("decision"),
            reviewer_notes=latest_review.get("notes"),
            reviews=row["reviews"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            inference_seconds=result.get("timings", {}).get("total_seconds"),
            profile=row["profile"],
            failure_reason=row["error"],
            abstention_reason="Behavior unavailable or not routed"
            if result and predicted_class(result) is None
            else None,
            result=result,
        )
        if references:
            entry["reference"] = row.get("reference")
        rows.append(entry)
    return dict(
        schema_version=1,
        generated_at=now(),
        external_handover=external,
        excluded_rights=excluded,
        records=rows,
    )


def csv_export(payload):
    columns = [
        "clip_id",
        "source_video_id",
        "source",
        "license",
        "assignment",
        "population",
        "clip_sha256",
        "item_id",
        "job_id",
        "status",
        "predicted_class",
        "execution_mode",
        "anomaly_scores",
        "review_decision",
        "reviewer_notes",
        "started_at",
        "finished_at",
        "inference_seconds",
        "profile",
        "failure_reason",
        "abstention_reason",
    ]
    if any("reference" in row for row in payload["records"]):
        columns.append("reference")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in payload["records"]:
        values = {}
        for key in columns:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                value = json.dumps(value, allow_nan=False)
            # Spreadsheet formula injection includes whitespace-prefixed formulas.
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            values[key] = value
        writer.writerow(values)
    return output.getvalue()
