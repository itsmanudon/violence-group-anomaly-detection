"""Reference-only metrics with explicit populations and inspectable denominators."""

import csv
import html
import io
import json
from collections import Counter

from surveillance.datasets.dcsass_protocol import CLASSES
from surveillance.municipal.catalog import encode, now


def prediction(result):
    # A routing bypass is not a verified Normal behavior classification.
    label = result["final_alert"].get("behavior")
    if label is None:
        windows = result["behavior"]["windows"]
        available = [w for w in windows if w.get("status") == "available"]
        if available:
            label = max(available, key=lambda w: w["anomaly_score"])["predicted_behavior"]
    return label


def records(catalog, *, include_references=False, item_ids=None):
    with catalog.connect() as db:
        rows = db.execute(
            "SELECT i.*,j.execution_mode,j.profile_hash,j.config_sha256,j.provenance,"
            "j.created_at AS submitted_at "
            "FROM items i JOIN jobs j USING(job_id) ORDER BY i.created_at,i.item_id"
        ).fetchall()
        reviews = [
            dict(r) for r in db.execute("SELECT * FROM reviews ORDER BY created_at,review_id")
        ]
    latest_review = {r["item_id"]: r for r in reviews}
    output = []
    for row in rows:
        if item_ids is not None and row["item_id"] not in item_ids:
            continue
        clip = catalog.clip(row["clip_id"])
        result = json.loads(row["result"]) if row["result"] else None
        review = latest_review.get(row["item_id"])
        ref = catalog.reference(row["clip_id"]) if include_references else None
        output.append(
            {
                "item_id": row["item_id"],
                "job_id": row["job_id"],
                "clip_id": row["clip_id"],
                "source_video_id": clip["source_video_id"],
                "source": clip["source"],
                "sha256": clip["sha256"],
                "assignment": clip["assignment"],
                "cohort": clip["cohort"],
                "annotation_status": clip["annotation_status"],
                "selection_method": clip["selection_method"],
                "license": clip["license"],
                "external_handover_permission": clip["external_handover_permission"],
                "geography": clip["geography"],
                "duration": clip["duration"],
                "status": row["state"],
                "execution_mode": row["execution_mode"],
                "predicted_class": prediction(result) if result else None,
                "anomaly_scores": result["anomaly"]["segment_scores"] if result else None,
                "anomaly": result["anomaly"] if result else None,
                "behavior": result["behavior"] if result else None,
                "alerts": result["alerts"] if result else None,
                "timings": result["timings"] if result else None,
                "model_provenance": json.loads(row["provenance"]),
                "profile_hash": row["profile_hash"],
                "config_sha256": row["config_sha256"],
                "inference_duration": result["timings"].get("total_seconds") if result else None,
                "failure_reason": row["error"],
                "abstention_reason": (
                    "behavior_unavailable_or_not_routed"
                    if result and prediction(result) is None
                    else None
                ),
                "submitted_at": row["submitted_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "reference": ref,
                "review": review,
                "review_history": [r for r in reviews if r["item_id"] == row["item_id"]],
            }
        )
    return output


def classification(truth, predicted, labels):
    n = len(truth)
    if not n:
        return {"support": 0, "accuracy": None, "macro_f1": None, "per_class": {}}
    per_class = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        per_class[str(label)] = {
            "support": sum(t == label for t in truth),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    return {
        "support": n,
        "accuracy": sum(t == p for t, p in zip(truth, predicted)) / n,
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(labels),
        "per_class": per_class,
    }


def union_duration(intervals):
    end, total = 0.0, 0.0
    for a, b in sorted(intervals):
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total


def metrics(catalog, profile_hash, *, cohort="untouched_evaluation", mode="on_demand"):
    rows = records(catalog, include_references=True)
    population = [
        r
        for r in rows
        if r["profile_hash"] == profile_hash
        and r["cohort"] == cohort
        and r["execution_mode"] == mode
    ]
    # One latest submitted item per clip; a failed rerun never silently resurrects an old success.
    latest = {r["clip_id"]: r for r in population}
    selected = list(latest.values())
    completed = [r for r in selected if r["status"] == "completed"]
    labeled = [
        r
        for r in completed
        if r["reference"] and r["reference"]["payload"].get("behavior") in CLASSES
    ]
    truth = [r["reference"]["payload"]["behavior"] for r in labeled]
    predicted = [r["predicted_class"] for r in labeled]
    covered = [r for r in labeled if r["predicted_class"] is not None]
    binary = [
        r
        for r in completed
        if r["reference"] and type(r["reference"]["payload"].get("violence")) is bool
    ]
    # Only Fighting is an explicit violence class; other behavior classes abstain.
    binary_covered = [r for r in binary if r["predicted_class"] in ("Normal", "Fighting")]
    temporal = [
        r
        for r in completed
        if r["reference"]
        and r["reference"]["payload"].get("temporal_complete") is True
        and "event_intervals" in r["reference"]["payload"]
    ]
    false_alerts, alert_count, false_seconds, negative_seconds = 0, 0, 0.0, 0.0
    for row in temporal:
        events = row["reference"]["payload"]["event_intervals"]
        alerts = [a["timestamp_interval"] for a in row["alerts"] if a.get("timestamp_interval")]
        false_alerts += sum(not any(a < y and b > x for x, y in events) for a, b in alerts)
        alert_count += len(alerts)
        overlap = [
            [max(a, x), min(b, y)] for a, b in alerts for x, y in events if max(a, x) < min(b, y)
        ]
        false_seconds += union_duration(alerts) - union_duration(overlap)
        negative_seconds += row["duration"] - union_duration(events)
    return {
        "population": {
            "cohort": cohort,
            "execution_mode": mode,
            "profile_hash": profile_hash,
            "unique_clips": len(selected),
            "completed": len(completed),
            "completed_behavior_abstentions": sum(r["predicted_class"] is None for r in completed),
            "completed_prediction_coverage": (
                sum(r["predicted_class"] is not None for r in completed) / len(completed)
                if completed
                else None
            ),
            "model_disagreement_windows": sum(
                bool(w.get("model_disagreement"))
                for r in completed
                for w in r["behavior"]["windows"]
            ),
            "failed_or_pending": len(selected) - len(completed),
            "reference_behavior_clips": len(labeled),
            "covered_behavior_clips": len(covered),
            "abstentions": len(labeled) - len(covered),
            "coverage": len(covered) / len(labeled) if labeled else None,
            "unlabeled_completed": sum(r["reference"] is None for r in completed),
            "item_ids": [r["item_id"] for r in selected],
        },
        "behavior_all_labeled_abstentions_incorrect": classification(truth, predicted, CLASSES),
        "behavior_covered_only": classification(
            [r["reference"]["payload"]["behavior"] for r in covered],
            [r["predicted_class"] for r in covered],
            CLASSES,
        ),
        "binary_violence": {
            "mapping": "Fighting=true; Normal=false; all other classes abstain",
            "reference_support": len(binary),
            "covered_support": len(binary_covered),
            "coverage": len(binary_covered) / len(binary) if binary else None,
            **classification(
                [r["reference"]["payload"]["violence"] for r in binary],
                [
                    True
                    if r["predicted_class"] == "Fighting"
                    else False
                    if r["predicted_class"] == "Normal"
                    else None
                    for r in binary
                ],
                [False, True],
            ),
        },
        "temporal_false_alerts": {
            "reference_support": len(temporal),
            "false_alert_count": false_alerts,
            "alert_count": alert_count,
            "negative_seconds": negative_seconds,
            "false_alert_fraction": false_alerts / alert_count if alert_count else None,
            "false_alerts_per_negative_hour": false_alerts * 3600 / negative_seconds
            if negative_seconds
            else None,
            "false_alert_seconds": false_seconds,
            "false_positive_time_fraction": false_seconds / negative_seconds
            if negative_seconds
            else None,
        },
        "composition": {
            "source_count": len({r["source_video_id"] for r in selected}),
            "assignment": dict(Counter(r["assignment"] for r in selected)),
            "selection_method": dict(Counter(r["selection_method"] for r in selected)),
            "geographic_confidence": dict(Counter(r["geography"]["confidence"] for r in selected)),
        },
        "limitations": "Curated libraries are not unbiased benchmarks. "
        "Uncalibrated research scores. "
        "Synthetic fixture results are software tests, never benchmark performance.",
        "generated_at": now(),
    }


def export_json(catalog, profile_hash, *, include_references=False, item_ids=None):
    return encode(
        {
            "schema_version": 1,
            "generated_at": now(),
            "records": records(catalog, include_references=include_references, item_ids=item_ids),
            "evaluation": metrics(catalog, profile_hash),
        }
    )


def export_csv(catalog, *, include_references=False, item_ids=None):
    rows = records(catalog, include_references=include_references, item_ids=item_ids)
    fields = [
        "clip_id",
        "item_id",
        "source_video_id",
        "source",
        "status",
        "assignment",
        "cohort",
        "annotation_status",
        "execution_mode",
        "predicted_class",
        "anomaly_scores",
        "reference",
        "review",
        "submitted_at",
        "started_at",
        "finished_at",
        "inference_duration",
        "profile_hash",
        "config_sha256",
        "model_provenance",
        "failure_reason",
        "abstention_reason",
        "license",
        "external_handover_permission",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        values = {}
        for field in fields:
            value = row[field]
            value = encode(value) if isinstance(value, (dict, list)) else value
            # Spreadsheet formula injection must not execute reviewer/source metadata.
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            values[field] = value
        writer.writerow(values)
    return output.getvalue()


def printable_report(catalog, profile_hash):
    data = metrics(catalog, profile_hash)
    rows = records(catalog)
    body = "".join(
        "<tr>"
        + "".join(
            "<td>" + html.escape(str(r.get(k) or "—")) + "</td>"
            for k in ("clip_id", "execution_mode", "status", "predicted_class", "failure_reason")
        )
        + "</tr>"
        for r in rows
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Municipal evaluation report</title>"
        "<style>body{font:15px Segoe UI,sans-serif;color:#213C4E;margin:36px}"
        "table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #CBD6DE}"
        "pre{white-space:pre-wrap}@media print{button{display:none}}</style>"
        "<h1>Municipal evaluation · research prototype</h1>"
        "<p>Use the browser Print command to save a PDF. "
        "No raw video or facial imagery is embedded.</p>"
        "<p>UCF ROC-AUC 0.7441 · behavior macro F1 0.2140 · cascade alert F1 0.4733. "
        "Historical protocols; these are not this library’s results. Human review required.</p>"
        "<h2>Untouched evaluation / actual inference</h2><pre>"
        + html.escape(json.dumps(data, indent=2))
        + "</pre><h2>Processing records (all populations, explicitly marked)</h2>"
        "<table><tr><th>Clip</th><th>Execution</th><th>Status</th><th>Behavior</th><th>Failure</th></tr>"
        + body
        + "</table></html>"
    )
