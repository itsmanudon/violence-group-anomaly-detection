"""Presentation of real cascade outputs; boxes belong only to analyzed reference frames."""

import cv2
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from surveillance.datasets.dcsass_protocol import CLASSES


def timeline_figure(result: dict) -> Figure:
    anomaly = result["anomaly"]
    figure = Figure(figsize=(12, 2.6), layout="constrained")
    axes = figure.subplots()
    times = [(a + b) / 2 for a, b in anomaly["timestamps"]]
    axes.plot(times, anomaly["segment_scores"], color="#287C8E", linewidth=2)
    axes.axhline(
        result["policy"]["anomaly_threshold"],
        color="#A84857",
        linestyle="--",
        linewidth=1,
        label="Routing threshold",
    )
    for index, window in enumerate(result["behavior"]["windows"]):
        axes.axvspan(
            *window["interval"],
            color="#B77B27",
            alpha=0.16,
            label="Behavior window" if index == 0 else None,
        )
    axes.set(
        xlabel="Video time (seconds)",
        ylabel="Anomaly score",
        ylim=(0, 1),
        xlim=(0, anomaly["metadata"]["duration_sec"]),
    )
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", alpha=0.15)
    axes.legend(loc="upper right", frameon=False)
    return figure


def reference_gallery(result: dict) -> list:
    output = []
    cap = cv2.VideoCapture(str(result["video_path"]))
    try:
        for window in result["behavior"]["windows"]:
            if "reference_frame" not in window:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, window["reference_frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            for index, (box, score) in enumerate(
                zip(window["boxes"], window["detector_scores"], strict=True), 1
            ):
                a, b, c, d = [int(v) for v in box]
                cv2.rectangle(frame, (a, b), (c, d), (180, 180, 40), 2)
                cv2.putText(
                    frame,
                    f"Actor {index}  {score:.2f}",
                    (a, max(12, b - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            label = window["predicted_behavior"] or "No actors — behavior unavailable"
            caption = (
                f"{window['reference_timestamp']:.2f}s · {window['actor_count']} actors · {label}"
            )
            output.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption))
    finally:
        cap.release()
    return output


def behavior_probabilities(result: dict) -> tuple[dict, str]:
    windows = result["behavior"]["windows"]
    available = [w for w in windows if w["status"] == "available"]
    if not available:
        return (
            {},
            "Behavior analysis unavailable" if windows else "No anomaly: behavior analysis skipped",
        )
    chosen = next(
        (w for w in available if w["interval"] == result["final_alert"]["timestamp_interval"]),
        available[0],
    )
    return dict(
        zip(CLASSES, chosen["probabilities"], strict=True)
    ), f"Detected behavior · {chosen['interval'][0]:.2f}–{chosen['interval'][1]:.2f}s"


def interval_rows(result: dict) -> list[list]:
    return [
        [
            f"{w['interval'][0]:.2f}–{w['interval'][1]:.2f}s",
            w["actor_count"],
            w["predicted_behavior"] or "Unavailable",
            round(w["anomaly_score"], 3),
            "Disagree" if w["model_disagreement"] else w["status"],
        ]
        for w in result["behavior"]["windows"]
    ]
