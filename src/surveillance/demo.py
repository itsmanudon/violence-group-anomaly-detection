"""One-command local Gradio surveillance research console."""

import argparse
import json
import os
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / "outputs/demo/gradio-temp"))

from surveillance.demo_rendering import (  # noqa: E402
    behavior_probabilities,
    interval_rows,
    reference_gallery,
    timeline_figure,
)
from surveillance.demo_service import DemoService  # noqa: E402
from surveillance.inference.loading import asset_status  # noqa: E402
from surveillance.training.sultani_trainer import select_device  # noqa: E402

CSS = """
body { background: #F0F4F7 !important; }
.gradio-container { max-width: 1280px !important; background: #F0F4F7;
 width: min(1280px, calc(100% - 48px)) !important;
 color: #213C4E; margin: 0 auto !important; }
.gradio-container .prose { color: #213C4E !important; }
.gradio-container .prose h1, .gradio-container .prose h3,
.gradio-container .prose p { color: #213C4E !important; }
.research-title h1 { font-family: Bahnschrift, 'Segoe UI', sans-serif; font-weight: 600;
 font-size: 2.15rem; letter-spacing: -.04em; margin-bottom: .3rem; }
.research-subtitle { color: #213C4E; opacity: .8; }
.evidence-strip { border-left: 4px solid #287C8E !important; }
.alert-state:not(.prose) { border-left: 4px solid #B77B27 !important;
 padding: 16px !important; }
.research-note { font-size: .85rem; opacity: .8; }
button.primary { background: #287C8E !important; border-color: #287C8E !important; }
button:focus-visible { outline: 3px solid #B77B27 !important; outline-offset: 3px; }
@media (max-width: 700px) { .research-title h1 { font-size: 1.6rem; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def create_demo(config: dict, root: Path = ROOT):
    import gradio as gr

    service = DemoService(config, root)
    statuses = asset_status(config, root)
    missing = [
        key.replace("_checkpoint", "").replace("_", " ")
        for key, present in statuses.items()
        if not present
    ]
    status_text = (
        "Live inference assets installed."
        if not missing
        else "Live inference awaits: **"
        + ", ".join(missing)
        + "**. Installed cached examples remain available."
    )
    if select_device(config.get("device", "auto")).type == "cpu":
        status_text += " CPU processing may be slow."
    outputs = Path(root) / config["outputs"]
    outputs.mkdir(parents=True, exist_ok=True)

    def analyze(video, selected, cached):
        try:
            result = service.analyze(video, selected, cached)
            probabilities, caption = behavior_probabilities(result)
            alert = result["final_alert"]
            title = {
                "no_anomaly": "No anomaly detected",
                "generic_anomaly": "General anomaly detected",
                "behavior_review": "Detected behavior: " + str(alert["behavior"]),
            }[alert["state"]]
            seconds = result["timings"]["total_seconds"]
            mode = (
                "Cached example"
                if result["execution_mode"] == "cached_example"
                else "On-demand inference"
            )
            artifact = outputs / (uuid.uuid4().hex + ".json")
            artifact.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            return (
                result["video_path"],
                f"### {title}\n{alert['reason']}",
                timeline_figure(result),
                reference_gallery(result),
                probabilities,
                caption,
                interval_rows(result),
                f"**{mode}** · Anomaly score **{result['anomaly']['overall_score']:.3f}** · "
                f"Inference **{seconds:.2f}s**",
                str(artifact),
                result,
            )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            return (
                video,
                f"### Processing unavailable\n{error}",
                None,
                [],
                {},
                "No behavior result",
                [],
                "Resolve the input or asset issue and try again.",
                None,
                {"error": str(error)},
            )

    with gr.Blocks(title="Surveillance behavior research", analytics_enabled=False) as app:
        gr.Markdown("# Surveillance behavior research", elem_classes="research-title")
        gr.Markdown(
            "Locate unusual activity. Inspect observable behavior.",
            elem_classes="research-subtitle",
        )
        gr.Markdown(status_text, elem_classes="research-note")
        with gr.Row():
            source = gr.Dropdown(
                ["Upload a video", *service.examples],
                value="Upload a video",
                label="Video source",
                scale=3,
            )
            cached = gr.Checkbox(value=True, label="Use precomputed result for examples", scale=2)
        with gr.Row():
            with gr.Column(scale=3):
                upload = gr.Video(
                    sources=["upload"], label="Surveillance video", height=340, include_audio=False
                )
            with gr.Column(scale=2):
                alert = gr.Markdown(
                    "### Ready to inspect\nUpload a video or select an installed example.",
                    elem_classes="alert-state",
                )
                probabilities = gr.Label(label="Detected behavior", num_top_classes=6)
                caption = gr.Markdown("Behavior probabilities appear after analysis.")
                button = gr.Button("Analyze video", variant="primary")
        summary = gr.Markdown("", elem_classes="evidence-strip")
        timeline = gr.Plot(
            label="Anomaly timeline · highlighted intervals receive behavior analysis"
        )
        gallery = gr.Gallery(
            label="Analyzed reference frames · actor boxes and detector confidence",
            columns=3,
            height=260,
            object_fit="contain",
        )
        table = gr.Dataframe(
            headers=["Interval", "Actors", "Detected behavior", "Anomaly score", "Model agreement"],
            datatype=["str", "number", "str", "number", "str"],
            interactive=False,
            label="Interval evidence",
        )
        with gr.Accordion("Inspect model outputs", open=False):
            result_json = gr.JSON(label="Separate anomaly and behavior outputs")
            artifact = gr.File(label="Download result JSON")

        def preview_source(name):
            entry = service.examples.get(name)
            if not entry:
                return None
            path = Path(root) / entry["video"]
            return str(path) if path.is_file() else None

        source.change(preview_source, inputs=source, outputs=upload, api_name="preview_example")
        gr.Markdown(
            "Research prototype; predictions may be wrong and require human review.\n\n"
            "Behavior classification does not identify a person's character or identity.",
            elem_classes="research-note",
        )
        button.click(
            analyze,
            inputs=[upload, source, cached],
            outputs=[
                upload,
                alert,
                timeline,
                gallery,
                probabilities,
                caption,
                table,
                summary,
                artifact,
                result_json,
            ],
            concurrency_limit=1,
            api_name="analyze_video",
        )
    app.research_theme = gr.themes.Base(
        font=["Segoe UI", "Arial", "sans-serif"], font_mono=["Consolas", "monospace"]
    ).set(
        body_background_fill="#F0F4F7",
        body_background_fill_dark="#F0F4F7",
        body_text_color="#213C4E",
        body_text_color_dark="#213C4E",
        body_text_color_subdued_dark="#213C4E",
        block_background_fill="#FFFFFF",
        block_background_fill_dark="#FFFFFF",
        block_border_color="#CBD6DE",
        block_border_color_dark="#CBD6DE",
        input_background_fill="#F0F4F7",
        input_background_fill_dark="#F0F4F7",
        block_title_text_color_dark="#213C4E",
        block_label_text_color_dark="#213C4E",
        block_label_background_fill_dark="#F0F4F7",
        block_info_text_color_dark="#213C4E",
        input_placeholder_color_dark="#213C4E",
        button_secondary_text_color_dark="#213C4E",
        table_text_color_dark="#213C4E",
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surveillance_demo.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    app = create_demo(config)
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        theme=app.research_theme,
        css=CSS,
        footer_links=[],
        allowed_paths=[str(ROOT / "data/raw"), str(ROOT / "data/examples"), str(ROOT / "outputs")],
    )


if __name__ == "__main__":
    main()
