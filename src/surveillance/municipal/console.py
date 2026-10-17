"""Six-screen municipal extension of the installed React-backed Gradio console."""

import json
import os
import shutil
import uuid
from pathlib import Path

from surveillance.demo import CSS
from surveillance.demo_rendering import (
    behavior_probabilities,
    interval_rows,
    reference_gallery,
    timeline_figure,
)
from surveillance.municipal.catalog import inside
from surveillance.municipal.evaluation import records


def create_console(service):
    import gradio as gr

    def metric_text(data):
        population = data["population"]
        score = data["behavior_all_labeled_abstentions_incorrect"]

        def display(value):
            return "No supported reference labels" if value is None else f"{value:.3f}"

        return (
            f"**{population['reference_behavior_clips']} verified behavior references** · "
            f"{population['covered_behavior_clips']} classified · "
            f"{population['abstentions']} abstentions\n\n"
            f"Accuracy: {display(score['accuracy'])} · Macro F1: {display(score['macro_f1'])}\n\n"
            f"Source recordings: {data['composition']['source_count']}. "
            "Selection and source composition are recorded in the detailed report. "
            "A curated library is not automatically an unbiased benchmark."
        )

    def library(search="", assignment="all", cohort="all"):
        clips = service.catalog.clips(search, None if assignment == "all" else assignment)
        clips = [c for c in clips if cohort == "all" or c["cohort"] == cohort]
        return [
            [
                c["clip_id"],
                c["source_video_id"],
                c["source"],
                c["assignment"],
                c["cohort"],
                c["annotation_status"],
                c["duration"],
                c["license"],
                c["external_handover_permission"],
                c["geography"]["confidence"],
            ]
            for c in clips
        ]

    def queue_rows():
        return [
            [
                j["job_id"],
                i["item_id"],
                i["clip_id"],
                i["state"],
                round(j["progress"] * 100),
                i["attempts"],
                i["error"] or "",
                j["execution_mode"],
            ]
            for j in service.queue.jobs()
            for i in j["items"]
        ]

    def overview():
        h = service.health()
        return (
            f"**{h['library_count']} clips** · **{h['reviews_pending']} reviews pending** · "
            f"**{h['jobs']['running']} running** · **{h['jobs']['queued']} queued**\n\n"
            "Default evaluation: untouched test clips, actual inference, current frozen profile. "
            "Unknown reference labels are excluded from metrics."
        )

    def refresh(search="", assignment="all", cohort="all"):
        clips = service.catalog.clips()
        jobs = service.queue.jobs()
        items = [
            (f"{i['clip_id']} · {j['execution_mode']} · {i['state']}", i["item_id"])
            for j in jobs
            for i in j["items"]
            if i["state"] == "completed"
        ]
        return (
            overview(),
            library(search, assignment, cohort),
            gr.update(choices=[c["clip_id"] for c in clips]),
            queue_rows(),
            gr.update(choices=[j["job_id"] for j in jobs]),
            gr.update(choices=items),
            service.evaluation(),
            metric_text(service.evaluation()),
        )

    def upload(paths, source, license):
        if not paths:
            raise gr.Error("Choose at least one local video")
        results = []
        for filename in paths:
            path = inside(filename, Path(os.environ["GRADIO_TEMP_DIR"]))
            temporary = service.catalog.temporary / (uuid.uuid4().hex + path.suffix.lower())
            try:
                if path.stat().st_size > service.catalog.max_bytes:
                    raise gr.Error("Video exceeds the configured size limit")
                shutil.copyfile(path, temporary)
                results.append(service.import_upload(temporary, source=source, license=license))
            finally:
                temporary.unlink(missing_ok=True)
        return results

    def inspect(item_id):
        if not item_id:
            raise gr.Error("Choose a completed processing item")
        item = service.item(item_id)
        result = item["result"]
        path = service.catalog.media_path(item["clip_id"])
        result["video_path"] = str(path)
        probs, caption = behavior_probabilities(result)
        mode = result["execution_mode"]
        disagreement = sum(bool(w["model_disagreement"]) for w in result["behavior"]["windows"])
        summary = (
            f"**Execution: {mode}** · "
            f"recorded inference {result['timings']['total_seconds']:.2f}s · "
            f"{disagreement} model disagreements.\n\n"
            "Scores and class probabilities are uncalibrated. Actor boxes describe analyzed "
            "reference frames. Human review is required."
        )
        return (
            str(path),
            summary,
            timeline_figure(result),
            reference_gallery(result),
            probs,
            caption,
            interval_rows(result),
            result,
            item_id,
            None,
            "",
            [],
        )

    def save_review(item_id, reviewer, decision, notes, events):
        parsed = json.loads(events) if isinstance(events, str) else events
        parsed = [pair for pair in (parsed or []) if any(value is not None for value in pair)]
        return {"review_id": service.review(item_id, reviewer, decision, notes, parsed or [])}

    with gr.Blocks(
        title="Municipal evaluation · local research console", analytics_enabled=False
    ) as ui:
        gr.Markdown("# Municipal evaluation", elem_classes="research-title")
        gr.Markdown(
            "Local evidence desk · observable behavior · independent human review",
            elem_classes="research-subtitle",
        )
        gr.Markdown(
            "Research prototype. UCF ROC-AUC **0.7441** · behavior macro F1 **0.2140** · "
            "cascade alert F1 **0.4733** under historical protocols. "
            "These systems cannot establish guilt or intent.",
            elem_classes="evidence-strip",
        )
        refresh_button = gr.Button("Refresh library and processing status")
        with gr.Tab("Dashboard"):
            dashboard = gr.Markdown(overview())
            evaluation_summary = gr.Markdown(metric_text(service.evaluation()))
            with gr.Accordion("Inspect evaluation denominators and per-class metrics", open=False):
                evaluation = gr.JSON(
                    service.evaluation(), label="Evaluation overview · references only"
                )
            asset_check = gr.Button("Verify frozen model assets")
            asset_status = gr.JSON(service.asset_report, label="Local inference readiness")
        with gr.Tab("Video library"):
            with gr.Row():
                search = gr.Textbox(label="Search clip or source")
                assignment = gr.Dropdown(
                    ["all", "train", "validation", "test", "demo", "upload"],
                    value="all",
                    label="Assignment",
                )
                cohort = gr.Dropdown(
                    [
                        "all",
                        "curated_demo",
                        "untouched_evaluation",
                        "arbitrary_upload",
                        "synthetic_fixture",
                    ],
                    value="all",
                    label="Population",
                )
            table = gr.Dataframe(
                library(),
                headers=[
                    "Clip ID",
                    "Source recording",
                    "Provenance",
                    "Assignment",
                    "Population",
                    "Annotation status",
                    "Seconds",
                    "License",
                    "Handover",
                    "Geography confidence",
                ],
                interactive=False,
                label="Local clip catalog",
            )
            selected = gr.Dropdown(
                choices=[c["clip_id"] for c in service.catalog.clips()],
                multiselect=True,
                label="Select clips to process",
            )
            mode = gr.Radio(
                ["on_demand", "cached_example"], value="on_demand", label="Execution mode"
            )
            submit = gr.Button("Queue selected clips", variant="primary")
            submission = gr.JSON(label="Submission receipt")
            with gr.Accordion("Import arbitrary local videos", open=False):
                gr.Markdown(
                    "Imports remain unlabeled. Dataset manifests and protected assignments use the "
                    "documented local importer. Nothing is downloaded automatically."
                )
                uploads = gr.File(
                    file_count="multiple",
                    type="filepath",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"],
                    label="Choose local videos",
                )
                source = gr.Textbox(value="unknown", label="Source / provenance")
                license = gr.Textbox(value="unknown", label="Rights / license")
                import_button = gr.Button("Import videos into local library")
                import_status = gr.JSON(label="Import results")
        with gr.Tab("Processing queue"):
            gr.Markdown(
                "GPU inference is sequential. Cancellation takes effect after the current clip. "
                "Restart recovery requeues interrupted work; completed results remain stored."
            )
            queue_table = gr.Dataframe(
                queue_rows(),
                headers=[
                    "Job",
                    "Item",
                    "Clip",
                    "State",
                    "Batch progress %",
                    "Attempts",
                    "Failure details",
                    "Execution",
                ],
                interactive=False,
            )
            job = gr.Dropdown(choices=[j["job_id"] for j in service.queue.jobs()], label="Job")
            with gr.Row():
                cancel = gr.Button("Cancel remaining clips")
                retry = gr.Button("Retry failed or cancelled clips")
            queue_status = gr.JSON(label="Queue action")
        with gr.Tab("Analysis workspace"):
            item_choice = gr.Dropdown(choices=[], label="Completed clip result")
            inspect_button = gr.Button("Open evidence")
            playback = gr.Video(label="Catalog video", include_audio=True)
            evidence = gr.Markdown(
                "Choose a completed clip after refreshing the queue.", elem_classes="evidence-strip"
            )
            timeline = gr.Plot(label="Sultani timeline · 32 temporal segments")
            gallery = gr.Gallery(label="Actor boxes at analyzed reference frames", columns=3)
            probabilities = gr.Label(label="Uncalibrated behavior model probabilities")
            caption = gr.Markdown()
            windows = gr.Dataframe(
                headers=["Interval", "Actors", "Behavior", "Anomaly score", "Agreement"],
                interactive=False,
                label="Routed windows and suspicious timestamps",
            )
            details = gr.JSON(label="Complete model evidence, provenance and stage timings")
        with gr.Tab("Human review"):
            active_item = gr.Textbox(label="Processing item under review", interactive=False)
            gr.Markdown(
                "Open a prediction in Analysis first. Reference labels stay hidden "
                "until explicitly "
                "revealed. Reviews are append-only and never replace verified reference labels."
            )
            reviewer = gr.Textbox(label="Reviewer name / operator identifier")
            decision = gr.Radio(
                ["correct", "incorrect", "uncertain", "not_assessable"],
                value="uncertain",
                label="Assessment",
            )
            notes = gr.Textbox(label="Reviewer notes", lines=3)
            events = gr.Dataframe(
                value=[],
                headers=["Start (seconds)", "End (seconds)"],
                datatype=["number", "number"],
                type="array",
                interactive=True,
                column_count=2,
                label="Observed event times · optional",
            )
            save = gr.Button("Save independent review", variant="primary")
            review_status = gr.JSON(label="Review receipt")
            reveal = gr.Button("Reveal reference labels · marks subsequent reviews unblinded")
            reference = gr.JSON(label="Verified reference · hidden by default")
        with gr.Tab("Results and exports"):
            with gr.Row():
                metric_cohort = gr.Dropdown(
                    [
                        "untouched_evaluation",
                        "curated_demo",
                        "arbitrary_upload",
                        "synthetic_fixture",
                    ],
                    value="untouched_evaluation",
                    label="Population",
                )
                metric_mode = gr.Dropdown(
                    ["on_demand", "cached_example", "synthetic_fixture"],
                    value="on_demand",
                    label="Execution mode",
                )
            calculate = gr.Button("Show population metrics and error examples")
            result_summary = gr.Markdown()
            metrics_output = gr.JSON(
                label="Accuracy, macro F1, precision/recall, abstention and false alerts"
            )
            errors = gr.JSON(label="Operator feedback and failed / incorrect / uncertain examples")
            export_item = gr.Textbox(label="Optional processing item ID for per-clip export")
            include_refs = gr.Checkbox(
                value=False, label="Include supplied reference labels · unblinds exported items"
            )
            format = gr.Radio(
                ["csv", "json", "html"], value="json", label="Export format · HTML is printable"
            )
            export_button = gr.Button("Create local export")
            download = gr.File(label="Download report · no video embedded")
            backup_button = gr.Button("Back up catalog and review history")
            backup = gr.File(label="SQLite backup · copy owned media separately")

        outputs = [
            dashboard,
            table,
            selected,
            queue_table,
            job,
            item_choice,
            evaluation,
            evaluation_summary,
        ]
        refresh_button.click(refresh, [search, assignment, cohort], outputs, api_name=False)
        for component in (search, assignment, cohort):
            component.change(library, [search, assignment, cohort], table, api_name=False)
        asset_check.click(service.check_assets, outputs=asset_status, api_name=False)
        import_button.click(upload, [uploads, source, license], import_status, api_name=False)
        submit.click(
            lambda ids, m: {"job_id": service.queue.submit(ids or [], uuid.uuid4().hex, m)},
            [selected, mode],
            submission,
            api_name=False,
        )
        cancel.click(
            lambda j: service.queue.cancel(j) or {"cancel_requested": True},
            job,
            queue_status,
            api_name=False,
        )
        retry.click(
            lambda j: {"job_id": service.queue.retry(j, uuid.uuid4().hex)},
            job,
            queue_status,
            api_name=False,
        )
        inspect_button.click(
            inspect,
            item_choice,
            [
                playback,
                evidence,
                timeline,
                gallery,
                probabilities,
                caption,
                windows,
                details,
                active_item,
                reference,
                notes,
                events,
            ],
            api_name=False,
        )
        save.click(
            save_review,
            [active_item, reviewer, decision, notes, events],
            review_status,
            api_name=False,
        )
        reveal.click(service.reveal, active_item, reference, api_name=False)

        def results(c, m):
            examples = [
                r
                for r in records(service.catalog)
                if r["cohort"] == c
                and r["execution_mode"] == m
                and (r["status"] == "failed" or r["review"])
            ]
            data = service.evaluation(c, m)
            return data, examples, metric_text(data)

        calculate.click(
            results,
            [metric_cohort, metric_mode],
            [metrics_output, errors, result_summary],
            api_name=False,
        )
        export_button.click(
            lambda f, refs, item: str(
                service.export(f, include_references=refs, item_ids=[item] if item else None)
            ),
            [format, include_refs, export_item],
            download,
            api_name=False,
        )
        backup_button.click(
            lambda: str(
                service.catalog.backup(
                    service.catalog.directory / "exports" / (uuid.uuid4().hex + ".sqlite3")
                )
            ),
            outputs=backup,
            api_name=False,
        )
        timer = gr.Timer(3)
        timer.tick(
            lambda: (overview(), queue_rows()), outputs=[dashboard, queue_table], api_name=False
        )

    ui.municipal_css = (
        CSS
        + """
    .gradio-container .prose strong, .tab-nav button { color:#213C4E !important; }
    .tab-nav button.selected { color:#287C8E !important; }
    @media print {button, footer {display:none !important}}
    """
    )
    ui.municipal_theme = gr.themes.Base(
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
        button_secondary_text_color="#213C4E",
        button_secondary_background_fill="#FFFFFF",
        button_secondary_background_fill_dark="#FFFFFF",
        button_secondary_background_fill_hover="#E4EDF2",
        button_secondary_background_fill_hover_dark="#E4EDF2",
        table_text_color_dark="#213C4E",
        table_even_background_fill_dark="#FFFFFF",
        table_odd_background_fill_dark="#F0F4F7",
    )
    return ui
