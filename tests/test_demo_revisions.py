def test_delayed_analysis_and_preview_responses_cannot_restore_old_video():
    from surveillance.demo_state import SessionRevisions

    gate = SessionRevisions()
    assert gate.observe("browser-a", 1)
    assert gate.current("browser-a", 1)
    # While analysis/preview A is delayed, the user uploads or selects B.
    assert gate.observe("browser-a", 2)
    assert not gate.current("browser-a", 1)
    assert not gate.observe("browser-a", 1)  # Old queued request starts late.
    assert gate.current("browser-a", 2)
    assert gate.observe("browser-b", 1)
    assert gate.current("browser-b", 1)
    assert gate.current("browser-a", 2)
    assert gate.observe(None, 0) and gate.current(None, 0)  # Stateless API calls.


def test_actual_callback_suppresses_delayed_result_after_new_source(tmp_path, monkeypatch):
    from threading import Event, Thread
    from types import SimpleNamespace

    import gradio as gr

    from surveillance.demo import create_demo
    from surveillance.inference.provenance import ASSET_KEYS

    started, release = Event(), Event()

    def delayed(*args):
        started.set()
        assert release.wait(5)
        return {}  # Must be suppressed before plotting or serializing this stale response.

    monkeypatch.setattr("surveillance.demo.DemoService.analyze", delayed)
    config = {key: key + ".pt" for key in ASSET_KEYS}
    config.update(examples="missing.json", outputs="outputs", device="cpu")
    app = create_demo(config, tmp_path)
    analyze = next(fn.fn for fn in app.fns.values() if fn.api_name == "analyze_video")
    preview = next(fn.fn for fn in app.fns.values() if fn.api_name == "preview_example")
    request = SimpleNamespace(session_hash="browser")
    results = []
    worker = Thread(
        target=lambda: results.append(analyze("a.mp4", "Upload a video", False, 1, request))
    )
    worker.start()
    try:
        assert started.wait(5)
        preview("Upload a video", 2, request)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert results == [(gr.skip(),) * 10]


def test_source_change_during_rendering_suppresses_finished_response(tmp_path, monkeypatch):
    from threading import Event, Thread
    from types import SimpleNamespace

    import gradio as gr

    from surveillance.demo import create_demo
    from surveillance.inference.provenance import ASSET_KEYS

    started, release = Event(), Event()

    def delayed_gallery(result):
        started.set()
        assert release.wait(5)
        return []

    result = {
        "final_alert": {"state": "no_anomaly", "reason": "Normal", "behavior": None},
        "timings": {"total_seconds": 1},
        "execution_mode": "on_demand",
        "video_path": "a.mp4",
        "anomaly": {"overall_score": 0.1},
    }
    monkeypatch.setattr("surveillance.demo.DemoService.analyze", lambda *args: result)
    monkeypatch.setattr("surveillance.demo.behavior_probabilities", lambda r: ({}, ""))
    monkeypatch.setattr("surveillance.demo.timeline_figure", lambda r: None)
    monkeypatch.setattr("surveillance.demo.reference_gallery", delayed_gallery)
    monkeypatch.setattr("surveillance.demo.interval_rows", lambda r: [])
    config = {key: key + ".pt" for key in ASSET_KEYS}
    config.update(examples="missing.json", outputs="outputs", device="cpu")
    app = create_demo(config, tmp_path)
    analyze = next(fn.fn for fn in app.fns.values() if fn.api_name == "analyze_video")
    preview = next(fn.fn for fn in app.fns.values() if fn.api_name == "preview_example")
    request = SimpleNamespace(session_hash="browser")
    results = []
    worker = Thread(
        target=lambda: results.append(analyze("a.mp4", "Upload a video", False, 1, request))
    )
    worker.start()
    try:
        assert started.wait(5)
        preview("Upload a video", 2, request)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert results == [(gr.skip(),) * 10]
