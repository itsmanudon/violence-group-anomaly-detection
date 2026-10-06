import pytest


def test_changing_video_source_clears_old_evidence_and_upload_selects_live_mode(tmp_path):
    gr = pytest.importorskip("gradio")
    from surveillance.demo import create_demo
    from surveillance.inference.provenance import ASSET_KEYS

    config = {key: key + ".pt" for key in ASSET_KEYS}
    config.update(examples="missing.json", outputs="outputs", device="cpu")
    app = create_demo(config, tmp_path)
    source = next(block for block in app.blocks.values() if isinstance(block, gr.Dropdown))
    video = next(block for block in app.blocks.values() if isinstance(block, gr.Video))
    alert = next(
        block
        for block in app.blocks.values()
        if isinstance(block, gr.Markdown) and "alert-state" in block.elem_classes
    )
    changes = [d for d in app.config["dependencies"] if (source._id, "input") in d["targets"]]
    assert any(alert._id in d["outputs"] for d in changes), (
        "Old prediction must clear on new source"
    )
    uploads = [d for d in app.config["dependencies"] if (video._id, "upload") in d["targets"]]
    assert any(source._id in d["outputs"] and alert._id in d["outputs"] for d in uploads)
    callback = app.fns[next(d["id"] for d in uploads if source._id in d["outputs"])].fn
    result = callback()
    assert result[0] == "Upload a video"
    assert "Ready to inspect" in result[1]
    preview = next(fn.fn for fn in app.fns.values() if fn.api_name == "preview_example")
    assert preview("Upload a video")[0] == gr.skip(), (
        "Source update must preserve freshly uploaded bytes"
    )
