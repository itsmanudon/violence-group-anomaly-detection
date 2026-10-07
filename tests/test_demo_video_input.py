import pytest


def test_uploaded_video_frames_are_passed_unchanged_without_requiring_ffmpeg(tmp_path, monkeypatch):
    gradio = pytest.importorskip("gradio")
    from gradio.data_classes import FileData

    from surveillance.demo import create_demo
    from surveillance.inference.provenance import ASSET_KEYS

    config = {key: key + ".pt" for key in ASSET_KEYS}
    config.update(examples="missing.json", outputs="outputs", device="cpu")
    app = create_demo(config, tmp_path)
    component = next(block for block in app.blocks.values() if isinstance(block, gradio.Video))
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture bytes; this test checks transport, not video inference")
    monkeypatch.setattr(
        "gradio.components.video.FFmpeg",
        lambda *args, **kwargs: pytest.fail(
            "Upload preprocessing must preserve original RGB frames and avoid mandatory transcoding"
        ),
    )
    assert component.preprocess(FileData(path=str(video))) == str(video)
