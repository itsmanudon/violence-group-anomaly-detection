import json

import cv2
import numpy as np

from surveillance.datasets.dcsass_audit import audit_dcsass


def test_real_decode_audit_records_short_clips_bad_labels_and_corrupt_video(tmp_path):
    source = tmp_path / "Abuse" / "Abuse001_x264.mp4"
    source.mkdir(parents=True)
    for index in (0, 1):
        path = source / f"Abuse001_x264_{index}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
        assert writer.isOpened()
        for frame in range(3):
            writer.write(np.full((24, 32, 3), frame * 20, dtype=np.uint8))
        writer.release()
    (source / "Abuse001_x264_2.mp4").write_bytes(b"corrupt")
    labels = tmp_path / "Labels"
    labels.mkdir()
    (labels / "Abuse.csv").write_text(
        "Abuse001_x264_0,Abuse,0\nAbuse001_x264_1,Abuse,\nAbuse001_x264_2,Abuse,1\n"
    )
    report = audit_dcsass(tmp_path, workers=1)
    assert report["total_clips"] == 3
    assert report["readable_clips"] == 2
    assert report["unreadable_clips"] == 1
    assert report["selected_class_counts"] == {"Normal": 1}
    assert report["source_count"] == 1
    rows = {r["clip_id"]: r for r in report["clips"]}
    assert rows["Abuse001_x264_0"]["num_frames"] == 3
    assert rows["Abuse001_x264_0"]["duration_sec"] == 0.3
    assert rows["Abuse001_x264_1"]["binary_label"] is None
    assert rows["Abuse001_x264_1"]["exclusion_reason"] == "missing_or_invalid_annotation"
    assert not rows["Abuse001_x264_2"]["selected"]
    assert report["duplicates"] == [["Abuse001_x264_0", "Abuse001_x264_1"]]
    json.dumps(report, allow_nan=False)
