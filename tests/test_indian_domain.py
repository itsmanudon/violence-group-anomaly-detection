import csv
import json

import cv2
import numpy as np
import pytest

from surveillance.datasets.indian_domain import (
    benchmark_inventory,
    import_annotations,
    inspect_header,
    near_duplicate,
    normalize_label,
)


def make_video(path, seed=0):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (32, 24))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8))
    writer.release()


def annotations(path, rows, header=None):
    header = header or ["clip_id", "file_path", "label", "violence_start", "violence_end"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize(
    "raw,expected",
    [(" Violence ", "violence"), ("NON-VIOLENCE", "non_violence"), ("not sure", "not_sure")],
)
def test_normalization(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize("raw", ["normal", "abuse", "1", "0", "uncertain", "violent", ""])
def test_ambiguous_labels_rejected(raw):
    with pytest.raises(ValueError, match="unsupported"):
        normalize_label(raw)


def test_header_inspected_before_mapping(tmp_path):
    path = annotations(
        tmp_path / "labels.csv",
        [{"ID": "x", "Video": "x.avi", "Class": "violence"}],
        ["ID", "Video", "Class"],
    )
    assert inspect_header(path) == ["ID", "Video", "Class"]
    with pytest.raises(ValueError, match="mapping"):
        import_annotations(path, tmp_path, tmp_path / "bad")
    make_video(tmp_path / "x.avi")
    report = import_annotations(
        path,
        tmp_path,
        tmp_path / "good",
        columns={"clip_id": "ID", "file_path": "Video", "label": "Class"},
    )
    assert report["accepted_binary_clips"] == 1
    assert report["unresolved_provenance"]
    stage = json.loads((tmp_path / "good/import.json").read_text())
    assert stage["records"][0]["label"] == 1
    assert stage["records"][0]["temporal_annotations"] is None
    assert "split" not in stage["records"][0]
    assert stage["records"][0]["raw"]["Class"] == "violence"


@pytest.mark.parametrize("header", ["", "clip_id,label,label", "clip_id,,label"])
def test_invalid_header(tmp_path, header):
    path = tmp_path / "bad.csv"
    path.write_text(header)
    with pytest.raises(ValueError, match="header"):
        inspect_header(path)


def test_quarantine_and_content_duplicates(tmp_path):
    make_video(tmp_path / "a.avi")
    (tmp_path / "copy.avi").write_bytes((tmp_path / "a.avi").read_bytes())
    rows = [
        {"clip_id": "positive", "file_path": "a.avi", "label": "violence"},
        {"clip_id": "copy", "file_path": "copy.avi", "label": "non_violence"},
        {"clip_id": "uncertain", "file_path": "absent.avi", "label": "not_sure"},
        {"clip_id": "missing", "file_path": "absent.avi", "label": "violence"},
        {"clip_id": "duplicate", "file_path": "a.avi", "label": "violence"},
        {"clip_id": "duplicate", "file_path": "a.avi", "label": "non_violence"},
        {"clip_id": "", "file_path": "a.avi", "label": "violence"},
    ]
    path = annotations(tmp_path / "labels.csv", rows)
    report = import_annotations(path, tmp_path, tmp_path / "v1")
    assert report["accepted_binary_clips"] == 2
    assert report["quarantined_rows"] == 5
    assert report["quarantine_reasons"]["uncertain_label"] == 1
    assert report["quarantine_reasons"]["duplicate_clip_id"] == 2
    assert report["duplicate_candidates"] == [{"pair": ["copy", "positive"], "kind": "exact"}]
    with pytest.raises(FileExistsError):
        import_annotations(path, tmp_path, tmp_path / "v1")


@pytest.mark.parametrize(
    "label,start,end,accepted",
    [
        ("violence", "", "", True),
        ("non_violence", "", "", True),
        ("violence", "0.2", "0.8", True),
        ("violence", "0.2", "", False),
        ("violence", "nan", "1", False),
        ("violence", "0", "2", False),
        ("non_violence", "0", "0.5", False),
    ],
)
def test_temporal_semantics(tmp_path, label, start, end, accepted):
    make_video(tmp_path / "a.avi")
    csv_path = annotations(
        tmp_path / "a.csv",
        [
            {
                "clip_id": "a",
                "file_path": "a.avi",
                "label": label,
                "violence_start": start,
                "violence_end": end,
            }
        ],
    )
    result = import_annotations(csv_path, tmp_path, tmp_path / "import")
    assert result["accepted_binary_clips"] == int(accepted)
    if accepted:
        row = json.loads((tmp_path / "import/import.json").read_text())["records"][0]
        assert row["temporal_annotations"] == ([[0.2, 0.8]] if start else None)


def test_path_readability_and_provenance_conflict(tmp_path):
    (tmp_path / "broken.avi").write_bytes(b"not a video")
    path = annotations(
        tmp_path / "a.csv",
        [
            {"clip_id": "escape", "file_path": "../escape.avi", "label": "violence"},
            {"clip_id": "broken", "file_path": "broken.avi", "label": "violence"},
            {"clip_id": "empty", "file_path": "", "label": "violence"},
        ],
    )
    report = import_annotations(path, tmp_path, tmp_path / "import")
    assert report["accepted_binary_clips"] == 0
    assert report["quarantined_rows"] == 3


def test_visual_screening_and_benchmark_schema(tmp_path):
    assert near_duplicate(["0000000000000000"] * 8, ["0000000000000001"] * 8)
    assert not near_duplicate(["0000000000000000"] * 8, ["ffffffffffffffff"] * 8)
    manifest = tmp_path / "old.jsonl"
    manifest.write_text(json.dumps({"source_video_id": "ucf:one", "sha256": "a" * 64}) + "\n")
    inventory = benchmark_inventory([manifest], "Explicit historical inventory; fixture")
    assert inventory["source_video_ids"] == ["ucf:one"]
    assert inventory["video_sha256"] == ["a" * 64]
    assert inventory["missing_video_hashes"] == []


def test_malformed_row_width_quarantines_raw_extras(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("clip_id,file_path,label\nx,x.avi,violence,extra\ny,y.avi\n")
    report = import_annotations(path, tmp_path, tmp_path / "import")
    assert report["quarantined_rows"] == 2
    quarantine = json.loads((tmp_path / "import/quarantine.json").read_text())
    assert quarantine[0]["raw"]["__extra_fields__"] == ["extra"]


def test_provenance_conflict_quarantines_and_preserves_original(tmp_path):
    path = annotations(
        tmp_path / "a.csv",
        [{"clip_id": "x", "file_path": "x.avi", "label": "violence", "source_video_id": "one"}],
        ["clip_id", "file_path", "label", "source_video_id"],
    )
    report = import_annotations(
        path, tmp_path, tmp_path / "import", provenance={"x": {"source_video_id": "two"}}
    )
    assert report["quarantined_rows"] == 1
    assert report["quarantine_reasons"]["conflicting provenance: source_video_id"] == 1
