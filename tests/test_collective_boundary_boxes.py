"""Visible boundary actors remain supervised, with explicit correction provenance."""

import pytest
from test_collective_validation import sequence

from surveillance.datasets.collective import parse_collective_annotations, prepare_collective
from surveillance.datasets.collective_validation import validate_collective
from surveillance.experiments.protocol import load_protocol


def test_visible_box_clipping_preserves_order_labels_and_audits_all_rows(tmp_path):
    path = tmp_path / "annotations.txt"
    path.write_text("1 -2 1 12 10 2\n1 15 4 6 9 6\n2 -1 0 5 5 1\n")
    corrections = []
    scenes = parse_collective_annotations(
        path, (12, 20), box_policy="clip_to_image", corrections=corrections
    )
    assert scenes[1]["actor_boxes"] == [[0, 1 / 12, 0.5, 11 / 12], [0.75, 1 / 3, 1, 1]]
    assert scenes[1]["actor_labels"] == [0, 4]
    assert scenes[1]["group_label"] == 0
    assert len(corrections) == 3
    assert corrections[0]["raw_xywh"] == [-2, 1, 12, 10]
    assert corrections[0]["clipped_xyxy"] == [0, 1, 10, 11]
    assert corrections[0]["annotation_line"] == 1
    assert corrections[0]["selected_supervised"] is True
    assert corrections[2]["selected_supervised"] is False
    assert path.read_text().startswith("1 -2")


@pytest.mark.parametrize("raw", ["1 20 0 2 4 2", "1 -5 0 5 4 2", "1 0 0 0 4 2", "1 0 0 4 -2 2"])
def test_clipping_rejects_fully_outside_and_nonpositive_boxes(tmp_path, raw):
    path = tmp_path / "annotations.txt"
    path.write_text(raw)
    with pytest.raises(ValueError, match="positive|visible"):
        parse_collective_annotations(path, (12, 20), box_policy="clip_to_image")


def test_default_remains_strict_and_clipping_without_audit_warns(tmp_path):
    path = tmp_path / "annotations.txt"
    path.write_text("1 -1 0 5 5 2\n")
    with pytest.raises(ValueError, match="inside"):
        parse_collective_annotations(path, (12, 20))
    with pytest.warns(UserWarning, match="Clipped 1"):
        parse_collective_annotations(path, (12, 20), box_policy="clip_to_image")
    with pytest.raises(ValueError, match="box_policy"):
        parse_collective_annotations(path, (12, 20), box_policy="ignore")


def test_canonical_duplicates_after_clipping_fail(tmp_path):
    path = tmp_path / "annotations.txt"
    path.write_text("1 -1 0 6 5 2\n1 -2 0 7 5 3\n")
    with pytest.raises(ValueError, match="duplicate"):
        parse_collective_annotations(path, (12, 20), box_policy="clip_to_image", corrections=[])


def test_validator_and_preparation_use_same_audited_policy(tmp_path):
    for sid in (1, 5):
        folder = sequence(tmp_path, sid)
        (folder / "annotations.txt").write_text("1 -1 0 6 5 2\n11 16 9 5 4 6\n")
    report = validate_collective(
        tmp_path, (1,), require_full_split=False, box_policy="clip_to_image"
    )
    corrections = []
    rows = prepare_collective(
        tmp_path, [1], [5], False, box_policy="clip_to_image", corrections=corrections
    )
    assert report["valid"] and report["invalid_boxes"] == []
    assert report["box_policy"] == "clip_to_image"
    assert report["box_corrections"] == corrections
    assert len(corrections) == 4
    assert report["counts"]["actors"] == sum(len(r.actor_labels) for r in rows) == 4
    assert report["warnings"]


def test_protocol_policy_is_explicit_and_invalid_policy_rejected(tmp_path):
    from pathlib import Path

    import yaml

    original = yaml.safe_load(Path("configs/experiments/collective_protocol.yaml").read_text())
    path = tmp_path / "protocol.yaml"
    original["dataset"]["annotation_box_policy"] = "clip_to_image"
    path.write_text(yaml.safe_dump(original))
    assert load_protocol(path)["dataset"]["annotation_box_policy"] == "clip_to_image"
    original["dataset"]["annotation_box_policy"] = "ignore"
    path.write_text(yaml.safe_dump(original))
    with pytest.raises(ValueError, match="annotation_box_policy"):
        load_protocol(path)


def test_real_preparation_propagates_policy_and_manifest_validation(tmp_path):
    from surveillance.experiments.preparation import prepare_data

    for sid in range(1, 45):
        folder = sequence(tmp_path, sid)
        (folder / "annotations.txt").write_text("1 -1 0 6 5 2\n11 16 9 5 4 6\n")
    output = tmp_path / "manifest.jsonl"
    protocol = {
        "dataset": {
            "root": str(tmp_path),
            "manifest": str(output),
            "validation_sequences": [1, 2, 3],
            "annotation_box_policy": "clip_to_image",
        },
        "output_root": str(tmp_path / "run"),
    }
    report = prepare_data(protocol)
    checked = validate_collective(tmp_path, manifest=output, box_policy="clip_to_image")
    assert checked["valid"]
    assert checked["box_corrections"] == report["box_corrections"]
    assert [checked["splits"][s]["sequences"] for s in ("train", "val", "test")] == [29, 3, 12]
    assert (tmp_path / "run" / "dataset_validation.json").exists()
