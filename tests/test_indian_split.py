import json
from dataclasses import replace

import pytest
from test_indian_domain import annotations, make_video

from surveillance.datasets.common import read_manifest, write_manifest
from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import (
    benchmark_inventory,
    import_annotations,
    write_new_json,
)
from surveillance.datasets.indian_split import (
    assign_groups,
    freeze_split,
    source_groups,
    verify_split,
)


def split_fixture(tmp_path):
    raw, provenance = [], {}
    for i in range(12):
        key = f"c{i:02d}"
        make_video(tmp_path / f"{key}.avi", seed=i)
        raw.append(
            {
                "clip_id": key,
                "file_path": f"{key}.avi",
                "label": "violence" if i % 2 else "non_violence",
            }
        )
        provenance[key] = {
            "source_video_id": f"source:{i}",
            "incident_id": f"incident:{i}",
            "source": "synthetic test generator",
        }
    csv_path = annotations(tmp_path / "labels.csv", raw)
    import_annotations(csv_path, tmp_path, tmp_path / "import", provenance=provenance)
    imported = tmp_path / "import/import.json"
    historical = tmp_path / "historical.jsonl"
    historical.write_text(
        json.dumps({"source_video_id": "historical:unrelated", "sha256": "a" * 64})
    )
    inventory_path = tmp_path / "inventory.json"
    write_new_json(inventory_path, benchmark_inventory([historical], "synthetic fixture scope"))
    stage = json.loads(imported.read_text())
    review = {
        "import_sha256": sha256(imported),
        "benchmark_inventory_sha256": sha256(inventory_path),
        "benchmark_scope_complete": True,
        "benchmark_scope_evidence": "synthetic scope only",
        "excluded_clips": {},
        "clip_reviews": {
            r["clip_id"]: {
                "annotation_approved": True,
                "provenance_approved": True,
                "benchmark_independent": True,
                "evidence": "synthetic fixture inspection",
            }
            for r in stage["records"]
        },
        "candidate_decisions": [
            {"pair": c["pair"], "relation": "independent", "evidence": "different generated scenes"}
            for c in stage["duplicate_candidates"]
        ],
    }
    review_path = tmp_path / "review.json"
    write_new_json(review_path, review)
    return imported, inventory_path, review_path, tmp_path / "split"


def test_freeze_roundtrip_and_no_overwrite(tmp_path):
    args = split_fixture(tmp_path)
    receipt = freeze_split(*args)
    rows, verified = verify_split(args[-1])
    assert receipt == verified
    assert receipt["clip_counts"] == {"train": 8, "val": 2, "test": 2}
    assert all(
        count > 0 for counts in receipt["class_counts"].values() for count in counts.values()
    )
    assert sorted(r.video_id for r in rows if r.split == "test") == receipt["test_clip_ids"]
    assert freeze_split(*args[:-1], tmp_path / "split2") == receipt
    with pytest.raises(FileExistsError):
        freeze_split(*args)
    path = args[-1] / "manifest.jsonl"
    rows[0] = replace(rows[0], split="test" if rows[0].split == "train" else "train")
    write_manifest(rows, path)
    with pytest.raises(ValueError, match="manifest changed"):
        verify_split(args[-1])


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("review", "stale"),
        ("source", "provenance"),
        ("benchmark", "overlap"),
        ("unreviewed", "Unresolved"),
        ("candidate", "Unresolved"),
    ],
)
def test_reject_unresolved_inputs(tmp_path, mutation, error):
    args = split_fixture(tmp_path)
    stage, review = json.loads(args[0].read_text()), json.loads(args[2].read_text())
    if mutation == "review":
        review["import_sha256"] = "b" * 64
    elif mutation == "source":
        stage["records"][0]["source_video_id"] = "unknown"
    elif mutation == "benchmark":
        stage["records"][0]["source_video_id"] = "historical:unrelated"
    elif mutation == "unreviewed":
        review["clip_reviews"]["c00"]["provenance_approved"] = False
    else:
        stage["duplicate_candidates"].append({"pair": ["c00", "c01"], "kind": "visual_candidate"})
        review["candidate_decisions"] = []
    args[0].write_text(json.dumps(stage))
    if mutation != "review":
        review["import_sha256"] = sha256(args[0])
    args[2].write_text(json.dumps(review))
    with pytest.raises(ValueError, match=error):
        freeze_split(*args)
    assert not args[-1].exists()


def test_connected_groups_and_conflicting_duplicate_labels():
    rows = [
        {
            "clip_id": f"c{i}",
            "source_video_id": f"s{i}",
            "incident_id": f"i{i}",
            "sha256": str(i) * 64,
            "label": i % 2,
        }
        for i in range(5)
    ]
    rows[1]["source_video_id"] = rows[0]["source_video_id"]
    rows[2]["incident_id"] = rows[1]["incident_id"]
    rows[3]["sha256"], rows[3]["label"] = rows[2]["sha256"], rows[2]["label"]
    assert source_groups(rows, [["c3", "c4"]]) == [[f"c{i}" for i in range(5)]]
    rows[3]["label"] = 1
    with pytest.raises(ValueError, match="conflicting"):
        source_groups(rows, [])


def test_stratification_is_order_independent_and_reports_group_ratios():
    rows = [{"clip_id": f"c{i}", "label": i % 2} for i in range(20)]
    groups = [[r["clip_id"]] for r in rows]
    a = assign_groups(rows, groups, 7, (0.7, 0.15, 0.15))
    assert a == assign_groups(list(reversed(rows)), groups, 7, (0.7, 0.15, 0.15))
    assert list(a.values()).count("train") == 14
    with pytest.raises(ValueError, match="three independent"):
        assign_groups(rows[:4], groups[:4], 7, (0.7, 0.15, 0.15))


def test_changed_media_rejected(tmp_path):
    args = split_fixture(tmp_path)
    freeze_split(*args)
    row = read_manifest(args[-1] / "manifest.jsonl")[0]
    from pathlib import Path

    Path(row.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="video changed"):
        verify_split(args[-1])


def test_three_mixed_class_incidents_keep_every_partition_nonempty():
    rows = [
        {"clip_id": f"g{group}-{label}", "label": label} for group in range(3) for label in (0, 1)
    ]
    groups = [[f"g{group}-0", f"g{group}-1"] for group in range(3)]
    assignments = assign_groups(rows, groups, 7, (0.7, 0.15, 0.15))
    assert set(assignments.values()) == {"train", "val", "test"}
    for group in groups:
        assert assignments[group[0]] == assignments[group[1]]
