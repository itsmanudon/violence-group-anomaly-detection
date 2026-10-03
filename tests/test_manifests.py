from dataclasses import replace

import pytest

from surveillance.datasets.common import (
    Record,
    check_leakage,
    read_manifest,
    split_records,
    write_manifest,
)


def test_roundtrip_and_leakage(tmp_path):
    rows = [
        Record("dcsass", "clip1", "shared", "a.mp4", "train", 1),
        Record("ucf_crime", "clip2", "shared", "b.mp4", "test", 1),
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(rows, path)
    assert read_manifest(path) == rows
    with pytest.raises(ValueError, match="leakage"):
        check_leakage(rows)
    check_leakage([rows[0], replace(rows[1], split="train")])


def test_split_is_deterministic_and_grouped():
    rows = [Record("dcsass", str(i), str(i // 2), f"{i}.mp4", "train", i % 2) for i in range(40)]
    a = split_records(rows, seed=7)
    assert a == split_records(list(reversed(rows)), seed=7)
    check_leakage(a)
    assert {r.split for r in a} == {"train", "val", "test"}


def test_invalid_manifest(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"dataset": "x"}\n')
    with pytest.raises(ValueError, match="line 1"):
        read_manifest(path)


@pytest.mark.parametrize("ratios", [(0.85, 0.15, 0), (0, 0.5, 0.5), (0.3, 0, 0.7)])
def test_zero_ratio_split_never_receives_rounding_remainder(ratios):
    rows = [Record("ucf_crime", str(i), str(i), f"{i}.mp4", "train", i % 2) for i in range(3)]
    result = split_records(rows, ratios=ratios)
    zero_splits = {name for name, ratio in zip(("train", "val", "test"), ratios) if ratio == 0}
    assert not {r.split for r in result} & zero_splits
