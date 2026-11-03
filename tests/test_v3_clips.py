import numpy as np
import pytest

from surveillance.datasets.v3_clips import (
    evenly,
    grid_windows,
    intervals_to_labels,
    safe_name,
    split_sources,
    window_frames,
    xd_kind,
    xd_source,
)


def test_window_frames_rounds_two_seconds():
    assert window_frames(25.0) == 50
    assert window_frames(29.97) == 60


def test_evenly_takes_all_when_short_and_spaces_otherwise():
    assert evenly([1, 2], 3) == [1, 2]
    assert evenly(list(range(10)), 2) == [2, 7]
    assert evenly(list(range(9)), 3) == [1, 4, 7]


def test_grid_windows_violent_and_matched_normal_with_margin():
    labels = np.zeros(40, dtype=bool)
    labels[20:30] = True
    violent, normal = grid_windows(labels, width=5, margin=3)
    assert violent == [20, 25]
    # 15 and 30 touch violence within the 3-frame margin; 10 and 35 stay clear of it.
    assert normal == [0, 5, 10, 35]


def test_grid_windows_margin_excludes_after_violence():
    labels = np.zeros(20, dtype=bool)
    labels[4] = True
    _, normal = grid_windows(labels, width=5, margin=2)
    assert normal == [10, 15]


def test_intervals_to_labels_inclusive_and_validated():
    labels = intervals_to_labels(10, [2, 3, 7, 7])
    assert labels.nonzero()[0].tolist() == [2, 3, 7]
    with pytest.raises(ValueError):
        intervals_to_labels(10, [2])
    with pytest.raises(ValueError):
        intervals_to_labels(10, [5, 20])


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A", "normal"),
        ("v=S-7rRLrxnVQ__#1_label_B4-0-0", "violent"),
        ("x__#1_label_B1-B5-0", "violent"),
        ("The.Fast.and.the.Furious.2001__#00-29-00_00-29-52_label_B2-G-0", "excluded"),
        ("v=u5SF4SlqNDQ__#00-00-00_00-01-00_label_G-0-0", "excluded"),
        ("x__#1_label_B1-B6-0", "excluded"),
    ],
)
def test_xd_kind(name, kind):
    assert xd_kind(name) == kind


def test_xd_source_and_safe_name():
    assert xd_source("v=cO1UefhG7AY__#1_label_G-0-0") == "v=cO1UefhG7AY"
    assert safe_name("v=a#b c") == "v_a_b_c"


def test_split_sources_stratified_and_deterministic():
    sources = {f"n{i}": False for i in range(10)} | {f"v{i}": True for i in range(5)}
    first = split_sources(sources, 0.2, 7)
    assert first == split_sources(sources, 0.2, 7)
    assert sum(first[f"n{i}"] == "val" for i in range(10)) == 2
    assert sum(first[f"v{i}"] == "val" for i in range(5)) == 1
