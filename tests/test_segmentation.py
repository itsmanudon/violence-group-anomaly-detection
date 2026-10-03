import numpy as np
import pytest

from surveillance.video.segmentation import aggregate_segments, segment_indices


@pytest.mark.parametrize("length", [1, 5, 31, 32, 33, 101, 10001])
def test_segments_cover_input_in_order(length):
    groups = segment_indices(length, 32)
    assert len(groups) == 32
    assert all(len(g) > 0 for g in groups)
    flat = np.concatenate(groups)
    assert set(flat) == set(range(length))
    assert np.all(np.diff(flat) >= 0)
    if length >= 32:
        np.testing.assert_array_equal(flat, np.arange(length))
        assert max(map(len, groups)) - min(map(len, groups)) <= 1
    assert all(np.array_equal(a, b) for a, b in zip(groups, segment_indices(length, 32)))


def test_aggregation_and_empty():
    result = aggregate_segments(np.ones((5, 4)), 32)
    np.testing.assert_allclose(result, 0.5)
    with pytest.raises(ValueError):
        segment_indices(0)
