import numpy as np
import pytest

from surveillance.inference.anomaly_pipeline import AnomalyPipeline
from surveillance.models.sultani.model import SultaniScorer
from surveillance.video.segmentation import c3d_segment_frame_ranges


def test_c3d_frame_mapping_follows_feature_units_and_clamps_padding():
    assert c3d_segment_frame_ranges(65, num_segments=2) == [(0, 32), (32, 65)]
    assert c3d_segment_frame_ranges(5, num_segments=4) == [(0, 5)] * 4
    ranges = c3d_segment_frame_ranges(638)
    assert len(ranges) == 32
    assert ranges[0] == (0, 16)
    assert ranges[-1][1] == 638
    assert all(a[1] == b[0] for a, b in zip(ranges[:-1], ranges[1:]))


def test_repeated_short_unit_timestamps_merge_as_one_suspicious_interval():
    model = SultaniScorer(feature_dim=4, hidden_dims=[3], dropout=0)
    pipeline = AnomalyPipeline(model, num_segments=4)
    result = pipeline.predict_features(
        np.ones((4, 4), dtype=np.float32), 0.5, threshold=0, timestamps=[(0.0, 0.5)] * 4
    )
    assert result.suspicious_intervals == [(0.0, 0.5)]
    assert result.timestamps == [(0.0, 0.5)] * 4
    with pytest.raises(ValueError):
        pipeline.predict_features(
            np.ones((4, 4), dtype=np.float32), 0.5, timestamps=[(0.0, 1.0)] * 4
        )
