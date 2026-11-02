import json

import pytest
from test_cascade_evaluation import record

from surveillance.datasets.dcsass_audit import sha256
from surveillance.evaluation.cascade import summarize_cascade
from surveillance.evaluation.cascade_failures import attribute_failures, load_registered_cascade


def test_failure_attribution_does_not_call_bypass_a_person_detection_failure():
    rows = [record("miss", 3, 0.1), record("fp", 0, 0.9, "no_actors")]
    report = attribute_failures(rows)
    assert report["counts"]["abnormal_bypassed"] == 1
    assert report["counts"]["routed_no_actors"] == 1
    assert report["per_class"]["Fighting"]["bypassed"] == 1
    assert "routed_no_actors" not in report["per_class"]["Fighting"]


def test_registered_predictions_are_verified_before_analysis(tmp_path):
    (tmp_path / "clips").mkdir()
    row = record("clip", 0, 0.1)
    path = tmp_path / "clips/clip.json"
    path.write_text(json.dumps(row))
    (tmp_path / "registration.json").write_text(json.dumps({"provenance": {"fixture": True}}))
    (tmp_path / "progress.json").write_text(json.dumps({"clip": sha256(path)}))
    (tmp_path / "metrics.json").write_text(json.dumps(summarize_cascade([row])))
    (tmp_path / "completion.json").write_text(
        json.dumps(
            {"complete": True, "clips": 1, "metrics_sha256": sha256(tmp_path / "metrics.json")}
        )
    )
    assert load_registered_cascade(tmp_path)[0] == [row]
    path.write_text(json.dumps({**row, "binary_label": 1}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_registered_cascade(tmp_path)
