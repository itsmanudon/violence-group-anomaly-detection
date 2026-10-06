import json
from pathlib import Path

import pytest

from surveillance.experiments.dcsass_cache import write_json


def test_transient_replace_lock_retries_without_losing_previous_receipt(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    write_json(path, {"old": "verified"})
    original = Path.replace
    attempts = []

    def locked(source, destination):
        attempts.append(1)
        if len(attempts) <= 2:
            assert json.loads(path.read_text()) == {"old": "verified"}
            raise PermissionError("Simulated Windows scanner lock")
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr("surveillance.experiments.dcsass_cache.time.sleep", lambda delay: None)
    write_json(path, {"old": "verified", "new": "verified"})
    assert len(attempts) == 3
    assert json.loads(path.read_text()) == {"old": "verified", "new": "verified"}


def test_permanent_replace_denial_is_bounded_and_preserves_both_versions(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    write_json(path, {"old": "verified"})
    attempts = []

    def locked(source, destination):
        attempts.append(1)
        raise PermissionError("Persistent write denial")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr("surveillance.experiments.dcsass_cache.time.sleep", lambda delay: None)
    with pytest.raises(PermissionError, match="Persistent"):
        write_json(path, {"new": "verified"})
    assert len(attempts) == 8
    assert json.loads(path.read_text()) == {"old": "verified"}
    assert json.loads(path.with_suffix(".json.tmp").read_text()) == {"new": "verified"}
