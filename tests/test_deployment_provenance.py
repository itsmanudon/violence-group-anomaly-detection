import json

import pytest

from surveillance.datasets.dcsass_audit import sha256
from surveillance.inference.provenance import verify_selected_checkpoint


@pytest.mark.parametrize("changed", ["none", "missing", "unfinished", "preflight", "weights"])
def test_selection_receipt_requires_frozen_nonpreflight_identical_weights(tmp_path, changed):
    model = tmp_path / "best.pt"
    model.write_bytes(b"selected weights")
    receipt = {
        "checkpoint_sha256": sha256(model),
        "selection_frozen": True,
        "dry_run": False,
    }
    if changed == "unfinished":
        receipt["selection_frozen"] = False
    elif changed == "preflight":
        receipt["dry_run"] = True
    elif changed == "weights":
        model.write_bytes(b"different weights")
    if changed != "missing":
        (tmp_path / "selection.json").write_text(json.dumps(receipt))
    if changed == "none":
        assert verify_selected_checkpoint(model) == receipt
    else:
        with pytest.raises(ValueError, match="selection"):
            verify_selected_checkpoint(model)
