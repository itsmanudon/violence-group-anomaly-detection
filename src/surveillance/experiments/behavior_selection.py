"""One predeclared E-C initialization comparison, selected on validation only."""

from pathlib import Path

import yaml

from surveillance.inference.provenance import verify_selected_checkpoint


def select_initialization_control(transfer: Path, random: Path) -> dict:
    runs = [Path(transfer), Path(random)]
    configs = [yaml.safe_load((run / "resolved_config.yaml").read_text()) for run in runs]
    initializations = [config.pop("initialization", "collective_transfer") for config in configs]
    if initializations != ["collective_transfer", "random"] or configs[0] != configs[1]:
        raise ValueError("Controlled runs must differ only in Actor-Transformer initialization")
    selections = [verify_selected_checkpoint(run / "best.pt") for run in runs]
    # Strictly higher macro F1 wins; retain the prospective transfer baseline on a tie.
    winner = int(selections[1]["validation"]["macro_f1"] > selections[0]["validation"]["macro_f1"])
    return {
        "schema_version": 1,
        "decision_gate": "E-C",
        "selection_frozen": True,
        "selection_metric": "validation_macro_f1",
        "held_out_used_for_selection": False,
        "selected_run": str(runs[winner]),
        "checkpoint_sha256": selections[winner]["checkpoint_sha256"],
        "initialization": initializations[winner],
        "comparisons": [
            {
                "run": str(run),
                "initialization": init,
                "checkpoint_sha256": saved["checkpoint_sha256"],
                "selected_iteration": saved["selected_iteration"],
                "validation_macro_f1": saved["validation"]["macro_f1"],
                "validation_accuracy": saved["validation"]["accuracy"],
            }
            for run, init, saved in zip(runs, initializations, selections)
        ],
        "limitation": "Only two covered Fighting validation clips from one source; "
        "macro F1 is sensitive to individual minority-class predictions.",
    }
