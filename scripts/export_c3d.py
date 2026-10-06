"""Convert the approved local OpenMMLab Sports-1M C3D checkpoint to FC6."""

import argparse
import json
from pathlib import Path

from _common import run_cli

from surveillance.features.c3d_conversion import export_openmmlab_c3d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_openmmlab_c3d(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    run_cli(main)
