"""Extract I3D Mixed_4f actor RoIs using a local compatible export."""

from _actor_features import extract_features
from _common import run_cli

if __name__ == "__main__":
    run_cli(lambda: extract_features("rgb"))
