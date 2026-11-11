"""Pack the files a teammate needs to run the demo model (no training data) into one zip.

The zip keeps repository-relative paths, so it is unpacked at the root of a fresh clone. It
holds the five frozen cascade models, their receipts, the v3 pilot head, the nine demo example
videos and their result caches, plus MANIFEST.sha256. Run `scripts/demo_preflight.py` after
unpacking; the demo also re-checks every model hash when it loads.

    python scripts/pack_demo_bundle.py --output ~/Desktop/violence-demo-bundle.zip
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSET_KEYS = (
    "sultani_checkpoint",
    "c3d_checkpoint",
    "behavior_checkpoint",
    "detector_checkpoint",
    "i3d_checkpoint",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_files(spec_path: Path) -> list[Path]:
    profile = yaml.safe_load((ROOT / "configs/demo/professor_demo.yaml").read_text())
    files = []
    for key in ASSET_KEYS:
        model = Path(profile[key])
        files.append(model)
        for sidecar in (model.with_suffix(".json"), model.parent / "selection.json"):
            if (ROOT / sidecar).is_file():
                files.append(sidecar)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    files.append(Path(spec["checkpoint"]))
    for entry in json.loads((ROOT / profile["examples"]).read_text(encoding="utf-8")):
        files.append(Path(entry["video"]))
        if entry.get("result_cache"):
            files.append(Path(entry["result_cache"]))
    unique = sorted(set(files), key=str)
    missing = [str(path) for path in unique if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError("Missing files, cannot pack: " + ", ".join(missing))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spec", type=Path, default=ROOT / "runs/indian_pilot/demo/indian_pilot_spec_v3.json"
    )
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    pilot = ROOT / spec["checkpoint"]
    if sha256(pilot) != spec["checkpoint_sha256"]:
        raise ValueError(f"{pilot} does not match the SHA256 pinned in {args.spec}")
    files = bundle_files(args.spec)
    manifest = []
    output.parent.mkdir(parents=True, exist_ok=True)
    # Weights do not compress; storing keeps packing and unpacking fast.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(ROOT / path, path.as_posix())
            manifest.append(f"{sha256(ROOT / path)}  {path.as_posix()}")
            print(f"added {path.as_posix()}")
        archive.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    size = output.stat().st_size / 1e6
    print(f"Wrote {output} ({len(files)} files, {size:.0f} MB)")


if __name__ == "__main__":
    main()
