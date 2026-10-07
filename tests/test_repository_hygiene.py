import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ignored(paths):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v", "-z", "--stdin"],
        input=("\0".join(paths) + "\0").encode(),
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    fields = result.stdout.decode().split("\0")[:-1]
    assert len(fields) % 4 == 0
    return {
        path
        for source, _, pattern, path in zip(
            fields[::4], fields[1::4], fields[2::4], fields[3::4], strict=True
        )
        if source == ".gitignore" and not pattern.startswith("!")
    }


def test_heavy_generated_private_and_os_files_are_protected():
    paths = {
        ".venv/lib/site-packages/library.py",
        "src/surveillance/__pycache__/model.pyc",
        "data/raw/ucf/video.mp4",
        "data/features/actors/features.npz",
        "data/examples/reference.jpg",
        "checkpoints/external/upstream/model.py",
        "runs/experiment/metrics.json",
        "outputs/demo/result.json",
        "downloads/dataset.zip",
        "weights/model.safetensors",
        ".env",
        ".env.local",
        ".aws/credentials",
        ".DS_Store",
        "data/._metadata",
        "Thumbs.db",
        "Desktop.ini",
        ".vscode/settings.json",
        ".idea/workspace.xml",
        "configs/surveillance_demo.local.yaml",
    }
    assert ignored(paths) == paths


def test_source_protocol_templates_and_docs_remain_trackable():
    paths = {
        "src/surveillance/demo.py",
        "scripts/verify_demo_assets.py",
        "configs/surveillance_demo.yaml",
        "configs/demo_examples_ucf_v1.json",
        "docs/macos-replication.md",
        "tests/test_repository_hygiene.py",
        "data/splits/ucf_authors_provenance.json",
        "data/manifests/.gitkeep",
        ".env.example",
        "README.md",
        "pyproject.toml",
    }
    assert not ignored(paths)
