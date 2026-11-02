"""Compare frozen C3D FC6 and I3D Mixed_4f features on CPU and MPS for a few clips.

Engineering evidence for choosing the Indian pilot extraction device; not a model
evaluation. I3D is pooled over one full-frame box per window (scene-level).
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from surveillance.datasets.dcsass_audit import sha256  # noqa: E402
from surveillance.features.c3d import C3DExtractor  # noqa: E402
from surveillance.features.i3d import I3DActorExtractor  # noqa: E402
from surveillance.video.actor_window import read_actor_window  # noqa: E402
from surveillance.video.decode import iter_clips  # noqa: E402


def decoded_frames(path):
    """Frames that actually decode; some container headers overstate the count."""
    return sum(len(clip) for clip in iter_clips(path, 1, strict=False))


def i3d_scene(extractor, path, device, windows=4):
    frames = decoded_frames(path)
    edges = np.linspace(0, frames, windows + 1).astype(int)
    box = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]], device=device)
    valid = torch.ones(1, 1, dtype=torch.bool, device=device)
    out = []
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        window = read_actor_window(path, frames, start=int(start), end=int(end))
        out.append(extractor(window["frames"][None].to(device), box, valid)[0, 0].cpu().numpy())
    return np.stack(out)


def run(device, clips, c3d_ckpt, i3d_ckpt):
    c3d = C3DExtractor(c3d_ckpt, device, 4, (104, 117, 128), "rgb")
    i3d = I3DActorExtractor(i3d_ckpt).to(device).eval()
    results = {}
    for clip in clips:
        timing = {}
        for name, fn in (
            ("c3d", lambda p=clip: c3d.extract_video(p, 32, strict_frame_count=False)),
            ("i3d", lambda p=clip: i3d_scene(i3d, p, torch.device(device))),
        ):
            if device == "mps":
                torch.mps.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                array = fn()
            if device == "mps":
                torch.mps.synchronize()
            timing[name] = time.perf_counter() - started
            results.setdefault(str(clip), {})[name] = array
        results[str(clip)]["timing"] = timing
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, nargs="+", required=True)
    parser.add_argument("--c3d", type=Path, default=Path("checkpoints/c3d_fc6_openmmlab_v1.pt"))
    parser.add_argument(
        "--i3d", type=Path, default=Path("checkpoints/i3d_mixed4f_collective_rgb_v1.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refuse to overwrite {args.output}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run("cpu", args.clips[:1], args.c3d, args.i3d)  # warm-up, excluded from timing
        cpu = run("cpu", args.clips, args.c3d, args.i3d)
        run("mps", args.clips[:1], args.c3d, args.i3d)
        mps = run("mps", args.clips, args.c3d, args.i3d)
    fallbacks = sorted({str(w.message) for w in caught if "MPS" in str(w.message)})
    report = {"evidence_kind": "device_parity_engineering_only", "torch": torch.__version__}
    report["checkpoints"] = {"c3d": sha256(args.c3d), "i3d": sha256(args.i3d)}
    report["mps_fallback_warnings"] = fallbacks
    report["clips"] = []
    for clip in map(str, args.clips):
        entry = {
            "clip": clip,
            "timing_cpu_s": cpu[clip]["timing"],
            "timing_mps_s": mps[clip]["timing"],
        }
        for name in ("c3d", "i3d"):
            a, b = cpu[clip][name], mps[clip][name]
            diff = np.abs(a - b)
            entry[name] = {
                "shape": list(a.shape),
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "max_abs_value": float(np.abs(a).max()),
                "cosine_min": float(
                    min(
                        (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
                        for x, y in zip(a.reshape(len(a), -1), b.reshape(len(b), -1), strict=True)
                    )
                ),
            }
        report["clips"].append(entry)
    args.output.mkdir(parents=True)
    (args.output / "parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
