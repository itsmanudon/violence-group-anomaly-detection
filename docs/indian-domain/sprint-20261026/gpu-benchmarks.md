# Phase 2 actual RTX 4070 measurements

Receipt: `runs/indian-domain/sprint-20261026-v1/rtx4070-cascade/profile.json`.
Script: `scripts/profile_surveillance_sprint.py --output <new-directory> --repeats 5`.
This is inference-only engineering work on the existing pinned curated
`Robbery020_x264_10` demonstration asset, **not accuracy evaluation or selection**.
Five checkpoint identities and preprocessing are in the receipt, matching the
unchanged deployment config. No Indian footage used. No checkpoint downloaded.

GPU NVIDIA GeForce RTX 4070 Laptop GPU; Python 3.13.5, torch 2.13.0+cu126,
CUDA build 12.6. Total VRAM 8,585,216,000 bytes; free before loading
7,411,335,168 bytes. Peak PyTorch allocation 1,020,069,376 bytes (972.81 MiB),
reserved 1,226,833,920 bytes (1,170 MiB); device free afterward
6,150,946,816 bytes. Allocator peaks are not total system VRAM usage.

All five models remain resident; one complete first inference warms the path,
then five sequential FP32 runs synchronized before/after end-to-end timing.
C3D batch four, 120 source frames, 30 FPS, four seconds, native 320×240.
C3D uses 16-frame units mapped into 32 segments; repeated padded/mapped units
are not independent observations. Actor preprocessing uses ten consecutive
clamped frames/window, resized 480×720; I3D maps/features/RoIs keep original
contracts. Device/caching/power contention and OS page cache are uncontrolled.

| Quantity | Median seconds | p95 seconds |
| --- | ---: | ---: |
| Complete uncached pipeline (decode/features each run) | 1.298047 | 1.321201 |
| Decode across pipeline windows | 0.311560 | 0.342196 |
| C3D preprocessing | 0.058621 | 0.067040 |
| C3D forward + transfer boundary | 0.063076 | 0.067445 |
| Sultani stage | 0.001425 | 0.001507 |
| Actor preprocessing | 0.047536 | 0.048858 |
| Person detection across selected windows | 0.523168 | 0.547380 |
| I3D across covered windows | 0.197636 | 0.200526 |
| Actor transformer across covered windows | 0.012322 | 0.014288 |
| Cached Sultani only, resident [32,4096] | 0.000621 | 0.000903 |
| Cached actor transformer only, last covered two-actor window | 0.003942 | 0.004189 |
| CPU review assessment | 0.000207 | 0.000300 |

Model loading: **2.470153 s**, excludes CUDA initialization but includes pin and
selection-receipt verification. First complete inference: **3.020937 s**. These
are single startup observations, not distributions or guaranteed cold disk reads.
Warm throughput 0.770 clips/s for this fixed clip/policy only. Stage wall timers
use existing boundaries, include transfers/synchronization, and separate stage
quantiles must not be added. Cached head rows exclude decode/backbones/transfers;
they are not a full cached cascade and cannot be compared as identical workloads.

Five warm runs had exactly equal segment scores, boxes, probabilities and alerts.
Assessment preserved the legacy alert. This verifies repeatability on one asset,
not general accuracy or cross-device determinism. No speculative AMP change was
introduced: previous tiny-head FP16 measurements already showed overhead.

NOT MEASURED: Indian throughput/metrics, full cached video pipeline, online event
detection latency, continuous camera-hour false alarms, new architecture accuracy.
The historical 10.93 ms synthetic training timing remains a different workload.
Recorded script hash identifies execution bytes; later formatting/extra metadata
fields in the profiler do not retroactively modify this immutable receipt.
