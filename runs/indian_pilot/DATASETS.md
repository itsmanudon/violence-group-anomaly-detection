# Candidate datasets for v3 (checked 2026-11-03)

Links only; nothing below has been downloaded. Footage, features and URL lists stay on the SSD
under `/Volumes/Seagate_2TB/violence-training/` and never go into git.

## What makes a dataset useful here

v1 showed the head learns *which collection* a clip came from. v2 only improved because DCSASS
cuts violent and normal clips from the **same recording**. A new dataset helps only if it can be
used the same way:

- **Matched:** temporal (start/end) annotations, so the non-violent parts of a violent video
  become negatives from the same camera. Clip-level fight/no-fight sets from different sources
  repeat the v1 confound.
- **Surveillance:** fixed CCTV, not movies, sports or phone footage.
- **Not already seen:** fingerprint against UCF-Crime/DCSASS (DCSASS is cut from UCF-Crime)
  with `scripts/pilot_source_groups.py fingerprint` before use; overlapping sources join the
  existing group.

## Ranked

| # | Dataset | What it is | Matched? | Access | Use |
|---|---|---|---|---|---|
| 1 | **IITB-Corridor** ([project](https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction/), [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Rodrigues_Multi-timescale_Trajectory_Prediction_for_Abnormal_Human_Activity_Detection_WACV_2020_paper.pdf)) | Indian (IIT Bombay campus), one fixed corridor camera, 483,566 frames; anomalies include fighting, chasing, robbery | Yes: normal and abnormal from the same camera | Free for research; the [GitHub README](https://github.com/Rodrigues-Royston/Multi-timescale-Trajectory-Prediction-for-Abnormal-Human-Activity-Detection) says to email the author for a download link. Also listed on [OpenDataLab](https://opendatalab.com/OpenDataLab/IITB_Corridor/download) | Indian matched **test** set. Staged and single-scene, so weak as training data |
| 2 | **NTU CCTV-Fights** ([page](https://rose1.ntu.edu.sg/dataset/cctvFights/)) | 1,000 real fight videos (CCTV and mobile), 7.2 GB, frame-level start/end of each fight | Yes, via the annotated fight intervals | Register on ROSE Lab, submit the request form, accept the release agreement; academic, non-commercial | Training (CCTV subset), matched like DCSASS |
| 3 | **UBI-Fights** ([listing](https://hyper.ai/en/datasets/20095)) | 1,000 videos, 80 h, frame-level annotation, fight and normal videos | Yes | Distributed from `socia-lab.di.ubi.pt`; the site did not load from here today | Training |
| 4 | **ExtrAnom** ([paper](https://arxiv.org/html/2605.25806v1), [repo](https://github.com/A24CS09005/ExtrAnom)) | IIT Bhubaneswar, 2026: 1,001 videos (500 normal, 501 anomalous): harassment, chain snatching, kidnapping, stalking, assassination; CCTV, mobile and handheld | Not stated | On request, research only | Possibly Indian-heavy (country not stated); request and inspect |
| 5 | **XD-Violence** ([project](https://roc-ng.github.io/XD-Violence/), [HF mirror](https://huggingface.co/datasets/jherng/xd-violence)) | 4,754 untrimmed videos, 217 h; fighting, shooting, riot, abuse, accidents, explosions | Only the 800 test videos have frame annotations; train is video-level | HF mirror downloads directly: videos ~85.5 GB; precomputed C3D features ~48 GB (extractor compatibility unknown, re-extract to be safe). Original licence not stated | Mixed movies/sports/CCTV, so weakest fit; use the annotated test videos for matched segments |
| 6 | SCVD ([Kaggle](https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd)) | 1.28 GB preprocessed CCTV clips: violence, weaponized violence | No | Kaggle account; CC BY-NC-SA 4.0 | Low value: unmatched |
| 7 | Surveillance Camera Fight ([GitHub](https://github.com/seymanurakti/fight-detection-surv-dataset)) | 300 two-second YouTube CCTV clips, 150 fight / 150 non-fight | No | Public | Low value: tiny, unmatched |
| 8 | RWF-2000 ([GitHub](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)) | 2,000 surveillance clips | No | **Videos withheld** for privacy; not obtainable | None |

## Other anomaly-detection datasets (from the ExtrAnom comparison table)

Most of these label *any* unusual event, not violence, so they add few violent positives.

| Dataset | Verdict |
|---|---|
| **UCA** ([project](https://xuange923.github.io/Surveillance-Video-Understanding), [GitHub](https://github.com/Xuange923/Surveillance-Video-Understanding)) | **Useful.** Manual timestamps and event descriptions for 1,854 UCF-Crime videos. Not new footage, but its event times let us cut more matched violent/normal segments from UCF-Crime videos we already have. They share sources with DCSASS, so keep DCSASS source groups |
| ↳ fetched 2026-11-03 | Clone at `/Volumes/Seagate_2TB/violence-training/raw/uca` (commit `e478cb5`). 1,854 videos, all present in the SSD UCF-Crime copy; violent categories Abuse 50, Assault 48, Fighting 50, Robbery 149. Timestamps cover every described event (normal ones too), so a violent segment must be picked by a pre-registered sentence rule and spot-checked |
| RareAnom ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323002674)) | Maybe. 2,200 multi-source real-world videos, 17 rare anomaly types, temporal labels; same IIT Bhubaneswar group as ExtrAnom. Class list and access not verified; ask the authors alongside ExtrAnom |
| UBnormal | Low. Synthetic (Cinema 4D) scenes; "fighting" is one of 22 types. Rendering gap to real CCTV |
| ADOC, Street Scene, Subway Entrance, UCSD Ped1/Ped2, CUHK Avenue | No. Single-scene benchmarks whose anomalies are jaywalking, bikes, wrong direction, running, thrown objects; little or no violence |
| UCF-Crime, CCTV-Fights, XD-Violence, ExtrAnom | Covered above |

## Web-sourced Indian CCTV clips

Usable only as matched data: an uncut single-camera recording with calm footage before the
incident, so both labels come from one video. News packages, compilations and phone video bring
back the source confound. Candidate URLs are kept on the SSD
(`/Volumes/Seagate_2TB/violence-training/web_candidates/`), not in git. Someone on the team must
watch each one and record the incident start/end; segments are then cut from those times.
Downloading from YouTube is against its terms of service; that call is the team's.

## Rules for v3 (before any training)

1. Pre-register a v3 protocol (data, split, metric, one-time test) as `DCSASS_PROTOCOL.md` was.
2. Group by source recording; a recording's segments never cross splits.
3. Fresh test data only: the Indian v1 test and the DCSASS v2 test are spent.
