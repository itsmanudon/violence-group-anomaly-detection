# Phase 1: feasible interventions and verified references

References checked online on 26 October 2026. Search retrieved primary CVF/PMLR
metadata/abstracts; direct CVF HTML opens for Actor-Transformers and RTFM returned
403, so full-text review is not claimed. No model/checkpoint/code downloads.
No published benchmark improvement is projected onto this project.

| Rank / option | Reference and task fit | Cost, access, risk and labels |
| --- | --- | --- |
| 1: coverage and cross-window agreement assessment | Geifman & El-Yaniv, **SelectiveNet**, ICML 2019: distinguishes risk from coverage. Our deterministic postprocessing is not SelectiveNet training or its risk guarantee. [Primary paper](https://proceedings.mlr.press/v97/geifman19a.html) | Reuse available six-class windows, O(windows × 6), no new weights/dependencies; no learning needed for software contract, approved validation needed to establish utility. Conservative abstention may lower behavior coverage substantially. |
| 2: temporal persistence/hysteresis and recall-constrained calibration | Sultani, Chen & Shah, **Real-World Anomaly Detection in Surveillance Videos**, CVPR 2018, weak-label MIL with temporal regularization. Post-hoc alert persistence differs from its training loss. [Primary paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html) | Existing local causal temporal rules/calibration; linear score cost, no backbone change. Needs eligible validation and truthful interval clocks; short actions can be suppressed, overlapping padded segments need care. Not selected for deployment without recall evidence. |
| 3: lightweight temporal/interaction head on cached actor features | Gavrilyuk, Sanford, Javan & Snoek, **Actor-Transformers for Group Activity Recognition**, CVPR 2020; actor-specific pose and 3D CNN representations. [Primary PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gavrilyuk_Actor-Transformers_for_Group_Activity_Recognition_CVPR_2020_paper.pdf) | Existing RGB actor transformer and binary adapter usable; new cache/orchestration and independent binary labels still needed. Head cheaper than extraction, no measured local new-head cost yet. Pose temporal associations are absent from current RGB contract; adding skeletons is substantial work. |
| 4: stronger weak supervision/temporal dependencies | Tian, Pang, Chen, Singh, Verjans & Carneiro, **Weakly-Supervised Video Anomaly Detection With Robust Temporal Feature Magnitude Learning**, ICCV 2021. [Primary paper](https://openaccess.thecvf.com/content/ICCV2021/html/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.html) | New trained temporal magnitude function, dilated convolution/self-attention. Existing L2-normalized C3D bags discard raw magnitude; direct substitution is inappropriate. New training/access/representation audit; high engineering risk before deadline. |
| 5: pretrained representation replacement, future only | Wang, Huang, Zhao, Tong, He, Wang, Wang & Qiao, **VideoMAE V2: Scaling Video Masked Autoencoders With Dual Masking**, CVPR 2023. [Primary paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html) | Different representation/pretraining, no compatible checkpoint installed/verified here. [Author repository](https://github.com/OpenGVLab/VideoMAEv2) is discoverable, but weight access, licenses, memory and Indian transfer not validated. Recaching/backbone replacement exceeds bounded scope. |

Licensing: publication access is not a dataset/model-use license. Repository,
checkpoint and training-data usage terms for new candidates were **not verified**;
none is installed, copied or made a dependency. Existing pinned local assets are
reused under their existing receipts; no new permission is invented.

Costs above are complexity/engineering judgments, not measured GPU performance.
Selected assessment adds no GPU tensor or training parameter. Phase 2 will measure
the current frozen pipeline on permitted local demonstration footage strictly for
runtime, plus cached assessment cost. Historical test labels will not select any
policy, confidence cutoff or model. No architectural novelty is claimed.
