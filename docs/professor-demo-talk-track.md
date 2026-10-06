# Professor demonstration talk track

1. **Problem:** surveillance footage is lengthy. We seek observable unusual or
   aggressive behavior that a human can review. We do not identify criminals or
   infer someone's character.
2. **Paper one:** Sultani et al. (2018) learn a weakly supervised anomaly scorer
   from positive/normal video bags. C3D features provide a 32-segment timeline.
3. **Paper two:** Gavrilyuk et al. (2020) model relationships between actors using
   a Transformer. We use the RGB representation with automatic Faster R-CNN actors.
4. **Why cascade:** anomaly scoring proposes suspicious intervals; actor analysis
   runs inside those intervals and adds an interpretable behavior label. Both
   model outputs remain visible when they disagree.
5. **Evidence:** on Collective, RGB reached 77.55% group accuracy with GT boxes and
   74.32% with detected boxes. Fixed pose-heavy fusion underperformed RGB. Those
   results selected the representation; they are not violence-recognition metrics.
6. **Surveillance adaptation:** DCSASS has 6,491 valid human-centric clips from
   203 separated original sources. Classes are Normal, Abuse, Assault, Fighting,
   Robbery and Vandalism. No actor action labels were invented.
7. **Normal example:** select **Normal: no-alert example**, show playback, timeline
   and actor-analysis bypass. Disclose its supplemental post-hoc selection and
   identify cached versus live mode. The original Normal false alert remains available.
8. **Anomaly example:** select **Robbery example**; explain score, interval, actors and
   six-class probabilities. An alert requests review; probabilities are uncalibrated.
9. **Limitations:** actor coverage is about 69% on test. Minority labels are weak;
   Fighting validation has only two covered clips. Behavior accuracy is 51.02%,
   macro F1 is 0.2140. Show **Limitation: high-confidence mistake**, where anomaly
   scoring alerts but behavior predicts Normal, or the retained false alert.
10. **Sultani protocol:** the provisional DCSASS run has clip/bag labels, not frame
    ground truth. Held-out bag ROC-AUC is 0.6583; normal-clip false positives are
    35.35% at the frozen 0.5 threshold. UCF-Crime frame validation is pending download. Modern C3D
    channel-mean preprocessing differs from the original Caffe volume mean.
11. **Future research:** stronger independent surveillance validation, multi-seed
    evaluation, detector-domain adaptation and privacy-preserving deployment.

The demonstration illustrates the implemented research cascade and its failures;
it does not establish reliable violence detection or an exact paper reproduction.
