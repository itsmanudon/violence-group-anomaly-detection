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
7. **Normal example:** show playback, anomaly timeline and the no-anomaly bypass
   if the actual selected result takes that path. Identify cached versus live mode.
8. **Anomaly example:** explain score, timestamp interval, detected actors and
   six-class probabilities. An alert requests review; probabilities are uncalibrated.
9. **Limitations:** actor coverage is about 69% on test. Minority labels are weak;
   Fighting validation has only two covered clips. Show an actual disagreement,
   false alarm or no-actor failure without concealing it.
10. **Sultani protocol:** the provisional DCSASS run has clip/bag labels, not frame
    ground truth. UCF-Crime benchmark validation is pending download. Modern C3D
    channel-mean preprocessing differs from the original Caffe volume mean.
11. **Future research:** stronger independent surveillance validation, multi-seed
    evaluation, detector-domain adaptation and privacy-preserving deployment.

Before presenting, substitute the final selected-control and Sultani measured
results from their frozen receipts. This talk track does not imply those pending
experiments or a successful violence clip have already been demonstrated.
