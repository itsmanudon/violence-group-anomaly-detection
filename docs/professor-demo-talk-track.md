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
7. **Normal example:** select **Normal activity (previous bypass example)**, show playback, timeline
   and actor-analysis bypass. Disclose its supplemental post-hoc selection and
   identify cached versus live mode. The original Normal false alert remains available.
8. **Anomaly example:** select **Robbery: correct cascade example (curated)** alongside
   the original **Robbery example**, which the UCF scorer misses. Explain score,
   interval, actors and probabilities. An alert requests review; probabilities
   are uncalibrated. Presentation selection never changes the model or threshold.
9. **Limitations:** standalone middle-frame actor coverage is 69.22% on test,
   with conditional behavior accuracy 51.02% and macro F1 0.2140. The deployed
   cascade covers 311/991 clips (31.38% overall), with conditional accuracy 45.66%
   and macro F1 0.1888. Minority labels are weak; Fighting validation has only two
   covered clips. Show **Limitation: high-confidence mistake**, where anomaly
   scoring alerts but behavior predicts Normal, or the retained false alert.
10. **Sultani protocol:** the deployed UCF baseline has 0.7441 held-out frame
    ROC-AUC on 290 videos. At the frozen 0.5 threshold, precision is 0.1912,
    recall 0.5189 and negative-frame false positives 18.02%. This modern-C3D,
    source/content-safe experiment is not an exact paper reproduction. DCSASS
    clip-alert metrics remain separate from UCF temporal ground truth.
11. **Future research:** stronger independent surveillance validation, multi-seed
    evaluation, detector-domain adaptation and privacy-preserving deployment.

The demonstration illustrates the implemented research cascade and its failures;
it does not establish reliable violence detection or an exact paper reproduction.
