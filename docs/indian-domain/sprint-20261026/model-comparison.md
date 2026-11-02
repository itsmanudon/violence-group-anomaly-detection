# Phase 2 matched comparison readiness

| Mode | Variable | Recognition outcome |
| --- | --- | --- |
| A0 legacy | Existing final-alert representative covered window | Historical original results only; new matched validation NOT MEASURED |
| A1 coverage_consistency | Abstain from six-class suggestion unless all routed windows covered and agreeing | NOT MEASURED on eligible validation |
| A2 coverage | Remove agreement condition from A1 | NOT MEASURED on eligible validation |

No complete eligible joint cascade validation predictions were found. The 692
behavior validation examples' aggregate selection receipt cannot reconstruct
anomaly routing, multiple windows or coverage. Existing held-out predictions were
used only for descriptive Phase 1 attribution, never comparison/tuning. Synthetic
comparison fixtures demonstrate contracts, not scientific recognition outcomes.

When a legitimate all-split manifest and matching full validation outputs exist:

```powershell
.venv/Scripts/python.exe scripts/compare_cascade_review.py --manifest <all-split-human-manifest.jsonl> --predictions <complete-val-cascade-predictions.json> --config configs/surveillance_demo_ucf_v1.yaml --output runs/indian-domain/review-comparison-new-version
```

Placeholders deliberately name unavailable inputs; no population is invented.
Predictions need explicit `split=val`, original source/video hashes, both labels
and result identity. Preserve raw test data outside this command. The report
provides identical full-population binary precision/recall/F1/ROC/PR-AUC/FPR,
plus mode-specific behavior confusion, coverage and all-population class recall.
Source confidence intervals remain unavailable until appropriate independent
validation sources and a prospectively registered uncertainty procedure exist.
No binary false-positive reduction is claimed: the new sidecar preserves alerts.
