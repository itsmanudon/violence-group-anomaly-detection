# Milestone 2A execution plan and decisions

User specification is the design authority; proceed without another approval.
Starting point: clean main at a7f5d99; all 28 Milestone 1 tests passed before edits.
Work branch: feat/actor-transformer-baseline. No commits or pushes in this task.

1. Collective: verify original annotation columns and published split against
   primary sources; write parser tests, scene JSONL, feature dataset and padded
   batches. Keep existing binary anomaly records unchanged.
2. Actor model: independent pose/RGB projections, sinusoidal x/y encoding,
   post-norm encoder with explicit padding mask, per-actor head, masked max and
   group head. Early concatenate/project; late weighted-average branch probabilities
   (pose:RGB = 2:1, as stated in the paper).
   Test padding invariance, permutations, single actors, gradients and fusion.
3. Backbones: validated local feature-only HRNet-W32/I3D exports, no downloads or
   vendored research repositories. Explicit crop/coordinate transforms and
   torchvision RoIAlign. Test geometry and synthetic exported networks.
4. Training: reuse seed/device helpers, add actor config loader, iteration Adam
   schedule, joint loss, TensorBoard, validation best checkpoint and resume with
   manifest-byte fingerprint. Precomputed and raw local-backbone paths share model input.
5. Evaluate/infer: group and actor metrics excluding padding, serializable output
   and optional attention. Add CLI and offline synthetic smoke workflow.
6. Update documentation, run full pytest/Ruff/diff checks, review all changes.

Shared interfaces: samples are dictionaries with actor_boxes [N,4] normalized
xyxy, actor_labels [N], group_label scalar, optional pose_features [N,Dp] and
rgb_features [N,Dr]. Batches pad N and expose actor_valid_mask [B,N] (True=valid),
actor_labels [B,N] with -100 padding, group_labels [B]. All-empty scenes fail.
Raw samples add frames [T,3,H,W] float RGB [0,1]; batch contains frames [B,T,3,H,W].
Model returns group_logits [B,Cg], actor_logits [B,N,Ca], optional attention
[B,L,H,N,N] (per branch for late fusion). Padded values are cleared before use.

Research choices requiring documentation: center-frame static features for both
train/test when only center annotations exist; normalized centers scaled to a
configurable reference image grid; early concat projection; probability-average
late fusion with configurable pose weight. Raw backbones require vetted, feature-only local exports and explicit
preprocessing provenance. No real-dataset accuracy claim until evaluated.
