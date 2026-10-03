# Milestone 1 implementation record

The user's detailed specification is the design authority. Initial inspection:
clean main, one initial commit, one-line README only. Python 3.13.5, uv, pytest,
and Ruff available; no usable Python 3.11 launcher, PyTorch, OpenCV, or TensorBoard.
No existing source conventions or user data to preserve.

Implementation sequence:

1. Package/src layout, ignore rules, and mathematical/data tests.
2. JSONL records, source grouping, segmentation, scorer, paired MIL, feature loader.
3. Streaming decoding, explicit C3D transforms, local FC6 checkpoint adapter.
4. Reproducible training, checkpoints, TensorBoard, frame/bag metrics, inference, plots.
5. Local preparation, UCF official lists/annotations, joint split CLI, experiment CLIs.
6. Synthetic verification, editable installation, CLI smoke checks, Ruff, full pytest,
   diff/status review. No commits, pushes, publications, or model/dataset downloads.

Decisions: released network's second hidden layer is linear. Requested Adagrad
0.001 overrides released script's 0.01. Paired MIL reductions average over bags;
released all-pairs/batch scaling is not copied. Squared-weight penalty is explicit.
Epoch schedule and pair sampler are implementation defaults. Short inputs repeat
ordered units because 32 disjoint nonempty partitions are impossible with N<32.
Unknown sources require maps. DCSASS requires actual normalized clip labels.
Frame evaluation never derives abnormal frame truth from weak bag labels.
Pretrained C3D compatibility/preprocessing remain externally supplied requirements.

Verification focuses on cross-dataset source isolation, temporal coverage, loss
math/gradients, checkpoint roundtrip/resumption, finite/shape features, tiny-video
decoding, frame annotation conventions, and reduced evaluation labeling.

Final verification: editable installation succeeded; 28 tests passed; Ruff lint
and formatting checks passed. Installed CLIs completed a synthetic 4096-dimensional
train/evaluate/infer smoke run with JSON and a visually inspected timeline PNG.
Expected missing-data errors were checked. Review found and fixed a zero-test-ratio
rounding bug (largest-remainder allocation now preserves zero-ratio empty splits)
and clarified extending the epoch target before resuming a completed run.
All changes remain uncommitted on main; no pretrained model or dataset downloaded.
