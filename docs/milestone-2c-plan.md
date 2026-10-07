# Milestone 2C execution plan

Initial state: clean `feat/automatic-actor-detection` at `8f56b11`. The requested
branch did not exist; created `feat/collective-real-benchmark` after all **234
tests passed before edits**. No real Collective tree or pretrained checkpoints
were found in the repository. CUDA is unavailable (CPU PyTorch build).

1. Validate actual sequences, images, annotations, vocabulary, split membership
   and leakage. Keep the documented 32/12 source list explicit; its IDs come from
   related author code, not a published exact Actor-Transformers split listing.
2. Resolve one experiment protocol with three seeds, frame/features/optimizer
   settings, metrics and source-only validation holdout. Freeze immutable JSON
   receipts by content hash; later parameter changes invalidate them.
3. Inspect local backbone features on bounded scenes; strengthen GT and detected
   cache provenance without changing model mathematics.
4. Tune only confidence on explicitly validation-only broad candidate detections,
   using detector F1 subject to minimum recall. Freeze the selection before test.
5. Orchestrate single-modality GT runs before fusion and detected comparison,
   recording resolved configs, environment, predictions, metrics and errors.
6. Aggregate compatible seed populations, retaining individual values and sample
   standard deviation; produce confusion and confidence summaries.
7. Run an offline synthetic orchestration smoke and full pytest/Ruff/diff checks.

Real-data performance remains unmeasured until datasets and vetted feature/detector
weights exist. Synthetic/dry-run outputs are explicitly non-benchmark. Never tune
using test predictions, choose a mode from test accuracy, or combine incompatible
GT/matched-actor populations. No architecture/cascade/UI expansion or git publishing.
