# HANDOVER — Indian pilot fine-tune (HCI violence project)

Handover date: 2026-11-03 (IST). Deadline: **Nov 10 professor presentation**.

Give this whole file to the Mac Claude Code session. Save it in the repo as `docs/handover/HANDOVER.md` and commit it, so the context survives session compaction. Re-read it when unsure.

---

## 1. What you are and what you're allowed to do

You are now the working agent for this project. You have the full context below, and you are expected to make engineering decisions on your own. You do not need approval for routine choices.

**Log every non-trivial decision** in `runs/indian_pilot/DECISIONS.md` with:
- the date
- what you decided
- why
- what alternatives you rejected

This log matters as much as the code, because the presentation depends on being able to justify choices.

**Hard stops.** Message me and wait before doing any of these:
1. Destructive git operations: force-push, history rewrite, deleting branches, or merging the two municipal platform versions.
2. Modifying or deleting original media, existing frozen receipts, or earlier `runs/` outputs.
3. **Touching the Indian test split before the evaluation protocol is committed** (§6, Phase D). Evaluate on test exactly once.
4. Anything that would put footage, data CSVs, or weights into git, or share data externally.
5. If a result shows the pilot is fundamentally invalid. Example: the collection-shortcut probe (§6, Phase C) shows the model is mostly learning collection identity *and* no mitigation works. In that case, report it rather than presenting the result.

**Checkpoints.** Report at these points, without waiting for a reply unless a hard stop applies:
- after setup
- after the split is frozen
- after the validation results
- after the test evaluation
- when the demo is ready

Keep reports short: numbers, decisions, and problems.

## 2. Project background (condensed)

The HCI course project is "violence/goon detection in crowds" on video. We operationalize it as **observable violent events** only. We never infer a person's identity, intent, character, affiliation, or protest legitimacy. Peaceful protests are not positives, and everything goes to human review.

Three technical tracks exist. **Only track 1 matters now.**

1. **This repo's five-stage pipeline.** Video → C3D FC6 (Sports1M) → Sultani MIL anomaly scorer (UCF-Crime) → Faster R-CNN people → I3D Mixed_4f actor RGB → Actor-Transformer six-class behavior head (DCSASS) → React console for human review.
   - Its honest baseline numbers are weak, and that is fine to show:
     - UCF frame ROC-AUC 0.744.
     - The full UCF→DCSASS cascade on 991 protected clips had alert ROC-AUC 0.512, F1 0.473, and Normal FPR 40.2%.
     - The six-class head had zero correct Abuse/Assault/Fighting predictions on its covered test clips.
2. **A separate X3D/YOLO/GINEConv "Q3" quality-aware fusion prototype** in another local repo. It tied the scene-only model at 96.33% on a 436-video test and calibrated worse. **Not used here.**
3. **"Pose to Protect" / FedAvg federated learning.** This is a referenced external paper and a proposal only, never implemented. **No federated work in this sprint.**

**Sprint direction (decided):**
- Keep the existing five-stage pipeline and six-class model.
- No new architecture, no federated learning, no new repo.
- Compute: the Mac (M4 Pro, 48 GB). The Legion is busy with other projects, and the H100 is unavailable.

## 3. The goal, stated precisely

Produce a **pilot fine-tuned binary violence model on Indian footage** that is *decent and working, not robust*. Present it honestly, with limitations. The core claim to support with evidence:

> "The model works modestly on Indian footage, and it measurably improves with more data. More annotated data is the path forward."

That claim is supported by:
1. a fine-tuned head beating the zero-shot existing cascade on the same frozen Indian test set, and
2. **a learning curve**, where test performance rises as the training data grows.

A modest result with a clean protocol beats an inflated result. A sharp professor will ask about leakage, so make the answer airtight.

## 4. Current state (verified by the previous Mac session)

**Machine and assets**
- All 5 pinned weights and 9 demo clips are present on the Mac, and their SHA256 hashes match the pins.
- The repo venv is Python 3.12. **Keep 3.12 if the pins work.** Don't rebuild on 3.11 just because an earlier prompt said so.
- UCF-Crime, DCSASS, and Collective are on `/Volumes/Seagate_2TB/`. They are not required for the pilot and should be treated as read-only. See the optional variant in §6, Phase C.
- The Indian data was transferred from the Legion through Google Drive (folder `My Drive/violence-pilot-transfer/2026-11/`, with a SHA256 `MANIFEST.csv`), with `/Volumes/Seagate_2TB/violence-datasets/` as the intended destination. **Verify it:**
  - Every file in the manifest is present and its hash matches.
  - Reconcile files on disk vs inventory rows vs annotation rows.
  - Expected counts: 523 items (521 video + 2 audio-only); labels 48 violence / 465 nonviolence / 8 uncertain-or-blank; QC 504 candidates (39 violence / 465 nonviolence) / 17 holds.
  - Report any mismatch.

**Git**
- This checkout: `feat/municipal-evaluation-platform` at `472c7d8` (redesigned review UI on top of `4b479f3`, the Mac's platform version), based on `main` `01d1ef1`.
- The Legion pushed `legion/municipal-oct25`, with three commits:
  - `208c75f` and `4d7c27a`: the Legion's version of the municipal platform. It conflicts heavily with the Mac's version (about +4,900 / −2,900 lines). **Do not merge** these. Reconciling the two platform versions is deferred until after Nov 10.
  - `27d31f0`: the Oct 25 Indian-domain work. 70 files, 68 of them new:
    - modules `indian_domain`, `indian_readiness`, `indian_split`, `indian_binary` (model / training / evaluation), and review/failure analysis
    - 20 scripts, including `freeze_indian_split.py` and `train_indian_binary.py`
    - 125 tests
  - It does not import either municipal platform version.
- **Decision: cherry-pick `27d31f0` onto `feat/municipal-evaluation-platform`.** It was tested in a throwaway worktree: it applies cleanly, and the result passes 151 tests with 2 skipped.
  - Do it on a new branch, `pilot/indian-binary`, created from `472c7d8`, so the UI branch stays untouched.
  - **Reuse the existing split and training code** rather than writing new versions. Extend or patch it where §6 requires.

**Critical data facts from the Legion's notes / `counts.json`**
- No Indian split or test set has been frozen, and no Indian training or inference has run. The test set is clean.
- The readiness gates mark `training_eligible: 0` for all 521 clips, because source and provenance aren't cleared.
  - The pilot **overrides** this deliberately. Implement the override as an explicit, named config flag, for example `pilot_override: team_labels_unadjudicated`, not by editing the gates.
  - State the override plainly in the report.
- Source and provenance columns are mostly blank. Source groups must be reconstructed (§6, Phase B).
- **⚠️ Label/collection confound (the single biggest validity risk).** All 352 clips with documented Indian origin are labeled nonviolence, and all 48 violence clips have unknown geography. So a classifier can score well by learning *which collection a clip came from* instead of *whether it shows violence*.
  - Your protocol must measure this and mitigate it (§6, Phases B–D).

## 5. Pilot definition

- **Eligible clips:** video, readable, labeled violence or nonviolence, not in the 17 holds, and not uncertain or blank.
- **Labels:** team-annotated and **not adjudicated**. The report calls this a pilot.
- **Collection tag:** give every clip a `collection` field, using documented-Indian vs unknown-geography at minimum, plus finer groupings if filenames or batches reveal them. This tag drives the confound controls.

## 6. Plan

Suggested timeline (adjust if needed, and log it):
- Nov 3–4: A + B + C-split
- Nov 5–6: features + training
- Nov 7: test evaluation + demo
- Nov 8–9: buffer + presentation material

### Phase A — Setup
1. Create `pilot/indian-binary` from `472c7d8` and cherry-pick `27d31f0`. Run the tests.
2. Verify the Indian data per §4.
3. Run `scripts/demo_preflight.py`, then do an upload-mode demo smoke test on CPU with one Indian clip.
4. **MPS check.**
   - Set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
   - Extract C3D and I3D features for 5 clips on both MPS and CPU.
   - Compare max and mean absolute difference, and timing. Record which ops fell back to CPU.
   - Pick MPS or CPU for extraction and log the decision.
5. Checkpoint report.

### Phase B — Source groups
- Inspect `indian_split` first: does it already group by source? Extend it rather than duplicating it.
- Reconstruct groups by combining:
  - **perceptual hashes** of about 1 frame/sec, linking clips that share near-identical frames (tune the Hamming threshold on obvious pairs and log it)
  - shared filename, URL, or ID patterns
  - and, as weak hints, duration, resolution, and file timestamps
- Union-find the links into groups. Unlinked clips become singletons.
- **Review every group that contains a violence clip yourself**, by viewing thumbnails with the Read tool. Merge clips that are clearly from the same incident or camera.
  - Log each judgment in `manifests/v2-pilot/violence_cluster_review.md`, so I can audit it later.
  - When unsure, merge. Over-grouping is safe; under-grouping leaks.
- Output `manifests/v2-pilot/source_groups.csv`: `clip, group_id, collection, link_reasons, label`.

### Phase C — Split, features, training

**Split**
- Use `freeze_indian_split.py`, extended as needed. Make a group-wise split of about 70/15/15, with no group crossing splits.
- **Stratify by label × collection.** Val and test must each contain:
  - violence clips: aim for 6 or more, and report the actual counts
  - **nonviolence clips from the same unknown-geography collection as the violence clips**, as well as documented-Indian nonviolence
- If that's impossible, choose the best feasible split and document why.
- Ambiguous groups go to train.
- Write `manifests/v2-pilot/split_v1.csv` and its SHA256, then commit the split ("Freeze Indian pilot split v1"). **The test set is locked from here.**
- Checkpoint report.

**Features**
- Extract once and cache to `runs/indian_pilot/features/`, recording the extractor hashes. Use C3D FC6 32-segment bags, the same as the pipeline, and optionally I3D pooled features.
- Extraction of all eligible clips can run as a background job.

**Training** (use `train_indian_binary.py`, extended as needed)
- Freeze the backbones. Train a small binary head on cached features: a 1–2 layer MLP, or a MIL max-pool head.
- Use class-weighted BCE or focal loss.
- Select the checkpoint and decision threshold on **validation only**.

**Learning curve**
- Train on nested subsets of train *groups*: 25%, 50%, 75%, 100%.
- Run 3 seeds per size, and keep the same subsets across seeds.

**Collection-shortcut probe** (required)
- Train the same head to predict `collection` instead of violence. If collection is near-perfectly predictable from the features, the confound is live.
- Then report the violence model's metrics **within the unknown-geography collection alone**, on validation. That subset holds both violence and nonviolence clips, so a collection shortcut can't help there. This within-collection number is the honest headline.
- Optional mitigation, to try if the probe is bad: train violence vs nonviolence only within the unknown-geography collection, then compare.

**Baseline**
- Compute zero-shot scores from the existing cascade (the Sultani anomaly score, and the cascade alert) on validation.

**Optional variant** (only if ahead of schedule)
- Train on Indian train data plus DCSASS violence/normal clips from the Seagate drive.
- Compare against Indian-only on the *same* frozen Indian test set. This is secondary evidence for "more data helps."

- Checkpoint report with all validation results.

### Phase D — Frozen protocol, one-time test, demo
1. **Commit `runs/indian_pilot/EVAL_PROTOCOL.md` before touching test.** It fixes:
   - the chosen models
   - the thresholds
   - the metrics
   - the subsets to report (overall, and within unknown-geography)
2. Evaluate on test once, comparing the zero-shot cascade against the fine-tuned head. Report:
   - PR-AUC and ROC-AUC
   - precision, recall, and F1 at the validation-chosen threshold
   - recall at a fixed FPR
   - confusion matrix
   - bootstrap 95% CIs
   - the same metrics within the unknown-geography collection
   - the **learning curve on test**: mean ± std over seeds versus train size
3. Pick 5 correct predictions and 5 failures, each with a one-line likely cause.
4. **Demo**
   - Add the fine-tuned head to the existing console as an optional "Indian pilot" binary score, shown alongside the existing outputs, with the human-review framing kept.
   - Use upload mode on a few Indian test clips.
   - Verify the live scores match the offline evaluation scores.
   - Run it end to end on the Mac and record the exact launch commands.
5. Write `runs/indian_pilot/REPORT.md`, written for the presentation:
   - setup and data counts
   - the pilot override and unadjudicated labels
   - how source groups were reconstructed
   - the label/collection confound, and what the probe showed
   - the results with CIs
   - the learning curve
   - failure cases
   - what more and better data would fix
6. Commit everything except media, features, and weights.

## 7. Principles

- Keep the populations and denominators of different metrics separate:
  - component metrics vs end-to-end metrics
  - validation vs test
  - overall vs within-collection
- Never tune on test. Never re-run test and report the better run.
- A CSV label is an annotation, not adjudicated ground truth. Call it what it is.
- Report negative and weak results plainly. A weak but honest number is the deliverable.
- When a decision is a judgment call, make it, log it, and move on. Don't wait for me on routine choices.
