# Pose to Protect — Codex Sequential Sprint Bundle

This directory contains **one orchestrator** and **four numbered tasks** for a *single* Codex agent to execute sequentially, October 26–27, 2026.

## Files

- `00_START_HERE.md` — give this one to Codex; directs the entire run
- `01_FAILURE_ANALYSIS_AND_RESEARCH.md`
- `02_MODEL_OPTIMIZATION_AND_GPU.md`
- `03_DATASET_QUALITY_AUTOMATION.md`
- `04_INTEGRATION_AND_HANDOFF.md`

## Setup

1. Extract/copy this **entire folder**, intact, into the repository, e.g. `D:\Github Repos\violence-group-anomaly-detection\docs\agent-sprints\20261026\`.
2. Open Codex with `D:\Github Repos\violence-group-anomaly-detection` as its workspace.
3. Provide **only** the launch instruction below (you don't need to paste all four task descriptions):

> Read `docs/agent-sprints/20261026/00_START_HERE.md`, then execute files `01` through `04` sequentially in this same checkout and session. Finish each phase, write its local handoff, and continue to the next without asking for routine approval. Preserve all pre-existing uncommitted work. Do not commit, push, merge, deploy, approve an Indian-domain split, or train on unapproved Indian clips. Return one consolidated sprint report at the end.

The numbered files use *relative file names*, so the agent should read them from the same directory as the orchestrator. If the repo root is different, adapt the folder path in the launch instruction.

## Practical notes

- Run as **one long Codex job with sequential phases**, not four concurrent agents editing the same uncommitted working tree.
- Codex must have access to local CSVs, media, checkpoints, and GPU if any phase is to use them; absent inputs must be reported, not assumed.
- Research browsing depends on capabilities made available to the local agent. Lack of internet must not lead to invented citations.
- This bundle itself does **not** run code, change the repository, or launch training. It supplies instructions for the agent to do the authorized work locally.
