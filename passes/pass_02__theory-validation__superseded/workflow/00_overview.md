# Phase 00: Overview and Workspace Setup

## Goal

Initialize the research workflow, verify prerequisites, and confirm the strategy before beginning experimental work.

## Background

This workflow implements the Fix-and-Scale research strategy for validating the BP four-term decomposition applied to AutoResearch agent architectures. The core insight from prior work is that evaluation non-determinism (time-based random seed in train.py) was the root cause of nearly every downstream problem: noise, selection bias, degenerate mode distributions, and unreplicable results.

The workflow has four main phases:
1. Make evaluation deterministic (eliminates noise floor)
2. Power calibration on one cell pair (go/no-go gate)
3. Full 2x2 experiment with mode identification
4. Theorem revision based on evidence

With conditional branches to CIFAR-100 escalation or structured search if needed.

## Tasks

1. Verify the repo is clean and on the correct branch:
   ```bash
   cd {{repo_root}}
   git status
   git log --oneline -5
   ```

2. Verify the key files exist:
   - `autoresearch/prepare.py` (evaluation harness)
   - `autoresearch/train.py` (baseline training script)
   - `scripts/compute_decomposition.py` (decomposition code)
   - `scripts/label_modes.py` (mode labeling)
   - `configs/experiment_d00.yaml` through `experiment_d11.yaml`

3. Verify the workflow directory is initialized:
   ```bash
   ls workflow/phases/ workflow/scripts/ workflow/artifacts/
   ```

4. Read and confirm the current state of the codebase:
   - What dataset is being used? (CIFAR-10)
   - What is the agent budget? (30 min per agent)
   - What is the training timeout? (120 seconds)
   - How is the random seed set? (time-based — the problem we fix next)

## Required Inputs

- Clean git repo with the autoresearch pipeline
- Python environment with torch, numpy

## Expected Outputs

- Confirmation that all prerequisite files exist
- A log entry confirming workspace readiness
- No code changes in this phase

## Success Criteria

- All listed files exist and are readable
- The repo is on a clean branch (no uncommitted changes to key files)
- The workflow state file is initialized

## Failure Modes

- Missing files: check git history, ensure correct branch
- Dirty repo state: commit or stash changes before proceeding

## Next Phase

On success: proceed to `01_deterministic_eval`
