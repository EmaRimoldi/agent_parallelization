# Phase 03: Full 2x2 Experiment

## Goal

Run all 4 architecture cells (d00, d10, d01, d11) with deterministic evaluation, extended budget, and enough replicates for the full BP decomposition.

## Background

The calibration phase confirmed that architecture differences are detectable with deterministic evaluation. This phase collects the full dataset needed for the four-term decomposition across all cells.

## Tasks

### 1. Configure all 4 cells

For each config file (`experiment_d00.yaml` through `experiment_d11.yaml`):
- Set `time_budget_minutes: 60`
- Set `train_time_budget_seconds: 60`
- Verify all other settings are consistent

### 2. Run 5 replicates of each cell

```bash
for cell in d00 d10 d01 d11; do
  for rep in 1 2 3 4 5; do
    python scripts/run_parallel_experiment.py \
      --config configs/experiment_${cell}.yaml \
      --output-dir runs/full_${cell}_rep${rep}
  done
done
```

Total: 20 experiment runs.

With 60-min budget and 60s training timeout:
- d00, d10 (1 agent): ~20 attempts per agent per rep → ~100 per cell
- d01, d11 (2 agents): ~40 attempts per rep → ~200 per cell

### 3. Verify completeness

For each run, check:
- `mode_*/agent_*/results/training_runs.jsonl` exists and has entries
- At least 10 training runs with val_bpb values per agent

```bash
python workflow/scripts/check_completeness.py --pattern "runs/full_*"
```

### 4. Record metadata

```bash
python workflow/run.py measure '{"full_2x2_cells": 4, "full_2x2_reps": 5, "full_2x2_budget_minutes": 60}'
python workflow/run.py log "Full 2x2 experiment completed"
```

## Required Inputs

- Deterministic prepare.py and train.py (from Phase 01)
- Experiment configs for all 4 cells
- Sufficient compute and API budget for 20 experiment runs

## Expected Outputs

- 20 experiment directories: `runs/full_{d00,d10,d01,d11}_rep{1-5}`
- Each containing training_runs.jsonl, turns.jsonl, and snapshot data
- ~100-200 training runs per cell across all reps

## Success Criteria

- All 20 experiment directories contain valid data
- Each cell has ≥ 80 total training runs across 5 reps
- No more than 1 failed replicate per cell

## Failure Modes

- API rate limiting: space runs out or reduce parallelism
- Agent crashes: check logs, retry failed reps
- Insufficient training runs: extend budget or add more reps

## Next Phase

On completion: proceed to `03a_mode_labeling`
