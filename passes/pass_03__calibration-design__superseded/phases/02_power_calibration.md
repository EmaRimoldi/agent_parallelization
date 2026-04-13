# Phase 02: Power Calibration — d00 vs d10

## Goal

Run a focused calibration experiment on one cell pair (d00 = single/no memory, d10 = single/memory) with deterministic evaluation to determine whether architecture differences are detectable at all.

## Background

With deterministic evaluation, each agent-produced `train.py` maps to exactly one val_bpb. The only source of variance is the agent's strategy choices. This phase tests whether the d00 vs d10 architecture difference produces measurably different outcomes.

This is a go/no-go gate: if we cannot detect any architecture effect with clean evaluation on the simplest comparison, the CIFAR-10 substrate is too simple and we escalate.

## Tasks

### 1. Configure the calibration experiments

Modify configs to use reduced training timeout for more attempts:

For `configs/experiment_d00.yaml` and `configs/experiment_d10.yaml`:
- Set `time_budget_minutes: 45` (up from 30)
- Set `train_time_budget_seconds: 60` (down from 120)
- This gives ~15 training attempts per agent instead of ~5

### 2. Run 5 replicates of each cell

Execute the experiment runner for each cell, 5 times:

```bash
for rep in 1 2 3 4 5; do
  python scripts/run_parallel_experiment.py \
    --config configs/experiment_d00.yaml \
    --output-dir runs/calibration_d00_rep${rep}
done

for rep in 1 2 3 4 5; do
  python scripts/run_parallel_experiment.py \
    --config configs/experiment_d10.yaml \
    --output-dir runs/calibration_d10_rep${rep}
done
```

Alternatively, use the calibration wrapper script:
```bash
python workflow/scripts/run_calibration.py --repo-root {{repo_root}} --reps 5
```

### 3. Collect results

For each run, the agent will have produced training_runs.jsonl with val_bpb values. Since evaluation is deterministic, each val_bpb is the true score of that code modification.

### 4. Save raw data

```bash
python workflow/run.py log "Calibration runs completed for d00 and d10, 5 reps each"
```

## Required Inputs

- Deterministic `autoresearch/train.py` and `autoresearch/prepare.py` (from Phase 01)
- Experiment configs `configs/experiment_d00.yaml`, `configs/experiment_d10.yaml`
- The experiment runner `scripts/run_parallel_experiment.py`

## Expected Outputs

- 10 experiment directories: `runs/calibration_d00_rep{1-5}`, `runs/calibration_d10_rep{1-5}`
- Each containing `mode_*/agent_*/results/training_runs.jsonl` with val_bpb values
- ~75 total training runs per cell (15 per agent x 5 reps)

## Success Criteria

- All 10 experiment directories contain valid training_runs.jsonl files
- Each cell has at least 50 training runs with val_bpb values
- No experiment crashed or timed out without producing results

## Failure Modes

- Agent crashes: check API keys, model availability, rate limits
- Training crashes: check that the deterministic modifications didn't break anything
- Too few runs: increase time_budget_minutes or reduce train_time_budget_seconds

## Next Phase

On completion: proceed to `02a_analyze_calibration`
