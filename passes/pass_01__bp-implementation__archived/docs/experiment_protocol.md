# Experiment Protocol

## Running a matched experiment

### Step 1: Run parallel mode

```bash
python scripts/run_parallel_experiment.py \
  --time-budget 30 \
  --train-budget 300 \
  --experiment-id my_exp_001
```

This launches 2 agents on GPU 0 and GPU 1 simultaneously, each with a 30-minute
budget and 300s per training run. Wall-clock time: ~30 minutes.

### Step 2: Run single-agent-longer mode

```bash
python scripts/run_single_long_experiment.py \
  --time-budget 30 \
  --train-budget 300 \
  --experiment-id my_exp_001_single
```

This launches 1 agent on GPU 0 with a 60-minute budget (2×30). Total compute
matches the parallel run. Wall-clock time: ~60 minutes.

### Step 3: Compare

```bash
python scripts/compare_experiments.py \
  runs/experiment_my_exp_001 \
  runs/experiment_my_exp_001_single
```

## Output structure

```
runs/
  experiment_<id>/
    config.json
    mode_parallel/
      agent_0/
        workspace/         # git worktree (isolated)
        logs/
          run_agent.log    # agent session log
        results/
          trajectory.jsonl
          results.tsv
          metadata.json
          snapshots/
      agent_1/             # same structure
      aggregate/
        combined_summary.json
        comparison_table.csv
        experiment_report.txt
    mode_single_long/
      agent_0/             # same structure
      aggregate/
    final_comparison/
      parallel_vs_single.csv
      final_report.md
```

## Environment requirements

- `claude` CLI installed and authenticated
- `uv` installed (for running train.py)
- 2 CUDA-capable GPUs for parallel mode (or 1 for single mode)
- `autoresearch` submodule initialized: `git submodule update --init`
- Data prepared: `cd autoresearch && uv run prepare.py`

## Known best baseline

- val_bpb = 1.1020746984708296
- Run: `exp_smart_20260330_063836 / agent_0 / iter0003_s350_bpb1.1021.py`
- Hyperparameters: EMBEDDING_LR=0.8, UNEMBEDDING_LR=0.005, MATRIX_LR=0.06,
  WEIGHT_DECAY=0.1, WARMDOWN_RATIO=0.4
