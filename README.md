# Agent Parallelization

Parallel agent experiment framework using Claude Code sub-agents to search hyperparameters faster through independent exploration.

## How it works

![Workflow overview](docs/workflow_overview.svg)

N independent Claude Code sub-agents run in parallel, each with its own isolated git worktree, GPU allocation, time budget, and output directory. Agents never communicate during the run — they explore the hyperparameter space independently and their best results are merged at the end.

### Agent lifecycle

```
Orchestrator
│
├─ spawns N processes simultaneously
│
├─ Agent 0 ──────────────────────────────────────────────────────────►
│    │  bash start_gpu_worker.sh   →  SLURM allocates GPU
│    │  ┌─ gpu_allocated_at written ── budget clock starts here ──┐
│    │  │                                                          │ T minutes
│    │  │  loop:                                                   │
│    │  │    edit train.py → bash run_on_worker.sh → val_bpb       │
│    │  │    keep if improved, git reset if not                    │
│    │  └──────────────────────────────────────────────────────────┘
│    │  bash stop_gpu_worker.sh   →  GPU released
│
├─ Agent 1 ──────────────────────────────────────────────────────────►
│    │  (same flow, different GPU, independent git branch)
│
└─ Collector → Merger → final merged train.py
```

**Budget accounting:** the T-minute clock starts from the moment SLURM allocates the GPU, not from when the agent process starts. This ensures all agents get a fair T minutes of actual research time regardless of queue wait.

### Training loop (per agent)

Each agent runs this loop autonomously until interrupted:

1. Form a hypothesis — one scalar hyperparameter change
2. Edit `train.py`, `git commit`
3. `python save_snapshot.py` — log hypothesis before training
4. `bash run_on_worker.sh` — blocks until training completes (~5 min), prints `val_bpb`
5. Keep commit if `val_bpb` improved, `git reset --hard HEAD~1` if not
6. `python update_snapshot.py` — record outcome and next step
7. Repeat

The 5-minute training budget is wall-clock time **excluding compilation** — training runs are directly comparable across different hyperparameter configurations.

## Modes

| Mode | Command | Description |
|---|---|---|
| Parallel search | `run-parallel` | N independent agents × T budget |
| Single long search | `run-single-long` | 1 agent × 2T budget (control) |
| Capacity benchmark | `python scripts/benchmark_parallel_capacity.py` | Find empirical upper bound on N |
| Merge phase | `python scripts/run_merge_phase.py` | Aggregate best results after parallel search |

## Quick Start

```bash
# Run two parallel agents for 30 minutes each
run-parallel --time-budget 30 --train-budget 360

# Run single agent for 60 minutes (matched budget)
run-single-long --time-budget 30 --train-budget 360

# Or use a config file
run-parallel --config configs/experiment.yaml
```

## Requirements

- Python ≥ 3.10
- Claude Code CLI (`claude`) in PATH
- SLURM cluster (configure partition/gres in `configs/experiment.yaml`)
- `uv` package manager

## Configuration

Edit `configs/experiment.yaml` to set agents, budget, model, and SLURM parameters:

```yaml
agents:
  n: 2
  model: claude-haiku-4-5-20251001
  time_budget_minutes: 30
  train_time_budget_seconds: 300

slurm:
  partition: pi_tpoggio
  gres: gpu:1
  time: "00:10:00"
```

## Architecture

```
src/agent_parallelization_new/
  config.py              — ExperimentConfig, AgentConfig dataclasses
  orchestrator.py        — launches and monitors sub-agents
  budgeting.py           — wall-clock budget tracking (starts at GPU allocation)
  snapshotting.py        — saves train.py snapshots on every change
  reasoning_trace.py     — structured per-step reasoning logs
  merger.py              — aggregates trajectories into a merged train.py
  resource_benchmark.py  — empirical parallelism upper-bound estimation
  agents/                — isolated subprocess runner, Claude CLI wrapper
  outputs/               — schema, collector, evaluator, reporter
  compatibility/         — SLURM training harness, script generators
  utils/                 — git worktree management, log parsing
```

## Docs

- [Parallel Capacity](docs/parallel_capacity.md)
- [Merge Protocol](docs/merge_protocol.md)
- [Workflow Diagram](docs/workflow_diagram.md)
