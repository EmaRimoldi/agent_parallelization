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

## Open questions / known issues

### Q: How many runs fit in a 10-minute budget?

**Answer: ~1 complete run.**

Each training run occupies ~390s of wall-clock time even though the training budget is 300s:

| Phase | Time |
|---|---|
| Python import + model build | ~5s |
| `torch.compile` (step 0, `dt ≈ 31s`) | ~31s |
| Steps 1–10 (excluded from budget counter) | ~19s |
| **Actual training (steps 11→end)** | **~301s** |
| Final `evaluate_bpb()` | ~15s |
| NFS polling + worker loop overhead | ~5s |
| Worker `sleep 2` + `run.result` roundtrip | ~3s |
| **Total wall-clock per run** | **~379s** |

With 600s agent budget: 600 / 379 ≈ **1.6 runs → effectively 1**.

To fit 2 runs in 10 minutes, either reduce `train_time_budget_seconds` to ~180s (two runs × ~270s ≈ 540s), or increase the agent budget to 15 minutes.

### Q: Does `torch.compile` happen on every run?

**No** — on the second and subsequent runs within the same experiment, the kernel cache in `/tmp/torchinductor_<user>/` (node-local) is reused. Recompilation only happens on the first run per experiment (or when the SLURM worker lands on a different node).

Note: the training budget counter (`if step > 10: total_training_time += dt`) already excludes steps 0–10, so compile time does **not** eat into the 300s training budget. It still consumes ~50s of wall-clock time per run.

### Q: Why is the workspace so large?

The workspace is a git worktree (~source files only, no git history duplication). The `.venv` is now a symlink to `autoresearch/.venv` — a single shared 7 GB environment. Previously a new 7 GB venv was created per workspace because `autoresearch/.venv` did not exist; fixed by running `uv sync` in `autoresearch/`.

### Q: Why can't I see the CoT (chain of thought) of the sub-agent?

The `claude` CLI (`--print --output-format text`) exposes only final assistant text, not internal thinking. To expose reasoning, replace the subprocess `claude --print` call in `agents/claude_agent_runner.py:_invoke_claude_turn` with a direct Anthropic SDK call using `thinking={"type": "enabled", "budget_tokens": N}`.

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

## TODO

- [ ] **Agent-driven merge phase** — the current merge (`run_merge_phase.py`) is deterministic: it parses scalar hyperparameters with regex and picks the best value found per parameter. This is brittle and misses interactions between parameters. Replace it with a Claude agent that reads all agent trajectories, reasoning traces, and snapshots, reasons about which changes were causal vs. incidental, and produces a merged `train.py` by judgement rather than by greedy per-parameter selection.

## Docs

- [Parallel Capacity](docs/parallel_capacity.md)
- [Merge Protocol](docs/merge_protocol.md)
- [Workflow Diagram](docs/workflow_diagram.md)
