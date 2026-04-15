# Architecture

## Overview

`agent_parallelisation_new` runs parallel ML research experiments using Claude Code
sub-agents as the LLM backend, replacing the original OpenClaw + Gemini system.

## Component map

```
launcher.py
  → ExperimentConfig (config.py)
  → Orchestrator (orchestrator.py)
      → IsolatedAgentProcess × N (agents/isolated_agent_process.py)
          → ClaudeAgentRunner (agents/claude_agent_runner.py)
              → `claude --print` CLI calls
              → BudgetTracker (budgeting.py)
      → workspace.py (git worktree per agent)
      → training_harness.py (run_training.sh, check_training.sh)
  → collector.py (reads per-agent output files after all finish)
  → reporter.py (writes final_report.md)
```

## Independence guarantees

Agents are isolated at three levels:
1. **Directory**: each agent has a separate git worktree under `runs/.../agent_N/workspace/`
2. **Process**: each agent runs in a separate Python `multiprocessing.Process`
3. **Session**: each agent gets a unique session ID; no session sharing

The orchestrator never reads one agent's results and passes them to another.
Aggregation happens only after all agents finish.

## Budget accounting

```
Parallel mode (Mode 1):
  agent_0: budget = T
  agent_1: budget = T
  total compute = 2T  (2 GPUs simultaneously)
  wall-clock ≈ T

Single-agent-longer mode (Mode 2):
  agent_0: budget = 2T
  total compute = 2T
  wall-clock ≈ 2T

Both modes consume identical total compute — the parallel mode trades
wall-clock time for exploration diversity.
```

## Merge phase

The merge phase (combining two agents' best changes) was implemented in the original
system but **never produced a result better than the better parallel agent** in any
of the 5 runs where it completed. It is intentionally omitted from this system.

Future extension: if needed, implement in `experiment_modes/merge_phase.py` following
the spec in `configs/prompts/merge_phase.md` from the original repo.
