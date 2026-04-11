# Task 4: Implement d10 — single agent with external memory

Read `IMPLEMENTATION_GUIDE.md` → Task 4 for full specification.

## What to do

### 4a. Add config flag
In `src/agent_parallelization_new/config.py`, add to `AgentConfig`:
```python
use_external_memory: bool = False
```

### 4b. Add memory injection to agent runner
In `src/agent_parallelization_new/agents/claude_agent_runner.py`:

1. Add method `_build_memory_context(self) -> str` that reads `agent_dir/reasoning/trace.jsonl` and produces a compact markdown table:
```
# Experiment Log
| # | change | bpb | Δ | best |
|---|--------|-----|---|------|
| 1 | LR 3e-4→1e-4 | 1.341 | -0.02 | ✓ |
```

2. In the turn message construction for turns > 0 (the "Continue the research…" message), prepend the memory context if `self.config.use_external_memory` is True:
```python
if self.config.use_external_memory:
    memory = self._build_memory_context()
    if memory:
        turn_msg = f"{memory}\n\n---\n\n{turn_msg}"
```

### 4c. Create new experiment mode
Create `src/agent_parallelization_new/experiment_modes/single_agent_memory.py` — identical to `single_agent_double_budget.py` but sets `use_external_memory=True` on the agent config.

### 4d. Add launcher entry point
In `src/agent_parallelization_new/launcher.py`, add `main_single_memory()` that routes to the new mode.

### 4e. Register CLI
In `pyproject.toml`, add:
```toml
run-single-memory = "agent_parallelization_new.launcher:main_single_memory"
```

### 4f. Add YAML support
In `configs/experiment.yaml`, document `single_memory` as a valid mode.

## Success criteria
- `AgentConfig` has `use_external_memory` field
- `_build_memory_context()` method exists and produces a compact table from trace.jsonl
- New experiment mode `single_memory` exists and is launchable via `run-single-memory`
- `pytest -q` passes with no new failures
- Changes committed with message "Task 4: Implement d10 single agent with external memory"

Do NOT proceed to other tasks.
