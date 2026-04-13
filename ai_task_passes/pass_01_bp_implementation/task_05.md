# Task 5: Implement d11 — parallel agents with shared memory

Read `docs/guides/IMPLEMENTATION_GUIDE.md` → Task 5 for full specification.

## What to do

### 5a. Add config flag
In `src/agent_parallelization_new/config.py`, add to `AgentConfig`:
```python
use_shared_memory: bool = False
```

### 5b. Create shared log file in workspace setup
In `src/agent_parallelization_new/utils/workspace.py`, in `create_workspace()`:
- When mode is `parallel_shared`, create `experiment_dir/shared_results_log.jsonl` (if not exists)
- Symlink it into each agent's workspace as `workspace/shared_results_log.jsonl`

### 5c. Write to shared log after each experiment
In `src/agent_parallelization_new/agents/claude_agent_runner.py`, add method `_append_shared_log(self, step, hypothesis, val_bpb, accepted)` that appends a JSON record to the shared log with file locking (`fcntl.flock`).

Call this method after each `update_snapshot.py` completes (detect via the watcher or after the agent reports a result).

### 5d. Read shared log before each turn
Add method `_build_shared_memory_context(self) -> str` that reads `workspace/shared_results_log.jsonl` and produces a compact markdown table showing all agents' results.

Inject into turn message when `self.config.use_shared_memory` is True.

### 5e. Create new experiment mode
Create `src/agent_parallelization_new/experiment_modes/parallel_shared_memory.py` — identical to `parallel_two_agents.py` but sets `use_shared_memory=True` on all agents and passes the shared log path to workspace creation.

### 5f. Register in launcher and pyproject.toml
Add `main_parallel_shared()` entry point and `run-parallel-shared` CLI command.

## Success criteria
- `AgentConfig` has `use_shared_memory` field
- Shared JSONL log is created at experiment level and symlinked into each workspace
- File locking prevents corruption from concurrent writes
- Each agent sees the other's results in its turn message
- New mode `parallel_shared` is launchable via `run-parallel-shared`
- `pytest -q` passes with no new failures
- Changes committed with message "Task 5: Implement d11 parallel agents with shared memory"

Do NOT proceed to other tasks.
