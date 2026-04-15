# Task 3: Add structured per-training-run wall-clock logging

Read `docs/guides/IMPLEMENTATION_GUIDE.md` → Task 3 for full specification.

## What to do

Edit `src/agent_parallelization_new/agents/claude_agent_runner.py`:

1. In the watcher that detects `run.result` (around line 318–341), after detecting a completed training run, write a structured JSONL record to `agent_dir/results/training_runs.jsonl`:

```python
training_run_record = {
    "run_index": self._training_run_count,
    "turn": self.turn_count,
    "started_at": run_wall_start,
    "finished_at": time.time(),
    "wall_seconds": time.time() - run_wall_start,
    "val_bpb": parsed_val_bpb,
    "status": "success" or "crash",
}
```

2. Also parse `training_seconds` from `logs/train_current.out` using the existing `log_parser.py` and include it in the record.

3. Initialize `self._training_run_count = 0` and increment it each time a run completes.

## Success criteria
- `training_runs.jsonl` is created in `agent_dir/results/`
- Each record has timing and metric data
- `pytest -q` passes with no new failures
- Changes committed with message "Task 3: Structured per-training-run wall-clock logging"

Do NOT proceed to other tasks.
