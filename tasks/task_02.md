# Task 2: Add per-turn token and context tracking

Read `IMPLEMENTATION_GUIDE.md` → Task 2 for full specification.

## What to do

Edit `src/agent_parallelization_new/agents/claude_agent_runner.py`:

1. At runner initialization, create a turns log path:
```python
self.turns_log_path = self.agent_dir / "results" / "turns.jsonl"
self._cumulative_chars = 0
```

2. After each call to `_run_turn()`, append a JSONL record to `turns.jsonl` containing:
   - `turn`, `timestamp`, `model`
   - `input_tokens`, `output_tokens` (from JSON usage if available, else estimate as chars // 4)
   - `system_prompt_chars`, `turn_msg_chars`, `response_chars`
   - `context_fill_ratio` (cumulative chars / 4 / 200000, capped at 1.0)
   - `wall_clock_seconds` (elapsed time for this turn)

3. Add method `_estimate_context_fill(self) -> float` that computes c/K from `self._cumulative_chars`.

4. Increment `self._cumulative_chars` after each turn with the total chars of system prompt + turn message + response.

5. At end of run (in `_write_metadata()`), add aggregate fields to `metadata.json`:
   - `total_input_tokens`, `total_output_tokens`, `total_turns`
   - `avg_context_fill`, `final_context_fill`

## Success criteria
- `turns.jsonl` is created in `agent_dir/results/` with one record per turn
- Each record has all specified fields
- `metadata.json` includes the new aggregate token/context fields
- `pytest -q` passes with no new failures
- Changes committed with message "Task 2: Add per-turn token and context tracking"

Do NOT proceed to other tasks.
