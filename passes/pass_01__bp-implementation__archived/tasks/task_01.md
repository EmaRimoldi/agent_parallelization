# Task 1: Fix model passthrough in agent runner

Read `docs/guides/IMPLEMENTATION_GUIDE.md` → Task 1 for full specification.

## What to do

Edit `src/agent_parallelization_new/agents/claude_agent_runner.py`:

1. Find the command construction (around line 392) where `cmd = ["claude", "--print", ...]` is built.

2. Add `--model` passthrough from `self.config.model`:
```python
if self.config.model:
    cmd += ["--model", self.config.model]
```

3. Switch output format from `"text"` to `"json"` to enable token tracking in the next task:
```python
"--output-format", "json",
```

4. Add a helper method `_parse_claude_output(self, raw_stdout: str) -> tuple[str, dict]` that parses the JSON output and extracts the text response and usage metadata. Handle the case where JSON parsing fails by falling back to raw text with empty usage dict.

5. Update `_run_turn()` to use the parsed output instead of raw stdout.

## Success criteria
- `self.config.model` is passed to the `claude` CLI via `--model`
- Output is parsed from JSON format
- A `_parse_claude_output` method exists and handles JSON decode errors gracefully
- `pytest -q` passes with no new failures (some pre-existing failures are expected)
- Changes committed with message "Task 1: Fix model passthrough in agent runner"

Do NOT proceed to other tasks.
