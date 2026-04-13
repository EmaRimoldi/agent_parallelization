# Task 6: Post-hoc mode labeling

Read `docs/guides/IMPLEMENTATION_GUIDE.md` → Task 6 for full specification.

## What to do

Create `scripts/label_modes.py` with the complete code from the implementation guide.

The script must:
1. Accept `--experiment-dir` as argument
2. For each agent in the experiment, read all snapshots from `snapshots/step_*/train.py`
3. Diff each snapshot against the baseline (or previous accepted step)
4. Classify the diff into one of 5 modes using keyword matching: `optimizer`, `lr_schedule`, `architecture`, `batch_data`, `other`
5. Write results to `agent_dir/results/mode_labels.jsonl`

Use the `MODE_KEYWORDS` dictionary and `classify_diff()` function from the implementation guide verbatim.

## Success criteria
- `scripts/label_modes.py` exists and runs without errors on a completed experiment
- `mode_labels.jsonl` is produced with correct fields: `step`, `mode`, `diff_lines_changed`, `hypothesis`, `val_bpb_after`, `accepted`
- Committed with message "Task 6: Post-hoc mode labeling script"

Do NOT proceed to other tasks.
