# Phase 03a: Mode Labeling Upgrade and Execution

## Goal

Upgrade the mode labeling system to a two-level scheme (subsystem x change_type) and apply it to all full 2x2 experiment runs.

## Background

The current 6-category keyword labeling is too coarse: accepted-mode posteriors collapse to point masses with 1-2 accepted edits. A finer labeling scheme with more categories, applied to more data, should produce non-degenerate mode distributions.

## Tasks

### 1. Upgrade `scripts/label_modes.py`

Add a two-level classification system:

**Level 1 — Subsystem** (which part of train.py was changed):
- `optimizer`: optimizer type, LR, weight_decay, momentum, betas
- `scheduler`: schedule type, warmup, min_lr, cosine parameters
- `architecture`: depth, width, channels, normalization, activation
- `regularization`: dropout, weight_decay (when used as regularizer), label smoothing, gradient clipping
- `data_augmentation`: augmentation methods, batch size, data loading
- `training_loop`: epochs, gradient accumulation, mixed precision, loss function

**Level 2 — Change type** (what kind of edit):
- `tune`: modify a numeric hyperparameter
- `add`: add a new component (layer, scheduler, augmentation)
- `remove`: remove a component
- `replace`: swap one component for another (e.g., SGD → AdamW)

**Combined mode**: `{subsystem}_{change_type}`, e.g., `optimizer_tune`, `architecture_add`

### 2. Improve diff-based classification

Instead of just keyword matching on hypothesis text, analyze the actual diff:
- Parse the changed lines
- Identify which code region was modified (optimizer setup, model definition, data loading, etc.)
- This is more reliable than hypothesis text

### 3. Apply to all experiment runs

```bash
for dir in runs/full_*; do
  python scripts/label_modes.py --experiment-dir "$dir"
done
```

### 4. Verify mode coverage

Check that the labeling produces non-degenerate distributions:
```bash
python workflow/scripts/check_mode_coverage.py --pattern "runs/full_*" \
  --output workflow/artifacts/mode_coverage.json
```

Report:
- Total labeled steps per cell
- Number of distinct modes per cell
- Number of accepted edits per mode per cell
- Mode distribution per cell

### 5. Record results

```bash
python workflow/run.py measure '{"mode_labeling": "two_level", "modes_per_cell": {...}}'
```

## Required Inputs

- All experiment directories from Phase 03
- Updated `scripts/label_modes.py`

## Expected Outputs

- `mode_labels.jsonl` files in each agent's results directory
- `workflow/artifacts/mode_coverage.json`
- Updated measurements in state

## Success Criteria

- Each cell has ≥ 4 distinct modes with ≥ 2 labeled steps
- At least 2 cells have overlapping accepted modes (required for φ estimation)
- Mode distributions differ across cells (required for G > 0)

## Failure Modes

- All edits fall into one mode: the labeling is too coarse or the agent's behavior is too uniform. Consider increasing the number of subsystem categories or using unsupervised clustering.
- No accepted edits have mode labels: ensure the labeling script processes snapshot metadata correctly.

## Next Phase

On completion: proceed to `03b_decomposition`
