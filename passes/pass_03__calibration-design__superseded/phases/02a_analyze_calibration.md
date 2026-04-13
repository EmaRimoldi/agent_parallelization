# Phase 02a: Analyze Calibration Results

## Goal

Compute effect sizes, mode diversity counts, and cost variance from the calibration data to inform the Phase 02b decision gate.

## Background

Phase 02 produced 5 replicates each of d00 and d10 with deterministic evaluation. This phase analyzes the data to determine whether the architecture difference is detectable and whether mode distributions are non-degenerate.

## Tasks

### 1. Label modes for all calibration runs

```bash
for rep in 1 2 3 4 5; do
  python scripts/label_modes.py --experiment-dir runs/calibration_d00_rep${rep}
  python scripts/label_modes.py --experiment-dir runs/calibration_d10_rep${rep}
done
```

### 2. Compute per-cell statistics

For each cell, aggregate all training_runs.jsonl across 5 reps:

- **val_bpb**: mean, std, min, max across all runs
- **Best val_bpb**: the true best (deterministic, so no selection bias)
- **Training runs count**: total number of completed training attempts
- **Mode distribution**: count of runs per mode category

Use the analysis script:
```bash
python workflow/scripts/analyze_calibration.py \
  --repo-root {{repo_root}} \
  --output workflow/artifacts/calibration_analysis.json
```

### 3. Compute effect size

Cohen's d = (mean_d10 - mean_d00) / pooled_std

Record:
```bash
python workflow/run.py measure '{"cohens_d": <value>, "d00_mean": <value>, "d10_mean": <value>, "d00_n": <n>, "d10_n": <n>}'
```

### 4. Compute mode diversity

For each cell, count the number of distinct modes with at least 2 accepted edits.

Record:
```bash
python workflow/run.py measure '{"d00_mode_count": <n>, "d10_mode_count": <n>, "d00_modes": [...], "d10_modes": [...]}'
```

### 5. Compute cost variance and Jensen gap

For each cell, compute the within-cell cost variance and Jensen gap on both axes, following the same methodology as the prior experiment_02_cost_variance analysis.

Record:
```bash
python workflow/run.py measure '{"d00_jensen_token": <v>, "d10_jensen_token": <v>, "d00_jensen_wall": <v>, "d10_jensen_wall": <v>}'
```

### 6. Write analysis summary

Write a brief summary to `workflow/artifacts/calibration_summary.md` covering:
- Sample sizes
- Effect size and confidence interval
- Mode distribution per cell
- Jensen gap values
- Recommendation for the decision gate

## Required Inputs

- Calibration experiment directories from Phase 02
- `scripts/label_modes.py`
- `workflow/scripts/analyze_calibration.py`

## Expected Outputs

- `workflow/artifacts/calibration_analysis.json` (machine-readable)
- `workflow/artifacts/calibration_summary.md` (human-readable)
- Updated measurements in workflow state

## Success Criteria

- All calibration runs are included in the analysis
- Effect size, mode counts, and Jensen gaps are computed and recorded
- Summary document is written

## Failure Modes

- Missing data: some reps may have failed. Analyze what is available and note the gap.
- Zero accepted edits in some reps: note this as a mode diversity issue.

## Next Phase

On completion: proceed to `02b_decision_gate`
