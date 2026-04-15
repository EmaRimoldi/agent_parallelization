# Phase 03b: Compute Full Decomposition

## Goal

Run the corrected BP-style decomposition on all replicates using the upgraded mode labels and deterministic evaluation data.

## Background

With deterministic evaluation and upgraded mode labeling, the decomposition should now produce non-degenerate estimates for φ, G, and ε. Each term has a clear estimator aligned with BP's definitions.

## Tasks

### 1. Run the decomposition for each replicate

```bash
for rep in 1 2 3 4 5; do
  python scripts/compute_decomposition.py \
    --d00 runs/full_d00_rep${rep} \
    --d10 runs/full_d10_rep${rep} \
    --d01 runs/full_d01_rep${rep} \
    --d11 runs/full_d11_rep${rep} \
    --output workflow/results/decomposition_rep${rep}.json
done
```

### 2. Aggregate across replicates

For each cell and each term, compute:
- Mean across 5 reps
- Standard error (sample std / sqrt(5), t-distribution CI with df=4)
- 95% CI using t₄,0.025 = 2.776

Write to: `workflow/results/decomposition_aggregate.json`

### 3. Check for non-degeneracy

For each term, check:
- **cost_token, cost_wall**: should be non-NaN for all cells (always computable)
- **φ**: how many reps have non-NaN φ? (requires mode overlap)
- **G**: how many reps have non-NaN G? (requires accepted posterior)
- **ε**: how many reps have non-NaN ε? (requires accepted + routing distributions)

Record:
```bash
python workflow/run.py measure '{"phi_defined_reps": <n>, "G_defined_reps": <n>, "epsilon_defined_reps": <n>}'
```

### 4. Compute Jensen remainders

For each cell and axis:
- R_α = log E[κ_α] - E[log κ_α]

These should match the prior cost variance analysis. Verify consistency.

### 5. Write decomposition summary

Create `workflow/results/decomposition_summary.md`:
- Table of all terms by cell, averaged across reps
- Which terms are non-zero and significant
- Which terms are still NaN/degenerate
- Assessment of whether the decomposition is non-trivial

## Required Inputs

- All full 2x2 experiment directories with mode labels
- `scripts/compute_decomposition.py`

## Expected Outputs

- `workflow/results/decomposition_rep{1-5}.json`
- `workflow/results/decomposition_aggregate.json`
- `workflow/results/decomposition_summary.md`

## Success Criteria

- Decomposition computed for all 5 reps without crashes
- At least one of φ, G, ε is non-NaN in ≥ 3 of 5 reps for at least one cell
- Cost terms are consistent across reps (std < 30% of mean)

## Failure Modes

- All φ, G, ε remain NaN: mode overlap is still insufficient despite upgrade. Consider structured search fallback.
- Cost terms vary wildly: check that evaluation is still deterministic (re-run Phase 01a).
- Decomposition script crashes: check for division by zero, empty mode distributions.

## Next Phase

On completion: proceed to `03c_identity_check`
