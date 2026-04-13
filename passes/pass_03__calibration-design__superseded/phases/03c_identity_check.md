# Phase 03c: Verify Accounting Identity

## Goal

Check whether the four-term decomposition identity holds: Δ_α ≈ cost_α + φ_α + G - ε + R_α for each cell and axis.

## Background

The BP decomposition is an algebraic identity. If our estimators are correct and the packed-family assumptions hold, the left-hand side (observed Δ) should equal the right-hand side (sum of estimated terms) to within estimation error.

## Tasks

### 1. Compute observed Δ for each cell

For each cell d relative to baseline d00:
- Δ_token(d) = best_val_bpb(d00) - best_val_bpb(d) [or mean, depending on definition]
- Δ_wall(d) = analogous for wall-clock-adjusted metric

Note: with deterministic evaluation, "best" is unambiguous — it's the code modification with the lowest val_bpb.

### 2. Compute the right-hand side

For each cell d and each rep:
- RHS_α = cost_α + φ_α + G - ε + R_α

Use values from `workflow/results/decomposition_rep{1-5}.json`.

### 3. Compute the residual

residual_α = Δ_α - RHS_α

This should be small (< tolerance) if the identity holds.

### 4. Set tolerance

The tolerance should be the larger of:
- The estimation uncertainty (SE of the RHS terms)
- 0.05 (a practical threshold for this scale)

### 5. Evaluate for each cell and axis

| Cell | Axis | Δ | RHS | Residual | Within tolerance? |
|------|------|---|-----|----------|-------------------|
| d10 | token | ... | ... | ... | ... |
| d10 | wall | ... | ... | ... | ... |
| d01 | token | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

### 6. Record results

```bash
python workflow/run.py measure '{"identity_residuals": {...}, "identity_holds": <bool>}'
```

### 7. Make the decision

**Case 1: Identity holds** (all residuals within tolerance)
→ The decomposition works. Proceed to theorem update.

**Case 2: Identity fails because terms are degenerate** (NaN terms make RHS undefined)
→ The decomposition is underidentified. Consider structured search fallback.

**Case 3: Identity fails with bounded remainder** (residuals are nonzero but bounded)
→ The decomposition approximately works. The remainder becomes part of the theorem. Proceed to theorem update.

```bash
python workflow/run.py decide <identity_holds|identity_fails_degenerate|identity_fails_remainder>
```

## Required Inputs

- `workflow/results/decomposition_rep{1-5}.json`
- `workflow/results/decomposition_aggregate.json`

## Expected Outputs

- `workflow/results/identity_check.json`
- `workflow/results/identity_check_summary.md`
- Decision recorded in state

## Decision Branches

- **identity_holds**: → `06_theorem_update`
- **identity_fails_degenerate**: → `05_structured_search`
- **identity_fails_remainder**: → `06_theorem_update` (with remainder explicitly in theorem)

## Failure Modes

- Cannot compute Δ: need a clear definition of the performance gap metric. Use mean val_bpb per cell.
- Large residuals: could indicate estimator bugs, violated assumptions, or a genuine gap in the theory.
