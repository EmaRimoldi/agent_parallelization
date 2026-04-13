# Task 7: Four-term decomposition estimators

Read `IMPLEMENTATION_GUIDE.md` → Task 7 for full specification.

## What to do

Create `scripts/compute_decomposition.py` with the complete code from the implementation guide.

The script must:
1. Accept `--d00`, `--d10`, `--d01`, `--d11` as experiment directory arguments
2. Load all instrumented data (turns.jsonl, training_runs.jsonl, mode_labels.jsonl) from each cell
3. Compute for each cell vs d00 baseline:
   - `κ̄_token`: mean total tokens per attempt
   - `κ̄_wall`: mean wall-clock seconds per attempt
   - `log(κ̄₀/κ̄)` on both axes (cost term)
   - `φ`: within-mode competence (placeholder 0.0 for now)
   - `G`: information gain from entropy of mode distribution
   - `ε`: routing mismatch as KL divergence from pooled prior
   - `Δ_token` and `Δ_wall`: full decomposition
4. Test hypotheses H1–H6
5. Compute κ̄_token stratified by c/K quartile bins (for H5)
6. Print a formatted table to stdout
7. Save full results to `decomposition_results.json`

Use the functions `compute_kappa_token`, `compute_kappa_wall`, `compute_kappa_by_context_bin`, `compute_mode_distribution`, `entropy`, `kl_divergence`, `compute_decomposition`, `test_hypotheses` from the implementation guide.

## Success criteria
- `scripts/compute_decomposition.py` runs without errors on valid experiment directories
- Produces readable table on stdout with all 4 terms for d10, d01, d11
- `decomposition_results.json` is valid JSON with decomposition values and hypothesis test results
- Committed with message "Task 7: Four-term decomposition estimators"

Do NOT proceed to other tasks.
