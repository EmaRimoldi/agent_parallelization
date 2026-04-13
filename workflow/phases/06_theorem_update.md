# Phase 06: Revise Theorem Based on Evidence

## Goal

Update the theorem statement, assumptions, and proof sketch based on the experimental findings from Phases 02-03.

## Background

The experiments have now produced data on:
- Whether architecture effects are detectable (Phase 02)
- Whether the decomposition is non-degenerate (Phase 03b)
- Whether the identity holds (Phase 03c)
- The magnitude of the Jensen remainder (all phases)

This phase synthesizes those findings into a revised theoretical statement.

## Tasks

### 1. Collect all evidence

Read from state and artifacts:
- `workflow/results/decomposition_aggregate.json`
- `workflow/results/identity_check.json`
- `workflow/artifacts/calibration_analysis.json`
- Measurements in `workflow/state.json`

### 2. Assess each assumption

For each of the four theorem assumptions, determine if it is:
- **Verified**: empirical evidence supports it
- **Plausible**: evidence is consistent but not definitive
- **Violated**: evidence contradicts it
- **Untested**: insufficient data

| Assumption | Status | Evidence |
|-----------|--------|----------|
| 1. Packed family (non-degenerate modes) | ? | Mode coverage data |
| 2. Latent-loss regularity | Verified (if deterministic) | σ_eval = 0 |
| 3. Bounded cost variance | ? | Jensen gap measurements |
| 4. Mode overlap for φ | ? | Cross-cell mode overlap |

### 3. Revise the theorem statement

Based on the evidence:

**If the identity holds with non-degenerate terms:**
- State the theorem as a conditional identity with verified assumptions
- List which assumptions were empirically verified vs assumed
- Give the Jensen remainder as an explicit, measured bound

**If the identity holds but terms are partially degenerate:**
- State which terms were identifiable and which collapsed
- Formulate the theorem as a partial decomposition
- Note which conditions are needed for full identification

**If the identity fails:**
- Identify where the accounting breaks down
- Propose a weaker theorem (e.g., cost-term-only, or two-term)
- Diagnose whether the failure is theoretical or methodological

### 4. Write the revised theorem

Write `workflow/results/revised_theorem.md` containing:
- The strongest defensible theorem statement
- Required assumptions (with empirical status)
- The proof sketch (from BP inheritance)
- The estimation protocol
- Open questions

### 5. Update the revised LaTeX if applicable

If the findings support updating `theory_validation_bp_20260412/theory/autoresearch_bp_revised.tex`, note the specific changes needed.

## Required Inputs

- All results from Phases 02-03
- The current theorem from `theory_validation_bp_20260412/theory/autoresearch_bp_revised.pdf`
- BP framework (Theorems 19, 21, 32)

## Expected Outputs

- `workflow/results/revised_theorem.md`
- Updated assumption status table
- Recommendation for LaTeX updates

## Success Criteria

- Every assumption has an empirical status assessment
- The theorem statement is consistent with the evidence
- The gap between what is proved and what is conjectured is explicit

## Failure Modes

- Evidence is contradictory: document the contradiction and propose resolution experiments
- Not enough evidence for a theorem: state the strongest conditional claim possible

## Next Phase

On completion: proceed to `07_final_report`
