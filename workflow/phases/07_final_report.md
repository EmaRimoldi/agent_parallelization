# Phase 07: Generate Final Report

## Goal

Synthesize all experimental evidence, theoretical revisions, and methodological findings into a comprehensive final report suitable for publication or project handoff.

## Background

The workflow has now completed either the main path (deterministic eval → calibration → full 2x2 → theorem) or one of the branch paths (CIFAR-100 escalation, structured search). This phase collects everything.

## Tasks

### 1. Compile the evidence inventory

List all artifacts produced across phases:
- State measurements
- Decomposition results
- Identity check results
- Mode coverage analysis
- Cost variance / Jensen gap analysis
- Decision gate rationale
- Theorem revision

### 2. Write the executive summary

In 1 page:
- What was the research question?
- What was the experimental approach?
- What were the key findings?
- What is the strongest defensible theorem?
- What remains open?

### 3. Write the methodology section

Document the exact experimental protocol:
- Deterministic evaluation setup (seeds, DataLoader config)
- Agent architecture descriptions (d00, d10, d01, d11)
- Budget parameters (time, training timeout, replicates)
- Mode labeling scheme
- Decomposition estimators
- Statistical methods (t-distribution CIs, Cohen's d, bootstrap)

### 4. Write the results section

Present:
- Calibration results (effect sizes, mode counts)
- Full decomposition table (all terms by cell)
- Identity check (residuals)
- Jensen remainder analysis
- Hypothesis test outcomes (H1'-H6')

### 5. Write the discussion

Address:
- What the decomposition reveals about agent architecture
- Which theoretical predictions were confirmed/falsified
- Comparison with the original (noisy) pilot
- Limitations and open questions
- Recommendations for future work

### 6. Package the report

Write to `workflow/reports/final_report.md` with all sections.

Create a companion `workflow/reports/final_report_data.json` with all machine-readable results.

### 7. Update the theory validation bundle

Copy key outputs to `theory_validation_bp_20260412/`:
- Updated `analysis/final_verdict.md`
- New `analysis/deterministic_results_summary.md`
- Updated decomposition JSONs

## Required Inputs

- All phase artifacts and results
- Workflow state with all measurements
- Revised theorem from Phase 06

## Expected Outputs

- `workflow/reports/final_report.md`
- `workflow/reports/final_report_data.json`
- Updated theory validation bundle

## Success Criteria

- Report covers all major findings
- All claims are supported by referenced data
- The report is self-contained (a reader can understand without prior context)
- The recommended next steps are actionable

## Failure Modes

- Missing data from earlier phases: note gaps and work around them
- Contradictory findings: present both sides and explain

## Next Phase

This is the final phase. The workflow is complete.

After completion:
```bash
python workflow/run.py complete
```

The workflow state will show status `__done__`.
