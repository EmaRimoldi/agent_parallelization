# Theory Validation Bundle

This folder is a self-contained bundle for auditing, stress-testing, and refining the AutoResearch BP theory.

Current status:

> **promising but not yet rigorous**

The theorem has been narrowed, the protocol and estimator layers have been upgraded, and targeted follow-up experiments have been run. The bundle now supports a much cleaner review than the original pilot-only package.

## Best Reading Order

If you want the shortest path to the current conclusion:

1. `analysis/final_verdict.md`
2. `analysis/reanalysis_summary.md`
3. `theory/autoresearch_bp_revised.pdf`
4. `analysis/handoff_note.md`

If you want the core theory + empirical trace:

1. `analysis/validation_strategy.md`
2. `analysis/formal_theory_audit.md`
3. `analysis/theorem_refactor_summary.md`
4. `analysis/protocol_upgrade_spec.md`
5. `analysis/estimator_design.md`
6. `analysis/experiment_01_replicated_means.md`
7. `analysis/experiment_02_cost_variance.md`
8. `analysis/experiment_03_context_sweep.md`
9. `theory/autoresearch_bp_revised.pdf`

## What Changed Since The First Bundle

This bundle now includes:

- a narrower single-axis theorem with explicit Jensen remainder
- protocol upgrades for incumbent reevaluation and provenance logging
- corrected mode-labeling and decomposition estimators
- repeated incumbent evaluations across all four cells
- explicit Jensen-gap analysis for token and wall-clock cost
- a context-pressure feasibility analysis

## Folder Structure

### `theory/`

- `autoresearch_bp.pdf`
  The original AutoResearch theory PDF supplied for validation.
- `autoresearch_bp_extracted.txt`
  Plain-text extraction of the original theory PDF.
- `BP.pdf`
  The Beneventano--Poggio source paper.
- `BP_extracted.txt`
  Plain-text extraction of the BP paper.
- `autoresearch_bp_revised.tex`
  Current LaTeX source of the narrowed theory note.
- `autoresearch_bp_revised.pdf`
  Current best theory PDF.

### `analysis/`

Key review files:

- `final_verdict.md`
- `reanalysis_summary.md`
- `handoff_note.md`

Formal / theorem files:

- `formal_theory_audit.md`
- `theorem_refactor_summary.md`
- `validation_strategy.md`

Protocol / estimator files:

- `protocol_compliance_audit.md`
- `protocol_upgrade_spec.md`
- `protocol_upgrade_smoke_check.md`
- `estimator_design.md`
- `estimator_validation_note.md`

Experiment summaries:

- `experiment_01_replicated_means.md`
- `experiment_02_cost_variance.md`
- `experiment_03_context_sweep.md`

Machine-readable outputs:

- `noise_assay_interpretation.json`
- `context_pressure_metrics.json`
- `mode_label_coverage.json`
- `protocol_compliance.json`
- `estimator_validation_rep1.json`
- `corrected_decomposition_rep1.json`
- `corrected_decomposition_rep2.json`
- `corrected_decomposition_rep3.json`

### `artifacts/`

Original pilot artifacts:

- `pilot_summary.md`
- `pilot_raw_data.json`
- `decomposition_rep1.json`
- `decomposition_rep2.json`
- `decomposition_rep3.json`
- `pilot_mapping.json`

### `experiments/noise_assay/`

Targeted verifier-noise follow-up:

- `baseline/`
- `best_d10/`
- `noise_summary.json`

Purpose:

- show that single-shot selection is unreliable
- test whether the pilot-selected best `d10` candidate replicates

### `experiments/followup_01/`

Targeted post-refactor experiments:

- `run_replicated_means.py`
- `analyze_cost_variance.py`
- `replicated_means_summary.json`
- `cost_variance_summary.json`
- `context_sweep_feasibility.json`
- `replicated_means/`
  Raw per-run logs from the repeated incumbent evaluations.

Purpose:

- estimate repeated incumbent means per cell
- quantify Jensen-gap risk for `kappa_bar`
- assess whether a real H5 context sweep is currently feasible

### `code/`

- `compute_decomposition.py`
- `aggregate_pilot.py`
- `label_modes.py`

These are included so a future reviewer can inspect the exact estimator and aggregation logic used by the bundle.

### `logs/`

- `pilot_phase2.log`
- `pilot_phase2_attempt1.log`

Batch logs from the original pilot execution.

## Main Findings In One Paragraph

The original AutoResearch BP note was too strong. The current best theorem is a single-axis BP reduction with explicit extra assumptions and a Jensen remainder. The protocol now has a real reevaluation path and corrected estimator scaffolding, so the decomposition no longer collapses for implementation reasons alone. However, repeated incumbent evaluations still show overlapping uncertainty across the main cells, wall-clock Jensen gaps are large enough to matter, and the pilot still does not generate enough accepted-mode support for stable `phi`, `G`, and `epsilon` estimation. So the theory is now cleaner and more defensible, but still not validated empirically in this substrate.

## Important Caveat

This bundle is intended for analysis and review, not as a minimal rerun package. It keeps raw evidence, scripts, and summaries together so an external reviewer can reconstruct the logic of the verdict without asking for missing context.
