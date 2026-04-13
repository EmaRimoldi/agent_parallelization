# Task 04: Targeted High-Value Experiments

## Goal

Run the smallest set of experiments that most strongly determine whether the theorem is empirically identifiable and whether the architecture contrasts are real.

## Inputs

- `theory_validation_bp_20260412/analysis/final_verdict.md`
- `theory_validation_bp_20260412/analysis/validation_strategy.md`
- reviewer conclusion already summarized in the validation bundle

## Experiment Priority

Run experiments in this order unless a prerequisite fails.

### Experiment 1: Replicated cell means

Objective:

- estimate stable cell-level performance without best-of-N optimism

Minimum target:

- at least 5 repeated evaluations per cell

Preferred adaptive target:

- keep running until either:
  - confidence intervals separate enough to support a real contrast, or
  - uncertainty remains too large and you conclude the substrate is too noisy

Use means and uncertainty, not only best values.

### Experiment 2: Within-architecture cost variance

Objective:

- estimate whether using averaged `kappa_bar` is reasonable

Compute:

- mean
- variance
- Jensen gap proxy or delta-method approximation

### Experiment 3: Context-pressure sweep

Objective:

- enter a regime where H5 is actually testable

Vary one or more of:

- available context budget
- prompt payload size
- memory-table size
- history retention policy

The point is to force `c/K` into higher bins.

### Experiment 4: Optional follow-up if estimator pipeline is ready

Only if Tasks 02 and 03 succeeded well enough:

- rerun a small 2x2 comparison with corrected logging and corrected estimators

## Required Outputs

Create a subdirectory:

- `theory_validation_bp_20260412/experiments/followup_01/`

Store:

- configs or commands used
- raw logs
- summarized metrics
- a markdown note per experiment

At minimum write:

- `theory_validation_bp_20260412/analysis/experiment_01_replicated_means.md`
- `theory_validation_bp_20260412/analysis/experiment_02_cost_variance.md`
- `theory_validation_bp_20260412/analysis/experiment_03_context_sweep.md`

Each note must contain:

- claim tested
- exact procedure
- results
- interpretation
- whether failure points to theory or experiment

## Completion Checklist

- [ ] Replicated means experiment run and summarized
- [ ] Cost variance analysis run and summarized
- [ ] Context-pressure sweep run and summarized or explicitly declared infeasible with reason
- [ ] All outputs saved under `theory_validation_bp_20260412/`
