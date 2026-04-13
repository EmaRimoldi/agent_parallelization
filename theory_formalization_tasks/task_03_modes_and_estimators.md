# Task 03: Modes And Correct Estimators

## Goal

Replace the currently broken / placeholder decomposition estimators with definitions that are at least structurally aligned with BP.

## Inputs

- `theory_validation_bp_20260412/analysis/formal_theory_audit.md`
- `theory_validation_bp_20260412/analysis/protocol_compliance_audit.md`
- current `scripts/compute_decomposition.py`
- current `scripts/label_modes.py`

## Required Work

### A. Operational mode definition

Define an observable mode system that is explicit and reproducible.

Possible ingredients:

- diff class
- hypothesis text
- changed subsystem
- tool-use pattern
- edit/evaluate trajectory features

The mode definition must avoid circularity as much as possible.
Do not define modes in a way that trivially forces the decomposition to be non-degenerate.

### B. `phi` estimator

Implement a real estimator for within-mode competence differences.

It does not need to be perfect, but it must not be a hardcoded zero.
If sample size is too small, it may return `nan` with justification. That is acceptable.

### C. `G` estimator

Replace the entropy-difference placeholder with something aligned to mutual information:

- either a direct empirical `I(S; D)` estimator,
- or a clearly labeled approximation that explicitly states what `D` is and why the approximation is being used.

### D. `epsilon` estimator

Replace `KL(mode_distribution || prior)` with an estimator tied to routing mismatch:

- identify what counts as routing allocation `q_D`
- identify what counts as posterior `pi_D`
- estimate `E_D[KL(pi_D || q_D)]`

If exact estimation is impossible, implement the cleanest justified proxy and document the gap.

### E. Token-cost calibration

Either:

- validate the chars//4 proxy empirically on a sample, or
- replace it with a better token count path.

## Deliverables

- updated `scripts/label_modes.py`
- updated `scripts/compute_decomposition.py`
- estimator note:
  `theory_validation_bp_20260412/analysis/estimator_design.md`

The note must include a table with columns:

- quantity
- BP definition
- implemented estimator
- assumptions
- known bias / limitation

## Validation

Run the estimators on at least one existing pilot repetition and confirm:

- they run without crashing
- they no longer trivially collapse for implementation reasons alone
- if they still collapse, explain whether that is due to data insufficiency rather than code design

Write:

- `theory_validation_bp_20260412/analysis/estimator_validation_note.md`

## Completion Checklist

- [ ] Mode definition documented
- [ ] `phi` no longer hardcoded to zero
- [ ] `G` no longer uses the old incorrect entropy-difference formula
- [ ] `epsilon` no longer uses the old incorrect KL-vs-prior formula
- [ ] Estimator design note written
- [ ] Validation note written
