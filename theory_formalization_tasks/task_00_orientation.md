# Task 00: Orientation And Guardrails

## Goal

Build a shared understanding of the current state before changing theory, code, or experiments.

## Read First

Read these files in order:

1. `theory_validation_bp_20260412/README.md`
2. `theory_validation_bp_20260412/analysis/final_verdict.md`
3. `theory_validation_bp_20260412/analysis/formal_theory_audit.md`
4. `theory_validation_bp_20260412/analysis/protocol_compliance_audit.md`
5. `theory_validation_bp_20260412/analysis/noise_assay_interpretation.json`
6. `theory_validation_bp_20260412/theory/autoresearch_bp_revised.pdf`
7. `theory_validation_bp_20260412/theory/autoresearch_bp.pdf`
8. `theory_validation_bp_20260412/theory/BP.pdf`

## Reviewer Findings You Must Treat As Binding Until Overturned

The last review established the following working assumptions:

1. The BP algebra itself is not the main issue.
2. The dangerous gaps are:
   - reward/loss bridge,
   - averaged `kappa_bar` vs fixed `kappa`,
   - architecture indexing vs BP model indexing,
   - broken estimators for `phi`, `G`, and `epsilon`.
3. The current pilot does not validate the four-term decomposition.
4. The noise level is large enough that single-shot best-of-N evaluation is unreliable.
5. The next step is not another large pilot. It is:
   - theorem tightening,
   - protocol/logging repair,
   - estimator repair,
   - then targeted experiments.

## Required Output

Create a short note:

- `theory_validation_bp_20260412/analysis/task_00_orientation_note.md`

The note must contain:

- your current understanding of the theorem status
- the top 5 blockers
- the top 3 next actions
- what you consider the success condition for the whole queue

## Completion Checklist

- [ ] All required files read
- [ ] Orientation note written
- [ ] The note is consistent with the existing validation bundle
- [ ] No theory or code changes made yet unless needed for note-taking only
