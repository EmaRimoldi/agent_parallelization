# Task 01: Theorem Refactor

## Goal

Tighten the theory so that every formal claim is either:

- explicitly inherited from BP,
- explicitly assumed,
- or clearly labeled empirical / conjectural.

## Inputs

- `theory_validation_bp_20260412/theory/autoresearch_bp_revised.tex`
- `theory_validation_bp_20260412/analysis/formal_theory_audit.md`
- `theory_validation_bp_20260412/theory/BP.pdf`

## Required Work

1. Decide whether the best current theorem should be:
   - a single-axis theorem with remainder terms, or
   - a two-axis theorem with an explicit axis-sharing assumption.

2. Update the revised theory source to make the following points precise:
   - what object is theorem-level: latent loss, threshold success, or reward transform
   - what is inherited from BP Theorem 21 / 32
   - what is newly assumed for AutoResearch
   - where Jensen / averaging error enters
   - whether `phi/G/epsilon` are axis-shared or axis-indexed

3. If you keep any strong assumption, justify why it is acceptable as a theorem assumption rather than a hidden convenience.

4. Add a theorem-status table to the theory note that distinguishes:
   - proved algebra
   - structural assumption
   - estimation assumption
   - empirical hypothesis

## Deliverables

At minimum:

- updated `theory_validation_bp_20260412/theory/autoresearch_bp_revised.tex`
- regenerated `theory_validation_bp_20260412/theory/autoresearch_bp_revised.pdf`
- updated root copy `autoresearch_bp_revised.pdf`
- new note `theory_validation_bp_20260412/analysis/theorem_refactor_summary.md`

The summary note must say:

- what theorem form you chose
- what changed relative to the previous version
- what gaps remain even after refactor

## Completion Checklist

- [ ] Theory source updated
- [ ] PDF rebuilt successfully
- [ ] Refactor summary written
- [ ] The theorem is weaker or equal in strength to what is actually justified
- [ ] No theorem-level claim depends on broken estimators
