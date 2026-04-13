# Phase 02b: Decision Gate — Proceed, Extend, or Escalate

## Goal

Evaluate the calibration results against pre-specified criteria and decide which path to take.

## Background

This is the critical go/no-go decision point. The calibration data tells us whether the CIFAR-10 substrate and current agent setup produce enough signal for the BP decomposition to be testable.

## Decision Criteria

Read the measurements from state and evaluate:

### Criterion 1: Architecture Effect

| Condition | Decision |
|-----------|----------|
| Cohen's d > 0.5 (medium effect) | Strong signal — proceed |
| 0.3 < Cohen's d ≤ 0.5 (small-medium) | Adequate signal — proceed |
| 0.1 < Cohen's d ≤ 0.3 (small) | Weak signal — extend budget or escalate |
| Cohen's d ≤ 0.1 (negligible) | No signal — escalate to CIFAR-100 |

### Criterion 2: Mode Diversity

| Condition | Decision |
|-----------|----------|
| Both cells have ≥ 3 distinct modes with ≥ 2 edits each | Good diversity — proceed |
| Both cells have ≥ 2 modes but < 3 | Marginal — extend budget |
| One or both cells have ≤ 1 mode | Degenerate — escalate or structured search |

### Criterion 3: Sample Size

| Condition | Decision |
|-----------|----------|
| ≥ 50 runs per cell | Adequate |
| 30–50 runs per cell | Marginal — extend budget |
| < 30 runs per cell | Insufficient — extend budget |

### Combined Decision Matrix

| Effect (d) | Modes | Runs | Decision |
|------------|-------|------|----------|
| d > 0.3 | ≥ 3 | ≥ 50 | **proceed** to full 2x2 |
| d > 0.3 | < 3 | any | **extend_budget** (60 min, re-run calibration) |
| d > 0.3 | any | < 30 | **extend_budget** |
| d ≤ 0.3 | any | ≥ 50 | **escalate_cifar100** |
| d ≤ 0.1 | ≤ 1 | any | **structured_search** |

## Tasks

### 1. Read the measurements

```bash
python workflow/run.py status
```

The relevant measurements are:
- `cohens_d`
- `d00_mode_count`, `d10_mode_count`
- `d00_n`, `d10_n`

### 2. Evaluate against criteria

Apply the decision matrix above.

### 3. Record the decision

```bash
python workflow/run.py decide <proceed|extend_budget|escalate_cifar100|structured_search>
```

### 4. Document the rationale

Write a brief rationale to `workflow/artifacts/decision_gate_rationale.md`:
- Which criteria were met/failed
- The specific numbers that drove the decision
- Any qualifications or concerns

## Required Inputs

- Measurements from Phase 02a in the workflow state
- `workflow/artifacts/calibration_analysis.json`
- `workflow/artifacts/calibration_summary.md`

## Expected Outputs

- A decision recorded in the workflow state
- `workflow/artifacts/decision_gate_rationale.md`

## Decision Branches

- **proceed**: → `03_full_2x2_run`
- **extend_budget**: → `02c_extended_calibration` (re-runs Phase 02 with longer budget)
- **escalate_cifar100**: → `04_escalation_cifar100` (switches dataset)
- **structured_search**: → `05_structured_search` (changes agent interface)

## Failure Modes

- Measurements missing: return to Phase 02a
- Ambiguous case (borderline criteria): prefer extending budget over escalating — it's cheaper
