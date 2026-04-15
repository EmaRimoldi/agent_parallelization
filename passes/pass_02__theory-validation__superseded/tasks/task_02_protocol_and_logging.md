# Task 02: Protocol And Logging Upgrade

## Goal

Repair the experimental protocol so that the quantities required by the theorem can, in principle, be estimated from future runs.

## Inputs

- `archives/pass_02_theory_validation_bundle_20260412/analysis/protocol_compliance_audit.md`
- `archives/pass_02_theory_validation_bundle_20260412/analysis/final_verdict.md`
- current implementation under `src/`, `scripts/`, and `templates/`

## Required Work

Implement the smallest logging/protocol changes that unblock rigorous estimation.

### A. Re-evaluation protocol

Add a real incumbent re-evaluation path:

- when a candidate appears to beat the incumbent, repeat evaluation
- keep repeated evaluations tied to the same commit / candidate id
- store all repeats in structured logs
- record whether promotion happened after re-evaluation or only after a single noisy win

### B. Cost variance logging

Add enough data to estimate not only average per-step cost but also variation:

- per-turn wall-clock
- per-turn token counts or the best available proxy
- per-run / per-agent aggregate cost variance summaries if useful

### C. Routing evidence

Add logging that can later support `q_D` estimation, for example:

- which mode / strategy the agent intended to pursue
- hypothesis category before edit
- whether a shared-memory suggestion or prior result influenced the step

Do not fake posterior quantities. Log observables from which they can later be approximated.

### D. Evaluation provenance

Ensure every training run and every repeated evaluation can be linked to:

- experiment id
- agent id
- turn
- candidate / commit
- baseline or reevaluation flag

## Deliverables

- code changes implementing the protocol upgrades
- a new protocol note:
  `archives/pass_02_theory_validation_bundle_20260412/analysis/protocol_upgrade_spec.md`

The note must list:

- what was added
- what theorem/estimator requirement it satisfies
- what remains unobservable even after the upgrade

## Validation

Run a small smoke test that produces logs with the new fields and verify the fields are actually populated.

Write:

- `archives/pass_02_theory_validation_bundle_20260412/analysis/protocol_upgrade_smoke_check.md`

## Completion Checklist

- [ ] Re-evaluation path exists in code
- [ ] Cost variance is recoverable from logs
- [ ] Candidate identity is stable across repeated evaluations
- [ ] Smoke check confirms new fields are populated
- [ ] Upgrade spec note written
