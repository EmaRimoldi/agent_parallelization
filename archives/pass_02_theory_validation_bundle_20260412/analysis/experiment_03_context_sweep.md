# Experiment 03: Context-Pressure Sweep

## Claim Tested

Whether the current framework can enter a regime where H5 is actually testable, i.e. where `c/K` reaches substantially higher bins than the original pilot.

## Exact Procedure

Artifacts:

- `archives/pass_02_theory_validation_bundle_20260412/analysis/context_pressure_metrics.json`
- `archives/pass_02_theory_validation_bundle_20260412/experiments/followup_01/context_sweep_feasibility.json`

Procedure performed in this iteration:

1. Use the existing pilot context-pressure audit as the empirical baseline.
2. Compute the multiplicative increase in effective context pressure needed to reach:
   - `50%` fill
   - `75%` fill
3. Assess whether the current repository exposes a controllable mechanism to induce that increase under real agent turns.

Derived feasibility numbers:

- `d00`: current max `0.2283`; need about `2.19x` to reach `50%`, `3.29x` to reach `75%`
- `d10`: current max `0.2173`; need about `2.30x` to reach `50%`, `3.45x` to reach `75%`
- `d01`: current max `0.2426`; need about `2.06x` to reach `50%`, `3.09x` to reach `75%`
- `d11`: current max `0.2115`; need about `2.36x` to reach `50%`, `3.55x` to reach `75%`

## Results

I am declaring a **full theorem-relevant context sweep infeasible in this iteration**.

Reason:

1. H5 is about live agent-turn context pressure, not just offline token accounting.
2. The current framework does not expose a clean experiment knob for:
   - shrinking the actual Claude context budget `K`,
   - padding live prompts in a controlled way across otherwise identical runs,
   - or forcing retention policy changes while holding the rest of the system fixed.
3. Running a synthetic filler experiment without real agent turns would test token arithmetic, not the actual theorem-relevant mechanism.

## Interpretation

This is not evidence against H5.

It is evidence that:

- the current instrumentation can measure low-regime context pressure,
- but the framework still lacks a controlled experimental interface for pushing the agent into high-pressure regimes.

## Does Failure Point To Theory Or Experiment?

This points to **experiment design / instrumentation**, not to the theorem itself.

The theorem currently says context pressure may matter through the cost term. The problem here is not a contradiction. The problem is that the current implementation still does not let us excite that mechanism cleanly enough to test it.

## Next Required Change If H5 Becomes Priority

To make H5 testable in a later iteration, the framework needs one explicit control surface such as:

1. a configurable prompt-padding parameter,
2. a configurable effective context ceiling,
3. or a configurable history-retention policy in the runner.

Without one of these, context-pressure testing will remain observational rather than interventional.
