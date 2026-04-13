# Final Verdict

## Theorem Status

The strongest current classification is:

> **promising but not yet rigorous**.

That is stronger than the original draft, because the theorem has now been narrowed and the estimator layer has been repaired structurally. It is still short of rigorous validation, because the current substrate does not yet identify the full decomposition in a stable way.

## Executive Conclusion

After the follow-up work in this queue, the state of the project is now:

1. the theorem statement is materially better than before;
2. the protocol is materially better than before;
3. the estimator code is materially better than before;
4. the empirical identification problem remains.

So the right conclusion is no longer “the code forces a fake zero decomposition,” but it is still not “the theorem is validated.”

The narrowest honest statement is:

> AutoResearch still looks compatible with a BP-style decomposition under explicit assumptions, and the single-axis theorem with a Jensen remainder is now the right formal object, but the present CPU pilot still does not identify the full `phi + G - epsilon` structure well enough to count as a validated theorem-package.

## What Changed In This Iteration

### 1. The theorem was narrowed

See:

- `theory_validation_bp_20260412/theory/autoresearch_bp_revised.pdf`
- `theory_validation_bp_20260412/analysis/theorem_refactor_summary.md`

The base theorem is now:

- single-axis rather than shared two-axis by default
- written on latent-loss threshold success
- explicit about the Jensen remainder for `kappa_bar`
- separate from the estimator layer

This is the strongest defensible theorem form on the current evidence.

### 2. The protocol is no longer missing the critical reevaluation path

See:

- `theory_validation_bp_20260412/analysis/protocol_upgrade_spec.md`
- `theory_validation_bp_20260412/analysis/protocol_upgrade_smoke_check.md`

The framework now logs:

- stable `candidate_id`
- reevaluation events
- promotion-after-reevaluation vs provisional wins
- turn-level token/wall-cost variation
- routing-evidence observables

This closes a real implementation gap.

### 3. The estimator layer no longer collapses because of broken code

See:

- `theory_validation_bp_20260412/analysis/estimator_design.md`
- `theory_validation_bp_20260412/analysis/estimator_validation_note.md`
- `theory_validation_bp_20260412/analysis/corrected_decomposition_rep1.json`
- `theory_validation_bp_20260412/analysis/corrected_decomposition_rep2.json`
- `theory_validation_bp_20260412/analysis/corrected_decomposition_rep3.json`

Important update:

- `phi` is no longer hardcoded to zero
- `G` is no longer the old entropy-difference placeholder
- `epsilon` is no longer KL against the pooled prior
- token cost now uses observed tokens when present and calibrates the fallback proxy

The decomposition does **not** reduce to cost-only for implementation reasons anymore.

## What The New Evidence Shows

### A. Repeated means still do not separate the main cells cleanly

See:

- `theory_validation_bp_20260412/analysis/experiment_01_replicated_means.md`
- `theory_validation_bp_20260412/experiments/followup_01/replicated_means_summary.json`

Repeated incumbent evaluations (5 per cell total) give:

- `d10`: mean `0.8412`, 95% CI `[0.8096, 0.8729]`
- `d11`: mean `0.8737`, 95% CI `[0.8280, 0.9194]`
- `d00`: mean `0.8739`, 95% CI `[0.8512, 0.8967]`
- `d01`: mean `0.9008`, 95% CI `[0.8313, 0.9703]`

Interpretation:

- `d10` still looks best on mean
- `d01` still looks weakest
- the intervals overlap too much to support a sharp architecture claim among `d10`, `d11`, and `d00`

So the ranking signal is still weak.

### B. Jensen-gap risk is real, especially on wall-clock

See:

- `theory_validation_bp_20260412/analysis/experiment_02_cost_variance.md`
- `theory_validation_bp_20260412/experiments/followup_01/cost_variance_summary.json`

Empirical Jensen gaps:

- token axis: about `0.022` to `0.046`
- wall-clock axis: about `0.209` to `0.269`

This is strong evidence that the remainder term is not cosmetic on the wall-clock axis.

### C. H5 is still under-instrumented

See:

- `theory_validation_bp_20260412/analysis/experiment_03_context_sweep.md`
- `theory_validation_bp_20260412/analysis/context_pressure_metrics.json`
- `theory_validation_bp_20260412/experiments/followup_01/context_sweep_feasibility.json`

Observed max context fill remains around `0.21` to `0.24`.
Reaching even `50%` fill would require roughly `2.1x` to `2.4x` more effective pressure.

So H5 remains untested by intervention.

### D. Corrected decomposition is still mostly data-limited

Across the corrected decomposition reruns:

- `phi` is generally `NaN` because accepted-mode overlap across cells is missing
- `G` is sometimes `NaN` because some cells still have no accepted-mode posterior
- `epsilon` is non-trivial in at least one case (`d01` in reps 1 and 2: about `0.6931`)

Interpretation:

- the estimator stack is now capable of producing non-zero non-cost terms
- the current pilot data still do not support stable full-term estimation

So the decomposition is no longer broken by code design, but it is still underidentified in practice.

## What Is Now Supported

The current bundle supports these claims:

1. The single-axis theorem with explicit assumptions and Jensen remainder is the right formal baseline.
2. Re-evaluation is mandatory in this substrate.
3. Averaged wall-clock `kappa_bar` carries a non-trivial Jensen risk.
4. The corrected estimator stack no longer forces `phi = G = epsilon = 0`.
5. The present substrate still lacks enough stable accepted-mode evidence for full decomposition identification.

## What Is Still Not Supported

The current bundle still does **not** support:

1. claiming the full theorem is rigorously proved in this implementation setting;
2. claiming the pilot validated the four-term decomposition empirically;
3. claiming sharp architecture dominance among the best cells;
4. claiming H5 has been tested properly.

## Decision

The right decision is:

> keep the theorem in its narrowed single-axis form, keep the corrected protocol and estimators, and treat the current empirical package as promising but underidentified rather than validated or refuted.

## Next Step

If the project continues, the next highest-value move is:

1. add a real control surface for context pressure,
2. produce more accepted and reevaluated candidates per cell,
3. and only then rerun a smaller corrected 2x2 study.

Without those changes, further claims will mostly recycle the same substrate-noise limitation.
