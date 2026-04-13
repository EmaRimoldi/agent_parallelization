# Task 00 Orientation Note

## Theorem Status

The current AutoResearch BP result is best treated as a **conditional specialization of BP**, not a completed theorem. The algebraic four-term identity is inherited from BP once a packed-family representation is granted, but the AutoResearch bridge still depends on extra assumptions that are not yet fully justified or empirically checked. The present pilot therefore supports a narrower statement: cost-channel contrasts are measurable and verifier noise is operationally important, but the full `phi + G - epsilon` decomposition is not yet identified.

## Top 5 Blockers

1. The reward-to-loss bridge is still incomplete: BP's continuous-reward machinery has not been fully matched to thresholded latent loss under noisy evaluation.
2. The `kappa_bar` substitution is not yet rigorous: BP uses fixed per-step `kappa`, while the current theory uses run-averaged context-dependent cost without an explicit Jensen remainder or bound.
3. The empirical mode structure is underdefined and undermeasured: mode labels are incomplete, which collapses `phi`, `G`, and `epsilon`.
4. The current estimators are not BP-valid: `phi` is hardcoded to zero, and the implemented `G` and `epsilon` formulas do not match the BP definitions.
5. The pilot protocol is not strong enough for architecture ranking: single best-of-N evaluations are selection-biased and the noise assay shows repeated evaluation is necessary.

## Top 3 Next Actions

1. Narrow the theorem to the strongest defensible form, likely single-axis first, with explicit assumptions and an explicit Jensen-style remainder or bounded approximation term.
2. Repair the protocol and instrumentation so the logged objects match the theorem-level objects, especially repeated evaluation, per-step cost variance, routing/allocation traces, and mode labels.
3. Replace the current decomposition estimators with definitions that actually target BP quantities, then run targeted replicated experiments instead of another broad pilot.

## Success Condition For The Queue

The queue succeeds only if the repository ends with a **narrower but genuinely defensible theorem-package**: a revised PDF whose assumptions are explicit, a validation bundle updated to reflect the corrected theory, estimators that are either mathematically correct or clearly flagged as partial, and targeted experiments that can distinguish theory failure from protocol failure. If the decomposition remains empirically non-identifiable after these repairs, success means stating that clearly and reducing the claim to the smaller result that the evidence actually supports.
