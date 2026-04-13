# Theorem Refactor Summary

## Chosen Theorem Form

I replaced the previous conditional two-axis theorem with a **conditional single-axis theorem with an explicit Jensen remainder**.

For a fixed axis `alpha`, the theorem now states:

- the BP four-term identity is inherited on that axis once AutoResearch is treated as an architecture-indexed packed family,
- theorem-level success is defined by a latent-loss threshold rather than by noisy single-shot evaluation,
- arithmetic mean cost enters only after paying an explicit remainder `R_alpha` induced by replacing the exact effective BP cost with a reported run-average cost.

This is the strongest current statement that stays inside what is actually justified.

## What Changed Relative To The Previous Version

1. The theorem is now **single-axis by default**.
   The earlier revised note still assumed that `phi`, `G`, and `epsilon` were shared across wall-clock and cost axes. That assumption has been removed from the base theorem and downgraded to a conditional corollary for future work.

2. The theorem-level object is now **threshold success on latent loss**.
   This is cleaner than relying on one-shot noisy loss values and avoids quietly treating the noisy evaluator as if it were already the verifier used by the theorem.

3. The `kappa_bar` substitution is now **explicitly non-free**.
   The updated theorem introduces a Jensen remainder `R_alpha(d | d0) = J_alpha(d) - J_alpha(d0)` so the paper no longer hides the averaging error inside the main identity.

4. The note now includes a **claim-status table**.
   It distinguishes proved algebra, structural assumptions, estimation assumptions, and empirical hypotheses.

5. The theorem no longer depends on the current estimator implementation.
   The note explicitly separates the theorem from the current broken estimation layer.

## Remaining Gaps Even After Refactor

1. The theorem still depends on an explicit **architecture-indexed packed-family assumption**. That bridge is now honest, but it is still assumed rather than proved.
2. The latent-loss threshold formulation still needs a stronger operational link to the actual repeated-evaluation protocol used in experiments.
3. The Jensen remainder is now visible, but it is not yet measured from logs. The protocol still needs within-architecture cost-variance estimates.
4. The current repository still lacks correct estimators for `phi_alpha`, `G_alpha`, and `epsilon_alpha`.
5. The stronger shared-axis statement remains unverified and should not be used unless later evidence supports it.
