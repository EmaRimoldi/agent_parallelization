# Estimator Design

## Operational Mode Definition

The mode system is now explicitly observable and reproducible.

Implemented in:

- `scripts/label_modes.py`

For each snapshot step, the script logs:

- `hypothesis`
- `expected_effect`
- `hypothesis_category`
- `diff_category`
- `baseline_diff_category`
- `mode`
- `mode_source`
- `candidate_commit`
- `candidate_id`
- `accepted`
- `val_bpb_after`

Mode selection rule:

1. classify the textual hypothesis into one of:
   - `optimization`
   - `regularization`
   - `architecture`
   - `data_pipeline`
   - `memory_or_coordination`
   - `other`
2. classify the accepted-reference diff and baseline-reference diff into the same category set
3. choose the final `mode` using:
   - hypothesis category if non-`other`
   - else accepted-diff category if non-`other`
   - else baseline-diff category if non-`other`
   - else `other`

This avoids direct dependence on outcome while keeping the mode system tied to observable edit intent and changed subsystem.

## Estimator Table

| quantity | BP definition | implemented estimator | assumptions | known bias / limitation |
| --- | --- | --- | --- | --- |
| `mode` | latent strategy / mode `S` | observable category from hypothesis text plus diff evidence | hypothesis text is informative about intended search direction; diffs expose changed subsystem | still a proxy for latent modes; coarse categories may merge distinct strategies |
| `phi_alpha` | `E_s[log(T_alpha(d0,s) / T_alpha(d,s))]` | weighted log-ratio of `attempts-to-first-accepted-success` per overlapping mode, using global accepted-mode prior weights | accepted snapshot step is a proxy for within-mode certified effort | sparse accepted edits often make overlap empty; uses step index rather than theorem-level certified time |
| `G_alpha` | `I(S; D)` | per-design pointwise MI contribution `KL(pi_D || pi_global)` where `pi_D` is accepted-mode distribution and `pi_global` is pooled accepted-mode prior | equal-weight design prior; accepted modes approximate the design-conditioned posterior | this is a per-design contribution, not the full pooled MI scalar by itself; undefined when a design has no accepted-mode posterior |
| `epsilon_alpha` | `E_D[KL(pi_D || q_D)]` | `KL(pi_D || q_D_proxy)` where `pi_D` is accepted-mode distribution and `q_D_proxy` is the distribution of all proposed mode labels in that design | all proposed mode labels approximate routing allocation; accepted modes approximate posterior mass | proposal counts are only a routing proxy, not true compute allocation; sensitive to small sample size |
| `kappa_token` | per-step token cost | observed `(input_tokens + output_tokens)` when available, else calibrated `chars/4` proxy | turn logs with true token counts are representative enough to calibrate fallback proxy | older runs without observed token counts still depend on proxy calibration |
| `kappa_wall` | per-step wall-clock cost | mean and std of turn-level `wall_clock_seconds` | turn-level agent call duration is the relevant wall-cost proxy | folds agent-side overhead and planning into one scalar |

## Why This Is Structurally Better Than Before

1. `phi` is no longer hardcoded to `0.0`.
2. `G` is no longer an entropy difference against a pooled prior; it is now a KL-based pointwise mutual-information contribution.
3. `epsilon` is no longer `KL(mode_distribution || prior)`; it now compares posterior-vs-routing proxy distributions within the same design.
4. Token cost now uses observed token counts whenever available and calibrates the old `chars/4` fallback empirically.

## Remaining Gaps

1. None of these estimators proves identifiability.
2. `phi_alpha` is still the weakest estimator because the current pilot rarely has overlapping accepted modes across cells.
3. `q_D` remains a routing proxy, not the true allocation used in BP.
4. The design still needs more accepted and reevaluated candidates before `G_alpha` and `epsilon_alpha` become stable enough for theorem-level use.
