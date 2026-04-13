# Formal Theory Audit

## Scope

This audit reads [autoresearch_bp.pdf](../theory/autoresearch_bp.pdf) against [BP.pdf](../theory/BP.pdf) and classifies each central claim as:

- formally inherited from BP,
- plausible but under-justified in the AutoResearch specialization,
- or purely empirical / conjectural.

## Central Statement In The AutoResearch PDF

The central mathematical claim is **Theorem 1 (Two-axis model-aware decomposition)** in `autoresearch_bp.pdf` pages 4-5:

```text
Delta_wall(d) = log(kappa_bar_wall(d0) / kappa_bar_wall(d)) + phi(d0,d) + G(d) - epsilon(d)
Delta_cost(d) = log(kappa_bar_cost(d0) / kappa_bar_cost(d)) + phi(d0,d) + G(d) - epsilon(d)
```

with the claim that:

- the two axes share the same `phi`, `G`, and `epsilon`,
- only the cost term changes,
- and the result follows by applying BP Lemma 18 / Theorem 21 to each axis separately.

The paper then derives:

- Pareto dominance as the meaning of "better" (Section 6.1),
- main-effect / interaction contrasts on the 2x2 design (Section 6.2),
- predictive crossover conditions for memory and parallelism (Section 6.5),
- and pilot hypotheses H1-H5 (Section 7.6).

## Dependencies In The AutoResearch PDF

The AutoResearch derivation depends on:

- Definitions 1-7 in `autoresearch_bp.pdf`
- Assumption 1 (AutoResearch modes)
- BP Assumption 20 (model-dependent packed family)
- BP Theorem 21 (model-aware decomposition)
- BP Corollary 26 (crossover condition)
- BP Appendix A / Theorem 32 (continuous-reward extension)

## What Is Already Formally Supported

The following parts are genuinely inherited from BP, provided the AutoResearch setup is brought into BP's assumptions without changing the algebra:

1. **Single-axis four-term decomposition**
   BP Theorem 21 proves
   `Delta = log(kappa0 / kappa) + phi + G - epsilon`
   under a model-dependent packed family.

2. **Routing-information split**
   BP Lemma 18 and Theorem 19 justify the `G - epsilon` split via the cross-entropy identity.

3. **Crossover logic**
   BP Corollary 26 really does give the generic "cost + competence > information + routing penalty" decision rule.

4. **Continuous-reward version**
   BP Appendix A / Theorem 32 extends the same decomposition to a continuous-reward setting.

These are real formal supports, not hand-waving.

## What Remains Under-Justified Or Conjectural

The AutoResearch PDF adds several non-trivial bridges that are **not** proved in the text.

### 1. Reward-to-loss bridge is missing

BP Appendix A is written for bounded **rewards** `R(x,y) in [0, R_max]`, with higher better.
AutoResearch instead defines a noisy **loss** `V(y, omega) >= 0`, lower better, then defines:

- latent loss `L(y) = E[V(y, omega)]`,
- success at threshold `q*` as `L(y) <= q*`,
- certified times in terms of reaching that threshold.

This is not the same object as BP's continuous reward-rate theorem.

What is missing:

- either an explicit reduction from loss-threshold success to BP's binary verification formalism,
- or an explicit monotone transformation from loss to reward together with a proof that the decomposition is preserved under that transformation,
- or a rewritten theorem stated directly for thresholded latent loss.

As written, the proof line "apply BP Appendix A" is too short.

### 2. BP compares models; AutoResearch varies full architectures

BP Theorem 21 indexes decomposition terms by model `M`.
AutoResearch varies a full architecture `d = (M, pi, Omega, W, K, p)` while nominally holding base model fixed.

This can be made rigorous by treating `d` as the effective BP index, but the paper does not state that reduction explicitly.

What is missing:

- a proposition saying "replace BP's model index by the full design tuple / architecture class",
- together with the induced definitions of `kappa(d)`, `t0(d,s)`, `P_d(D | S)`, and `q_d`.

This is fixable, but currently implicit.

### 3. Context-dependent per-step cost is replaced by run-averaged `kappa_bar`

BP assumes a fixed per-step `kappa(M)`.
AutoResearch introduces `kappa(M, c/K)`, where cost depends on context fill, then moves to run averages `kappa_bar_wall(d)` and `kappa_bar_cost(d)`.

That substitution is not proved.

Why it matters:

- if `kappa` depends on path history, the log-time factorization is no longer automatically identical to BP's fixed-cost case,
- averaging can be valid as an estimator, but only under extra stationarity / concentration assumptions.

This is one of the main formal gaps.

### 4. The latent mode prior is estimated from successful edits

Assumption 1 says the latent mode prior `pi` is estimated empirically from successful edits.
In BP, the latent-mode family is part of the task structure.

This creates a possible post-treatment / selection issue:

- the prior is being estimated from outputs of the very systems being compared,
- and only from successful edits, not the full attempt distribution.

That may still be a usable estimator, but it is not the same as assuming the latent mode prior is fixed ex ante.

### 5. The incumbent re-evaluation protocol is operational, not theorem-level

The paper correctly recognizes upward selection bias under noisy verifier evaluations.
But the text only gives an operational protocol ("re-run twice more and compare means with pooled standard error"), not a theorem establishing consistency or error control for the certified-time estimator under this protocol.

So:

- the need for re-evaluation is mathematically motivated,
- the exact inferential guarantee of the proposed protocol is not proved.

### 6. Two-axis decomposition is plausible but not literally inherited "for free"

The paper claims the two axes share the same `phi`, `G`, and `epsilon` and differ only in the cost term.
This is plausible if both wall-clock and cost certified times satisfy the same multiplicative log form with the same latent structure.

But that requires assumptions not stated explicitly:

- same latent modes on both axes,
- same within-mode competence term on both axes,
- same routing posterior on both axes,
- and axis-specific pricing only through `kappa`.

This is likely the right formulation, but it should be stated as an explicit assumption or proposition.

## What Is Purely Empirical / Predictive

The following are not formal theorems in the current text. They are empirical predictions:

- H1-H5 in Section 7.6 of `autoresearch_bp.pdf`
- the claim that external memory primarily acts through context-pressure relief
- the claim that shared workspace lowers routing mismatch enough to offset coordination
- the dominance / interaction expectations for the 2x2 pilot

These rise or fall with the experimental protocol; they are not established by the algebra alone.

## Formal Verdict

The AutoResearch theory is **not yet a fully self-contained proved theorem**.

Current status:

- **Core decomposition structure:** genuinely supported by BP.
- **AutoResearch specialization:** plausible but not fully proved.
- **Most important missing bridge:** BP continuous reward vs AutoResearch thresholded latent loss.
- **Second most important missing bridge:** fixed per-step `kappa` in BP vs context-dependent path-average `kappa_bar` in AutoResearch.

The honest classification is:

> The current PDF contains a mathematically credible specialization of BP, but not yet a fully rigorous proof of the AutoResearch theorem as stated.

That means the likely end state is not "the theory is nonsense"; it is "the theory needs reformulation / tightening before it can be called proved."
