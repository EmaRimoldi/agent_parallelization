# Pass 02 — Theory Validation: Experiment Summary

**Status**: Superseded (by Pass 03)
**Period**: April 2026 (second AI pass)
**Objective**: Tighten the BP four-term decomposition theorem, fix broken estimators, and test whether the corrected framework can identify non-trivial decomposition terms on the existing CIFAR-10 substrate.

---

## Research Question

Pass 01 built the experimental infrastructure and ran a first pilot, but found that the decomposition collapsed to a single term (cost only): phi, G, and epsilon were all zero or NaN. Was this because the *theory* is wrong, or because the *implementation* was broken?

Pass 02 is a theory-validation pass. It asks three specific questions:

1. **Is the theorem statement defensible?** The original two-axis shared decomposition had hidden assumptions. Can we narrow it to something rigorous?
2. **Are the estimators structurally sound?** phi was hardcoded to zero, G used an entropy-difference placeholder, epsilon compared against the wrong distribution. Can we fix them?
3. **With corrected estimators, does the pilot data now produce non-trivial decomposition terms?**

**Background from Pass 01**: The BP four-term decomposition is:

```
Delta = log(kappa_0 / kappa) + phi + G - epsilon
```

Pass 01 measured log(kappa_0/kappa) but found phi = G = epsilon = 0 across all cells. Two possible explanations: (a) the framework genuinely doesn't apply (negative result), or (b) the estimators were broken and the pilot was too underpowered to produce signal. Pass 02 investigates (b).

## What Pass 02 Changed

### 1. Theorem refactoring

The theorem was narrowed from a two-axis shared model to a **single-axis decomposition with an explicit Jensen remainder**:

- **Before**: The decomposition was assumed to hold simultaneously on both token and wall-clock axes, with phi, G, epsilon shared across axes. The kappa_bar substitution (replacing per-step cost with an average) was treated as free.
- **After**: The decomposition holds on one axis at a time. Each claim is tagged as: *inherited from BP* (proved algebra), *structural assumption* (architecture-indexed packed family), *estimation assumption* (proxy validity), or *empirical hypothesis* (untested). The Jensen remainder R_alpha is now explicit — using kappa_bar instead of per-step kappa incurs a measurable error term.

This is the strongest statement that stays within what is actually justified. The two-axis shared version is downgraded to a conditional corollary for future work.

### 2. Protocol upgrades

The experimental framework now logs four new observables that Pass 01 lacked:

| New feature | What it enables |
|-------------|-----------------|
| **Stable candidate_id** | Track the same code snapshot across re-evaluations (different training runs of the same train.py). Before: identity was lost after each run. |
| **Re-evaluation events** | Distinguish provisional wins (first training of a new snapshot) from confirmed wins (same snapshot re-evaluated and still good). Before: all evaluations were treated as independent. |
| **Turn-level cost variance** | Record per-turn token count and wall-clock time separately. Before: only aggregate totals were available, hiding within-run variance. |
| **Routing-evidence fields** | Log the agent's stated hypothesis, expected_effect, and strategy_category per turn. Before: mode labels were inferred post-hoc from diffs only. |

These changes close real implementation gaps: re-evaluation is necessary for any threshold-based success definition, cost variance is needed for the Jensen remainder, and routing evidence improves mode labeling.

### 3. Estimator redesign

| Term | Before (broken) | After (corrected) |
|------|------------------|--------------------|
| **phi** | Hardcoded to 0.0 | Mode-weighted log-ratio of attempts-to-first-accepted-success per overlapping mode. Returns NaN when no modes overlap (honest about missing data). |
| **G** | Entropy-difference placeholder against pooled prior | KL(pi_D \|\| pi_global): pointwise mutual-information contribution comparing the design cell's accepted-mode distribution to the global pooled distribution. |
| **epsilon** | KL(mode_dist \|\| prior) — wrong comparison | KL(pi_D \|\| q_D): compares accepted-mode distribution to proposed-mode distribution within the same cell. This is what BP actually requires: routing mismatch is about the gap between what the agent *tried* and what *worked*. |
| **kappa_token** | Always chars/4 proxy | Observed API token counts when available; calibrated chars/4 fallback with empirical correction factor (~0.9998). |

The key structural improvement: the estimators can now produce non-zero values for phi, G, and epsilon. They no longer collapse by design.

## Experimental Design

Pass 02 ran three targeted follow-up experiments on the existing pilot data, plus one new data collection:

### Experiment 1: Replicated incumbent re-evaluation

**Purpose**: Test whether the single-shot val_bpb values from the pilot are reliable. In the pilot, each cell's "best" result was the minimum val_bpb from a single training run. But train.py with SEED=42 is not perfectly deterministic across runs (batch ordering, dropout sampling), so the same code may produce different val_bpb values.

**Protocol**: For each of the 4 cells x 3 pilot replicates, select the best candidate (the train.py snapshot that achieved the best pilot val_bpb). Re-evaluate it 2 times for rep1 and rep2, 1 time for rep3 (total: 5 evaluations per cell). Report mean and 95% CI.

### Experiment 2: Within-architecture cost variance (Jensen gap)

**Purpose**: Quantify the error from using kappa_bar (average cost per turn) instead of the per-step kappa that BP requires. The theorem now has an explicit Jensen remainder R_alpha. Is this remainder negligible or material?

**Protocol**: Load all turn-level logs from the 3 pilot replicates for each cell. Compute the empirical Jensen gap: log(E[kappa]) - E[log(kappa)]. Also compute the delta-method approximation: 0.5 * Var(kappa) / E[kappa]^2. Report both on token and wall-clock axes.

### Experiment 3: Context pressure feasibility (H5)

**Purpose**: Assess whether context pressure — the hypothesis that kappa increases as the agent's context window fills up — can be tested interventionally on this substrate.

**Protocol**: For each cell, find the maximum observed context fill ratio (context_tokens / context_window). Compute how much the agent's context would need to grow to reach 50% and 75% fill.

### Noise assay

**Purpose**: Measure the verifier noise floor. How much does val_bpb vary when the same train.py is re-run?

**Protocol**: Run the unmodified baseline train.py 5 times. Run the best d10 candidate train.py 5 times. Report mean, std, and range for each.

---

## Key Results

### 1. Replicated means: ranking signal is weak

Re-evaluating each cell's best candidate 5 times:

| Cell | Mode | Mean val_bpb | Std | 95% CI | N |
|------|------|-------------|-----|--------|---|
| d10 | Single / Memory | **0.8412** | 0.0361 | [0.810, 0.873] | 5 |
| d00 | Single / No Memory | 0.8739 | 0.0260 | [0.851, 0.897] | 5 |
| d11 | Parallel / Memory | 0.8737 | 0.0521 | [0.828, 0.919] | 5 |
| d01 | Parallel / No Mem | 0.9008 | 0.0793 | [0.831, 0.970] | 5 |

Per-replicate breakdown (mean of re-evaluations per pilot rep):

| Cell | Rep 1 mean | Rep 2 mean | Rep 3 mean | Pilot best (single-shot) |
|------|-----------|-----------|-----------|--------------------------|
| d00 | 0.859 | 0.883 | 0.886 | 0.871 / 0.800 / 0.799 |
| d10 | 0.865 | 0.817 | 0.842 | 0.829 / 0.761 / 0.755 |
| d01 | 0.994 | 0.853 | 0.810 | 1.010 / 0.802 / 0.837 |
| d11 | 0.885 | 0.905 | 0.787 | 0.801 / 0.804 / 0.796 |

![Replicated means](figures/fig01_replicated_means.png)

**Figure 1 interpretation**: d10 (memory) has the lowest mean, but the confidence intervals of all four cells overlap substantially. d01 shows extreme variance (rep1 mean = 0.994 vs rep3 = 0.810), driven by the unlucky d01/rep1 candidate that scored 1.010 in the pilot. The critical observation is the gap between single-shot pilot values and re-evaluated means: d10/rep2 was 0.761 in the pilot but 0.817 on re-evaluation (+0.056), d10/rep3 was 0.755 but 0.842 (+0.087). Single-shot selection systematically overestimates quality because it selects *lucky* training runs. This regression-to-the-mean effect is a fundamental problem for the decomposition: the "best val_bpb" from the pilot was partly signal and partly noise.

### 2. Jensen gap: kappa_bar is NOT a free substitution

| | Token axis | | Wall-clock axis | |
|---|---|---|---|---|
| **Cell** | **Empirical gap** | **Delta-method** | **Empirical gap** | **Delta-method** |
| d00 | 0.022 | 0.028 | 0.216 | 0.166 |
| d10 | 0.046 | 0.074 | 0.209 | 0.180 |
| d01 | 0.035 | 0.048 | 0.269 | 0.209 |
| d11 | 0.040 | 0.069 | 0.256 | 0.168 |

![Jensen gap](figures/fig02_jensen_gap.png)

**Figure 2 interpretation**: The wall-clock Jensen gap (~0.21-0.27) is roughly 10x larger than the token-axis gap (~0.02-0.05). This means that using the average wall-clock cost per turn instead of per-step costs introduces a systematic error that is comparable to, or larger than, the architecture-level contrasts we are trying to measure. The token axis is more benign but still non-negligible for d10 (0.046). This directly validates the theorem refactor: the Jensen remainder R_alpha cannot be dropped from the decomposition. It also suggests that wall-clock-axis claims will require either (a) the explicit remainder in the decomposition, or (b) per-step rather than averaged cost data.

### 3. Corrected decomposition: still mostly NaN

With the redesigned estimators applied to the same pilot data:

| Cell | Rep | cost_token | cost_wall | phi | G | epsilon |
|------|-----|-----------|----------|-----|---|---------|
| d10 | 1 | -0.153 | +0.111 | NaN | NaN | NaN |
| d10 | 2 | -0.165 | -0.477 | NaN | NaN | NaN |
| d10 | 3 | +0.000 | +0.124 | NaN | NaN | NaN |
| d01 | 1 | -0.231 | +0.007 | NaN | 0.0 | 0.693 |
| d01 | 2 | -0.011 | -0.043 | NaN | 0.0 | 0.693 |
| d01 | 3 | +0.018 | -0.060 | NaN | NaN | NaN |
| d11 | 1 | -0.185 | +0.029 | NaN | NaN | NaN |
| d11 | 2 | -0.090 | -0.444 | NaN | NaN | NaN |
| d11 | 3 | +0.007 | -0.093 | NaN | NaN | NaN |

![Corrected decomposition](figures/fig03_corrected_decomposition.png)

**Figure 3 interpretation**: Only the cost term is computable, and even it is unstable — d10 and d01 flip sign across reps on both axes. phi is NaN everywhere because no cell has overlapping accepted modes with the baseline (d00 had zero accepted edits). G is non-NaN only for d01 reps 1-2 (where the single accepted mode matched the global prior exactly, yielding G=0). epsilon = 0.693 (= ln(2)) appears in d01 reps 1-2 because the accepted distribution is concentrated on one mode while the proposed distribution has two modes — this is the only non-trivial non-cost term in the entire decomposition, and it is a degenerate case.

**Bottom line**: The corrected estimators are *structurally capable* of producing non-zero phi, G, epsilon (unlike Pass 01 where they were hardcoded to zero), but the pilot data is still too sparse to compute them. The data-limitation is: too few accepted edits, too little mode diversity, and zero overlap between cells.

### 4. Noise assay: verifier noise is substantial

| Condition | Mean val_bpb | Std | Range | N |
|-----------|-------------|-----|-------|---|
| Baseline (unmodified train.py) | 0.825 | 0.036 | 0.093 | 5 |
| Best d10 candidate (agent-modified) | 0.869 | 0.050 | 0.120 | 5 |

![Noise assay](figures/fig04_noise_assay.png)

**Figure 4 interpretation**: The same train.py produces val_bpb ranging from 0.781 to 0.874 across 5 runs (baseline range = 0.093). The best d10 candidate is even noisier (range = 0.120), and its mean (0.869) is *worse* than the baseline mean (0.825), despite having been selected as the "best" candidate from the pilot. This means the d10 pilot champion (val_bpb = 0.755) was likely a lucky outlier rather than a genuinely superior configuration. The noise floor (~0.04-0.05 std) is comparable to the inter-cell differences we are trying to measure (~0.03-0.05), making single-shot comparisons unreliable.

**Why val_bpb varies for the same code**: Although train.py sets SEED=42, PyTorch's DataLoader shuffle, dropout, and batch ordering introduce non-determinism. The agent's code edits determine the configuration (hyperparameters, architecture), but the stochastic training process means that the same configuration can produce meaningfully different val_bpb values. This is the "noisy verifier" problem that the revised theorem addresses via threshold-based success rather than point comparisons.

### 5. Context pressure: H5 is untestable on this substrate

| Cell | Max context fill | Multiplier to 50% | Multiplier to 75% |
|------|-----------------|-------------------|-------------------|
| d00 | 22.8% | 2.19x | 3.28x |
| d10 | 21.7% | 2.30x | 3.45x |
| d01 | 24.3% | 2.06x | 3.09x |
| d11 | 21.2% | 2.36x | 3.55x |

![Context pressure](figures/fig05_context_pressure.png)

**Figure 5 interpretation**: No cell exceeds 25% context fill. Testing H5 (kappa increases monotonically with context fill) would require agents to operate at 50%+ fill, which needs 2-2.4x more effective context pressure. The context bins used by the decomposition (0-25%, 25-50%, 50-75%, 75-100%) have data only in the first bin — all higher bins are null. This means the kappa-vs-context-fill relationship is unobserved. H5 cannot be tested observationally on the current substrate; it would require an intervention (longer runs, bigger substrates, or artificially injected context) to push agents into higher fill regimes.

---

## Hypothesis Verdicts (corrected estimators)

| Hypothesis | Rep 1 | Rep 2 | Rep 3 | Verdict |
|-----------|-------|-------|-------|---------|
| H1: parallelism helps wall-clock only | No | No | No | **0/3 — no support** |
| H2: memory helps both axes | No | No | Yes | **1/3 — inconsistent** |
| H3: shared memory lowers epsilon | No | No | No | **0/3 — no support** (epsilon undefined in d11) |
| H4: epsilon exceeds log(2) in parallel | No | No | No | **0/3 — no support** |
| H5: context pressure dominant | — | — | — | **Untestable** (observational only) |
| H6: d11 dominates d00 on both axes | No | No | No | **0/3 — no support** |

Hypothesis test results are essentially unchanged from Pass 01. The corrected estimators did not improve the empirical picture because the underlying data is still too sparse.

## Conclusions

### What Pass 02 achieved

1. **Theorem is tighter and more honest**: The single-axis form with explicit Jensen remainder is a genuine improvement. Every claim is now tagged with its epistemic status (proved, assumed, or empirical). The two-axis shared version is correctly downgraded.

2. **Protocol captures re-evaluation and cost variance**: The framework now supports the observables needed for the revised theorem (candidate identity, re-evaluation events, turn-level costs). This is infrastructure that future passes can build on.

3. **Estimators are no longer structurally broken**: phi, G, epsilon can now produce non-zero values. The old code *forced* a cost-only decomposition; the new code honestly reports NaN when data is insufficient, which is the correct behavior.

4. **Jensen remainder is empirically material**: The wall-clock Jensen gap (~0.21-0.27) is comparable to the signal we're measuring. This validates the theorem refactor and establishes that kappa_bar cannot be used as a drop-in for kappa.

5. **Noise floor quantified**: Single-shot val_bpb has std ~0.04-0.05, comparable to inter-cell differences. Any future experiment must use repeated evaluation and threshold-based success criteria.

### What Pass 02 did NOT achieve

1. **The decomposition is still not identified**: phi, G, epsilon remain NaN in 7/9 cell-rep combinations. The corrected estimators prove the code isn't the bottleneck — the data is.

2. **No cell separation**: Re-evaluated means show d10 < d00 < d11 < d01, but all confidence intervals overlap. No architecture claim is statistically supported.

3. **H5 is still untested**: Context pressure requires an intervention, not just observation. The current substrate doesn't push agents hard enough.

4. **The verdict didn't upgrade**: Despite all improvements, the project status remains "promising but not yet rigorous."

### Why "promising but not yet rigorous"

The narrowest honest statement:

> AutoResearch still looks compatible with a BP-style decomposition under explicit assumptions, and the single-axis theorem with a Jensen remainder is now the right formal object, but the present CPU pilot still does not identify the full phi + G - epsilon structure well enough to count as a validated theorem-package.

### Three blockers for "rigorous"

1. Accepted-mode overlap too sparse for stable phi estimation
2. Some cells have no accepted posterior, so G and epsilon are undefined
3. Context pressure not experimentally controllable (H5 needs intervention)

## Implications for Later Passes

- **Pass 03** introduced phased execution (execution, monitoring, analysis), probe-based experiments, and configuration routing to address the data sparsity problem.
- **Pass 04** redesigned the 2x2 with confound controls (fixed seeds, CPU pinning, task headroom) informed by Pass 01-02 findings.

The key insight from Pass 02: the bottleneck is **data quality** (too few accepted edits, too little mode diversity, too noisy a verifier), not theory or code. Future passes must generate more diverse, successful agent edits before the decomposition can be meaningfully tested.
