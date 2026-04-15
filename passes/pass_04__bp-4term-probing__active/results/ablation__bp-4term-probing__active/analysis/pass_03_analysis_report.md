# Phase 03 Analysis Report: BP 2×2 Design Audit

## Analysis Question

Does the BP 2×2 experimental design provide a valid test of the Beneventano-Poggio four-term decomposition? Specifically: are the observed performance differences between cells (d00 > d10 > d01 > d11) attributable to architecture properties, or are they confounded by experimental design choices?

## Key Findings

### Finding 1: CPU Contention Confound (Confound 1)

**Strength: STRONG (p < 0.001)**

Training time is significantly inflated for d11 (Kruskal-Wallis H=56.3, p < 0.001). d11 median training time is 228s — 3.2x higher than d10 (70s) and 2.2x higher than d00 (103s). Mann-Whitney U confirms d11 is significantly slower than all other cells (all p < 0.001). Importantly, d01 (parallel, no sharing) does NOT show significant contention vs d00 (p=0.45), implicating the shared memory mechanism specifically.

### Finding 2: Agent Homogeneity (Confound 2)

**Strength: STRONG (descriptive, no inferential test possible)**

In d01, both agents independently discover the same strategy categories with Jaccard similarity 0.75-1.00 across all 3 reps. Keyword overlap is 50-60%. Strategy entropy is near-identical across all cells (1.87-2.01), confirming that parallelism provides no diversification.

### Finding 3: Memory Anchoring (Confound 3)

**Strength: MODERATE**

d10 shows 75% longer max strategy streaks (mean 4.2 vs 2.4 for d00). No significant early-vs-late strategy shift in any cell (all chi-squared p > 0.18). Memory depth shows no correlation with performance (Spearman r=-0.23, p=0.066). Evidence is consistent with anchoring hypothesis but does not definitively rule out routing.

### Finding 4: Task Ceiling (Confound 4)

**Strength: STRONG**

Only 12.2% of all 237 runs beat baseline. Regularization has 0% win rate across 50 attempts. Improvement magnitude degrades monotonically: d00 (delta=0.049) > d10 (0.016) > d01 (0.007) > d11 (0). The narrow improvement window means luck dominates over strategy.

### Finding 5: Budget Insufficiency (Confound 5)

**Strength: MODERATE-STRONG**

The "Run-9 Wall" — zero improvements before run index 9 in single-agent cells — defines a minimum exploration threshold. d11 agents average only 7.5 runs each (due to CPU contention), meaning they run out of budget before crossing this threshold. The 0% success rate in d11 may reflect budget inadequacy, not architectural failure.

## Strongest Supported Comparisons

1. **d00 vs d10** (memory effect in isolation): Cleanest comparison, same hardware conditions. Memory slightly hurts (Cohen's d = +0.52 on mean-best-per-rep).
2. **d00 vs d01** (parallelism effect, no CPU issue): d01 shows no significant CPU contention vs d00. Performance degrades slightly, attributable to agent homogeneity (G ≈ 0).

## Comparisons Undermined by Confounds

1. **Any comparison involving d11**: CPU contention, budget insufficiency, and low sample size (2 reps) make d11 results uninterpretable.
2. **d01 vs d11** (shared memory effect in parallel): Cannot separate memory effect from CPU contention effect.

## Main Caveats

- d11 has only 2 complete reps (d11_rep3 was still running during analysis)
- Unbalanced design (5/5/3/2 reps per cell)
- Strategy categories are self-labeled by the agent
- The deterministic evaluation (SEED=42, MAX_STEPS=585) means no stochastic effects
- All cells use the same LLM, temperature, and prompt

## What Changed in Experimental Understanding

Before this audit, the working hypothesis was "both memory and parallelism hurt performance." After the audit, the revised understanding is: **the experiment does not provide a valid test of the BP framework** because five systematic confounds compromise the comparison. The results are informative about what conditions the BP framework requires (diversity, structured memory, sufficient budget, adequate task headroom), but they do not constitute evidence for or against the framework itself.
