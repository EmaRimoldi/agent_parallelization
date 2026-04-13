# Phase 02a: Calibration Analysis Report

**Date**: 2026-04-13
**Experiment**: BP 2x2 Power Calibration (d00 vs d10)
**Branch**: `bp-2x2-instrumentation`

## Analysis Question

Does the single-agent memory-enabled architecture (d10) produce measurably different optimization outcomes than the single-agent memoryless architecture (d00) on the CIFAR-10 substrate, and is the signal sufficient to justify the full 2x2 experiment?

## Key Findings

### 1. Memory does NOT improve best-of-rep performance

| Metric | d00 (no memory) | d10 (memory) |
|--------|-----------------|--------------|
| Best-of-rep mean | **0.8905** | 0.9151 |
| Best-of-rep std | 0.0477 | 0.0222 |
| Reps beating baseline | 3/5 (60%) | 2/5 (40%) |
| Best overall | **0.8240** | 0.8755 |

**Cohen's d = +0.66 (medium, favoring d00)**. The memoryless agent actually achieves better best-of-rep results. This is the opposite of the expected direction.

However, Welch's t-test is not significant (p=0.338) due to small sample size (n=5 per cell). The effect is suggestive but not conclusive.

### 2. Memory increases iteration throughput

| Metric | d00 | d10 |
|--------|-----|-----|
| Total runs | 47 | 69 |
| Mean runs/rep | 9.4 (std=6.0) | 13.8 (std=8.2) |
| Max runs in a rep | 15 | 20 |

d10 agents consistently produce more training iterations per session. Memory appears to reduce deliberation time between runs, enabling faster iteration.

### 3. Memory dramatically reduces cost variance

| Metric | d00 | d10 |
|--------|-----|-----|
| Wall-clock mean | 134.5s | 83.1s |
| Wall-clock std | 85.5s | 26.6s |
| CV (wall-clock) | 0.636 | 0.320 |
| Jensen gap R_α (wall) | **0.151** | **0.041** |
| Jensen gap R_α (train) | 0.136 | 0.045 |

The Jensen remainder is **3.7x larger for d00**. Memory regularizes per-step cost — the agent spends more consistent time per iteration when it has history to draw on. d00 has extreme outlier runs (up to 495s) where the agent gets stuck deliberating.

### 4. Mode diversity is comparable

Both cells explore 5 strategy categories with similar distributions. The dominant modes are `optimization` and `other`, with `regularization`, `architecture`, and `data_pipeline` as secondary modes. Both cells have 4 promoted (successful) runs.

### 5. Memory depth does not correlate with performance in d10

Pearson r = 0.040, p = 0.747. Having more memory entries available does not predict better val_bpb. The agent uses memory to iterate faster but not to search smarter.

## Summary of Strongest Supported Comparisons

1. **d00 > d10 on best-of-rep** (Cohen's d = 0.66, but p = 0.34 — underpowered)
2. **d10 > d00 on iteration throughput** (13.8 vs 9.4 runs/rep)
3. **d00 >> d10 on cost variance** (Jensen gap 0.151 vs 0.041)
4. **d00 ≈ d10 on mode diversity** (5 categories each, similar distributions)
5. **Memory depth ⊥ val_bpb** (r = 0.04 in d10)

## Main Caveats

- **n=5 per cell**: Severely underpowered for detecting moderate effects on best-of-rep
- **Strategy labeling**: The `strategy_category` field is agent-reported, not independently verified via code diff analysis
- **d00's best results are driven by 2 high-performing reps** (rep1 and rep5), while d10's one strong rep (rep2) is more modest — high variance masks the picture
- **Memory content not controlled**: d10's memory includes full history including failures, which may cause information overload
- **Baseline anchoring**: 3/5 d10 reps never improved over the deterministic baseline, vs 2/5 for d00

## What Changed in Understanding

The initial hypothesis was that memory (d10) would improve optimization by enabling the agent to avoid repeating failed strategies. The data shows a more nuanced picture:

- Memory **does** help with iteration efficiency (κ axis) — lower cost, more runs, less variance
- Memory **does not** help with outcome quality (Δ axis) — if anything, it slightly hurts
- This suggests the ε (routing mismatch) term may be **negative** in practice: the agent conditions on its history but routes to suboptimal strategies (e.g., becoming fixated on incremental improvements rather than bold architecture changes)

## Decision Gate Inputs

For Phase 02b:
- **Cohen's d**: +0.66 (medium effect, but in wrong direction for d10)
- **Mode diversity**: 5 categories per cell, 4 with ≥2 runs each (d00), 4 with ≥2 runs each (d10)
- **Sample size**: d00=47 runs, d10=69 runs (>50 for d10, <50 for d00)
- **Jensen gap**: d00=0.151, d10=0.041 (3.7x difference)
