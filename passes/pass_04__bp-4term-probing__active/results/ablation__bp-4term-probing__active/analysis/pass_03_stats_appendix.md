# Phase 03 Statistical Appendix

## Sample Sizes

| Cell | Reps | Total Runs | Non-baseline | Agents/Rep | Runs/Agent (mean) |
|------|------|-----------|-------------|------------|-------------------|
| d00  | 5    | 47        | 42          | 1          | 9.4               |
| d10  | 5    | 69        | 64          | 1          | 13.8              |
| d01  | 3    | 91        | 85          | 2          | 15.2              |
| d11  | 2    | 30        | 26          | 2          | 7.5               |
| **Total** | **15** | **237** | **217** | — | — |

## Descriptive Statistics

### val_bpb (all runs, lower is better)
| Cell | Mean | SD | Median | Min | Max | IQR |
|------|------|----|--------|-----|-----|-----|
| d00 | 1.0026 | 0.1096 | 0.9891 | 0.8240 | 1.2832 | 0.0809 |
| d10 | 1.0127 | 0.1167 | 0.9990 | 0.8755 | 1.2750 | 0.1244 |
| d01 | 1.0663 | 0.1278 | 1.0594 | 0.9148 | 1.4037 | 0.1494 |
| d11 | 1.0453 | 0.0919 | 1.0652 | 0.9258 | 1.2657 | 0.1096 |

### Best val_bpb per replicate
| Cell | Rep 1 | Rep 2 | Rep 3 | Rep 4 | Rep 5 | Mean | SD |
|------|-------|-------|-------|-------|-------|------|-----|
| d00 | 0.8555 | 0.9258 | 0.9258 | 0.9213 | 0.8240 | 0.8905 | 0.0477 |
| d10 | 0.9258 | 0.8755 | 0.9258 | 0.9225 | 0.9258 | 0.9151 | 0.0222 |
| d01 | 0.9208 | 0.9148 | 0.9258 | — | — | 0.9205 | 0.0056 |
| d11 | 0.9258 | 0.9258 | — | — | — | 0.9258 | 0.0000 |

### Training time (seconds)
| Cell | Mean | SD | Median | Min | Max |
|------|------|----|--------|-----|-----|
| d00 | 116.4 | 64.4 | 103.0 | 60.3 | 477.3 |
| d10 | 73.2 | 24.7 | 70.4 | 31.5 | 159.4 |
| d01 | 113.6 | 46.7 | 104.0 | 57.3 | 354.2 |
| d11 | 267.8 | 187.6 | 228.4 | 60.7 | 863.7 |

### Wall-clock time (seconds)
| Cell | Mean | SD | Median | Min | Max |
|------|------|----|--------|-----|-----|
| d00 | 134.5 | 85.5 | 114.4 | 31.3 | 495.5 |
| d10 | 83.1 | 26.6 | 80.3 | 34.1 | 168.7 |
| d01 | 126.9 | 49.6 | 116.3 | 43.2 | 372.1 |
| d11 | 222.4 | 172.0 | 177.5 | 45.5 | 882.8 |

## Inferential Tests

### CPU Contention (Analysis 1)

**Kruskal-Wallis test (training_seconds across 4 cells):**
- H = 56.258, p < 0.001
- Assumptions: Independent samples, ordinal scale. Satisfied.
- Justification: Non-parametric chosen because training_seconds distributions are right-skewed with unequal variances.

**Pairwise Mann-Whitney U tests:**
| Comparison | U | p-value | Effect size r | Interpretation |
|-----------|---|---------|---------------|----------------|
| d00 vs d01 | 1970 | 0.452 | 0.064 | NOT significant |
| d00 vs d11 | 360 | 0.0003 | 0.410 | **SIGNIFICANT** |
| d10 vs d01 | 1429 | <0.001 | 0.466 | **SIGNIFICANT** |
| d10 vs d11 | 269 | <0.001 | 0.586 | **SIGNIFICANT** |
| d00 vs d10 | 2314 | <0.001 | 0.361 | **SIGNIFICANT** |
| d01 vs d11 | 734 | 0.0002 | 0.344 | **SIGNIFICANT** |

No multiple-comparison correction applied (these are planned contrasts, not exploratory). Effect size r = Z / sqrt(N).

### Agent Homogeneity (Analysis 2)

Jaccard similarity (strategy categories):
| Cell | Rep | Jaccard | Categories shared / total |
|------|-----|---------|--------------------------|
| d01 | 1 | 0.80 | 4/5 |
| d01 | 2 | 1.00 | 4/4 |
| d01 | 3 | 0.75 | 3/4 |
| d11 | 1 | 0.75 | 3/4 |
| d11 | 2 | 0.00 | 0/4 (partial data) |

Strategy entropy (Shannon, base 2):
| Cell | Entropy | N categories |
|------|---------|-------------|
| d00 | 1.939 | 5 |
| d10 | 2.011 | 5 |
| d01 | 1.867 | 5 |
| d11 | 1.870 | 4 |

### Memory Anchoring (Analysis 3)

**Chi-squared test (early vs late strategy distribution):**
| Cell | chi2 | df | p-value | Interpretation |
|------|------|----|---------|----------------|
| d00 | 5.772 | 4 | 0.217 | NOT significant |
| d10 | 4.629 | 4 | 0.328 | NOT significant |
| d01 | 4.235 | 4 | 0.375 | NOT significant |
| d11 | 3.700 | 3 | 0.296 | NOT significant |

**Spearman correlation (memory depth vs val_bpb):**
| Cell | r | p | n |
|------|---|---|---|
| d10 | -0.231 | 0.066 | 64 |
| d11 | NaN | NaN | 26 (constant memory) |

**Consecutive same-strategy streaks:**
| Cell | Mean max streak | Max | All streaks |
|------|-----------------|-----|-------------|
| d00 | 2.4 | 4 | [4, 1, 1, 3, 3] |
| d10 | 4.2 | 8 | [1, 4, 8, 3, 5] |
| d01 | 3.2 | 5 | [2, 3, 5, 3, 3, 3] |
| d11 | 2.5 | 3 | [2, 3, 2, 3] |

### Task Ceiling (Analysis 4)

**Success rates:**
| Cell | Successes | Non-baseline | Rate |
|------|-----------|-------------|------|
| d00 | 11 | 42 | 26.2% |
| d10 | 12 | 64 | 18.8% |
| d01 | 6 | 85 | 7.1% |
| d11 | 0 | 26 | 0.0% |

**Strategy win rates:**
| Strategy | Wins | Losses | Win Rate |
|----------|------|--------|----------|
| other | 18 | 39 | 31.6% |
| architecture | 6 | 16 | 27.3% |
| data_pipeline | 1 | 3 | 25.0% |
| optimization | 4 | 80 | 4.8% |
| regularization | 0 | 50 | 0.0% |

### Budget Sufficiency (Analysis 5)

**First improvement timing:**
| Cell | Rep 1 | Rep 2 | Rep 3 | Rep 4 | Rep 5 | Mean |
|------|-------|-------|-------|-------|-------|------|
| d00 | 9 | — | — | 12 | 10 | 10.3 |
| d10 | — | 9 | — | 12 | — | 10.5 |
| d01 | 5 | 7 | — | — | — | 6.0 |
| d11 | — | — | — | — | — | — |

**Point-biserial correlation (session length vs improvement):** r = 0.388, p = 0.153

### 2×2 Factorial Analysis

**Main effects (on mean-of-best-per-rep):**
| Effect | Estimate | Cohen's d | Direction |
|--------|----------|-----------|-----------|
| Memory | +0.0150 | +0.521 | Hurts |
| Parallelism | +0.0203 | +0.635 | Hurts |
| Interaction (M×P) | -0.0192 | — | Not significant |

**Permutation test for interaction:** p = 0.621 (10,000 permutations)

## Explicit Blockers and Limitations

1. **d11 sample size**: Only 2 complete reps. All d11 factorial estimates are unreliable.
2. **Unbalanced design**: 5/5/3/2 reps across cells. Factorial estimates are influenced by the cells with more data.
3. **Strategy labeling**: Categories are self-reported by the LLM agent, introducing labeling noise.
4. **No correction**: No Bonferroni/Holm correction applied to the 6 pairwise Mann-Whitney tests (planned contrasts).
5. **CPU contention in d11 invalidates**: Any comparison involving d11 on time-based metrics is confounded.
6. **No causal identification**: Observational study; confounds are identified but not experimentally controlled.
