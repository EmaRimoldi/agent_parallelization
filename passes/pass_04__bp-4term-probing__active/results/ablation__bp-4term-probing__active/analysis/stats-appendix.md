# Statistical Appendix

## Dataset Summary

| Metric | Value |
|--------|-------|
| Probes executed | 16 (P01-P13, P15-P17) |
| Probes missing | 2 (P14, P18) |
| Total runs | 299 |
| Runs with valid val_bpb | 293 |
| Runs with null val_bpb | 7 (all in P13/agent_1) |
| Baseline runs | 25 |
| Non-baseline runs | 268 |
| Below-baseline runs | 6 (2.0%) |
| Baseline val_bpb | 0.925845 |
| Model | claude-haiku-4-5-20251001 (all probes) |
| Training budget | 60s (except P04: 30s, P01: ~315s due to bug) |

## Descriptive Statistics by Probe

| Probe | n | Mean | Std | Median | IQR | Best | Worst | <BL |
|-------|---|------|-----|--------|-----|------|-------|-----|
| P01 | 7 | 0.958 | 0.054 | 0.926 | 0.046 | 0.923 | 1.064 | 1 |
| P02 | 14 | 1.136 | 0.129 | 1.125 | 0.269 | 0.980 | 1.314 | 0 |
| P03 | 8 | 1.070 | 0.111 | 1.043 | 0.222 | 0.936 | 1.223 | 0 |
| P04 | 10 | 1.447 | 0.243 | 1.491 | 0.339 | 1.103 | 1.879 | 0 |
| P05 | 7 | 1.051 | 0.131 | 0.974 | 0.214 | 0.919 | 1.267 | 1 |
| P06 | 10 | 1.047 | 0.101 | 1.002 | 0.153 | 0.948 | 1.250 | 0 |
| P07 | 18 | 1.055 | 0.095 | 1.067 | 0.144 | 0.906 | 1.246 | 1 |
| P08 | 14 | 1.390 | 0.160 | 1.461 | 0.183 | 0.982 | 1.548 | 0 |
| P09 | 29 | 1.190 | 0.107 | 1.219 | 0.105 | 0.971 | 1.472 | 0 |
| P10 | 14 | 1.090 | 0.109 | 1.087 | 0.227 | 0.960 | 1.241 | 0 |
| P11 | 21 | 1.816 | 0.362 | 1.831 | 0.458 | 0.934 | 2.305 | 0 |
| P12 | 41 | 1.049 | 0.122 | 1.004 | 0.187 | 0.914 | 1.462 | 2 |
| P13 | 37 | 1.852 | 1.045 | 1.675 | 0.271 | 0.961 | 7.876 | 0 |
| P15 | 13 | 1.501 | 0.230 | 1.524 | 0.194 | 0.880 | 1.823 | 1 |
| P16 | 21 | 1.216 | 0.123 | 1.184 | 0.174 | 0.962 | 1.497 | 0 |
| P17 | 29 | 1.064 | 0.170 | 1.000 | 0.103 | 0.955 | 1.690 | 0 |

## Inferential Tests

### Test 1: Shared memory effect (P12 vs P09)

- **Comparison**: P12 (shared memory FIXED, 45m) vs P09 (diverse, no memory, 30m)
- **Normality**: P12 Shapiro-Wilk p=0.0001 (non-normal), P09 p=0.190
- **Test**: Mann-Whitney U (one-sided: P12 < P09)
- **Result**: U=210.0, **p<0.001**
- **Effect size**: Rank-biserial r=0.647 (large)
- **Interpretation**: P12 has significantly lower val_bpb distribution

### Test 1b: P12 vs P13 (same budget, different memory)

- **Comparison**: P12 (shared, 45m) vs P13 (no memory, 45m, both diverse)
- **Test**: Mann-Whitney U (one-sided: P12 < P13)
- **Result**: U=63.0, **p<0.001**
- **Effect size**: Rank-biserial r=0.917 (very large)
- **Caveat**: P13 has extreme outliers (7.88 bpb), inflating effect

### Test 2: Temperature effect on iteration count

- **Comparison**: Paired within-experiment (agent_0 low-temp vs agent_1 high-temp)
- **Pairs**: P02, P07, P09, P12, P17 (n=5 pairs)
- **Test**: Wilcoxon signed-rank (one-sided: low < high)
- **Result**: p=0.0625 (not significant at alpha=0.05)
- **Descriptive**: Mean low-temp runs=9.2, high-temp runs=17.2
- **Caveat**: n=5 pairs is very low power. Effect is directionally consistent but underpowered.

### Test 3: Homogeneous vs diverse agents

- **Comparison**: P01+P10 (homo) vs P02+P09 (diverse, no memory)
- **Test**: Mann-Whitney U (two-sided)
- **Result**: U=189.0, **p<0.001**
- **Effect size**: Rank-biserial r=0.581 (large)
- **Direction**: Homo has LOWER mean val_bpb (better)
- **Caveat**: Pooling across budgets (15m and 30m). P01 had template bug (315s training).

### Test 4: Degradation trend in P11

- **Test**: Linear regression on val_bpb vs run_index
- **Result**: slope=0.0035/run, R²=0.003, p=0.804
- **Interpretation**: No significant LINEAR trend, but the oscillation pattern (degrade → revert → re-degrade) produces high variance that masks the directional drift.
- **P03 comparison**: slope=-0.004/run, p=0.845 (also non-significant)

### Test 5: Task ceiling

- **Success rate**: 5/269 = 1.86%
- **95% CI (Clopper-Pearson)**: [0.4%, 3.7%]
- **Near misses**: 31/269 within 5% of baseline
- **Far worse**: 85/269 >50% above baseline
- **Interpretation**: Very low success rate consistent with ceiling effect

### Test 6: Seeded search (P15 vs P11)

- **Comparison**: P15 (seeded LR hint, temp=1.2) vs P11 (unseeded, temp=1.2)
- **Test**: Mann-Whitney U (one-sided: P15 < P11)
- **Result**: U=61.0, **p=0.004**
- **Caveat**: P15 best (0.880) is the baseline run — the seed was injected into the first message, making run 1 itself an LR=1.5e-3 run.

## Multiple Comparison Correction

6 primary tests were conducted. Applying Bonferroni correction (alpha=0.05/6=0.0083):
- Test 1 (P12 vs P09): p<0.001 — **significant after correction**
- Test 1b (P12 vs P13): p<0.001 — **significant after correction**
- Test 2 (temp effect): p=0.063 — not significant
- Test 3 (homo vs diverse): p<0.001 — **significant after correction**
- Test 4 (P11 trend): p=0.804 — not significant
- Test 6 (seeded): p=0.004 — **significant after correction**

## Explicit Blockers and Limitations

1. **No seed-level replication**: Each configuration was run once. Run-level statistics have high variance due to stochastic agent behavior.
2. **Memory bugs**: P05-P08 memory visibility was 0% despite configuration. These probes cannot test memory effects.
3. **Missing P14**: The planned private-memory-only test at high temperature did not execute. Cannot assess private memory's correction effect.
4. **Budget confound**: Probes range from 15-45 min. Longer budgets mechanically produce more runs. Not corrected.
5. **Baseline variability**: Baseline runs range from 0.880 to 1.103 (P04 with 30s training). Baseline is not fixed.
6. **P01 template bug**: P01 used 315s/run training (pre-fix). Not directly comparable to 60s probes.
