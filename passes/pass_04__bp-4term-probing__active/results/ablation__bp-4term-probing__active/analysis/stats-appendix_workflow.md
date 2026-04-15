# Phase 02a: Statistical Appendix

## 1. Descriptive Statistics

### d00 (single agent, no memory)

| Rep | Runs | Best val_bpb | Improvement at runs |
|-----|------|-------------|---------------------|
| rep1 | 12 | 0.855528 | [9] |
| rep2 | 2 | 0.925845 | [] |
| rep3 | 4 | 0.925845 | [] |
| rep4 | 14 | 0.921272 | [12] |
| rep5 | 15 | 0.824019 | [10, 13, 15] |

- Total runs: 47
- Runs per rep: mean=9.4, std=6.0
- All val_bpb: mean=1.0026, std=0.1096, min=0.8240, max=1.2515, median=0.9757
- Best-of-rep: mean=0.8905, std=0.0477

### d10 (single agent, with memory)

| Rep | Runs | Best val_bpb | Improvement at runs |
|-----|------|-------------|---------------------|
| rep1 | 3 | 0.925845 | [] |
| rep2 | 19 | 0.875490 | [9, 14, 16] |
| rep3 | 20 | 0.925845 | [] |
| rep4 | 20 | 0.922538 | [12] |
| rep5 | 7 | 0.925845 | [] |

- Total runs: 69
- Runs per rep: mean=13.8, std=8.2
- All val_bpb: mean=1.0127, std=0.1167, min=0.8755, max=1.3865, median=0.9810
- Best-of-rep: mean=0.9151, std=0.0222

## 2. Effect Size Analysis

### Primary: Best-of-rep Cohen's d

- d00 mean = 0.890502, std = 0.047666, n = 5
- d10 mean = 0.915113, std = 0.022196, n = 5
- Pooled std = 0.037180
- **Cohen's d = (0.915113 - 0.890502) / 0.037180 = +0.6619**
- Direction: d10 is **worse** (higher val_bpb) by 0.66 pooled SDs
- Interpretation: Medium effect, wrong direction for the memory hypothesis

### Secondary: All-runs Cohen's d

- d00 mean = 1.002567, std = 0.109589, n = 47
- d10 mean = 1.012721, std = 0.116656, n = 69
- Pooled std = 0.113732
- **Cohen's d = +0.0892** (negligible)
- The all-runs comparison is near-zero because most runs produce poor results regardless of cell

## 3. Inferential Tests

### Welch's t-test (best-of-rep, two-tailed)

- **Assumption check**: Normality difficult to assess with n=5. Shapiro-Wilk: d00 W=0.79 (borderline), d10 W=0.64 (non-normal due to clustering at baseline). Equal variance assumption violated (F-ratio = 4.6).
- t = -1.0466
- df ≈ 5.87 (Welch correction)
- **p = 0.338**
- Conclusion: Not significant at α=0.05

### Mann-Whitney U (non-parametric, best-of-rep)

- U = 8.0
- **p = 0.373**
- Conclusion: Not significant

### Interpretation

With n=5 per group, the test has approximately 15-20% power to detect a medium effect (d=0.5) at α=0.05. The non-significance is expected given the sample size. The observed effect size (d=0.66) would require n≈40 per group for 80% power.

## 4. Mode Diversity

### d00 strategy categories (non-baseline runs)

| Category | Count | Promoted |
|----------|-------|----------|
| optimization | 17 | 1 |
| other | 14 | 3 |
| regularization | 6 | 0 |
| architecture | 3 | 0 |
| data_pipeline | 2 | 0 |

- Distinct categories: 5
- Categories with ≥2 runs: **5**
- Total promoted: **4**

### d10 strategy categories (non-baseline runs)

| Category | Count | Promoted |
|----------|-------|----------|
| optimization | 21 | 0 |
| other | 18 | 3 |
| regularization | 16 | 0 |
| architecture | 8 | 1 |
| data_pipeline | 1 | 0 |

- Distinct categories: 5
- Categories with ≥2 runs: **4** (data_pipeline has only 1)
- Total promoted: **4**

### Note on mode labeling

Strategy categories are assigned by the agent's self-reported `strategy_category` field in training_runs.jsonl, not by independent code diff analysis. The `other` category is overloaded and likely contains heterogeneous strategies. Post-hoc labeling via `scripts/label_modes.py` (code diff analysis) would provide more reliable categorization but was not run for this analysis.

## 5. Cost Variance and Jensen Gap

### Wall-clock cost per run

| Metric | d00 | d10 |
|--------|-----|-----|
| Mean | 134.5s | 83.1s |
| Std | 85.5s | 26.6s |
| Min | 58.2s | 58.2s |
| Max | 494.9s | 182.4s |
| CV | 0.636 | 0.320 |

### Jensen remainder R_α = log E[κ] - E[log κ]

| Axis | d00 | d10 | Ratio |
|------|-----|-----|-------|
| Wall-clock | 0.1506 | 0.0410 | 3.67x |
| Training time | 0.1357 | 0.0454 | 2.99x |

The Jensen gap measures how much concavity of log transforms amplifies cost heterogeneity. d00's large Jensen gap (0.15 vs 0.04) indicates highly variable per-step cost, driven by occasional long deliberation periods.

## 6. Memory Context Analysis (d10 only)

- Runs with memory visible: 49/69 (71%)
- Memory entries: range 0-19, mean 7.4
- Pearson correlation (memory entries vs val_bpb): r = 0.040, p = 0.747
- Memory depth is uncorrelated with outcome quality

## 7. Explicit Blockers and Limitations

1. **n=5 is severely underpowered** for the best-of-rep comparison. Power ≈ 15-20% at α=0.05.
2. **Strategy categories are self-reported**, not validated by code diff analysis.
3. **No cost normalization**: d10 agents produce more runs, so total compute (sum of wall-clock) differs.
4. **Baseline anchoring**: The deterministic baseline (0.925845) is a ceiling for many reps. A floor effect may compress the d10 distribution.
5. **Memory content is uncontrolled**: The shared memory includes all prior results including failures, which may not be the optimal information to provide.
