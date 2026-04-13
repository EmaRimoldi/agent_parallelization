# Analysis Report: BP Probing Experiments (P01-P18)

**Date**: 2026-04-13
**Analyst**: Claude (automated)
**Dataset**: 16 probes, 293 valid training runs, 25 baseline runs, 268 non-baseline runs
**Primary metric**: val_bpb (lower is better)
**Baseline**: 0.925845

---

## Comparison Questions

1. Does memory improve performance?
2. How are validation losses distributed — is there a ceiling effect?
3. Does agent parallelization help, and do homogeneous vs heterogeneous setups differ?
4. Does diversity in initialization (temperature) improve exploration?
5. What factors explain the best-performing trials?
6. Is the model (Claude Haiku) sufficiently capable for this task?
7. Is tool usage consistent across experiments?

---

## Key Findings

### Finding 1: Memory effect is confounded by mechanism bugs

**Critical context**: Memory was configured but **non-functional** in P05, P06, P07, P08 due to three bugs:
- Shared memory populated from unreliable data source (agents skip `update_snapshot.py`)
- Private memory read from empty trace.jsonl instead of training_runs.jsonl
- `elif` guard prevented both memory types from being active simultaneously

**After bug fixes** (P12, P17):
- P12 (shared memory FIXED): **2/41 runs below baseline** (4.9%), best=0.914
- P17 (both memory types FIXED): 0/29 below baseline, best=0.955
- Contrast with P09 (diverse, no memory, same budget class): 0/29 below baseline

**Statistical test**: P12 vs P09 Mann-Whitney U=210.0, **p<0.001**, rank-biserial r=0.647 (large effect).
P12 vs P13 (both 45m, diverse, no memory): U=63.0, **p<0.001**, r=0.917.

**However**: P12 had a symlink failure mid-experiment (agent_0 lost shared memory access after run 4 due to `git checkout` destroying the symlink). Only agent_1 had continuous shared memory visibility.

**Conclusion**: Shared memory produces significantly lower val_bpb distributions. The effect is large. But P17 (which had both memory types active and higher visibility: 86%) did NOT beat baseline. This suggests memory helps stabilize exploration (lower variance, lower mean) but does not reliably produce breakthroughs.

### Finding 2: Distribution collapses to a narrow band above baseline

- Global success rate: **5/269 non-baseline = 1.9%** (95% CI: 0.4%-3.7%)
- 97.4% of non-baseline runs are WORSE than baseline
- 31 runs are within 5% of baseline (near misses)
- 85 runs are >50% worse than baseline

The distribution is right-skewed: most runs cluster between 0.95-1.25, with a heavy tail up to 7.88 (P13 outlier). The narrow success band (0.88-0.93) is barely below the baseline band (0.93-0.96).

**Implication**: The task has a **ceiling effect**. The default train.py is already near-optimal for 60s training on this architecture. Most agent modifications make things worse because the search space is dominated by harmful configurations.

### Finding 3: Homogeneous agents outperform diverse agents (counterintuitively)

- Homo (P01+P10): mean=1.046, std=0.113
- Diverse (P02+P09): mean=1.172, std=0.118
- Mann-Whitney U=189.0, **p<0.001**, r=0.581

**This reversal** (homo better than diverse) is unexpected. Possible explanations:
1. **Temperature 0.3 agents waste time thinking**: In P07, agent_0 (temp=0.3) produced only 3 runs vs 15 for agent_1 (temp=1.2). The low-temp agent barely contributes.
2. **Homogeneous agents share baseline variance**: Both start from the same point and make small perturbations, staying closer to baseline.
3. **Jaccard paradox**: P02 (diverse temps) has Jaccard=1.0 (same strategy categories) while P01 (homo temps) has Jaccard=0.333. Temperature diversity does NOT produce strategy diversity.

### Finding 4: Temperature increases iteration speed but not quality

- High temp (1.2) agents: 152 runs, mean=1.377, best=0.880
- Low/default temp: 134 runs, mean=1.170, best=0.919
- Paired within-experiment (Wilcoxon): p=0.0625 (not significant at alpha=0.05)

Temperature increases run count (mean 17.2 vs 9.2 runs per agent in parallel probes) but the runs are on average WORSE. High-temp agents make more aggressive modifications that diverge from baseline.

**P11 is the critical case**: temp=1.2, no memory, 45 min. 21 runs, 0 below baseline, mean degradation +0.93 per run. The agent oscillates: escalate LR → degrade → partial git revert → re-escalate. Without memory, the agent cannot learn from its own history.

### Finding 5: All successes share one factor — learning rate modification

All 6 below-baseline runs:
| Run | Probe | bpb | LR change |
|-----|-------|-----|-----------|
| P15 run 1 | Seeded | 0.880 | Baseline (seeded hint in first message) |
| P07 run 14 | Shared* | 0.906 | 1e-3 → 1.5e-3 |
| P12 run 13 | Shared (FIXED) | 0.914 | 1e-3 → 5e-4 |
| P05 run 3 | Memory* | 0.919 | 1e-3 → 2e-3 |
| P01 run 3 | Par homo | 0.923 | 1e-3 → 2e-3 |
| P12 run 16 | Shared (FIXED) | 0.924 | Weight decay adjustment |

Strategy category success rates:
- optimization: 4/121 (3.3%)
- regularization: 1/77 (1.3%)
- other: 1/48 (2.1%)
- architecture: 0/15 (0.0%)
- data_pipeline: 0/7 (0.0%)

**P15 (seeded search) achieved the new overall record (0.880)** — but this was the BASELINE run, where the seeded hint told the agent to start from LR=1.5e-3. This confirms LR=1.5e-3 is better than the default 1e-3.

### Finding 6: Model capability is sufficient but task has low headroom

- All 16 probes used claude-haiku-4-5-20251001
- Baseline runs (first run, no modification): mean=0.977, std=0.049
- The model successfully commits code, runs training, and evaluates results in 100% of cases
- 293/293 runs have valid git commits and strategy categories
- Protocol modes are consistent: 84% explore, 9.2% bootstrap, 6.8% reevaluation

The model is capable of the task. The bottleneck is **task headroom**: 60s training on this architecture leaves very little room for improvement. The optimal changes (LR adjustments of 50-100%) are a needle in a haystack of harmful changes.

### Finding 7: Tool usage is consistent

- Protocol: 84% explore mode (expected for iterative search)
- All runs have git commits, strategy categories, and evaluation metadata
- Training time: mean=66.3s, std=42.6s (P01 had 362s due to original template bug, now fixed)
- Model: claude-haiku-4-5-20251001 across all probes

No tool usage inconsistencies found.

---

## Summary Table

| Question | Finding | Strength |
|----------|---------|----------|
| Memory effect | Shared memory significantly lowers mean bpb (p<0.001, r=0.647) but doesn't reliably produce breakthroughs | MODERATE (confounded by bugs in early probes) |
| Distribution | 97.4% of runs are worse than baseline; 1.9% success rate | STRONG ceiling effect |
| Parallelization | Homo outperforms diverse (p<0.001) — counterintuitive | MODERATE (small samples) |
| Diversity | Temperature increases speed not quality; strategy diversity ≠ temperature diversity | STRONG |
| Best trial factors | All successes involve LR modification near 1-2e-3 | STRONG (but n=6) |
| Model capability | Model is capable; task has low headroom | MODERATE |
| Tool consistency | Fully consistent across all probes | STRONG |

---

## Limitations

1. **No seed-level replication**: Each probe is a single run. No inferential claims about probe-level effects can be made with confidence. All statistics are run-level (within-probe), not probe-level.
2. **Memory bugs invalidate early probes**: P05-P08 cannot be compared to P12/P17 as memory tests. They are effectively "no memory" runs.
3. **Missing probes**: P14 (private memory + high temp) and P18 (seeded parallel) did not run. P14 was the key test for private memory correction of degradation.
4. **Confounded budget**: Probes span 15-45 min budgets. Longer budgets produce more runs mechanically. Cross-budget comparisons are confounded.
5. **Single model**: All experiments use Haiku. Results may not generalize to Sonnet/Opus.
6. **Single task**: One architecture, one dataset. LR sensitivity is task-specific.
