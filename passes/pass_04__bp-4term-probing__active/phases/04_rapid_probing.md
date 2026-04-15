# Phase 04: Rapid Probing — Experimental Redesign

## Goal

Identify which experimental settings produce informative signal for the BP framework by running short, targeted probe experiments. Each probe tests one variable at a time.

## Design Rationale

Phase 03 audit identified 5 confounds. This phase tests fixes for each:

| Confound | Fix | Probe IDs |
|----------|-----|-----------|
| Agent homogeneity (G≈0) | Temperature diversity | P01 vs P02 |
| Task ceiling (12% success) | Shorter training → more iterations | P03 vs P04 |
| Memory anchoring (ε<0) | Diversity + shared memory | P05 vs P06 |

## Probe Design

**Budget**: 15 min per probe (vs 45 min in Phase 02/03)
**Training**: 30-60s per run
**Reps**: 1 per probe (signal detection, not power)
**Execution**: SEQUENTIAL (eliminates CPU contention)
**Agents**: Haiku (same as Phase 02/03)

### Wave 1: Signal Detection (6 probes, ~90 min total)

| Probe | Mode | Agents | Temp | Train(s) | Variable Tested |
|-------|------|--------|------|----------|-----------------|
| P01 | parallel | 2 | null/null | 60 | Baseline parallel (homogeneous) |
| P02 | parallel | 2 | 0.3/1.2 | 60 | **Diversity injection** |
| P03 | single_long | 1 | null | 60 | Baseline single (reference) |
| P04 | single_long | 1 | null | 30 | **Task headroom** (2x iterations) |
| P05 | single_memory | 1 | null | 60 | Baseline memory |
| P06 | parallel_shared | 2 | 0.3/1.2 | 60 | **Diversity + shared memory** |

### Primary Metrics (Process-Based)

For 15-min probes, val_bpb improvement is unlikely. Instead measure:
1. **Strategy entropy** — Shannon entropy of strategy categories
2. **Per-agent Jaccard** — overlap between agent_0 and agent_1 strategies
3. **Switch rate** — fraction of consecutive runs that change strategy
4. **Training time** — contention measure
5. **Total iterations** — exploration depth
6. **Unique hypotheses** — exploration breadth

### Secondary Metrics
- Best val_bpb (if any improvement over baseline 0.925845)
- Success rate (fraction beating baseline)

## Decision Gate

After Wave 1, compare probe pairs:
- If P02 entropy > P01 entropy: diversity injection works → proceed with diverse agents
- If P04 iterations > P03 iterations × 1.5: shorter training increases headroom → use 30s
- If P06 shows different behavior from P05: shared memory + diversity → promising

## Wave 2 Design (contingent on Wave 1)

Based on Wave 1 results, run 4-6 more probes combining the best settings:
- Extend budget (30 min) for the most promising configuration
- Add 2 more reps for the configuration showing most signal
- Test combinations (e.g., diverse + short training + shared memory)

## Output

- `runs/probe_P{01-06}/` — experiment directories
- `workflow/artifacts/probe_wave1_analysis.json` — metrics
- `workflow/artifacts/probe_wave1_summary.md` — human-readable summary
- `workflow/logs/probe_execution.log` — full execution log
