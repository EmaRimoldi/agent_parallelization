# Phase 03: Design Audit Figure Catalog

Canonical figure directory: `results/figures/pass_03_design_audit/`

## Figure 1: CPU Contention Evidence (`figure-01-cpu-contention.{png,pdf}`)

### Panel A: Training Time per Run (Box Plot)
- **Purpose**: Show that parallel cells, especially d11, have inflated training times
- **Data source**: `training_seconds` from all training_runs.jsonl
- **Error bars**: Box=IQR, whiskers=1.5×IQR, dots=outliers
- **Caption requirements**: Report Kruskal-Wallis H and p-value; note d11 median
- **Key observation**: d11 median training time (228s) is 3.2x higher than d10 (70s)
- **Interpretation**: Shared memory + parallel agents on same CPU creates severe I/O contention
- **Caveat**: d01 shows no significant contention vs d00, so contention is d11-specific

### Panel B: Wall-Clock Time per Run (Box Plot)
- **Purpose**: Show total per-run time including agent deliberation
- **Data source**: `wall_seconds` from training_runs.jsonl
- **Key observation**: d11's wall time is also elevated but less extreme than training time
- **Interpretation**: Training (not thinking) is the bottleneck for d11

### Panel C: Training Time by Replicate (Error Bar)
- **Purpose**: Confirm contention is systematic, not rep-specific
- **Key observation**: d11 is consistently high across both available reps
- **Caveat**: d11 has only 2 reps; d00/d10 have higher rep-to-rep variance

## Figure 2: Agent Homogeneity Evidence (`figure-02-agent-homogeneity.{png,pdf}`)

### Panel A: Strategy Category Distribution (Grouped Bar)
- **Purpose**: Compare strategy mix across cells
- **Data source**: `strategy_category` from training_runs.jsonl (non-baseline)
- **Key observation**: All cells explore the same 4-5 categories with similar proportions
- **Interpretation**: Parallelism does not increase strategy diversity (G ≈ 0)

### Panel B: Per-Agent Strategy Breakdown — d01 (Butterfly Chart)
- **Purpose**: Show that both agents in d01 choose identical strategy types
- **Key observation**: Bars are nearly symmetric — both agents explore same categories
- **Interpretation**: Same-model agents are functionally duplicates

### Panel C: Per-Agent Strategy Breakdown — d11 (Butterfly Chart)
- **Purpose**: Show d11 also has agent overlap
- **Key observation**: Similar symmetry pattern as d01
- **Caveat**: d11 has fewer runs per agent; rep2 is partial

## Figure 3: Memory Anchoring Evidence (`figure-03-memory-anchoring.{png,pdf}`)

### Panel A: Strategy Switch Probability Over Time (Line Plot)
- **Purpose**: Test whether agents explore less as they accumulate memory
- **Data source**: Consecutive strategy_category changes per run index
- **Key observation**: d10 (memory) shows declining switch rates at later indices
- **Interpretation**: Memory may cause agents to "lock in" to familiar strategies
- **Caveat**: Sample sizes shrink at higher run indices; not all reps contribute equally

### Panel B: Memory Depth vs Performance (Scatter)
- **Purpose**: Test whether more memory entries improve val_bpb
- **Data source**: `memory_context_entries` + `shared_memory_context_entries` vs `val_bpb`
- **Key observation**: No correlation (Spearman r=-0.23, p=0.066)
- **Interpretation**: Memory quantity ≠ information quality. More entries don't help.

### Panel C: Cumulative Unique Strategies Over Runs (Line Plot)
- **Purpose**: Compare exploration breadth across cells
- **Key observation**: All cells plateau at 4-5 unique categories within 5-8 runs
- **Interpretation**: Memory does not unlock new strategy modes

## Figure 4: Task Ceiling Evidence (`figure-04-task-ceiling.{png,pdf}`)

### Panel A: val_bpb Distribution (Histogram)
- **Purpose**: Show how most modifications worsen performance
- **Data source**: All non-baseline val_bpb values
- **Key observation**: Baseline (0.926) is near the left edge; 87.8% of modifications are worse
- **Interpretation**: The search space has very few winners

### Panel B: Success Rate by Cell (Bar)
- **Purpose**: Compare fraction of runs beating baseline
- **Key observation**: Monotonic decrease: d00 (26%) > d10 (19%) > d01 (7%) > d11 (0%)
- **Interpretation**: More architectural complexity = lower success rate (or more confounds)

### Panel C: Strategy Win/Lose Counts (Horizontal Bar)
- **Purpose**: Show which strategy types actually work
- **Key observation**: Regularization: 0/50 wins. Optimization: 4/84 wins. "Other": 18/57 wins.
- **Interpretation**: The task constraints make regularization counterproductive

## Figure 5: Budget Sufficiency Evidence (`figure-05-budget-sufficiency.{png,pdf}`)

### Panel A: Optimization Trajectories (Best-so-far Line Plot)
- **Purpose**: Show when improvements occur in each session
- **Data source**: Cumulative minimum val_bpb per run index per rep
- **Key observation**: Improvements cluster at run 9+ for single-agent cells
- **Interpretation**: A minimum exploration threshold ("Run-9 Wall") must be crossed

### Panel B: First Improvement Timing (Dot/X Plot)
- **Purpose**: Show exactly when each rep first beats baseline
- **Key observation**: Single agents: run 9-12. d01: run 5-7. d11: never.
- **Interpretation**: d01 benefits from 2x parallelism (reaches threshold faster per agent)
- **Caveat**: Many reps never improve at all (shown as X markers)

### Panel C: Session Length vs Best Outcome (Scatter)
- **Purpose**: Test whether more runs = better outcomes
- **Key observation**: Positive trend (r=0.39, p=0.15) but not significant
- **Interpretation**: Budget matters but is not the sole determinant

## Figure 6: 2×2 Factorial Summary (`figure-06-2x2-summary.{png,pdf}`)

### Panel A: Best-of-Rep Performance (Dot + Mean Plot)
- **Purpose**: Show the primary outcome metric per cell
- **Key observation**: d00 has the widest spread AND the best outcomes
- **Interpretation**: Simple architecture + luck produces the best results

### Panel B: Per-Run Success Rate (Bar)
- **Purpose**: Compare success probability across cells
- **Key observation**: d11 = 0%, confirming its systematic failure

### Panel C: 2×2 Interaction Plot (Line)
- **Purpose**: Visualize the factorial design
- **Key observation**: Lines slope upward (memory hurts) and are approximately parallel (no interaction)
- **Interpretation**: Effects are additive; memory and parallelism independently degrade performance

### Panel D: Jensen Gap / Cost Variance (Bar)
- **Purpose**: Quantify cost heterogeneity per cell
- **Key observation**: d11 has the HIGHEST Jensen gap on wall-clock axis (due to CPU contention)
- **Interpretation**: d11's cost variance is inflated by contention, not architecture
- **Caveat**: This contradicts the earlier (pre-contention-correction) analysis that showed memory reducing Jensen gap

## Composite PDF Report

**`BP_2x2_Design_Audit_Report.pdf`** (13 pages)
- Contains all 6 figures with captions, full statistical analysis, 5 confound descriptions, and redesign recommendations
