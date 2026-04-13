# Phase 02a: Figure Catalog

Canonical figure directory: [`../../results/figures/pass_02_workflow_calibration/`](../../results/figures/pass_02_workflow_calibration/)

## Figure 1: Main Comparison (`figure-01-main-comparison.png`)

### Panel A: Best-of-rep Performance (Box + Strip Plot)
- **Purpose**: Compare the primary outcome metric between d00 and d10
- **Data source**: Best val_bpb per replicate from training_runs.jsonl
- **Error bars**: Box shows IQR; whiskers show range
- **Caption requirements**: Report n=5 per cell, baseline value, and note that lower is better
- **Key observation**: d00 has wider spread but lower mean; d10 clusters near baseline
- **Interpretation**: Memory does not improve best-of-rep. d00's variance includes both the worst and the best outcomes.
- **Caveat**: n=5 is too small for box plot quartiles to be reliable

### Panel B: All Runs Distribution (Histogram)
- **Purpose**: Show the full distribution of all training evaluations
- **Data source**: All val_bpb values from training_runs.jsonl
- **Key observation**: Both distributions are right-skewed with similar shape. Most runs produce val_bpb > baseline.
- **Interpretation**: The majority of agent modifications make things worse. The distributions overlap almost completely.
- **Caveat**: Runs within a rep are not independent (sequential search)

### Panel C: Iteration Count per Rep (Grouped Bar Chart)
- **Purpose**: Compare how many training iterations each architecture produces
- **Data source**: Run counts per replicate
- **Key observation**: d10 consistently produces more runs per rep, especially reps 3-4 (20 each)
- **Interpretation**: Memory enables faster iteration by reducing deliberation time

## Figure 2: Trajectories and Cost Analysis (`figure-02-trajectories-cost.png`)

### Panel A: Optimization Trajectories (Best-so-far Curves)
- **Purpose**: Show how optimization progress unfolds over sequential runs
- **Data source**: Cumulative minimum of val_bpb per run index
- **Key observation**: Improvements tend to occur late in the session (run 9+). Many reps plateau at baseline.
- **Interpretation**: The agent needs many exploratory failures before finding a productive direction. d00_rep5 shows the steepest descent (3 improvements in runs 10-15).
- **Caveat**: Run indices are not time-aligned between reps

### Panel B: Per-Run Cost Distribution (Histogram)
- **Purpose**: Compare wall-clock cost per training iteration
- **Data source**: wall_seconds from training_runs.jsonl
- **Key observation**: d00 has a long right tail (up to ~500s), d10 is concentrated around 60-100s
- **Interpretation**: Memory regularizes agent behavior — the agent spends less time "stuck" when it has history available
- **Caveat**: Wall-clock includes both thinking and training time

### Panel C: Strategy Mode Distribution (Horizontal Bar Chart)
- **Purpose**: Compare exploration diversity between architectures
- **Data source**: strategy_category from training_runs.jsonl (non-baseline runs)
- **Key observation**: Both cells explore the same 5 categories. d10 has more regularization and architecture attempts.
- **Interpretation**: Memory slightly shifts the exploration distribution but does not unlock new modes.
- **Caveat**: Categories are self-reported by the agent

## Figure 3: Jensen Gap and Memory Effect (`figure-03-jensen-memory.png`)

### Panel A: Jensen Remainder Comparison (Bar Chart)
- **Purpose**: Quantify cost heterogeneity via the BP framework's R_α term
- **Data source**: Computed from wall_seconds and training_seconds distributions
- **Key observation**: d00's Jensen gap is 3.7x larger than d10's on wall-clock axis
- **Interpretation**: This is the clearest architectural signal — memory stabilizes the per-step cost distribution, which directly affects the κ term in the BP decomposition
- **Caveat**: Jensen gap is sensitive to outliers in d00 (the 495s run)

### Panel B: Memory Depth vs Performance (Scatter Plot)
- **Purpose**: Test whether more memory context improves outcomes in d10
- **Data source**: memory_context_entries and val_bpb from d10 training_runs.jsonl
- **Key observation**: No correlation (r=0.04, p=0.75). Points scatter uniformly.
- **Interpretation**: The agent uses memory to iterate faster, not to make better decisions. Information quantity ≠ information quality.
- **Caveat**: Memory entries are cumulative within a rep, so later runs always have more entries
