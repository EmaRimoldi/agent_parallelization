# Figure Catalog

## Figure 1: Main Comparison (figure-01-main-comparison.pdf)

- **Purpose**: Show the full distribution of val_bpb across all 16 probes, grouped by experimental condition, with design factors visible.
- **Data source**: All 293 valid runs from runs/experiment_probe_P*/
- **Plotted variables**: Box-and-whisker of val_bpb per probe, with individual points overlaid. Stars mark below-baseline runs.
- **Color coding**: Gray=single no-memory, Blue=single+memory, Green=parallel, Purple=shared memory, Pink=full stack
- **Key observation**: Below-baseline successes (gold stars) appear only in P01, P05, P07, P12, P15. The vast majority of runs cluster above baseline. P11 and P13 show high variance and extreme degradation.
- **Interpretation**: The task has a strong ceiling effect. Most agent modifications harm performance. Only a few specific configurations occasionally break through. The broken-memory probes (P05-P08) are functionally equivalent to no-memory probes.
- **Caption requirements**: Must note that * marks broken-memory probes. Must include baseline value. Must note that P04 used 30s training and P01 had template bug.
- **Known caveats**: Mixed budgets across probes (15-45 min). Visual comparison across different n values may be misleading.

## Figure 2: Memory Effect (figure-02-memory-effect.pdf)

- **Purpose**: Analyze the memory mechanism's actual impact — distinguishing configured vs functional memory.
- **Panels**: (A) Runs vs best bpb scatter showing memory status, (B) P12 shared memory entries vs bpb, (C) P11 vs P12 trajectory comparison.
- **Data source**: All probes for panel A; P12 run-level data for panel B; P11 and P12 agent_1 for panel C.
- **Key observation**: Panel A shows no clear relationship between run count and best bpb — more runs do not help. Panel B shows P12 agent_1 achieving below-baseline results with high shared memory visibility (17+ entries). Panel C shows P11 diverging upward while P12 stays controlled.
- **Interpretation**: Memory does not improve exploration quantity. It improves exploration QUALITY by preventing repeated failures. P12 agent_1 (with shared memory) avoids the degradation spiral that dominates P11.
- **Caption requirements**: Must note that memory was broken in early probes. Panel B x-axis is the number of shared memory entries visible to the agent at each turn.
- **Known caveats**: P12 agent_0 lost shared memory access after run 4 (symlink destroyed by git checkout). Only agent_1 had continuous shared memory.

## Figure 3: Distribution by Condition (figure-03-distribution-by-condition.pdf)

- **Purpose**: Show the distribution shape (violin + strip plot) for each major experimental condition group.
- **Grouping**: Single no-memory, Single high-temp, Single memory*, Parallel homo, Parallel diverse, Parallel shared (fixed), Full stack.
- **Key observation**: All groups are right-skewed with most mass above baseline. The "Parallel shared (fixed)" group (P12 only) has the tightest distribution centered nearest to baseline. The "Single high-temp" group (P11, P15) has the widest spread.
- **Interpretation**: Shared memory reduces variance and shifts the distribution downward. High temperature increases variance without shifting the mean downward. Homogeneous parallel stays closer to baseline than diverse.
- **Caption requirements**: Must note sample sizes per group. Must note * for broken memory.
- **Known caveats**: Groups have very different n values (4 to 152). Violin width may be misleading.

## Figure 4: Temperature Effect (figure-04-temperature-effect.pdf)

- **Purpose**: Quantify temperature's effect on iteration speed and performance separately.
- **Panels**: (A) Temperature vs run count, (B) Temperature vs best bpb.
- **Data source**: Per-agent data from all parallel experiments (P02, P07, P09, P12, P13, P17).
- **Key observation**: Panel A shows a positive trend (higher temp → more runs) but not statistically significant (r=0.42). Panel B shows no clear relationship between temperature and best performance.
- **Interpretation**: Temperature increases iteration speed as expected (the G term), but this does not translate into better outcomes. High-temp agents explore faster but make more harmful modifications. The key insight: speed of exploration is decoupled from quality of exploration.
- **Caption requirements**: Must label each point with probe ID. Must note correlation coefficient.
- **Known caveats**: agent_0 temp varies across probes (0.3, 0.5, default). Not fully controlled.

## Figure 5: Strategy Categories (figure-05-strategy-categories.pdf)

- **Purpose**: Show which strategy categories agents use and which are successful.
- **Panels**: (A) Category frequency with success rate annotations, (B) val_bpb box plot by category.
- **Key observation**: Optimization is the most common category (121 runs) and the only one with >0% success rate (3.3%). Regularization (77 runs) has 1.3% success. Architecture (15 runs) and data_pipeline (7 runs) have 0%.
- **Interpretation**: With 60s training budgets, only fast-acting changes (LR) can show effect. Regularization, architecture changes, and data pipeline modifications need longer convergence time. The agent's strategy distribution is dominated by optimization and regularization — reasonable choices but with very low hit rates.
- **Caption requirements**: Must note that percentages are success rates (below baseline). Must note 60s training constraint.
- **Known caveats**: Strategy categorization is agent-reported. Some miscategorization is likely.

## Figure 6: Degradation and Convergence (figure-06-degradation-convergence.pdf)

- **Purpose**: Demonstrate the "G without epsilon = random walk" finding and compare convergence across configurations.
- **Panels**: (A) P11 full trajectory with oscillation cycles color-coded, (B) Cumulative best comparison across P03, P11, P12, P15.
- **Key observation**: Panel A shows P11's three degradation cycles (escalate → brief revert → re-escalate). Panel B shows P12 achieving steady cumulative improvement while P11 plateaus at its baseline. P15 (seeded) achieves the best start but doesn't improve beyond run 1.
- **Interpretation**: Without memory (P11), the agent cannot sustain corrections — it repeatedly makes the same class of mistakes (LR escalation). With shared memory (P12), the agent avoids repeating known failures. With seeding (P15), the initial configuration is optimal but further search degrades.
- **Caption requirements**: Must explain cycle coloring in panel A. Must note baseline line.
- **Known caveats**: P11 degradation trend is not statistically significant as linear (p=0.80) due to oscillation pattern.
