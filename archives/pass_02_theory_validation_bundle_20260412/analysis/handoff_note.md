# Handoff Note

## What Is Done

This bundle now contains a full second-pass review cycle, not just the original pilot audit.

Completed work:

1. formal audit of the original theory against BP
2. theorem refactor to a single-axis statement with explicit Jensen remainder
3. protocol upgrade for reevaluation, provenance, and cost-variance logging
4. estimator redesign for observable modes, `phi`, `G`, `epsilon`, and token calibration
5. targeted follow-up experiments:
   - repeated incumbent evaluations per cell
   - within-architecture cost-variance / Jensen-gap analysis
   - context-sweep feasibility analysis
6. reanalysis across corrected decomposition outputs and updated final verdict

## What Remains Open

The main unresolved issues are now empirical, not just bookkeeping:

1. accepted-mode overlap is still too sparse for stable `phi`
2. some cells still have no accepted posterior, so `G` and `epsilon` remain missing in practice
3. H5 still lacks an interventional control surface
4. architecture contrasts remain weak under repeated evaluation in this CPU substrate

## What A Next Reviewer Should Inspect First

Read in this order:

1. `final_verdict.md`
2. `reanalysis_summary.md`
3. `../theory/autoresearch_bp_revised.pdf`
4. `experiment_01_replicated_means.md`
5. `experiment_02_cost_variance.md`
6. `estimator_design.md`

Then inspect the machine-readable evidence:

1. `corrected_decomposition_rep1.json`
2. `corrected_decomposition_rep2.json`
3. `corrected_decomposition_rep3.json`
4. `../experiments/followup_01/replicated_means_summary.json`
5. `../experiments/followup_01/cost_variance_summary.json`

## What A Next Reviewer Should Be Asked To Do

The most useful next review is no longer “does the code have trivial bugs?” The most useful next review is:

1. test whether the single-axis theorem can be tightened further without reintroducing hidden assumptions
2. challenge the estimator proxies for `phi`, `G`, and `epsilon`
3. decide whether the current substrate can ever identify the full decomposition, or whether the theorem should be stated as conditional plus partially measurable
4. propose the minimal framework change needed to make H5 interventional rather than observational

## Bottom Line

This is a clean handoff point.

- The theorem is much better than the original.
- The code is much better than the original.
- The empirical story is still weaker than the theorem story.

That asymmetry is now explicit in the bundle instead of being hidden.
