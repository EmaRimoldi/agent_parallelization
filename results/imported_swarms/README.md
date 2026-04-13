# Imported Swarm Results

Historical analysis artifacts moved from `agents-swarms/analysis/` on 2026-04-13.

These results are conceptually close to the current shared-memory experiments because they study two agents optimizing `val_bpb` while sharing information. They are not directly equivalent to the current BP 2x2 runs: the imported swarm uses a stronger blackboard protocol with claim / publish / pull-best coordination, while native `d11` uses the current repo's `parallel_shared` shared-log and prompt-injection mechanism.

Contents:

- `analysis/analyze_swarm.py`: archived swarm analysis script.
- `analysis/experiment_exp_20260405_022850/`: report and figures for one historical swarm run.
- `analysis/model_comparison_20260406/`: archived model comparison analysis, figures, CSV, and JSON summaries.
- `analysis/swarms_vs_parallelisation/`: archived swarm-vs-parallelisation comparison, figures, and summary.

Interpretation guide:

- `analysis/swarms_vs_parallelisation/` compares a historical independent-parallel baseline against historical 2-agent swarm runs.
- `analysis/experiment_exp_20260405_022850/` explains one swarm run through trajectory plots, shared-memory event timing, cross-agent influence, parameter exploration, and attribution.
- `analysis/model_comparison_20260406/` compares Haiku 4.5, Sonnet 4.6, and Opus 4.6 in the historical 2-agent swarm setting.
- These artifacts are useful context for the imported blackboard implementation under `src/agent_parallelization_new/imported_swarms/`.
- They should not be read as already normalized `d00` / `d10` / `d01` / `d11` rows for the current BP decomposition.

No local `agents-swarms/runs/` directory was present when these artifacts were moved. If raw cloned swarm run directories are restored later, place them under `results/imported_swarms/runs/` so the imported analysis scripts resolve paths consistently.
