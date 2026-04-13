# Results

This directory contains the canonical tracked outputs of the **Pass 01 original 2x2 pilot**, the unified figure archive used across the repository, and imported historical swarm-analysis artifacts.

It is intentionally narrower than the rest of the repo:

- it is **not** the full archive of every experiment;
- it is **not** where the workflow calibration stores its raw artifacts;
- it is **not** the full Pass 02 theory-validation bundle.

## What Lives Here

| File or folder | Meaning |
| --- | --- |
| [`pilot_summary.md`](pilot_summary.md) | main human-readable summary of the original Pass 01 pilot |
| [`pilot_raw_data.json`](pilot_raw_data.json) | aggregated pilot data across all four cells and three repetitions |
| [`decomposition_rep1.json`](decomposition_rep1.json) | first repetition decomposition output from the original pilot pipeline |
| [`decomposition_rep2.json`](decomposition_rep2.json) | second repetition decomposition output |
| [`decomposition_rep3.json`](decomposition_rep3.json) | third repetition decomposition output |
| [`figures/`](figures/) | repository-wide tracked figure archive, now split by pass / phase |
| [`imported_swarms/`](imported_swarms/) | historical analyses moved from the cloned `agents-swarms` repository |

## Important Interpretation Note

The JSON decomposition files here are the **original Pass 01 outputs**. They are historically important, but they come from the earlier decomposition pipeline where:

- `phi` was effectively zero,
- `G` and `epsilon` were not yet correctly identified,
- the pilot was later judged too optimistic / too weakly identified on its own.

So these files should be read as **archival pilot outputs**, not as the final corrected theorem evidence.

## Where The Other Results Live

If you are looking for something else, use this map:

- raw experiment directories: [`../runs/`](../runs/)
- workflow calibration and decision-gate artifacts: [`../workflow/artifacts/`](../workflow/artifacts/)
- full second-pass theory-validation bundle: [`../archives/pass_02_theory_validation_bundle_20260412/`](../archives/pass_02_theory_validation_bundle_20260412/)
- figure index: [`figures/README.md`](figures/README.md)
- imported swarm analyses: [`imported_swarms/README.md`](imported_swarms/README.md)

## Imported Swarm Results

[`imported_swarms/`](imported_swarms/) contains the historical results that were originally under `agents-swarms/analysis/`.

These are related to the current experiments, but they are not the same BP 2x2 evidence set:

- Similarity: they study two agents optimizing `val_bpb` with shared information.
- Difference: they use an explicit swarm blackboard with claim / publish / pull-best coordination, whereas native `d11` uses the current repo's lighter `parallel_shared` log and prompt-injection mechanism.
- Difference: they include a model comparison across Haiku, Sonnet, and Opus in a 2-agent swarm setup.
- Difference: their historical budgets and raw-run assumptions differ from the current tracked `d00` / `d10` / `d01` / `d11` configs.

Treat them as archived swarm evidence and context for the imported blackboard implementation, not as directly normalized rows in the current BP 2x2 decomposition.

## Figures

All tracked figures now live under [`figures/`](figures/).

Subfolders:

- [`figures/pass_01_pilot/`](figures/pass_01_pilot/)
  original pilot figures from Pass 01
- [`figures/pass_02_workflow_calibration/`](figures/pass_02_workflow_calibration/)
  Phase 02 workflow calibration and gate figures from Pass 02
- [`figures/pass_02_archive_overview/`](figures/pass_02_archive_overview/)
  repository-level archive figures used by the top-level README

## Recommended Reading Order

If you opened this folder first:

1. Read [`pilot_summary.md`](pilot_summary.md).
2. If needed, inspect [`pilot_raw_data.json`](pilot_raw_data.json).
3. Then read the top-level [`../README.md`](../README.md) to see how these pilot results relate to Pass 02 follow-ups and the workflow calibration.
