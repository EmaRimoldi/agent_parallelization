# AI Task Passes

This directory groups the two major agentic task passes that produced the current codebase, experiment archive, and theory bundle.

## Structure

- `pass_01_bp_implementation/`
  First AI pass: buildout of the BP 2x2 instrumentation stack, substrate, experiment modes, and original pilot execution tasks.
- `pass_02_theory_formalization/`
  Second AI pass: theorem refactor, protocol upgrades, estimator repair, targeted follow-up experiments, and reviewer bundle finalization.
- `run_pass_02_theory_formalization.sh`
  Canonical sequential runner for the second pass task queue.

## Pass-to-Artifact Map

| AI pass | What the pass did | Main experiment families | Main outputs |
| --- | --- | --- | --- |
| `pass_01_bp_implementation` | built the CIFAR-10 substrate, mode routing, memory / parallel modes, decomposition scripts, and original pilot workflow | pre-pilot smoke runs, original 2x2 pilot | `autoresearch/`, `configs/`, `scripts/`, `runs/experiment_exp_*`, `runs/experiment_pilot_*`, `results/`, `runs/pilot_mapping.json` |
| `pass_02_theory_formalization` | narrowed the theorem, upgraded protocol/logging, repaired estimators, and reran targeted validation experiments | noise assay, replicated incumbent means, cost variance, context feasibility, deterministic calibration, d00/d10 workflow gate, ongoing second-pass exploration | `theory_validation_bp_20260412/`, `workflow/`, `theory_validation_bp_20260412/theory/autoresearch_bp_revised.pdf`, `runs/experiment_calibration_*` |

## Traceability Rule

If you want to know why an experiment or result exists:

1. identify the artifact family under `results/`, `workflow/`, `theory_validation_bp_20260412/`, or `runs/`
2. map it to the AI pass above
3. open the task Markdown files inside that pass to see the exact implementation or validation instructions that created it

The repository README now mirrors this same pass-based organization so the archive stays navigable from the top level.
