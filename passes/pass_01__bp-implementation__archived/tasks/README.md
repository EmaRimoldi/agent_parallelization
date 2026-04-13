# Pass 01: BP 2x2 Implementation Tasks

This directory is the first AI task pass. It contains the task queue that built the repository from the initial implementation guides through the original pilot experiment.

## Scope

This pass covered:

- CIFAR-10 substrate setup in `autoresearch/`
- model passthrough and JSON output parsing
- token, context, and training-run instrumentation
- external-memory and shared-memory experiment modes
- mode labeling and decomposition scripts
- CPU-only config routing
- the original Task 9 pilot execution and artifact aggregation

## Files

- `task_00.md` through `task_09.md`

## Main Artifact Families Created By This Pass

- `autoresearch/`
- `configs/`
- `scripts/label_modes.py`
- `scripts/compute_decomposition.py`
- `runs/experiment_exp_*`
- `runs/experiment_pilot_*`
- `results/`
- `runs/pilot_mapping.json`

## Runner

The original unattended runner remains at repository root:

```bash
./run_tasks.sh
```

It now reads from `ai_task_passes/pass_01_bp_implementation/` instead of the old top-level `tasks/` directory.
