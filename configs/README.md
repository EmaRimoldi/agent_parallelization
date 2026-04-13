# Configs

This directory contains the launcher and experiment configuration files for the BP 2x2 repository.

## What Lives Here

There are two kinds of config files:

- JSON defaults used by the older launcher paths.
- YAML experiment configs used by the current `d00` / `d10` / `d01` / `d11` experiment modes.

## File Guide

| File | Role | Notes |
| --- | --- | --- |
| [`agent_default.json`](agent_default.json) | legacy per-agent defaults | default model, agent time budget, train budget, CUDA device |
| [`experiment_default.json`](experiment_default.json) | legacy experiment defaults | default experiment-level settings such as `autoresearch_dir` and `results_root` |
| [`experiment.yaml`](experiment.yaml) | general template | full commented template showing every launcher option, including SLURM, memory flags, and per-agent overrides |
| [`experiment_cifar10.yaml`](experiment_cifar10.yaml) | CPU substrate baseline | compact local CPU config for the CIFAR-10 substrate, with `slurm.enabled: false` |
| [`experiment_d00.yaml`](experiment_d00.yaml) | single-agent baseline cell | `mode: single_long`, `n: 1`, no memory |
| [`experiment_d10.yaml`](experiment_d10.yaml) | single-agent external-memory cell | `mode: single_memory`, `use_external_memory: true` |
| [`experiment_d01.yaml`](experiment_d01.yaml) | two-agent parallel cell | `mode: parallel`, `n: 2`, no shared memory |
| [`experiment_d11.yaml`](experiment_d11.yaml) | two-agent shared-memory cell | `mode: parallel_shared`, `n: 2`, `use_shared_memory: true` |
| [`imported_swarms/experiment_imported_swarm.yaml`](imported_swarms/experiment_imported_swarm.yaml) | imported agents-swarms blackboard mode | additive `mode: imported_swarm`; does not replace `d11` |

## How To Read These Configs

The important fields are:

- `experiment.mode`
  selects the architecture family being tested.
- `agents.n`
  number of concurrent agents.
- `agents.model`
  Claude model used for the agent workers.
- `agents.time_budget_minutes`
  wall-clock budget per agent session.
- `agents.train_time_budget_seconds`
  budget for a single `autoresearch/train.py` run.
- `agents.use_external_memory`
  enables the single-agent memory table used in `d10`.
- `agents.use_shared_memory`
  enables the shared log used in `d11`.
- `slurm.enabled`
  `true` for cluster-backed workers, `false` for local CPU-only runs.

## Important Historical Note

These files are the **current tracked configs**, not a perfect snapshot of every historical run.

- The original Pass 01 pilot described in [`../docs/guides/PILOT_EXPERIMENT_PROTOCOL.md`](../docs/guides/PILOT_EXPERIMENT_PROTOCOL.md) used a `30 min` / `120 s` pilot regime.
- The currently tracked `experiment_d00.yaml` through `experiment_d11.yaml` reflect the later second-pass exploratory continuation, which currently uses `45 min` / `60 s`.

If you are trying to understand an old experiment, always cross-check:

- the run directory under [`../runs/`](../runs/)
- the protocol or workflow docs
- the archive bundle under [`../archives/`](../archives/)

## Which File To Use

- For a readable reference of all supported options, start with [`experiment.yaml`](experiment.yaml).
- For local CPU substrate work, start with [`experiment_cifar10.yaml`](experiment_cifar10.yaml).
- For the 2x2 cells, use the specific `experiment_d00.yaml` through `experiment_d11.yaml` files.
- For the imported blackboard swarm prototype, use [`imported_swarms/experiment_imported_swarm.yaml`](imported_swarms/experiment_imported_swarm.yaml).
