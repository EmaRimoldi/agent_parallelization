# Task 8: Config routing, CPU worker scripts, and per-cell YAML configs

Read `IMPLEMENTATION_GUIDE.md` → Task 8 and `CPU_SUBSTRATE_GUIDE.md` → "Compatibility with the framework" for full specification.

## What to do

### 8a. Mode routing in launcher

In `src/agent_parallelization_new/launcher.py`, ensure all 4 modes are routable:

```python
MODES = {
    "single_long": main_single_long,           # d00
    "single_memory": main_single_memory,        # d10
    "parallel": main_parallel,                  # d01
    "parallel_shared": main_parallel_shared,    # d11
}
```

### 8b. CPU-only worker scripts

Modify `src/agent_parallelization_new/compatibility/training_harness.py` to support a `local_cpu` mode.

Add a config flag `slurm.enabled` (default `true`). When `slurm.enabled: false`:

- `start_gpu_worker.sh` should just create `gpu_allocated_at` and echo a dummy worker ID
- `run_on_worker.sh` should run `python train.py` directly (no SLURM), parse val_bpb from output, write `run.result`
- `stop_gpu_worker.sh` should be a no-op

Use the exact script contents from `CPU_SUBSTRATE_GUIDE.md` → "Worker scripts" section.

### 8c. Create per-cell YAML configs

Create 4 config files, identical except for mode and agent flags:

**`configs/experiment_d00.yaml`**: mode=single_long, n=1
**`configs/experiment_d10.yaml`**: mode=single_memory, n=1
**`configs/experiment_d01.yaml`**: mode=parallel, n=2
**`configs/experiment_d11.yaml`**: mode=parallel_shared, n=2

All must share: `slurm.enabled: false`, same model, `time_budget_minutes: 30`, `train_time_budget_seconds: 120`.

### 8d. Update agent prompts for CPU

In `templates/agent_system_prompt.md`, parameterize GPU/CPU references using existing template placeholders. The training time reference should use `{{TRAIN_TIME_BUDGET_MIN}}` instead of hardcoded "5 minutes".

## Success criteria
- All 4 modes are launchable from CLI
- CPU-only worker scripts are generated when `slurm.enabled: false`
- 4 per-cell YAML configs exist and are valid
- `run-single-long --config configs/experiment_d00.yaml` works on a CPU-only machine
- `pytest -q` passes with no new failures
- Committed with message "Task 8: Config routing, CPU workers, per-cell configs"

Do NOT proceed to other tasks.
