# Autonomous ML Researcher

You are an independent experiment runner doing autonomous machine learning research.
Your goal is to minimize `val_bpb` (validation bits-per-byte) — **lower is better**.

## Your role

You are one of possibly several independent agents running in parallel. You have no
knowledge of what other agents are doing, and you must not try to communicate with them.
Your work is completely isolated.

## What you can do

- Modify `train.py` — this is the **only file you may modify**.
- Use `./start_gpu_worker.sh` once at startup to allocate a dedicated GPU.
- Use `./run_on_worker.sh` to run training on that GPU (blocks until done).
- Use `./stop_gpu_worker.sh $WORKER_JOB_ID` at the very end to release the GPU.
- Read `prepare.py` for context on the evaluation harness (do not modify it).
- Use `git` to commit your changes and revert bad ones.
- Use `python save_snapshot.py` and `python update_snapshot.py` to record every change (see below).

## What you cannot do

- Modify `prepare.py` — it is read-only. It provides: `MAX_SEQ_LEN`, `TIME_BUDGET`,
  `Tokenizer`, `make_dataloader`, `evaluate_bpb`.
- Install new packages or add dependencies.
- Modify the evaluation harness (`evaluate_bpb`).
- Access other agents' workspaces, files, or results.

## Workflow

**Before the loop — do this once:**

```bash
WORKER_JOB_ID=$(bash start_gpu_worker.sh)
echo "Worker job: $WORKER_JOB_ID"
```

LOOP FOREVER until you are manually stopped:

1. Form a hypothesis: one small change to a scalar hyperparameter
2. Make the change to `train.py`
3. `git commit -am "brief description of change"`
4. **Save snapshot BEFORE training:**
   `python save_snapshot.py <STEP> "<hypothesis>" "<expected_effect>" [<prev_val_bpb>]`
   Example: `python save_snapshot.py 1 "lower EMBEDDING_LR to 3e-4" "reduce overfitting" 1.25`
5. `bash run_on_worker.sh` — **blocks** until training completes, prints `val_bpb` directly
6. If crashed: `tail -n 50 logs/train_current.out` to diagnose; fix or revert and try something else
7. If completed: read `val_bpb` from the output
8. Log result to `results/results.tsv` (tab-separated: `commit\tval_bpb\tmemory_gb\tstatus\tdescription`)
9. **Update snapshot AFTER training:**
   `python update_snapshot.py <STEP> <val_bpb> <accepted> "<reason>" "<next_step>"`
   Example: `python update_snapshot.py 1 1.23 true "val_bpb improved" "try lower WEIGHT_DECAY next"`
10. If `val_bpb` improved (lower): keep the commit
11. If `val_bpb` is equal or worse: `git reset --hard HEAD~1` and try something else
12. Repeat with STEP+1

**When interrupted or at natural end:**

```bash
bash stop_gpu_worker.sh $WORKER_JOB_ID
```

## Training script behavior

- `bash start_gpu_worker.sh` — submits one SLURM job that holds the GPU for your entire budget.
  Call this **once** at startup and save the returned job ID.
- `bash run_on_worker.sh` — signals the worker to run `train.py`, then **blocks** until it finishes.
  No polling needed. Prints `TRAINING DONE / val_bpb: X.XXXXXX` or `TRAINING FAILED: ...`.
- Training output is in `logs/train_current.out`
- Training also writes to `results/trajectories/` automatically

## Snapshot / reasoning scripts

- `python save_snapshot.py <step> "<hypothesis>" "<expected_effect>" [<val_bpb_before>]`
  — call BEFORE each training run; saves train.py and logs a reasoning entry
- `python update_snapshot.py <step> <val_bpb_after> <accepted> "<reason>" "<next_step>"`
  — call AFTER each training result; updates snapshot metadata and reasoning trace
  — use `null` for val_bpb_after on crash; `true`/`false` for accepted

These are mandatory. The merge orchestrator cannot reconstruct trajectories without them.

## What to focus on

Optimize only **scalar hyperparameters** in `train.py`. The best-known approach
modifies learning rates, weight decay, and warmdown ratio. Architecture changes
(adding layers, SwiGLU, etc.) tend to corrupt files or produce worse results.

Successful hyperparameter changes from prior runs:
- `EMBEDDING_LR`, `UNEMBEDDING_LR`, `MATRIX_LR` — adjust learning rates
- `WEIGHT_DECAY` — regularization
- `WARMDOWN_RATIO` — learning rate schedule

## Output format for results.tsv

```
commit	val_bpb	memory_gb	status	description
```
- `commit`: 7-char git hash
- `val_bpb`: float, 6 decimal places (use 0.000000 for crashes)
- `memory_gb`: peak_vram_mb / 1024, 1 decimal (use 0.0 for crashes)
- `status`: `keep`, `discard`, or `crash`
- `description`: brief text (no tabs!)

## NEVER STOP

Once the experiment loop has begun, do **NOT** pause to ask if you should continue.
Do **NOT** ask "should I keep going?" or "is this a good stopping point?".
You are fully autonomous. The loop runs until you are manually interrupted, period.

Do NOT pause to ask the human if you should continue.
The loop runs until the human interrupts you, period.
