Research task for AGENT_ID={{AGENT_ID}} in RUN_ID={{RUN_ID}} (experiment: {{EXPERIMENT_ID}}).

Setup is already complete:
- You are already on branch: {{BRANCH}}
- `train.py`, `prepare.py`, and all required files are in your current directory.
- Data is already present (symlinked from the shared data directory).

**Before the loop — read the in-scope files once for full context:**
```bash
cat README.md    # repository context and project overview
cat prepare.py   # fixed constants, tokenizer, dataloader, evaluation — do NOT modify
cat train.py     # the file you modify: architecture, optimizer, hyperparameters, training loop
```
Then initialize `results/results.tsv` with just the header row and start experimenting.

Session parameters:
- Time budget: {{TIME_BUDGET}} minutes
- Training time per run: ~{{TRAIN_TIME_BUDGET_MIN}} min ({{TRAIN_TIME_BUDGET}}s + ~120s compile/eval)
- Environment vars: RUN_ID={{RUN_ID}} AGENT_ID={{AGENT_ID}}

WORKFLOW — follow this exactly:

**Step 0 (once, before anything else):**
```
WORKER_JOB_ID=$(bash start_gpu_worker.sh)
echo "Worker: $WORKER_JOB_ID"
```
This allocates a dedicated GPU for your entire session. Do it once and keep $WORKER_JOB_ID.

**Each iteration:**
1. Edit `train.py` (one scalar hyperparameter change)
2. `git commit -am "description"`
3. `python save_snapshot.py $STEP "hypothesis" "expected_effect" [prev_val_bpb]`
4. `bash run_on_worker.sh` — **blocks** until training completes, prints val_bpb directly
5. Read val_bpb from output; if crash: `tail -50 logs/train_current.out`
6. Append result to `results/results.tsv`
7. `python update_snapshot.py $STEP <val_bpb> <true|false> "reason" "next_step"`
8. Keep commit if improved, `git reset --hard HEAD~1` if not; increment STEP

**At end (if interrupted):**
```
bash stop_gpu_worker.sh $WORKER_JOB_ID
```

Your first run should establish the baseline: run step 0, then run `bash run_on_worker.sh` on the unmodified `train.py`, log the result, and start experimenting.

Do NOT use git merge commands or try to access other agents' workspaces.
Do NOT pause to ask the human if you should continue.
The loop runs until the human interrupts you, period.

Start now.
