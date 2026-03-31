Research task for AGENT_ID={{AGENT_ID}} in RUN_ID={{RUN_ID}} (experiment: {{EXPERIMENT_ID}}).

IMPORTANT — setup is already complete. Do NOT repeat the setup steps.
- You are already on branch: {{BRANCH}}
- `train.py`, `prepare.py`, and all required files are in your current directory.
- Data is already present (symlinked from the shared data directory).
- Initialize `results/results.tsv` with just the header row, then start experimenting immediately.

Session parameters:
- Time budget: {{TIME_BUDGET}} minutes
- Training time per run: {{TRAIN_TIME_BUDGET}}s (~{{TRAIN_TIME_BUDGET_MIN}} min)
- To run training: `./run_training.sh`
  This starts training in the background and returns immediately.
- To check if training is done: `./check_training.sh`
  It prints TRAINING RUNNING (with progress) or TRAINING DONE (with val_bpb).
- Training takes ~{{TRAIN_TIME_BUDGET}}s plus ~120s for compilation/eval. Wait for it.
- WORKFLOW: `./run_training.sh` → wait ~{{TRAIN_TIME_BUDGET_MIN}} min → `./check_training.sh` → read results → edit train.py → repeat.
- Environment vars are set: RUN_ID={{RUN_ID}} AGENT_ID={{AGENT_ID}}

Your first run should establish the baseline: run `./run_training.sh` on the unmodified `train.py`,
then read the result, log it, and start experimenting from there.

Do NOT use git merge commands or try to access other agents' workspaces.
Do NOT pause to ask the human if you should continue.
The loop runs until the human interrupts you, period.

Start now.
