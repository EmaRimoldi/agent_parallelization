# Validation & Pilot Experiment Protocol

## Purpose

This document is to be executed AFTER all Tasks (0–8) from `docs/guides/IMPLEMENTATION_GUIDE.md` are complete and committed. It has three phases: validate the instrumentation works end-to-end, run a miniature 2×2 pilot, and produce a first draft of results with the four-term decomposition.

---

## Phase 1: Validation — does everything work?

Run these checks in order. Each one must pass before moving to the next. If something fails, fix it before continuing.

### 1.1 Substrate sanity check

```bash
cd autoresearch
python prepare.py          # should print "Data ready."
python train.py            # should complete in ~2 min, print val_bpb: X.XXXXXX
```

Verify the output contains all expected fields: `val_loss`, `val_bpb`, `val_accuracy`, `training_seconds`, `total_seconds`, `total_steps`, `total_epochs`, `param_count`.

If `val_bpb` is missing, the framework's log parser will fail downstream. Do not proceed until this prints correctly.

### 1.2 Single agent smoke test (d00)

Run a single agent for 10 minutes (enough for ~3–4 training iterations):

```bash
run-single-long --time-budget 10 --config configs/experiment_cifar10.yaml
```

After it finishes, verify these files exist in the experiment directory (`runs/experiment_<id>/`):

```
mode_single_long/agent_0/results/metadata.json       # must have total_input_tokens, total_output_tokens, avg_context_fill
mode_single_long/agent_0/results/turns.jsonl          # one record per LLM turn
mode_single_long/agent_0/results/training_runs.jsonl  # one record per training run
mode_single_long/agent_0/results/trajectory.jsonl     # step/val_bpb pairs
mode_single_long/agent_0/snapshots/step_001/          # train.py + metadata.json
mode_single_long/agent_0/reasoning/trace.jsonl        # reasoning entries
mode_single_long/agent_0/workspace/results/results.tsv  # agent-written results
```

Open `turns.jsonl` and verify each record has: `turn`, `timestamp`, `input_tokens` (or estimate), `output_tokens` (or estimate), `context_fill_ratio`, `wall_clock_seconds`.

Open `training_runs.jsonl` and verify each record has: `run_index`, `wall_seconds`, `val_bpb`, `status`.

Open `metadata.json` and verify it has: `total_input_tokens`, `total_output_tokens`, `total_turns`, `avg_context_fill`, `final_context_fill`, `model`.

If any of these are missing or malformed, the decomposition script will fail. Fix before proceeding.

### 1.3 Single agent with memory smoke test (d10)

```bash
run-single-memory --time-budget 10 --config configs/experiment_cifar10.yaml
```

After it finishes, verify the same files as 1.2 exist. Additionally:

- Open `logs/task_runner/` or `run_agent.log` for this agent
- Confirm that from turn 2 onward, the agent's input message contains the experiment log table (lines starting with `| # | change | bpb |`)
- Compare `avg_context_fill` with d00's `avg_context_fill` — d10 should have a lower or similar value despite having extra context from the memory table, because the memory replaces information that would otherwise accumulate in the conversation context

### 1.4 Parallel agents smoke test (d01)

```bash
run-parallel --n-agents 2 --time-budget 10 --config configs/experiment_cifar10.yaml
```

Verify:
- Two agent directories exist: `mode_parallel/agent_0/` and `mode_parallel/agent_1/`
- Both have the full set of output files (turns.jsonl, training_runs.jsonl, etc.)
- The two agents' `reasoning/trace.jsonl` files show different experiments (they should not be identical since they work independently)
- No `shared_results_log.jsonl` exists at the experiment level (this is d01, not d11)

### 1.5 Parallel agents with shared memory smoke test (d11)

```bash
run-parallel-shared --n-agents 2 --time-budget 10 --config configs/experiment_cifar10.yaml
```

Verify:
- Two agent directories exist
- `shared_results_log.jsonl` exists at the experiment level and contains entries from BOTH agents (check `agent_id` field)
- From turn 2 onward, each agent's input message contains the shared experiment log with entries from the other agent
- The two agents' experiments show less overlap than in d01 (they should avoid duplicating each other's work thanks to shared info)

### 1.6 Mode labeling smoke test

Run the mode labeler on any completed experiment from above:

```bash
python scripts/label_modes.py --experiment-dir runs/experiment_<d00_id>
```

Verify:
- `mode_labels.jsonl` is created in each agent's `results/` directory
- Each entry has `step`, `mode`, `diff_lines_changed`, `hypothesis`, `val_bpb_after`, `accepted`
- The `mode` field is one of: `optimizer`, `lr_schedule`, `architecture`, `batch_data`, `other`
- Spot-check 5 entries: read the hypothesis and the assigned mode, verify they match

### 1.7 Decomposition script dry run

Create a fake decomposition run using the smoke test outputs:

```bash
python scripts/compute_decomposition.py \
    --d00 runs/experiment_<d00_id> \
    --d10 runs/experiment_<d10_id> \
    --d01 runs/experiment_<d01_id> \
    --d11 runs/experiment_<d11_id>
```

Verify:
- The script runs without errors
- The decomposition table prints with values for all 4 terms (some may be 0.0 or NaN on this small data — that's OK)
- `decomposition_results.json` is written
- Hypothesis test fields are all populated (values may be meaningless on 10-min runs)

If all seven checks pass, the instrumentation is validated. Proceed to Phase 2.

---

## Phase 2: Pilot 2×2 Experiment

### 2.1 Experiment parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Time budget per cell | 30 minutes | ~10–12 training iterations per agent at ~2.5 min/cycle |
| Number of agents (d01, d11) | 2 | Minimum for parallelism contrast |
| Number of agents (d00, d10) | 1 | Single agent cells |
| Model | Same for all cells (set in YAML) | M held fixed per paper |
| Training time per run | 120 seconds | CPU CIFAR-10 substrate |
| Independent repetitions | 3 per cell | Minimum for bootstrap CIs |
| Total cells | 4 | Full 2×2 factorial |
| Total runs | 12 (4 cells × 3 reps) | |

### 2.2 Create per-cell config files

Create four YAML configs. They should be identical except for `mode` and `agents.n`:

**`configs/experiment_d00.yaml`** — single_long, n=1, no memory flags
**`configs/experiment_d10.yaml`** — single_memory, n=1, use_external_memory=true
**`configs/experiment_d01.yaml`** — parallel, n=2, no shared memory
**`configs/experiment_d11.yaml`** — parallel_shared, n=2, use_shared_memory=true

All must share: same `agents.model`, same `agents.time_budget_minutes: 30`, same `agents.train_time_budget_seconds: 120`.

### 2.3 Run the pilot

Execute all 12 runs. Order does not matter, but for reproducibility, interleave cells across repetitions:

```bash
# Repetition 1
run-single-long    --time-budget 30 --config configs/experiment_d00.yaml  # d00 rep1
run-single-memory  --time-budget 30 --config configs/experiment_d10.yaml  # d10 rep1
run-parallel       --n-agents 2 --time-budget 30 --config configs/experiment_d01.yaml  # d01 rep1
run-parallel-shared --n-agents 2 --time-budget 30 --config configs/experiment_d11.yaml  # d11 rep1

# Repetition 2 (same commands, new experiment IDs auto-generated)
run-single-long    --time-budget 30 --config configs/experiment_d00.yaml
run-single-memory  --time-budget 30 --config configs/experiment_d10.yaml
run-parallel       --n-agents 2 --time-budget 30 --config configs/experiment_d01.yaml
run-parallel-shared --n-agents 2 --time-budget 30 --config configs/experiment_d11.yaml

# Repetition 3
run-single-long    --time-budget 30 --config configs/experiment_d00.yaml
run-single-memory  --time-budget 30 --config configs/experiment_d10.yaml
run-parallel       --n-agents 2 --time-budget 30 --config configs/experiment_d01.yaml
run-parallel-shared --n-agents 2 --time-budget 30 --config configs/experiment_d11.yaml
```

After each run, label modes:
```bash
python scripts/label_modes.py --experiment-dir runs/experiment_<id>
```

### 2.4 Estimated time and cost

- Per single-agent run: ~30 min wall-clock + ~5 min overhead = ~35 min
- Per parallel run: ~30 min wall-clock + ~5 min overhead = ~35 min (parallel agents overlap)
- Total for 12 runs: ~7 hours sequential
- Token cost: depends on plan. On Pro (~17.6k tokens/5hr window), this will span multiple sessions. On Max 5x, feasible in one sitting. On API, estimate ~$15–30 total in Sonnet tokens.

### 2.5 Post-pilot data collection

After all 12 runs complete, organize experiment directories by cell:

```bash
# Create a mapping file
cat > runs/pilot_mapping.json << 'EOF'
{
  "d00": ["runs/experiment_<d00_rep1>", "runs/experiment_<d00_rep2>", "runs/experiment_<d00_rep3>"],
  "d10": ["runs/experiment_<d10_rep1>", "runs/experiment_<d10_rep2>", "runs/experiment_<d10_rep3>"],
  "d01": ["runs/experiment_<d01_rep1>", "runs/experiment_<d01_rep2>", "runs/experiment_<d01_rep3>"],
  "d11": ["runs/experiment_<d11_rep1>", "runs/experiment_<d11_rep2>", "runs/experiment_<d11_rep3>"]
}
EOF
```

Fill in the actual experiment IDs.

---

## Phase 3: Analysis and First Draft of Results

### 3.1 Run the decomposition on each repetition

For each of the 3 repetitions, run:

```bash
python scripts/compute_decomposition.py \
    --d00 runs/experiment_<d00_repN> \
    --d10 runs/experiment_<d10_repN> \
    --d01 runs/experiment_<d01_repN> \
    --d11 runs/experiment_<d11_repN>
```

Save each output as `decomposition_rep1.json`, `decomposition_rep2.json`, `decomposition_rep3.json`.

### 3.2 Aggregate across repetitions

Write a script `scripts/aggregate_pilot.py` that:

1. Loads the 3 decomposition JSONs
2. For each term (cost_token, cost_wall, φ, G, ε) and each cell (d10, d01, d11), computes:
   - Mean across 3 reps
   - Standard deviation
   - 95% bootstrap confidence interval (resample 1000 times from the 3 values)
3. For each hypothesis H1–H6, counts how many reps support it (0/3, 1/3, 2/3, 3/3)
4. Produces a summary table and writes it to `results/pilot_summary.md`

### 3.3 Expected output format

The `pilot_summary.md` should contain:

#### Decomposition Table (mean ± std across 3 reps)

```
Cell   log(κ0/κ)_tok   log(κ0/κ)_wall   φ        G        -ε       Δ_tok    Δ_wall
─────────────────────────────────────────────────────────────────────────────────────
d10    X.XX ± X.XX     X.XX ± X.XX      X.XX     X.XX     X.XX     X.XX     X.XX
d01    X.XX ± X.XX     X.XX ± X.XX      X.XX     X.XX     X.XX     X.XX     X.XX
d11    X.XX ± X.XX     X.XX ± X.XX      X.XX     X.XX     X.XX     X.XX     X.XX
```

#### Hypothesis Verdicts

```
H1 (parallelism helps only wall-clock):         X/3 reps support
H2 (memory helps both axes):                    X/3 reps support
H3 (shared memory lowers ε):                    X/3 reps support
H4 (parallelism sensitive to coordination):     X/3 reps support
H5 (context pressure dominant):                 X/3 reps support
H6 (d11 dominates d00 on both axes):            X/3 reps support
```

#### Context Pressure Analysis (H5)

For each cell, the κ̄_token stratified by c/K quartile:

```
Cell   0-25%    25-50%   50-75%   75-100%   Monotone?
d00    XXXX     XXXX     XXXX     XXXX      yes/no
d10    XXXX     XXXX     XXXX     XXXX      yes/no
d01    XXXX     XXXX     XXXX     XXXX      yes/no
d11    XXXX     XXXX     XXXX     XXXX      yes/no
```

#### Raw Metrics

For each cell, basic stats:
- Total training runs completed
- Best val_bpb achieved
- Total tokens consumed
- Mean wall-clock per attempt

#### Interpretation

A brief paragraph per hypothesis interpreting the results:
- Which terms drove the outcome?
- Were there surprises?
- What would change with more data / longer runs?
- Does the decomposition add explanatory power beyond raw best-so-far curves?

#### Negative Result Criterion (Section 7.7 of the paper)

Check: can the raw best-so-far curves across all 4 cells be explained by a single scalar efficiency metric with R² > 0.9? If yes, the decomposition has added no explanatory power and this should be reported as a negative result. Compute R² by fitting `best_val_bpb = f(total_tokens_consumed)` as a simple regression across cells and reporting the fit.

### 3.4 Deliverables

At the end of Phase 3, the following files should exist:

```
results/
├── decomposition_rep1.json
├── decomposition_rep2.json
├── decomposition_rep3.json
├── pilot_summary.md          # main results document
├── pilot_raw_data.json       # all per-turn, per-run data aggregated
└── pilot_figures/             # optional: plots if matplotlib is available
    ├── kappa_by_context_bin.png
    ├── best_so_far_curves.png
    └── decomposition_bar_chart.png
```

### 3.5 Figures (optional but recommended if matplotlib is available)

If `matplotlib` is installed, generate:

1. **Best-so-far curves**: one line per cell (d00, d10, d01, d11), x-axis = cumulative tokens, y-axis = best val_bpb so far. This is the primary visual comparison.

2. **κ̄ by context bin**: bar chart, one group per cell, bars for each c/K quartile. Should show monotone increase for d00, flat or lower for d10.

3. **Decomposition stacked bar**: for each cell (d10, d01, d11), a stacked bar showing the 4 terms. Helps visualize which terms dominate.

4. **Mode distribution**: pie or bar chart showing the empirical π̂ (what fraction of accepted edits fell in each mode).

---

## Checklist before declaring Phase 3 complete

- [ ] All 12 runs completed without framework errors
- [ ] Mode labels assigned to all experiments
- [ ] 3 decomposition JSONs generated
- [ ] `pilot_summary.md` written with all sections
- [ ] Negative result criterion evaluated
- [ ] All deliverable files committed to the branch
- [ ] Summary interpretation written

If the pilot succeeds (decomposition terms are measurable, ≥4/6 hypotheses have consistent direction across reps, negative result criterion is not met), this constitutes preliminary evidence that the BP framework applies to continuous-verifier agentic settings. The next step is a full-scale experiment with 200 attempts per cell and 5 repetitions.

If the pilot fails (terms are noise-dominated, hypotheses flip across reps, R² > 0.9 for the single-scalar fit), report this as a negative result per Section 7.7 of the paper and analyze which of the three extensions (noisy verifier, wall-clock/cost split, context pressure) was the wrong abstraction.
