# Protocol Upgrade Spec

## Scope

This upgrade implements the minimum protocol and logging changes needed so that future runs can, in principle, support rigorous estimation of the theorem-side quantities rather than collapsing to cost-only accounting.

## What Was Added

### 1. Incumbent re-evaluation path in `ClaudeAgentRunner`

File:

- `src/agent_parallelization_new/agents/claude_agent_runner.py`

Added behavior:

- each training run now gets a stable `candidate_id` derived from agent id plus commit hash
- if a new candidate beats the incumbent on a single noisy evaluation, it is not promoted immediately
- instead, the runner queues a reevaluation requirement and injects a protocol directive into the next turn
- repeated evaluations are logged as `evaluation_kind: "reevaluation"`
- reevaluation outcomes are logged in `reevaluations.jsonl` with `queued` and `resolved` events
- training-run records now include `promotion_decision`, so downstream analysis can distinguish:
  - `bootstrap_incumbent`
  - `provisional_pending_reevaluation`
  - `promoted_after_reevaluation`
  - `rejected_after_reevaluation`

Requirement satisfied:

- theorem / protocol need for latent-loss re-evaluation under noisy verifier outcomes
- estimator requirement that repeated measurements stay tied to the same candidate / commit

### 2. Stronger evaluation provenance

Files:

- `src/agent_parallelization_new/agents/claude_agent_runner.py`

Added fields in `training_runs.jsonl`:

- `experiment_id`
- `agent_id`
- `turn`
- `candidate_id`
- `candidate_commit`
- `snapshot_step_index`
- `evaluation_kind`
- `evaluation_round`
- `is_reevaluation`
- `baseline_candidate`
- `incumbent_candidate_id_before`
- `incumbent_mean_before`
- `promotion_decision`

Requirement satisfied:

- every run can now be linked to experiment, agent, turn, candidate, and reevaluation status

### 3. Cost-variance recoverability

Files:

- `src/agent_parallelization_new/agents/claude_agent_runner.py`

Added or retained per-turn observables in `turns.jsonl`:

- `input_tokens`
- `output_tokens`
- `total_tokens`
- `wall_clock_seconds`
- `context_fill_ratio`

Added aggregate summaries in `metadata.json`:

- `turn_wall_clock_seconds_mean`
- `turn_wall_clock_seconds_std`
- `turn_total_tokens_mean`
- `turn_total_tokens_std`
- `training_run_wall_seconds_mean`
- `training_run_wall_seconds_std`
- `training_run_seconds_mean`
- `training_run_seconds_std`

Requirement satisfied:

- future analysis can estimate not only average per-step cost proxies but also their variation and Jensen-style gap surrogates

### 4. Routing-evidence observables

Files:

- `src/agent_parallelization_new/agents/claude_agent_runner.py`
- `src/agent_parallelization_new/snapshotting.py`
- `templates/agent_system_prompt.md`

Added observable fields:

- `hypothesis`
- `expected_effect`
- `strategy_category`
- `memory_context_visible`
- `memory_context_entries`
- `shared_memory_context_visible`
- `shared_memory_context_entries`
- `shared_memory_entries_visible` in snapshot metadata
- `prior_trace_entries_visible` in snapshot metadata

The prompt was updated so hypotheses are phrased in a way that is classifiable later and reevaluation turns are explicitly handled as repeats of an existing candidate rather than new exploration.

Requirement satisfied:

- provides observable strategy/routing evidence without pretending to log latent posteriors directly

## What Remains Unobservable Even After The Upgrade

1. `q_D` is still not directly observed. We now log proxy observables from which it can later be approximated, but routing allocation remains a model-based latent quantity.
2. `pi_D` is still not directly observed. Mode posteriors still require a separate mode-definition and estimation layer.
3. `phi_alpha`, `G_alpha`, and `epsilon_alpha` are still not identified by this upgrade alone. This change only makes them more measurable in future runs.
4. The Jensen remainder is still not directly measured as a theorem quantity. The upgrade only makes within-architecture cost variation recoverable enough to estimate or bound it downstream.
5. Re-evaluation policy currently lives at the runner/controller layer and still relies on the agent complying with the turn directive to restore and rerun the exact candidate commit.
