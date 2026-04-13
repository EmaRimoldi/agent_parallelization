# Protocol Upgrade Smoke Check

## Smoke-Test Setup

I ran a local synthetic smoke test that exercises the new protocol/logging path without needing the Claude CLI or a full training job.

Artifacts:

- `archives/pass_02_theory_validation_bundle_20260412/artifacts/protocol_smoke/smoke_summary.json`
- `archives/pass_02_theory_validation_bundle_20260412/artifacts/protocol_smoke/agent_smoke/results/training_runs.jsonl`
- `archives/pass_02_theory_validation_bundle_20260412/artifacts/protocol_smoke/agent_smoke/results/reevaluations.jsonl`
- `archives/pass_02_theory_validation_bundle_20260412/artifacts/protocol_smoke/agent_smoke/results/turns.jsonl`
- `archives/pass_02_theory_validation_bundle_20260412/artifacts/protocol_smoke/agent_smoke/results/metadata.json`
- `archives/pass_02_theory_validation_bundle_20260412/artifacts/protocol_smoke/agent_smoke/snapshots/step_001/metadata.json`
- `archives/pass_02_theory_validation_bundle_20260412/artifacts/protocol_smoke/agent_smoke/snapshots/step_002/metadata.json`

The smoke test created:

1. a baseline incumbent candidate A
2. a new candidate B that beats A on one primary evaluation
3. a reevaluation of candidate B using the same commit

Expected result:

- candidate B should be queued for reevaluation after the first win
- the reevaluation should stay tied to the same `candidate_id`
- promotion should occur only after the reevaluation event resolves

## Confirmed Fields

### Snapshot-level observables

The generated snapshot metadata contains:

- `strategy_category`
- `shared_memory_entries_visible`
- `prior_trace_entries_visible`

Observed examples:

- step 1: `strategy_category = "optimization"`
- step 2: `strategy_category = "regularization"`

### Training-run provenance

Each row in `training_runs.jsonl` contains:

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

Observed behavior:

- run 2 logged `promotion_decision = "provisional_pending_reevaluation"`
- run 3 logged `evaluation_kind = "reevaluation"`
- run 3 kept the same `candidate_id` as run 2
- run 3 logged `promotion_decision = "promoted_after_reevaluation"`

### Reevaluation event log

`reevaluations.jsonl` contains both:

- a `queued` event for candidate B
- a `resolved` event with `decision = "promoted_after_reevaluation"`

This confirms the reevaluation path exists in code and produces structured logs.

### Cost variance summaries

`metadata.json` now contains aggregate variation fields:

- `turn_wall_clock_seconds_mean`
- `turn_wall_clock_seconds_std`
- `turn_total_tokens_mean`
- `turn_total_tokens_std`
- `training_run_wall_seconds_mean`
- `training_run_wall_seconds_std`
- `training_run_seconds_mean`
- `training_run_seconds_std`

Observed values in the smoke test are non-null and populated.

## Verdict

The smoke test confirms all required protocol/logging upgrades are present at the artifact level:

- reevaluation path exists
- repeated runs stay tied to a stable candidate identity
- cost variation is recoverable from logs
- routing/strategy observables are present
- aggregate metadata contains non-degenerate variance summaries
