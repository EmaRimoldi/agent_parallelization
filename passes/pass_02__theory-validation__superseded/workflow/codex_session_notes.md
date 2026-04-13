# Codex Session Notes

## Scope and Constraints

- Read the full handoff in `workflow/artifacts/codex_handoff_prompt.md`.
- Confirmed the active calibration runner is still alive and untouched:
  - PID `36706`
  - command: `python workflow/scripts/run_calibration.py --repo-root . --reps 5 --cells d00 d10`
- Avoided touching in-progress `d10` experiment directories.
- Worked only on offline analysis prep, script validation, and next-phase readiness.

## Understanding of Current Experimental State

- The study is in **Phase 02 power calibration** on the `d00` vs `d10` contrast.
- `d00` is complete on disk with 5 reps and 47 total training runs.
- `d10` is partially complete and still being produced by the active runner.
- The intended decision point is Phase `02b_decision_gate`, based on:
  - effect size,
  - mode diversity,
  - sample size,
  - cost variance / Jensen gaps.

## Work Completed

### Workstream A: Analysis Infrastructure Validation

- Read:
  - `workflow/state.json`
  - `workflow/phases/*.md`
  - `workflow/scripts/analyze_calibration.py`
  - `workflow/scripts/check_decision_gate.py`
  - `workflow/scripts/run_calibration.py`
- Validated `workflow/scripts/analyze_calibration.py` against completed `d00` data only.
- Wrote smoke outputs:
  - `workflow/artifacts/calibration_analysis_d00_smoke.json`
  - `workflow/artifacts/calibration_analysis_d00_labeled.json`

### Workstream B: Mode Labeling Pipeline

- Found the original `scripts/label_modes.py` was aligned to sparse snapshot files, not to `training_runs.jsonl`.
- This caused two concrete failures on calibration data:
  - only a small fraction of attempts were labeled,
  - `accepted` was `null` in snapshot metadata, so accepted-mode counts collapsed to zero.
- Updated `scripts/label_modes.py` so that it now:
  - emits one label row per `training_runs.jsonl` record,
  - infers accepted edits from `promotion_decision == promoted_after_reevaluation`,
  - carries mode information across reevaluation rows via `candidate_id`,
  - uses snapshots only as optional diff enrichment,
  - adds a `training_loop` category and better `architecture` keyword coverage.
- Re-ran labeling on all completed `d00` reps.

### Workstream C: Four-Term Decomposition Readiness

- Verified the repo uses `scripts/compute_decomposition.py`, not `scripts/decompose_four_term.py`.
- Ran a decomposition smoke test using real `d00` data plus empty placeholders for the other cells.
- Confirmed the script:
  - parses current calibration logs,
  - returns clean `NaN` outputs rather than crashing when comparison cells are absent,
  - writes a machine-readable artifact:
    - `workflow/artifacts/decomposition_smoke_d00.json`

### Workstream D: d01/d11 Config Preparation

- Read:
  - `configs/experiment_d00.yaml`
  - `configs/experiment_d10.yaml`
  - `configs/experiment_d01.yaml`
  - `configs/experiment_d11.yaml`
  - `src/agent_parallelization_new/launcher.py`
  - `src/agent_parallelization_new/orchestrator.py`
  - `src/agent_parallelization_new/experiment_modes/*`
- Confirmed mode support exists for:
  - `parallel` (`d01`)
  - `parallel_shared` (`d11`)
- Updated only the non-running Phase 03 candidate configs:
  - `configs/experiment_d01.yaml`
  - `configs/experiment_d11.yaml`
- These are now pre-set to the Phase 03 budget:
  - `time_budget_minutes: 60`
  - `train_time_budget_seconds: 60`

### Workstream E: Visualization / Reporting Prep

- Figure plan prepared for the calibration summary:
  - box plot or violin plot for `val_bpb` by cell,
  - per-rep run trajectory plot (`run_index` vs `val_bpb`),
  - accepted mode distribution bar chart,
  - token vs wall Jensen-gap comparison.

## Key Findings

### 1. `workflow/state.json` is stale

The state file does not reflect the actual calibration progress on disk. It still reports older determinism measurements and does not record the completed `d00` calibration status described in the handoff prompt.

Practical implication:
- use on-disk artifacts in `runs/experiment_calibration_*` as the source of truth until state is synchronized.

### 2. Workflow-document drift is real

I found multiple mismatches between the DAG markdown and the repo as it exists now:

- Phase docs use `runs/calibration_*`, but actual dirs are `runs/experiment_calibration_*`.
- The handoff references `scripts/decompose_four_term.py`, but the live script is `scripts/compute_decomposition.py`.
- Phase 02/03 command examples use `--output-dir`, but the current launcher CLIs do not support that flag.
- `workflow/scripts/check_completeness.py` is referenced in Phase 03 but does not exist.
- `workflow/scripts/check_mode_coverage.py` is referenced in Phase 03a but does not exist.

Practical implication:
- follow the current Python launchers and actual on-disk naming, not the stale shell snippets in the phase docs.

### 3. The calibration analysis pipeline is now materially more usable

After patching `scripts/label_modes.py`, `d00` labeling now has full run coverage:

- total `d00` training runs: `47`
- total `d00` mode labels: `47`
- coverage ratio: `1.0`

### 4. `d00`-only calibration summary (current source-of-truth smoke)

From `workflow/artifacts/calibration_analysis_d00_labeled.json`:

- `n_runs = 47`
- `mean_val_bpb = 1.0025666`
- `std_val_bpb = 0.1095894`
- `min_val_bpb = 0.824019`
- `max_val_bpb = 1.251464`
- `distinct_modes = 6`
- `distinct_accepted_modes = 3`
- `modes_with_2plus_accepted = 1`
- `jensen_gap_token = 0.0109556`
- `jensen_gap_wall = 0.1961502`

Accepted promoted edits in `d00` now land in interpretable categories:

- `training_loop`: increase `MAX_STEPS` / epochs
- `architecture`: increase `BASE_CHANNELS`
- `optimization`: lower learning rate

### 5. Decision gate is not yet runnable

This is expected because `d10` is still in progress.

What is already clear from `d00` alone:
- the analysis scripts can run,
- the mode labeling issue was real and is partially fixed,
- the wall-clock Jensen gap is non-trivial,
- `modes_with_2plus_accepted` is still only `1` in `d00`, so diversity may remain a bottleneck even after `d10` completes.

## Bugs / Gaps That Still Need Attention

### High priority

- `workflow/state.json` should be synced to actual Phase 02 status once `d10` completes.
- The phase markdown files should be corrected for real paths / CLI usage.
- `check_completeness.py` and `check_mode_coverage.py` are still missing.

### Medium priority

- `analyze_calibration.py` currently treats each promoted reevaluation as an accepted label row, which is correct enough for calibration, but future decomposition work may want unique-candidate counting alongside row counting.
- Snapshot metadata remains sparse and often does not match the executed training-run commit; for calibration analysis this is now survivable because training-run labeling is primary.

## Immediate Next Actions Once `d10` Finishes

1. Run the corrected labeling script on all completed `d10` reps:
   ```bash
   for rep in 1 2 3 4 5; do
     python scripts/label_modes.py --experiment-dir runs/experiment_calibration_d10_rep${rep}
   done
   ```

2. Run the full calibration analysis:
   ```bash
   python workflow/scripts/analyze_calibration.py \
     --repo-root . \
     --output workflow/artifacts/calibration_analysis.json
   ```

3. Evaluate the gate:
   ```bash
   python workflow/scripts/check_decision_gate.py \
     --analysis workflow/artifacts/calibration_analysis.json
   ```

4. Sync `workflow/state.json` measurements from the analysis output before recording the decision.

5. If the decision is `proceed`, the prepared `d01/d11` configs are ready for the Phase 03 budget.

## Recommended Additional Experiments

### 1. Budget Sensitivity Study

Rationale:
- calibration run counts vary a lot by rep;
- memory advantage in `d10` may need more iterations to show up cleanly.

Proposal:
- run a small `d00`/`d10` grid at `{45, 60, 90}` minute agent budgets with the same `60s` train budget,
- compare:
  - number of attempts,
  - best and mean `val_bpb`,
  - promoted-mode diversity,
  - effect size trend with budget.

### 2. Random Search Baseline

Rationale:
- the BP framing is only interesting if agent-guided search outperforms naive exploration.

Proposal:
- build a small random baseline over the same parameter/edit families the agent is actually touching:
  - learning rate,
  - dropout,
  - depth,
  - base channels,
  - batch size,
  - max steps.
- Match the same evaluation budget and compare:
  - best `val_bpb`,
  - mean `val_bpb`,
  - diversity of promoted modes.

### 3. Optional if Gate Is Weak: CIFAR-100 Escalation Trigger

Rationale:
- if `d00` vs `d10` remains weak after full calibration, the substrate may still be too easy.

Proposal:
- treat weak effect size plus low accepted-mode diversity as a concrete trigger to escalate rather than continuing to collect more CIFAR-10 data by inertia.

## Files Modified This Session

- `scripts/label_modes.py`
- `configs/experiment_d01.yaml`
- `configs/experiment_d11.yaml`

## Artifacts Produced This Session

- `workflow/artifacts/calibration_analysis_d00_smoke.json`
- `workflow/artifacts/calibration_analysis_d00_labeled.json`
- `workflow/artifacts/decomposition_smoke_d00.json`
- `workflow/artifacts/codex_session_notes.md`

