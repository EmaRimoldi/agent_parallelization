# Task 9: Validation and pilot experiment

Read `PILOT_EXPERIMENT_PROTOCOL.md` for full specification. This is the final task.

## What to do

Execute all three phases of the pilot experiment protocol:

### Phase 1: Validation (Section 1.1–1.7 of the protocol)

Run all 7 validation checks in order:
1. Substrate sanity check (prepare.py + train.py)
2. d00 smoke test (10 min single agent)
3. d10 smoke test (10 min single agent with memory)
4. d01 smoke test (10 min parallel agents)
5. d11 smoke test (10 min parallel shared)
6. Mode labeling smoke test
7. Decomposition script dry run

For each check, verify the success criteria listed in the protocol. If any check fails, diagnose and fix before continuing.

### Phase 2: Pilot 2×2 (Section 2.1–2.5)

Run the full 2×2 pilot: 4 cells × 3 repetitions = 12 runs, each 30 minutes.

After each run, label modes with `python scripts/label_modes.py`.

Create `runs/pilot_mapping.json` with the actual experiment IDs.

### Phase 3: Analysis (Section 3.1–3.5)

1. Run decomposition on each of the 3 repetitions
2. Create `scripts/aggregate_pilot.py` per the specification
3. Generate `results/pilot_summary.md` with all required sections
4. Generate figures if matplotlib is available
5. Evaluate the negative result criterion

## Success criteria
- All validation checks pass
- 12 runs completed
- `pilot_summary.md` exists with decomposition table, hypothesis verdicts, context pressure analysis, raw metrics, interpretation, and negative result criterion evaluation
- All deliverables committed with message "Task 9: Pilot experiment results"
