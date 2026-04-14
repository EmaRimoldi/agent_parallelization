# Baseline Headroom Calibration

Use this before any reviewer-grade 2x2 run. The goal is to choose a baseline
and `q*` from controlled calibration evidence, not from the confirmatory 2x2.

The calibration is intentionally non-agentic:

- define healthy but mildly mis-tuned baseline candidates;
- apply a fixed edit panel across several strategy categories;
- run fixed-step `train.py` evaluations in isolated workspaces;
- select a baseline only if multiple categories can beat it.

## Command

Cheap smoke test:

```bash
uv run calibrate-baseline-headroom \
  --train-max-steps 2 \
  --baseline-ids lr_low_no_schedule \
  --edit-ids lr_1p5e3,batch64 \
  --out-dir runs/baseline_headroom_smoke
```

If `uv` is not installed in the local shell, use:

```bash
PYTHONPATH=src python -m agent_parallelization_new.baseline_calibration \
  --train-max-steps 2 \
  --baseline-ids lr_low_no_schedule \
  --edit-ids lr_1p5e3,batch64 \
  --out-dir runs/baseline_headroom_smoke
```

Full calibration starting point:

```bash
uv run calibrate-baseline-headroom \
  --train-max-steps 1170 \
  --train-time-budget 300 \
  --timeout 900 \
  --out-dir runs/baseline_headroom_calibration_fixed1170
```

The full default panel runs 5 baseline candidates and 10 controlled edits per
candidate, minus no-op edits. On CPU this can take hours at 1170 steps. Use
`--baseline-ids` and `--edit-ids` for a smaller screen when iterating.

Broader follow-up screen:

```bash
PYTHONPATH=src python -m agent_parallelization_new.baseline_calibration \
  --extended-panel \
  --train-max-steps 1170 \
  --train-time-budget 300 \
  --timeout 900 \
  --out-dir runs/baseline_headroom_calibration_extended_fixed1170
```

For custom screens, pass JSON lists of spec objects:

```json
[
  {
    "id": "custom_lr_low",
    "category": "baseline",
    "description": "Custom healthy baseline.",
    "changes": {"LEARNING_RATE": 0.0005, "USE_LR_SCHEDULE": false}
  }
]
```

Then run with `--baselines-json path/to/baselines.json` and/or
`--edits-json path/to/edits.json`.

The script writes:

- `baseline_headroom_plan.json`: exact trial plan;
- `baseline_headroom_results.json`: machine-readable results and gate decision;
- `baseline_headroom_trials.tsv`: flat table for quick inspection;
- `baseline_headroom_report.md`: reviewer-facing report.

## Gate

A candidate passes only if:

- its baseline run completes;
- at least 3 distinct strategy categories beat the baseline by `min_delta`;
- the completed-edit success rate is in the default 10-30% band.

If a candidate passes, the script proposes `q*` as the strictest threshold that
is still hit by the required number of winning categories. If no candidate
passes, do not proceed to the confirmatory 2x2; revise the task/baseline first.

## Default Baseline Candidates

- `lr_low_no_schedule`
- `lr_very_low_no_schedule`
- `narrow_lr_low`
- `no_batchnorm_lr_low`
- `overregularized_lr_low`

Add `--include-current-control` to evaluate the current repo baseline as a
diagnostic control. It is not intended as the headroom candidate.

## Default Edit Categories

- `optimizer_lr`
- `scheduler`
- `normalization_capacity`
- `regularization`
- `data_batch`

These categories are intentionally coarse. They are meant to test whether the
task has multi-modal headroom, not to optimize the final model.
