# Phase 01a: Verify Deterministic Evaluation

## Goal

Confirm that the modifications from Phase 01 produce fully deterministic evaluation: running the same `train.py` 5 times yields identical `val_bpb` values.

## Background

After fixing the random seeds, DataLoader configuration, and PYTHONHASHSEED, the training pipeline should be fully deterministic on CPU. This phase runs the verification script to confirm.

## Tasks

### 1. Run the verification script

```bash
cd {{repo_root}}
python workflow/scripts/verify_determinism.py
```

This script:
- Runs `train.py` in the `autoresearch/` directory 5 times
- Extracts `val_bpb` from each run's stdout
- Checks whether all 5 values are identical
- Reports PASS or FAIL with details

### 2. Record the results

If PASS:
```bash
python workflow/run.py measure '{"determinism_verified": true, "baseline_val_bpb": <value>, "determinism_runs": 5}'
python workflow/run.py decide pass
```

If FAIL:
```bash
python workflow/run.py measure '{"determinism_verified": false, "determinism_range": <max-min>}'
python workflow/run.py decide fail
```

## Required Inputs

- Modified `autoresearch/train.py` from Phase 01
- Modified `autoresearch/prepare.py` from Phase 01
- `workflow/scripts/verify_determinism.py`

## Expected Outputs

- Console output showing 5 val_bpb values
- PASS: all 5 identical → proceed to Phase 02
- FAIL: values differ → proceed to Phase 01b (debug)

## Success Criteria

- All 5 runs produce identical val_bpb (tolerance: 0.0)
- Script exits with code 0

## Failure Modes

- Values differ slightly (e.g., 1e-7): likely floating-point accumulation order. Check if the model uses any non-deterministic operations. Try `torch.use_deterministic_algorithms(True)` without `warn_only`.
- Values differ substantially (e.g., > 0.001): a major source of non-determinism was missed. Proceed to Phase 01b.
- Script crashes: check that train.py runs successfully on its own first.

## Decision Branches

- **pass**: All 5 runs identical → proceed to `02_power_calibration`
- **fail**: Values differ → proceed to `01b_debug_determinism`
