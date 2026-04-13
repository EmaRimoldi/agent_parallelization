# Estimator Validation Note

## Validation Run

I validated the updated estimators on pilot repetition 1 using:

- `runs/experiment_pilot_d00_rep1`
- `runs/experiment_pilot_d10_rep1`
- `runs/experiment_pilot_d01_rep1`
- `runs/experiment_pilot_d11_rep1`

Commands executed:

```bash
python scripts/label_modes.py --experiment-dir runs/experiment_pilot_d00_rep1
python scripts/label_modes.py --experiment-dir runs/experiment_pilot_d10_rep1
python scripts/label_modes.py --experiment-dir runs/experiment_pilot_d01_rep1
python scripts/label_modes.py --experiment-dir runs/experiment_pilot_d11_rep1

python scripts/compute_decomposition.py \
  --d00 runs/experiment_pilot_d00_rep1 \
  --d10 runs/experiment_pilot_d10_rep1 \
  --d01 runs/experiment_pilot_d01_rep1 \
  --d11 runs/experiment_pilot_d11_rep1 \
  --output theory_validation_bp_20260412/analysis/estimator_validation_rep1.json
```

Output:

- `theory_validation_bp_20260412/analysis/estimator_validation_rep1.json`

## What Passed

1. The mode-labeling script ran successfully and produced `mode_labels.jsonl` files for the tested replica.
2. The decomposition script ran successfully without crashing.
3. `phi` is no longer a hardcoded zero. In this replica it returns `NaN` with explicit diagnostics (`insufficient_mode_overlap`) rather than silently collapsing.
4. `G` no longer uses the old entropy-difference formula.
5. `epsilon` no longer uses the old KL-vs-prior formula.
6. Token calibration is populated from observed turn tokens and no longer relies blindly on raw `chars/4`.

## What The Replica Shows

### Non-trivial estimator behavior

The updated estimators do **not** collapse for implementation reasons alone:

- `epsilon_d01 = 0.6931`, because the posterior accepted-mode distribution (`architecture`) differs from the proposal/routing proxy (`50% architecture, 50% regularization`)
- `token_calibration_factor` is populated and near `1.0` rather than being absent
- `kappa_token_std` and `kappa_wall_std` are populated for all cells

This means the code path is now capable of producing non-degenerate terms.

### Why many fields are still `NaN`

The remaining `NaN` values are caused by data insufficiency in the pilot replica, not by placeholder code:

- `phi` is `NaN` because there is no overlap in accepted modes between baseline `d00` and the other cells on replica 1
- `G` is `NaN` for `d10` and `d11` because those cells have no accepted-mode posterior in replica 1
- `epsilon` is `NaN` for `d10` and `d11` for the same reason: no accepted posterior means no posterior-vs-routing KL can be estimated

For `d01`, `G = 0.0` is a real data outcome in this replica because the accepted-mode posterior matches the pooled accepted prior (`architecture` only), not because of the previous broken entropy-difference implementation.

## Bottom Line

The estimator layer is now structurally aligned with BP in a way the previous code was not.

Current status:

- code design no longer forces `phi = G = epsilon = 0`
- the present pilot still has too little accepted-mode support for stable full decomposition estimates

So the remaining collapse is now mainly a **data/regime problem**, not an **implementation bug**.
