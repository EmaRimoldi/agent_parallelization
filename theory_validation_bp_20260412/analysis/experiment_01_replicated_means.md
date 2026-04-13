# Experiment 01: Replicated Means

## Claim Tested

Whether the apparent cell contrasts survive repeated evaluation once best-of-N optimism is reduced.

## Exact Procedure

Artifacts:

- `theory_validation_bp_20260412/experiments/followup_01/run_replicated_means.py`
- `theory_validation_bp_20260412/experiments/followup_01/replicated_means_summary.json`
- raw logs under `theory_validation_bp_20260412/experiments/followup_01/replicated_means/`

Procedure:

1. For each cell (`d00`, `d10`, `d01`, `d11`) and each pilot replicate (`rep1`, `rep2`, `rep3`), select one incumbent candidate:
   - prefer the best accepted snapshot if present
   - otherwise fall back to the best logged training-run commit
2. Reconstruct `train.py` for that candidate.
3. Re-run each cell/rep incumbent under a fixed repeat plan `2 + 2 + 1`, for a total of **5 repeated evaluations per cell**.
4. Aggregate the resulting `val_bpb` values by cell using mean, std, standard error, and 95% CI.

Important limitation:

- this estimates the stability of the **selected incumbent candidates**, not the full search-process distribution of each architecture cell
- it is still much cleaner than the original best-of-N pilot because each chosen candidate is re-evaluated rather than trusted from one noisy win

## Results

Cell-level repeated-evaluation summary:

| cell | n | mean val_bpb | std | 95% CI |
| --- | --- | --- | --- | --- |
| `d10` | 5 | `0.8412` | `0.0361` | `[0.8096, 0.8729]` |
| `d11` | 5 | `0.8737` | `0.0521` | `[0.8280, 0.9194]` |
| `d00` | 5 | `0.8739` | `0.0260` | `[0.8512, 0.8967]` |
| `d01` | 5 | `0.9008` | `0.0793` | `[0.8313, 0.9703]` |

Most important observations:

- `d10` has the lowest repeated-evaluation mean
- `d00` and `d11` are extremely close on mean and their confidence intervals overlap heavily
- `d01` remains the weakest cell on mean and has the widest uncertainty band
- several selected incumbents degraded materially relative to their pilot-reported single best values

Example of residual optimism:

- `d10/rep3` selected incumbent had pilot value `0.7554`
- repeated evaluation for that same incumbent produced `0.8423`

## Interpretation

The follow-up repeated means do **not** support a strong ranking claim among `d10`, `d11`, and `d00`.

What they do support:

1. `d01` remains the least convincing cell.
2. `d10` still looks like the best practical candidate on current evidence.
3. The margin between `d10` and the nearby cells is too small, relative to uncertainty, to count as a decisive architecture contrast.

## Does Failure Point To Theory Or Experiment?

Primarily **experiment / substrate**.

Reason:

- the repeated means experiment shows that architecture ranking remains noisy even after incumbent reevaluation
- this weakens strong empirical claims, but it does not directly refute the theorem
- the result says the current CPU/CIFAR substrate is still too noisy for sharp architecture-level empirical discrimination
