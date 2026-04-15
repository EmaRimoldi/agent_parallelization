# Experiment 02: Within-Architecture Cost Variance

## Claim Tested

Whether replacing BP's fixed per-step `kappa` with an averaged `kappa_bar` is empirically harmless on the current pilot logs.

## Exact Procedure

Artifacts:

- `archives/pass_02_theory_validation_bundle_20260412/experiments/followup_01/analyze_cost_variance.py`
- `archives/pass_02_theory_validation_bundle_20260412/experiments/followup_01/cost_variance_summary.json`

Procedure:

1. Load all turn logs from all three pilot replicates for each cell.
2. Compute turn-level token-cost samples and wall-clock samples.
3. For each axis and cell, estimate:
   - mean
   - variance
   - standard deviation
   - empirical Jensen gap `log(E[kappa]) - E[log(kappa)]`
   - delta-method approximation `0.5 * Var(kappa) / E[kappa]^2`

## Results

### Token axis

| cell | mean | std | empirical Jensen gap | delta-method gap |
| --- | --- | --- | --- | --- |
| `d00` | `2234.9` | `532.7` | `0.0220` | `0.0284` |
| `d10` | `2460.9` | `948.3` | `0.0458` | `0.0742` |
| `d01` | `2407.4` | `745.5` | `0.0350` | `0.0479` |
| `d11` | `2439.6` | `905.2` | `0.0395` | `0.0688` |

### Wall-clock axis

| cell | mean | std | empirical Jensen gap | delta-method gap |
| --- | --- | --- | --- | --- |
| `d00` | `97.8` | `56.4` | `0.2156` | `0.1662` |
| `d10` | `105.0` | `63.1` | `0.2089` | `0.1804` |
| `d01` | `100.9` | `65.3` | `0.2691` | `0.2090` |
| `d11` | `115.5` | `67.0` | `0.2560` | `0.1680` |

## Interpretation

The token-axis averaging error is moderate but not negligible.

The wall-clock averaging error is large enough to matter:

- empirical Jensen gaps are around `0.21` to `0.27`
- these are comparable to, or larger than, several of the architecture-level contrast magnitudes we care about

So the current evidence strongly supports the reviewer's concern:

> treating `kappa_bar_wall` as if it were a drop-in replacement for BP's fixed `kappa` is not harmless in this substrate.

## Does Failure Point To Theory Or Experiment?

This points to **theory plus protocol** rather than a pure implementation issue.

Reason:

- the theory now needs the Jensen remainder explicitly
- the experiment needs to report variance, not just means
- the wall-clock axis is especially vulnerable to path-dependent averaging error

This experiment therefore supports the revised theorem form with an explicit remainder and argues against silently dropping that term.
