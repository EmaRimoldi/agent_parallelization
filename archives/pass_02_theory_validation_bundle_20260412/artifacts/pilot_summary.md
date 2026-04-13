# Pilot Summary

## Decomposition Table (mean +/- std across 3 reps)

| Cell | log(k0/k)_tok | log(k0/k)_wall | phi | G | -epsilon | delta_tok | delta_wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| d10 | -0.11 +/- 0.08 | -0.08 +/- 0.28 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | -0.00 +/- 0.00 | -0.11 +/- 0.08 | -0.08 +/- 0.28 |
| d01 | -0.07 +/- 0.11 | -0.03 +/- 0.03 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | -0.00 +/- 0.00 | -0.07 +/- 0.11 | -0.03 +/- 0.03 |
| d11 | -0.09 +/- 0.08 | -0.17 +/- 0.20 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | -0.00 +/- 0.00 | -0.09 +/- 0.08 | -0.17 +/- 0.20 |

Bootstrap 95% confidence intervals (mean of 1000 resamples):
- d10: tok [-0.17, 0.00], wall [-0.48, 0.12], G [0.00, 0.00], -epsilon [-0.00, -0.00]
- d01: tok [-0.23, 0.02], wall [-0.06, 0.01], G [0.00, 0.00], -epsilon [-0.00, -0.00]
- d11: tok [-0.18, 0.01], wall [-0.44, 0.03], G [0.00, 0.00], -epsilon [-0.00, -0.00]

## Hypothesis Verdicts

- H1 (parallelism helps only wall-clock): 1/3 reps support
- H2 (memory helps both axes): 1/3 reps support
- H3 (shared memory lowers epsilon): 0/3 reps support
- H4 (parallelism sensitive to coordination): 0/3 reps support
- H5 (context pressure dominant): 0/3 reps support
- H6 (d11 dominates d00 on both axes): 0/3 reps support

## Context Pressure Analysis (H5)

| Cell | 0-25% | 25-50% | 50-75% | 75-100% | Monotone? |
|---|---:|---:|---:|---:|---|
| d00 | nan | nan | nan | nan | yes |
| d10 | nan | nan | nan | nan | yes |
| d01 | nan | nan | nan | nan | yes |
| d11 | nan | nan | nan | nan | yes |

## Raw Metrics

| Cell | Total training runs | Best val_bpb | Total tokens | Mean wall-clock / attempt (s) |
|---|---:|---:|---:|---:|
| d00 | 5.00 +/- 2.94 | 0.81 +/- 0.01 | 40974.00 +/- 5106.45 | 136.29 +/- 0.24 |
| d10 | 5.33 +/- 2.62 | 0.78 +/- 0.03 | 41835.33 +/- 5006.73 | 135.78 +/- 0.18 |
| d01 | 12.67 +/- 3.86 | 0.82 +/- 0.01 | 85060.00 +/- 10165.37 | 151.04 +/- 19.89 |
| d11 | 13.33 +/- 3.30 | 0.80 +/- 0.00 | 75628.00 +/- 10217.44 | 137.44 +/- 0.81 |

## Interpretation

H1: The wall-clock term for d01 determines whether parallelism helped by latency reduction alone; token-side gains near zero or negative indicate coordination overhead without extra search efficiency.
H2: d10 is favorable when the memory table reduces token and wall costs simultaneously, implying better state compression than plain conversation accumulation.
H3/H4: epsilon captures coordination mismatch. Lower epsilon in d11 than d01 supports useful shared-memory routing; high epsilon suggests the parallel cell is paying coordination tax.
H5: The context-bin table tests whether token cost rises with context pressure in d00 and whether d10 flattens that curve. If it does, the memory mechanism is acting like a context compressor rather than extra baggage.
H6: d11 only clearly dominates when both delta_wall and delta_token stay positive across reps. If that fails, the shared-memory benefits are still conditional on coordination quality or search diversity.

## Negative Result Criterion

- Linear fit R^2(best_val_bpb ~ total_tokens): 0.32
- Negative result criterion met (R^2 > 0.9): no

## Figures

- results/figures/pass_01_pilot/best_so_far_curves.png
- results/figures/pass_01_pilot/kappa_by_context_bin.png
- results/figures/pass_01_pilot/decomposition_bar_chart.png

## Notes

- Repetitions aggregated: 3
- Cells use the Task 9 2x2 pilot mapping from runs/pilot_mapping.json.
