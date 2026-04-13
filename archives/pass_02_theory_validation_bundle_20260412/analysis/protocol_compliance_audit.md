# Protocol Compliance Audit

## Goal

Check whether the **actual pilot and instrumentation** are strong enough to validate the theorem claimed in `autoresearch_bp.pdf`.

This audit is separate from the formal audit:

- a theory can be sound but under-measured,
- and an experiment can be rich but target the wrong object.

## Compliance Summary

### What is present

- `autoresearch_bp.pdf` and `BP.pdf` are both available.
- The 12 pilot runs were completed.
- `runs/pilot_mapping.json` exists.
- Repetition-level decomposition outputs exist.
- Aggregate summary and figures exist.
- Turn-level and training-run logs exist for the pilot.

### What is missing or materially weakened

1. **Mode labels were not assigned to all pilot experiments**
   The pilot protocol required mode labels for all experiments before declaring Phase 3 complete.
   In practice, several pilot runs have snapshots but no `mode_labels.jsonl`.

2. **`phi`, `G`, and `epsilon` are effectively degenerate**
   In the current decomposition outputs:

   - `phi = 0.0` by implementation placeholder,
   - `G = 0.0` in all repetitions,
   - `epsilon = 0.0` in all repetitions.

   So the experiment did not empirically identify the full four-term decomposition.

3. **The cost-equalized condition from the theory was not run**
   The theoretical protocol asks for both:

   - wall-clock-equalized comparison,
   - cost-equalized comparison.

   The current pilot is effectively wall-clock-budgeted only.

4. **The cost axis differs from the paper's motivating definition**
   The implementation explicitly changed `tau_cost` to token-only cost and dropped GPU cost because the CPU substrate makes GPU cost irrelevant.

   This is a valid engineering adaptation, but it means the empirical study is **not a literal realization** of the paper's original cost definition.

5. **Incumbent re-evaluation was not implemented as the paper specifies**
   The paper asks for systematic re-evaluation of promising incumbents under a separate diagnostic budget.
   The pilot contains some repeated commit evaluations, but not a clear, disciplined implementation of the paper's rule.

6. **H5 is not really testable in the observed regime**
   The context-fill range in the pilot stays roughly within `0.01` to `0.24`.
   That means the quartile stratification proposed in the paper never reaches the high-pressure bins needed to test the monotonic context-pressure prediction in a meaningful way.

## Evidence

### 1. Four-term decomposition collapsed to cost-only in practice

From `results/decomposition_rep1.json`, `results/decomposition_rep2.json`, `results/decomposition_rep3.json`:

- `phi = 0.0` in all cells and repetitions,
- `G = 0.0` in all cells and repetitions,
- `epsilon = 0.0` in all cells and repetitions.

So empirically the decomposition behaved like:

```text
Delta ~= log(kappa0 / kappa)
```

This is a major limitation. It does **not** falsify the theorem, but it means the pilot does not yet validate the full theorem.

### 2. Mode-label coverage is incomplete

See `analysis/mode_label_coverage.json`.

Pilot-level summary:

- `d00`: only 1 mode-label file, 1 labeled row
- `d10`: 0 mode-label files
- `d01`: 2 mode-label files, 2 labeled rows
- `d11`: 3 mode-label files, 5 labeled rows

But snapshot metadata show more candidate edits than this, including some accepted ones.

Interpretation:

- the missing labels are a **measurement failure**,
- not direct evidence that the mode structure is absent.

### 3. Accepted-edit evidence is too sparse for stable posterior estimates

Across pilot snapshot metadata, only a handful of edits are marked accepted, and they are not distributed across cells in a way that would support robust estimation of:

- prior mode distribution,
- information term,
- routing mismatch term.

That makes `G` and `epsilon` numerically fragile even before considering missing labels.

### 4. Context-pressure coverage is shallow

See `analysis/context_pressure_metrics.json`.

Observed maxima by cell:

- `d00`: ~0.228
- `d10`: ~0.217
- `d01`: ~0.243
- `d11`: ~0.212

So the pilot almost never leaves the lowest quarter of context occupancy implied by the paper's 4-bin analysis.

Interpretation:

- H5 is **under-identified**, not cleanly falsified.

### 5. Cost-axis mismatch is explicit in the implementation docs

`docs/guides/IMPLEMENTATION_GUIDE.md` explicitly states:

- `tau_cost` is measured in tokens, not dollars,
- GPU cost is dropped from `kappa_cost`.

This is coherent for the CPU substrate, but it changes the interpretation of the cost axis relative to the motivating AutoResearch-H100 setting in the paper.

## Compliance Verdict

The current pilot is **good enough to test whether cost terms and coarse architectural contrasts show signal**.

It is **not good enough to claim full validation of the theorem as written**, because:

- the full four-term estimator was not identified,
- the cost-equalized condition was not run,
- H5 was not properly excited,
- and the re-evaluation protocol was not implemented in the explicit way the theory requires.

## Diagnostic Classification

This is best classified as:

> partial empirical validation of the cost-term side of the framework, with substantial protocol gaps for the `phi`, `G`, and `epsilon` channels.

That points to a theory/protocol mismatch, not a clean theorem refutation.
