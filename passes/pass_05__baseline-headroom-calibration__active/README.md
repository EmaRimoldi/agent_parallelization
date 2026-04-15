# Pass 05 - Baseline Headroom Calibration

**Status**: Active
**Period**: April 14, 2026
**Objective**: Find a healthy but non-trivial AutoResearch baseline before running the reviewer-grade BP 2x2.

## Research Question

Pass 04 showed that the previous task was too close to a narrow local optimum:
only a small learning-rate region reliably improved validation loss. Pass 05 asks:

**Can we choose a baseline and fixed-step evaluator where several distinct strategy categories can improve the model, without making the task trivially easy?**

This pass is deliberately non-agentic. It calibrates the task geometry before spending LLM budget on architecture comparisons.

## What Changed

Pass 05 added a controlled baseline-headroom calibration tool:

- fixed-step evaluator support with `AUTOSEARCH_MAX_STEPS`;
- isolated workspaces for every baseline/edit trial;
- baseline and edit panels covering optimizer/LR, scheduler, capacity, regularization, and batch/data;
- JSON/CSV/Markdown outputs for calibration screens;
- reviewer-grade cost and hitting-time instrumentation from the preceding protocol work.

The working `autoresearch/train.py` baseline was updated to the selected candidate:

```text
DEPTH = 3
BASE_CHANNELS = 30
FC_HIDDEN = 128
OPTIMIZER = adam
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.0
USE_LR_SCHEDULE = False
BATCH_SIZE = 128
AUTOSEARCH_MAX_STEPS = 1170
```

## Experiments

| screen | fixed steps | trials | purpose |
| --- | ---: | ---: | --- |
| `baseline_headroom_calibration_fixed1170` | 1170 | 43 | default healthy-mistuned baseline screen |
| `baseline_headroom_calibration_extended_targeted_fixed1170` | 1170 | 38 | broader model / optimizer / regularization screen |
| `baseline_refinement_custom_fixed585` | 585 | 40 | shorter-step refinement screen |
| `baseline_refinement_custom_fixed1170` | 1170 | 40 | intermediate-width / head / mild-dropout refinement |

Total controlled evaluations summarized here: **161**.

## Key Figures

![Baseline screen overview](results/figures/figure-01-baseline-screen-overview.png)

**Figure 1**: baseline quality and category-derived `q3` thresholds. The selected baseline is not the weakest; it sits near the better end while still preserving multiple useful intervention categories.

![Gate diagnostics](results/figures/figure-02-gate-diagnostics.png)

**Figure 2**: raw edit win rate versus number of winning strategy categories. The original 10-30% raw-win gate was too strict for this panel because several duplicate edits within the same category can win. The more useful diagnostic is category richness plus negative controls.

![Category improvement heatmap](results/figures/figure-03-category-improvement-heatmap.png)

**Figure 3**: best improvement by strategy category. Good candidates expose multiple positive categories while retaining negative or weak categories.

![Recommended baseline detail](results/figures/figure-04-recommended-baseline-detail.png)

**Figure 4**: detailed trial outcomes for `width30_lr_low`.

## Decision

Recommended baseline:

```text
baseline_id = width30_lr_low
run = refinement_fixed1170
baseline val_bpb = 0.841354
q* = 0.824
```

Winning categories:

| category | best trial | best val_bpb | improvement |
| --- | --- | ---: | ---: |
| data_batch | `width30_lr_low__data_batch__batch256` | 0.784812 | 0.056542 |
| normalization_capacity | `width30_lr_low__normalization_capacity__width32` | 0.823338 | 0.018016 |
| optimizer_lr | `width30_lr_low__optimizer_lr__lr_1p5e3` | 0.800896 | 0.040458 |

Negative / near-negative controls:

| trial | category | val_bpb | delta vs baseline |
| --- | --- | ---: | ---: |
| `width30_lr_low__scheduler__schedule_on` | scheduler | 0.845433 | -0.004079 |

## Candidate Comparison

| baseline | screen | steps | baseline val_bpb | raw wins | winning categories | q3 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| narrow_lr_low | default_fixed1170 | 1170 | 0.864447 | 4/8 | data_batch, normalization_capacity, optimizer_lr | 0.832826 |
| sgd_baseline | extended_fixed1170 | 1170 | 0.884132 | 3/6 | data_batch, optimizer_lr, regularization | 0.872697 |
| width30_lr_low | refinement_fixed1170 | 1170 | 0.841354 | 4/7 | data_batch, normalization_capacity, optimizer_lr | 0.823338 |
| mild_dropout_no_schedule | extended_fixed1170 | 1170 | 1.065839 | 4/7 | data_batch, optimizer_lr, optimizer_scheduler, regularization | 1.035120 |
| overregularized_lr_low | default_fixed1170 | 1170 | 0.966298 | 5/8 | data_batch, normalization_capacity, optimizer_lr, regularization | 0.899594 |
| no_batchnorm_lr_low | default_fixed1170 | 1170 | 1.078067 | 5/8 | data_batch, normalization_capacity, optimizer_lr | 0.963829 |
| fc96_lr_low | refinement_fixed1170 | 1170 | 0.851718 | 5/7 | data_batch, normalization_capacity, optimizer_lr | 0.834426 |
| dropout005_lr_low | refinement_fixed1170 | 1170 | 0.868005 | 5/7 | data_batch, optimizer_lr, regularization | 0.830335 |

## Why Not 585 Steps

The 585-step screens created broad headroom, but almost every reasonable edit won.
That is useful for debugging, but weak for confirmatory architecture claims. At 1170 steps, the task remains learnable while retaining more negative controls.

## Next Step

Run a small agentic pilot on `width30_lr_low` before the full 2x2:

```text
fixed-step evaluator
AUTOSEARCH_MAX_STEPS = 1170
serialized evaluator
q* = 0.824
separate agent_deliberation_wall_time and evaluator_wall_time
true independent replicates
```

## Artifacts

- Summary table: `results/tables/baseline_summary.csv`
- Trial table: `results/tables/trial_results.csv`
- Machine-readable summary: `results/tables/pass_05_summary.json`
- Source calibration reports remain under `runs/baseline_*`.
