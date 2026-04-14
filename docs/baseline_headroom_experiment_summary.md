# Baseline Headroom Experiment Summary

Generated after the baseline-headroom screens run on April 14, 2026.

## Experiments Run

| run | fixed steps | trials | purpose |
| --- | ---: | ---: | --- |
| `runs/baseline_headroom_calibration_fixed1170` | 1170 | 43 | default healthy-mistuned baseline screen |
| `runs/baseline_headroom_calibration_extended_targeted_fixed1170` | 1170 | 38 | broader model / optimizer / regularization screen |
| `runs/baseline_refinement_custom_fixed585` | 585 | 40 | shorter-step refinement screen |
| `runs/baseline_refinement_custom_fixed1170` | 1170 | 40 | intermediate-width / head / mild-dropout refinement |

Total controlled training evaluations: 161.

## Main Finding

The 585-step task is too permissive for confirmatory architecture experiments.
It creates broad headroom, but many reasonable edits win, so it is less useful
for testing whether agentic architecture improves search quality.

The best current candidate is:

```text
baseline_id = width30_lr_low
DEPTH = 3
BASE_CHANNELS = 30
FC_HIDDEN = 128
LEARNING_RATE = 5e-4
USE_LR_SCHEDULE = False
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.0
OPTIMIZER = adam
BATCH_SIZE = 128
AUTOSEARCH_MAX_STEPS = 1170
```

This baseline is not obviously broken: it keeps the original depth, batchnorm,
optimizer family, classifier head, weight decay, dropout, and batch size. It is
only mildly mis-tuned through width, learning rate, and schedule.

## Recommended Candidate

`width30_lr_low` at 1170 fixed steps:

| metric | value |
| --- | ---: |
| baseline val_bpb | 0.841354 |
| edit wins | 4 / 7 |
| winning categories | 3 |
| q* from third winning category | 0.823338 |

Winning categories:

| category | best edit | val_bpb | improvement |
| --- | --- | ---: | ---: |
| optimizer_lr | `lr_1p5e3` | 0.800896 | 0.040458 |
| normalization_capacity | `width32` | 0.823338 | 0.018016 |
| data_batch | `batch256` | 0.784812 | 0.056542 |

Negative / near-negative controls:

| edit | category | val_bpb | result |
| --- | --- | ---: | --- |
| `lr_1e3` | optimizer_lr | 0.847634 | worse |
| `adamw_1e3` | optimizer_lr | 0.878075 | worse |
| `schedule_on` | scheduler | 0.845433 | worse |

Recommended target for the next pilot:

```text
q* = 0.824
```

This includes the third winning category (`width32`, 0.823338) while still
requiring a real improvement over the baseline (0.841354).

## Why Not the Other Candidates

`narrow_lr_low` at 1170 is a valid backup, but `width30_lr_low` is less
obviously weakened and closer to the current task.

```text
narrow_lr_low:
baseline = 0.864447
winning categories = optimizer_lr, normalization_capacity, data_batch
raw wins = 4 / 8
q* = 0.832826
```

`overregularized_lr_low`, `mild_dropout_no_schedule`, `small_fc_lr_low`, and
`shallow_lr_low` expose multi-category headroom, but they are too easy or too
obviously damaged. They are useful diagnostics, not the best confirmatory
baseline.

`weak_regularization_no_schedule` is too strong/narrow: only `data_batch` wins.

`width28_lr_low`, `width24_lr_mid`, `fc96_lr_low`, and all 585-step refinements
were too permissive.

## Decision

Use `width30_lr_low` with `AUTOSEARCH_MAX_STEPS = 1170` for the next agentic
pilot.

Do not use the 585-step evaluator for the confirmatory 2x2. It makes the task
too easy and too dominated by broad early-training improvements.

Before the final 2x2, run a small agentic pilot on `width30_lr_low` with:

```text
q* = 0.824
fixed-step evaluator
serialized evaluator
separate evaluator_wall_time and agent_deliberation_wall_time
true independent replicates
```
