# Pass 02: Theory Formalization Task Queue

This directory is the second AI task pass. It is a sequential work queue for turning the current AutoResearch BP theory into a theorem that is both mathematically and experimentally defensible.

## Purpose

The current repository already contains:

- a validation bundle in `archives/pass_02_theory_validation_bundle_20260412/`
- a revised theory PDF in `archives/pass_02_theory_validation_bundle_20260412/theory/autoresearch_bp_revised.pdf`
- pilot results and a noise assay

Those artifacts establish that:

- the BP reduction is plausible,
- the current theorem is still conditional,
- the current estimators for `phi`, `G`, and `epsilon` are not yet valid,
- the current pilot is dominated by estimator gaps and selection bias,
- the next work should focus on theory tightening plus protocol / estimator repair before another large pilot.

## Task Ordering

The tasks are intentionally ordered and should be executed sequentially:

1. understand the current state and reviewer constraints
2. tighten the theorem
3. repair logging and protocol assumptions
4. define modes and correct estimators
5. run the highest-value follow-up experiments
6. recompute analysis and issue a clean verdict
7. regenerate the theory package and handoff bundle

## Files

- `task_00_orientation.md`
- `task_01_theorem_refactor.md`
- `task_02_protocol_and_logging.md`
- `task_03_modes_and_estimators.md`
- `task_04_targeted_experiments.md`
- `task_05_reanalysis_and_verdict.md`
- `task_06_finalize_bundle.md`
- `PROMPT_FOR_CODEX.md`

## Canonical Runner

Use the canonical pass runner:

```bash
./ai_task_passes/run_pass_02_theory_formalization.sh start
```

After finishing each task:

```bash
./ai_task_passes/run_pass_02_theory_formalization.sh complete "short completion note"
```

The runner will automatically advance to the next Markdown file and print it.

Useful commands:

```bash
./ai_task_passes/run_pass_02_theory_formalization.sh status
./ai_task_passes/run_pass_02_theory_formalization.sh current
./ai_task_passes/run_pass_02_theory_formalization.sh show
./ai_task_passes/run_pass_02_theory_formalization.sh list
./ai_task_passes/run_pass_02_theory_formalization.sh reset --force
```

Compatibility note:

- `./run_theory_formalization_tasks.sh ...` still works as a thin wrapper, but the canonical path now lives under `ai_task_passes/`.

## Completion Rule

Do not mark a task complete unless all of its listed deliverables exist and its checklist is satisfied.
