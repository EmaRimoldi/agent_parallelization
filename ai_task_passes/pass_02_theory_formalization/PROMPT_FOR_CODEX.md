# Prompt For Codex

Use the following prompt as-is with Codex:

```text
You are in the `agent_parallelization` repository.

Your job is to execute the theory-formalization queue stored in:

`ai_task_passes/pass_02_theory_formalization/`

Rules:

1. You must process the task files sequentially.
2. You must use the repository runner:
   - `./ai_task_passes/run_pass_02_theory_formalization.sh start`
   - `./ai_task_passes/run_pass_02_theory_formalization.sh show`
   - `./ai_task_passes/run_pass_02_theory_formalization.sh complete "short note"`
3. Do not skip tasks.
4. Do not mark a task complete until every deliverable in that Markdown file exists and its checklist is satisfied.
5. After completing one task, immediately advance to the next using the runner.
6. If the theory needs to be revised, update the LaTeX source and regenerate the PDF.
7. Be mathematically rigorous and experimentally rigorous.
8. Treat the existing validation bundle in `theory_validation_bp_20260412/` as the current source of truth unless you update it with stronger evidence.

Workflow:

1. Run:
   `./ai_task_passes/run_pass_02_theory_formalization.sh start`
2. Read the current task the runner prints.
3. Execute the task fully.
4. Validate that all deliverables exist.
5. Mark it complete:
   `./ai_task_passes/run_pass_02_theory_formalization.sh complete "what was done"`
6. The runner will print the next task automatically.
7. Repeat until the runner says all tasks are complete.

Useful commands:

- `./ai_task_passes/run_pass_02_theory_formalization.sh status`
- `./ai_task_passes/run_pass_02_theory_formalization.sh current`
- `./ai_task_passes/run_pass_02_theory_formalization.sh show`
- `./ai_task_passes/run_pass_02_theory_formalization.sh list`

Priority guidance:

- Tighten the theorem before large new experiments.
- Fix logging/protocol and estimators before trusting decomposition outputs.
- Prefer replicated means and estimator validity over another best-of-N pilot.
- If the strongest defensible theorem is weaker than the current one, weaken it explicitly rather than hiding the gap.

Success condition:

At the end there must be:

- a current revised theory PDF,
- a current validation bundle,
- corrected or clearly documented estimators,
- targeted experiments addressing the highest-value open questions,
- and a final verdict that is narrower but genuinely defensible.
```
