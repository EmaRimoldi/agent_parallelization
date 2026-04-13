# Prompt For Codex

Use the following prompt as-is with Codex:

```text
You are in the `agent_parallelization` repository.

Your job is to execute the theory-formalization queue stored in:

`theory_formalization_tasks/`

Rules:

1. You must process the task files sequentially.
2. You must use the repository runner:
   - `./run_theory_formalization_tasks.sh start`
   - `./run_theory_formalization_tasks.sh show`
   - `./run_theory_formalization_tasks.sh complete "short note"`
3. Do not skip tasks.
4. Do not mark a task complete until every deliverable in that Markdown file exists and its checklist is satisfied.
5. After completing one task, immediately advance to the next using the runner.
6. If the theory needs to be revised, update the LaTeX source and regenerate the PDF.
7. Be mathematically rigorous and experimentally rigorous.
8. Treat the existing validation bundle in `theory_validation_bp_20260412/` as the current source of truth unless you update it with stronger evidence.

Workflow:

1. Run:
   `./run_theory_formalization_tasks.sh start`
2. Read the current task the runner prints.
3. Execute the task fully.
4. Validate that all deliverables exist.
5. Mark it complete:
   `./run_theory_formalization_tasks.sh complete "what was done"`
6. The runner will print the next task automatically.
7. Repeat until the runner says all tasks are complete.

Useful commands:

- `./run_theory_formalization_tasks.sh status`
- `./run_theory_formalization_tasks.sh current`
- `./run_theory_formalization_tasks.sh show`
- `./run_theory_formalization_tasks.sh list`

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
