# Kickoff: Start the Autonomous Research Workflow

You are executing a phase-based research workflow for validating the BP four-term decomposition applied to AutoResearch agent architectures.

## How to proceed

1. Run `python workflow/run.py status` to see the current state.
2. Run `python workflow/run.py next` to get the next phase prompt.
3. Read the prompt carefully and execute all tasks listed in it.
4. When the phase is complete, run `python workflow/run.py complete` with any relevant measurements.
5. If the phase is a decision point, run `python workflow/run.py decide <choice>`.
6. Repeat from step 2 until the workflow is done.

## Rules

- Execute each phase fully before advancing.
- Record measurements with `python workflow/run.py measure '<json>'` for any quantitative findings.
- Log important observations with `python workflow/run.py log "<message>"`.
- If a phase fails, run `python workflow/run.py fail --reason "<reason>"`.
- Do not skip phases or advance state manually.
- Write all outputs to `workflow/artifacts/`, `workflow/results/`, or `workflow/reports/` as appropriate.

## Start now

```bash
python workflow/run.py next
```
