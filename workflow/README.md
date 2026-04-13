# Autonomous Research Workflow

Phase-based execution system for validating the BP four-term decomposition applied to AutoResearch agent architectures.

## Quick Start

```bash
# Initialize and see current status
python workflow/run.py status

# Display the next phase prompt (with injected context)
python workflow/run.py next

# After executing the phase instructions, mark it complete
python workflow/run.py complete

# For phases with branching decisions
python workflow/run.py decide <choice>
```

## How It Works

The workflow is a directed graph of phases. Each phase is a self-contained Markdown file with:
- Goal, background, and context
- Exact tasks to perform
- Required inputs and expected outputs
- Success criteria and failure modes
- What to do next (including conditional branches)

The orchestrator (`run.py`) manages:
- **State persistence**: tracks current phase, completed phases, decisions, measurements
- **Prompt rendering**: injects context from prior phases into each phase prompt
- **Branching logic**: routes to the correct next phase based on decisions
- **Logging**: records all actions and outputs

## Workflow Graph

```
00_overview → 01_deterministic_eval → 01a_verify_determinism
                                           ↓ pass
                                     02_power_calibration
                                           ↓
                                     02a_analyze_calibration
                                           ↓
                                     02b_decision_gate
                                      ↙    ↓    ↘
              04_escalation    03_full_2x2   05_structured
              (CIFAR-100)          ↓          search
                   ↓         03a_mode_labeling
              re-enter             ↓
              pipeline       03b_decomposition
                                   ↓
                             03c_identity_check
                                   ↓
                             06_theorem_update
                                   ↓
                             07_final_report
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `status` | Show current phase, completed phases, decisions, measurements |
| `next` | Render the next phase prompt with injected context |
| `complete [--output FILE] [--measurements JSON]` | Mark current phase done, optionally attach artifact and measurements |
| `fail [--reason TEXT]` | Mark current phase failed (retries up to 2 times, then branches) |
| `decide <choice>` | Record a branching decision at a decision-point phase |
| `log <message>` | Append a message to the current phase's log |
| `measure <JSON>` | Record measured quantities into persistent state |
| `render [--phase ID]` | Render any phase prompt without advancing state |
| `reset` | Reset to initial state (backs up current state first) |

## Directory Structure

```
workflow/
├── README.md              ← You are here
├── run.py                 ← Main orchestrator CLI
├── phases.json            ← Phase graph (manifest)
├── state.json             ← Persistent state (auto-created)
├── kickoff.md             ← Prompt to start the autonomous cycle
├── phases/                ← Phase instruction files
│   ├── 00_overview.md
│   ├── 01_deterministic_eval.md
│   ├── 01a_verify_determinism.md
│   ├── 01b_debug_determinism.md
│   ├── 02_power_calibration.md
│   ├── 02a_analyze_calibration.md
│   ├── 02b_decision_gate.md
│   ├── 03_full_2x2_run.md
│   ├── 03a_mode_labeling.md
│   ├── 03b_decomposition.md
│   ├── 03c_identity_check.md
│   ├── 04_escalation_cifar100.md
│   ├── 05_structured_search.md
│   ├── 06_theorem_update.md
│   └── 07_final_report.md
├── scripts/               ← Helper execution scripts
│   ├── verify_determinism.py
│   ├── analyze_calibration.py
│   └── check_decision_gate.py
├── prompts/               ← Rendered prompts (auto-generated)
├── artifacts/             ← Phase outputs and intermediate results
├── logs/                  ← Per-phase execution logs
├── results/               ← Final analysis results
└── reports/               ← Generated reports
```

## Typical Session

A typical interaction with Claude Code looks like:

```
Human: Start the next phase of the research workflow.
Claude: *runs `python workflow/run.py next`*
Claude: *reads the rendered prompt*
Claude: *executes the instructions (modifies files, runs scripts, writes analysis)*
Claude: *runs `python workflow/run.py complete --measurements '{"key": "value"}'`*
Claude: *runs `python workflow/run.py next` for the next phase*
```

For decision points:
```
Claude: *runs `python workflow/scripts/check_decision_gate.py --analysis ...`*
Claude: *runs `python workflow/run.py decide proceed`*
```

## Resuming After Interruption

The state file (`state.json`) persists between sessions. To resume:

```bash
python workflow/run.py status   # See where you left off
python workflow/run.py next     # Continue from current phase
```

## Resetting

```bash
python workflow/run.py reset    # Backs up state, starts fresh
```
