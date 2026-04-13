# Imported agents-swarms Integration Log

Date: 2026-04-13

## Running Log

- Analyzed the current repository first: `README.md`, `docs/architecture.md`, `configs/`, `src/agent_parallelization_new/`, `scripts/`, `workflow/`, and representative tests.
- Cloned `https://github.com/EmaRimoldi/agents-swarms.git` into `agents-swarms/`.
- Initialized the cloned repository's `autoresearch` submodule at commit `d195e2fea7cf2511bb2400634442885b6cb47939`.
- Compared the current implementation against the cloned `src/agent_swarms/` implementation.
- Imported the cloned swarm blackboard logic additively under `src/agent_parallelization_new/imported_swarms/`.
- Added a distinct experiment mode named `imported_swarm`; no native `d00`, `d10`, `d01`, or `d11` files were replaced.
- Added `run-imported-swarm` as a project script and a starter config at `configs/imported_swarms/experiment_imported_swarm.yaml`.
- Added focused tests for the imported blackboard implementation.
- Moved cloned historical analysis artifacts from `agents-swarms/analysis/` to `results/imported_swarms/analysis/`; no local `agents-swarms/runs/` directory was present.

## Current Repository Analysis

The current repository is a BP-style decomposition study for autonomous coding agents on a CPU AutoResearch CIFAR-10 substrate. The empirical design is a 2x2 matrix:

| Cell | Agents | Memory | Current mode |
| --- | --- | --- | --- |
| `d00` | 1 | none | `single_long` |
| `d10` | 1 | private external memory | `single_memory` |
| `d01` | 2 | none | `parallel` |
| `d11` | 2 | shared memory | `parallel_shared` |

The main implementation lives in `src/agent_parallelization_new/`. There is no separate swarm folder in the native implementation. The relevant logic is spread across:

- `config.py`: `use_external_memory` and `use_shared_memory` flags.
- `experiment_modes/`: wrappers for `single_memory`, `parallel`, and `parallel_shared`.
- `orchestrator.py`: creates isolated workspaces/processes and collects results after agents finish.
- `agents/claude_agent_runner.py`: multi-turn Claude CLI loop, private memory context, shared-memory prompt context, turn/training instrumentation, reevaluation bookkeeping.
- `utils/workspace.py`: workspace creation and `shared_results_log.jsonl` symlink setup for `parallel_shared`.
- `snapshotting.py` and `reasoning_trace.py`: per-agent structured traces used for memory, attribution, and merge analysis.
- `outputs/`, `scripts/compute_decomposition.py`, `scripts/label_modes.py`, and `workflow/`: result collection, analysis, calibration, and BP decomposition workflow.

Native memory sharing is lightweight and prompt-injected:

- `d10` private memory reads the current agent's `reasoning/trace.jsonl` and injects a compact experiment table into later turns.
- `d11` shared memory creates one experiment-level `shared_results_log.jsonl`, symlinks it into each workspace, appends completed trace entries with `fcntl.flock`, and injects a compact cross-agent table into later turns.
- There is no claim protocol, global-best sidecar, or coordinator CLI in the native shared-memory mode.

Native agent coordination is intentionally minimal. Agents are isolated by worktree, process, and session. The orchestrator does not route one agent's results to another; in `parallel_shared`, agents only see the shared log because it is present in their workspace and included in later prompts.

Native training and evaluation use generated worker scripts in each workspace. Agents may modify only `train.py`; `prepare.py` is fixed. The current substrate emits `val_bpb` as an alias for validation loss so downstream parsers can stay stable. Results are recorded through trajectory files, structured training-run logs, snapshots, reasoning traces, aggregate summaries, and workflow artifacts.

## Cloned Repository Analysis

The cloned repository, `agents-swarms/`, is a standalone `agent_swarms` Python package for collaborative Claude Code agents. It uses the same broad AutoResearch training pattern but makes the swarm subsystem explicit.

Key cloned and imported paths:

- `src/agent_swarms/shared_memory.py`: process-safe JSONL blackboard.
- `src/agent_swarms/coordinator.py`: CLI for `think`, `claim`, `publish`, `pull-best`, `reason`, and `log`.
- `src/agent_swarms/coordinator_local.py`: Python-class compatibility shim for the upstream autoresearch-at-home coordinator API.
- `src/agent_swarms/swarm_orchestrator.py`: creates one blackboard and passes it to all agents.
- `src/agent_swarms/swarm_agent_runner.py`: single-turn runner that lets each Claude process loop autonomously and interact with the coordinator.
- `src/agent_swarms/experiment_modes/swarm_two_agents.py`: swarm experiment entry point.
- `templates/` and `autoresearch/`: prompt/protocol material; `autoresearch` is a submodule.
- `results/imported_swarms/analysis/`: experiment analyses imported from the clone, including model comparison and swarm-vs-parallelisation summaries.

The cloned memory mechanism is stronger than the native shared log:

- Shared memory is an append-only JSONL blackboard.
- Entry types include `result`, `hypothesis`, `insight`, `status`, `claim`, `claim_release`, and `best`.
- Claims have TTLs and duplicate detection via token Jaccard similarity.
- Best `train.py` source is stored as a sidecar `best_<sha>.py` file so agents can pull the current global best.
- The coordinator logs memory access to `logs/coordinator.log`, which the runner streams into `run_agent.log`.

The cloned agent coordination loop is explicit:

```text
THINK -> REASON -> CLAIM -> pull-best -> RUN -> PUBLISH
```

This differs from the native repo's `parallel_shared` mode. Native `d11` gives agents visibility into a shared results table. The cloned swarm gives agents a blackboard API with work reservation, global-best transfer, insights, and hypotheses.

The cloned repo also contained results and analysis from earlier runs. Those artifacts now live under `results/imported_swarms/analysis/`. Its `swarms_vs_parallelisation/summary.md` reports lower `val_bpb` for swarm runs than a prior parallelisation run, and its model comparison analysis reports Haiku 4.5 as the best observed choice in that specific two-agent swarm setup. Those results are useful historical evidence, but they are not directly equivalent to the current repo's stricter BP calibration workflow.

## Comparison

| Area | Current repository | Cloned `agents-swarms` |
| --- | --- | --- |
| Main package | `agent_parallelization_new` | `agent_swarms` |
| Research framing | BP 2x2 decomposition, calibration, theorem workflow | Collaborative swarm for AutoResearch-at-home |
| Swarm-like mode | `parallel_shared` / `d11` | `swarm` |
| Memory primitive | Compact shared results JSONL table | Typed JSONL blackboard |
| Memory visibility | Prompt injection on later turns | Agent actively calls coordinator during a long turn |
| Coordination | No claims; no global best adoption during run | Claim/release, duplicate avoidance, global best pull, insights, hypotheses |
| Agent runner | Multi-turn controller loop with rich instrumentation | Single long Claude turn by default in swarm mode |
| Training harness | Current CPU/SLURM-compatible generated worker scripts | Similar worker-script model, originally more SLURM/GPU-oriented |
| Evaluation | `val_bpb` parsed from current CIFAR-10 substrate, deterministic workflow artifacts | `val_bpb` from AutoResearch-at-home style runs and analysis scripts |
| Outputs | `turns.jsonl`, `training_runs.jsonl`, snapshots, reasoning traces, decomposition artifacts | `trajectory.jsonl`, metadata, blackboard, coordinator logs, analysis figures |
| Experiment management | Configs for `d00/d10/d01/d11`, workflow DAG, calibration gates | `run-swarm` config and standalone analysis directories |
| Dependencies | Python package with Anthropic, pandas, matplotlib, PyYAML, tabulate | Similar dependencies |

## Imported Logic

The import is intentionally additive and reversible.

Added package:

- `src/agent_parallelization_new/imported_swarms/shared_memory.py`
- `src/agent_parallelization_new/imported_swarms/swarm_config.py`
- `src/agent_parallelization_new/imported_swarms/coordinator.py`
- `src/agent_parallelization_new/imported_swarms/coordinator_local.py`
- `src/agent_parallelization_new/imported_swarms/claude_agent_runner.py`
- `src/agent_parallelization_new/imported_swarms/swarm_agent_runner.py`
- `src/agent_parallelization_new/imported_swarms/swarm_orchestrator.py`
- `src/agent_parallelization_new/imported_swarms/workspace.py`
- `src/agent_parallelization_new/imported_swarms/launcher.py`

Added mode and entry points:

- `src/agent_parallelization_new/experiment_modes/imported_swarm.py`
- `run-imported-swarm` in `pyproject.toml`
- `configs/imported_swarms/experiment_imported_swarm.yaml`

Added prompt/protocol files:

- `templates/imported_swarms/agent_system_prompt.md`
- `templates/imported_swarms/agent_first_message.md`
- `templates/imported_swarms/collab.md`
- `templates/imported_swarms/program.md`

Added tests:

- `tests/test_imported_swarms_shared_memory.py`

Adaptations made during import:

- Package imports were rewritten from `agent_swarms.*` to `agent_parallelization_new.imported_swarms.*` or current package equivalents.
- The imported workspace setup is isolated in `imported_swarms/workspace.py` and runs after the current repository's normal workspace creation, so the existing workspace builder was not modified.
- The imported mode writes to `mode_imported_swarm/`, not `mode_parallel_shared/`.
- The imported prompts were adjusted to the current repository's worker-script and CIFAR-10 substrate context.
- The imported coordinator CLI is copied into each imported-swarm workspace as `coordinator.py`, while `coordinator_local.py` is also available for class-style compatibility.

## Redundancy and Structural Issues

Now that both implementations coexist, the main redundancies are:

- Two memory-sharing designs: native `shared_results_log.jsonl` and imported `shared_memory.jsonl` blackboard.
- Two Claude runners: current instrumented multi-turn runner and imported single-turn swarm runner.
- Two coordination protocols: native prompt-injected memory vs imported coordinator CLI.
- Duplicate config concepts: native `use_shared_memory` / `use_external_memory` flags vs imported `SwarmConfig`.
- Duplicate prompt families under `templates/` and `templates/imported_swarms/`.
- Duplicate output shapes: native decomposition tooling expects richer `turns.jsonl` and `training_runs.jsonl`; imported swarm currently produces the simpler trajectory/metadata style inherited from the cloned repo.
- The cloned repository remains as a nested Git repo at `agents-swarms/`, while the port exists under the current package. This is intentional for preservation, but it creates two source-of-truth locations until the import is reviewed.
- Historical analyses imported from the clone use older naming such as `agent_parallelisation_new` and prior run paths. They are useful archive material under `results/imported_swarms/analysis/`, not current workflow inputs.

## Recommended Reorganization Plan

Do not perform this cleanup automatically yet.

1. Preserve the current state until the imported mode has at least one smoke run.
2. Rename concepts rather than collapse them:
   - Native `parallel_shared` = `prompt_shared_memory`.
   - Imported `imported_swarm` = `blackboard_swarm`.
3. Move memory implementations into a clear namespace:
   - `memory/prompt_shared.py` for the current compact trace/table injection.
   - `memory/blackboard.py` for the imported typed JSONL blackboard.
4. Move runner variants into a clear namespace:
   - `runners/instrumented_multi_turn.py` for the native runner.
   - `runners/imported_single_turn.py` for the imported runner.
5. Normalize output schemas before using imported-swarm data in BP decomposition:
   - Either teach the imported runner to emit native `turns.jsonl` and `training_runs.jsonl`, or add explicit adapters in `scripts/compute_decomposition.py`.
6. Keep the cloned repo under `agents-swarms/` as archive/source material until the port is validated.
7. After validation, decide whether the nested clone should become:
   - an ignored external reference,
   - a Git submodule,
   - or a documented archive snapshot.
8. Only after that decision, consider merging shared utilities such as blackboard locking, best-source sidecars, and claim deduplication into native modes.

## Validation

Executed:

```bash
python -m pytest tests/test_imported_swarms_shared_memory.py
python -m agent_parallelization_new.imported_swarms.launcher --help
python -m pytest tests
```

Result:

- Targeted imported-swarm shared-memory tests passed.
- The full suite reported 82 passed and 9 failed. The failures were in pre-existing, untouched areas: `tests/test_capacity_benchmark.py` fails because a local multiprocessing worker cannot be pickled under the active spawn context, and `tests/test_merger.py` fails around hyperparameter return types plus an outdated `MergeOrchestrator.run(evaluate=False)` call signature.
- I did not run a full `run-imported-swarm` experiment because that would launch Claude agent sessions and training workers.
