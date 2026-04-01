"""Mode 1: N fully independent agents × T budget."""

from __future__ import annotations

from pathlib import Path

from agent_parallelization_new.config import ExperimentConfig
from agent_parallelization_new.orchestrator import Orchestrator
from agent_parallelization_new.outputs.collector import collect_experiment
from agent_parallelization_new.outputs.reporter import write_experiment_report


def run_parallel_experiment(
    config: ExperimentConfig,
    experiment_dir: Path,
    repo_root: Path,
    system_prompt: str,
    first_message_template: str,
) -> None:
    """Run Mode 1: N agents x T budget.

    Agents are:
    - Fully independent (no shared context, workspace, or files)
    - Launched simultaneously
    - All given the same time budget T
    - Results collected only AFTER all finish

    Total compute = N×T. Wall-clock time ≈ T.
    """
    assert config.mode == "parallel", f"Expected mode=parallel, got {config.mode}"
    assert len(config.agents) >= 1, f"Parallel mode expects at least 1 agent, got {len(config.agents)}"

    orchestrator = Orchestrator(config=config, repo_root=repo_root)
    orchestrator.run_parallel(
        experiment_dir=experiment_dir,
        system_prompt=system_prompt,
        first_message_template=first_message_template,
    )

    # Collect after all agents finish
    agent_ids = [a.agent_id for a in config.agents]
    summary = collect_experiment(
        experiment_dir=experiment_dir,
        experiment_id=config.experiment_id,
        mode="parallel",
        agent_ids=agent_ids,
    )

    mode_dir = experiment_dir / "mode_parallel"
    write_experiment_report(summary, mode_dir)

    return summary
