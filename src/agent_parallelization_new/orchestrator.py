"""Manages parallel process spawning for agent experiments.

Responsibilities:
- Initialize experiment directory
- Create isolated workspaces (one per agent)
- Launch agent processes concurrently
- Enforce wall-clock budget per agent
- Ensure zero cross-agent file access
- Wait for all agents to finish
- Hand off to collector.py

Must NOT:
- Read one agent's results and pass them to another
- Merge trajectories during the run
- Act as a central planner that changes agent behavior
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_parallelization_new.agents.isolated_agent_process import IsolatedAgentProcess
from agent_parallelization_new.config import AgentConfig, ExperimentConfig
from agent_parallelization_new.utils.workspace import create_workspace, destroy_workspace


class Orchestrator:
    """Coordinates multi-agent experiments.

    Modes supported:
      run_parallel()   — N independent agents in parallel (N from config.agents)
      run_single()     — 1 agent with 2× budget
      run_merge()      — post-hoc merge of a completed parallel run
    """

    POLL_INTERVAL_SEC = 10

    def __init__(self, config: ExperimentConfig, repo_root: Path):
        self.config = config
        self.repo_root = repo_root
        self.autoresearch_dir = repo_root / config.autoresearch_dir

    def run_parallel(
        self,
        experiment_dir: Path,
        system_prompt: str,
        first_message_template: str,
    ) -> None:
        """Launch all agents simultaneously. Block until all finish or budgets expire."""
        self._validate_gpu_assignments()
        mode_dir = experiment_dir / "mode_parallel"
        run_id = self.config.experiment_id

        # Write experiment manifest
        manifest_path = experiment_dir / "config.json"
        manifest_path.write_text(json.dumps(self.config.to_dict(), indent=2))

        # Set up workspaces and build processes
        processes: list[IsolatedAgentProcess] = []
        hard_deadlines: list[float] = []

        for agent_config in self.config.agents:
            agent_dir, workspace = self._setup_agent(agent_config, mode_dir, run_id)
            first_message = _render_first_message(
                template=first_message_template,
                agent_config=agent_config,
                run_id=run_id,
                experiment_id=self.config.experiment_id,
                workspace=workspace,
            )
            proc = IsolatedAgentProcess(
                config=agent_config,
                workspace=workspace,
                agent_dir=agent_dir,
                run_id=run_id,
                experiment_id=self.config.experiment_id,
                system_prompt=system_prompt,
                first_message=first_message,
            )
            processes.append(proc)
            hard_deadlines.append(
                time.monotonic() + agent_config.time_budget_minutes * 60 * 3
            )

        # Launch all agents simultaneously — no stagger, no communication
        for proc in processes:
            proc.start()

        print(f"[orchestrator] Launched {len(processes)} agent(s) simultaneously.")

        # Wait for all agents to finish or hit their hard deadlines
        self._wait_for_all(processes, hard_deadlines)

        print(f"[orchestrator] All {len(processes)} agents finished.")

    def run_single(
        self,
        experiment_dir: Path,
        system_prompt: str,
        first_message_template: str,
    ) -> None:
        """Launch one agent with double budget."""
        mode_dir = experiment_dir / "mode_single_long"
        run_id = self.config.experiment_id

        manifest_path = experiment_dir / "config.json"
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(self.config.to_dict(), indent=2))

        assert len(self.config.agents) == 1, "single_long mode expects exactly 1 agent"
        agent_config = self.config.agents[0]

        agent_dir, workspace = self._setup_agent(agent_config, mode_dir, run_id)
        first_message = _render_first_message(
            template=first_message_template,
            agent_config=agent_config,
            run_id=run_id,
            experiment_id=self.config.experiment_id,
            workspace=workspace,
        )
        proc = IsolatedAgentProcess(
            config=agent_config,
            workspace=workspace,
            agent_dir=agent_dir,
            run_id=run_id,
            experiment_id=self.config.experiment_id,
            system_prompt=system_prompt,
            first_message=first_message,
        )

        hard_deadline = time.monotonic() + agent_config.time_budget_minutes * 60 * 3
        proc.start()
        print(f"[orchestrator] Launched single agent {agent_config.agent_id}.")
        self._wait_for_all([proc], [hard_deadline])
        print("[orchestrator] Single agent finished.")

    def run_merge(
        self,
        experiment_dir: Path,
        source_mode: str = "parallel",
        evaluate: bool = False,
    ) -> None:
        """Run the merge phase on a completed parallel experiment.

        Parameters
        ----------
        experiment_dir : Path
            The experiment root produced by a previous parallel run.
        source_mode : str
            Which mode directory to read agent results from (default "parallel").
        evaluate : bool
            If True, attempt to evaluate the merged train.py via SLURM.
        """
        from agent_parallelization_new.merger import MergeOrchestrator

        print(f"[orchestrator] Starting merge phase for {experiment_dir.name}")
        merger = MergeOrchestrator(
            experiment_dir=experiment_dir,
            autoresearch_dir=self.autoresearch_dir,
            mode=source_mode,
        )
        results = merger.run(evaluate=evaluate, agent_based=True)
        print(
            f"[orchestrator] Merge complete. "
            f"best_individual={results.best_individual_val_bpb}, "
            f"merge={results.merge_val_bpb}, "
            f"merge_won={results.merge_won}"
        )

    def _setup_agent(
        self, agent_config: AgentConfig, mode_dir: Path, run_id: str
    ) -> tuple[Path, Path]:
        """Create workspace and result dirs for one agent. Returns (agent_dir, workspace)."""
        agent_dir = mode_dir / agent_config.agent_id
        workspace = agent_dir / "workspace"
        results_root = agent_dir / "results"

        branch_name = f"claude/{self.config.experiment_id}/{agent_config.agent_id}"

        create_workspace(
            autoresearch_dir=self.autoresearch_dir,
            workspace_path=workspace,
            branch_name=branch_name,
            train_budget_seconds=agent_config.train_time_budget_seconds,
            run_id=run_id,
            agent_id=agent_config.agent_id,
            results_root=results_root,
            slurm_partition=self.config.slurm_partition,
            slurm_gres=self.config.slurm_gres,
            slurm_time=self.config.slurm_time,
            agent_time_budget_minutes=agent_config.time_budget_minutes,
        )
        (agent_dir / "logs").mkdir(parents=True, exist_ok=True)
        return agent_dir, workspace

    def _validate_gpu_assignments(self) -> None:
        devices = [a.cuda_device for a in self.config.agents]
        if len(devices) != len(set(devices)):
            raise ValueError(
                f"Two agents assigned the same GPU: {devices}. "
                "Each agent must have a unique CUDA_VISIBLE_DEVICES."
            )

    def _wait_for_all(
        self,
        processes: list[IsolatedAgentProcess],
        hard_deadlines: list[float],
    ) -> None:
        """Poll until all processes finish or hard deadlines hit."""
        while True:
            now = time.monotonic()
            all_done = True
            for proc, deadline in zip(processes, hard_deadlines):
                if proc.is_alive():
                    if now >= deadline:
                        print(
                            f"[orchestrator] Hard deadline reached for "
                            f"{proc.config.agent_id}, sending SIGTERM."
                        )
                        proc.terminate()
                        time.sleep(2)
                        proc.kill()
                    else:
                        all_done = False
            if all_done:
                break
            time.sleep(self.POLL_INTERVAL_SEC)


def _render_first_message(
    template: str,
    agent_config: AgentConfig,
    run_id: str,
    experiment_id: str,
    workspace: Path,
) -> str:
    """Substitute template variables in the first message."""
    return (
        template
        .replace("{{AGENT_ID}}", agent_config.agent_id)
        .replace("{{RUN_ID}}", run_id)
        .replace("{{EXPERIMENT_ID}}", experiment_id)
        .replace("{{TIME_BUDGET}}", str(agent_config.time_budget_minutes))
        .replace("{{TRAIN_TIME_BUDGET}}", str(agent_config.train_time_budget_seconds))
        .replace("{{WORKSPACE}}", str(workspace))
        .replace("{{BRANCH}}", f"claude/{experiment_id}/{agent_config.agent_id}")
    )
