"""Imported swarm orchestrator with a shared JSONL blackboard.

This is a port of ``agents-swarms/src/agent_swarms/swarm_orchestrator.py`` into
the current package namespace. It intentionally uses a distinct
``mode_imported_swarm`` directory so it does not alter the native
``parallel_shared`` implementation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from agent_parallelization_new.config import AgentConfig, ExperimentConfig
from agent_parallelization_new.imported_swarms import IMPORTED_SWARM_MODE
from agent_parallelization_new.imported_swarms.shared_memory import SharedMemory
from agent_parallelization_new.imported_swarms.swarm_agent_runner import (
    IsolatedSwarmAgentProcess,
)
from agent_parallelization_new.imported_swarms.swarm_config import SwarmConfig
from agent_parallelization_new.imported_swarms.workspace import (
    install_imported_swarm_tools,
)
from agent_parallelization_new.orchestrator import Orchestrator
from agent_parallelization_new.utils.workspace import create_workspace


class ImportedSwarmOrchestrator(Orchestrator):
    """Coordinates a swarm run where agents communicate via a blackboard."""

    def __init__(
        self,
        config: ExperimentConfig,
        repo_root: Path,
        swarm_config: SwarmConfig,
    ) -> None:
        super().__init__(config=config, repo_root=repo_root)
        self.swarm_config = swarm_config
        self._swarm_memory_path: Path | None = None

    def run_swarm(
        self,
        experiment_dir: Path,
        system_prompt: str,
        first_message_template: str,
    ) -> None:
        """Launch all imported-swarm agents simultaneously."""
        self._validate_gpu_assignments()
        self.config.mode = IMPORTED_SWARM_MODE
        mode_dir = experiment_dir / f"mode_{IMPORTED_SWARM_MODE}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        run_id = self.config.experiment_id

        manifest_path = experiment_dir / "config.json"
        manifest_path.write_text(json.dumps(self.config.to_dict(), indent=2))

        shared_memory_path = mode_dir / self.swarm_config.shared_memory_file
        self._swarm_memory_path = shared_memory_path
        SharedMemory(
            path=shared_memory_path,
            max_context_entries=self.swarm_config.max_context_entries,
        )
        print(f"[imported-swarm] Shared blackboard: {shared_memory_path}", flush=True)

        processes: list[IsolatedSwarmAgentProcess] = []
        hard_deadlines: list[float] = []

        for agent_config in self.config.agents:
            agent_dir, workspace = self._setup_swarm_agent(agent_config, mode_dir, run_id)
            first_message = _render_swarm_first_message(
                template=first_message_template,
                agent_config=agent_config,
                run_id=run_id,
                experiment_id=self.config.experiment_id,
                workspace=workspace,
            )
            proc = IsolatedSwarmAgentProcess(
                config=agent_config,
                workspace=workspace,
                agent_dir=agent_dir,
                run_id=run_id,
                experiment_id=self.config.experiment_id,
                system_prompt=system_prompt,
                first_message=first_message,
                shared_memory_path=shared_memory_path,
            )
            processes.append(proc)
            hard_deadlines.append(
                time.monotonic() + agent_config.time_budget_minutes * 60 * 3
            )

        self._register_cleanup()
        self._processes = processes  # type: ignore[assignment]

        for proc in processes:
            proc.start()

        print(
            f"[imported-swarm] Launched {len(processes)} agent(s) simultaneously.",
            flush=True,
        )
        self._wait_for_imported_swarm(processes, hard_deadlines)
        print(f"[imported-swarm] All {len(processes)} agents finished.", flush=True)

    def _setup_swarm_agent(
        self,
        agent_config: AgentConfig,
        mode_dir: Path,
        run_id: str,
    ) -> tuple[Path, Path]:
        """Create one workspace and install imported-swarm coordination tools."""
        agent_dir = mode_dir / agent_config.agent_id
        workspace = agent_dir / "workspace"
        results_root = agent_dir / "results"
        branch_name = f"imported-swarm/{self.config.experiment_id}/{agent_config.agent_id}"

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
            use_slurm=self.config.slurm_enabled,
            agent_time_budget_minutes=agent_config.time_budget_minutes,
            experiment_mode=IMPORTED_SWARM_MODE,
        )
        if self._swarm_memory_path is None:
            raise RuntimeError("shared memory path was not initialized")
        install_imported_swarm_tools(
            repo_root=self.repo_root,
            workspace_path=workspace,
            swarm_memory_path=self._swarm_memory_path,
        )
        (agent_dir / "logs").mkdir(parents=True, exist_ok=True)
        return agent_dir, workspace

    def _wait_for_imported_swarm(
        self,
        processes: list[IsolatedSwarmAgentProcess],
        hard_deadlines: list[float],
    ) -> None:
        """Poll until all imported-swarm processes finish or hit deadlines."""
        while True:
            now = time.monotonic()
            all_done = True
            for proc, deadline in zip(processes, hard_deadlines):
                if proc.is_alive():
                    if now >= deadline:
                        print(
                            f"[imported-swarm] Hard deadline reached for "
                            f"{proc.config.agent_id}, sending SIGTERM.",
                            flush=True,
                        )
                        proc.terminate()
                        time.sleep(2)
                        proc.kill()
                    else:
                        all_done = False
            if all_done:
                break
            time.sleep(self.POLL_INTERVAL_SEC)


def _render_swarm_first_message(
    template: str,
    agent_config: AgentConfig,
    run_id: str,
    experiment_id: str,
    workspace: Path,
) -> str:
    """Substitute template variables for the imported swarm first message."""
    return (
        template
        .replace("{{AGENT_ID}}", agent_config.agent_id)
        .replace("{{RUN_ID}}", run_id)
        .replace("{{EXPERIMENT_ID}}", experiment_id)
        .replace("{{TIME_BUDGET}}", str(agent_config.time_budget_minutes))
        .replace("{{TRAIN_TIME_BUDGET}}", str(agent_config.train_time_budget_seconds))
        .replace("{{WORKSPACE}}", str(workspace))
        .replace("{{BRANCH}}", f"imported-swarm/{experiment_id}/{agent_config.agent_id}")
    )

