"""CLI entry point for the imported agents-swarms blackboard mode."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from agent_parallelization_new.config import ExperimentConfig
from agent_parallelization_new.experiment_modes.imported_swarm import (
    run_imported_swarm_experiment,
)
from agent_parallelization_new.imported_swarms import IMPORTED_SWARM_MODE
from agent_parallelization_new.imported_swarms.swarm_config import SwarmConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text()


def _render_template(
    template: str,
    train_budget_seconds: int,
    slurm_enabled: bool,
) -> str:
    train_min = max(1, train_budget_seconds // 60)
    compute_device = "GPU" if slurm_enabled else "CPU worker"
    resource_metric = "VRAM" if slurm_enabled else "memory"
    return (
        template.replace("{{TRAIN_TIME_BUDGET_MIN}}", str(train_min))
        .replace("{{COMPUTE_DEVICE}}", compute_device)
        .replace("{{RESOURCE_METRIC}}", resource_metric)
    )


def _make_experiment_id(prefix: str = IMPORTED_SWARM_MODE) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _load_swarm_config(path: Path | None) -> SwarmConfig:
    if path is None:
        return SwarmConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    return SwarmConfig.from_dict(raw.get("swarm", {}))


def main_imported_swarm(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run imported agents-swarms mode with a shared JSONL blackboard"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--time-budget", type=int, default=30)
    parser.add_argument("--train-budget", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default="runs")
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    config_path = Path(args.config) if args.config else None

    if config_path is not None:
        config = ExperimentConfig.from_yaml(config_path, repo_root=str(repo_root))
        config.mode = IMPORTED_SWARM_MODE
        if args.experiment_id is not None:
            config.experiment_id = args.experiment_id
    else:
        config = ExperimentConfig.make_n_parallel(
            experiment_id=args.experiment_id or _make_experiment_id(),
            n_agents=args.n_agents,
            time_budget_minutes=args.time_budget,
            train_time_budget_seconds=args.train_budget,
            repo_root=str(repo_root),
        )
        config.mode = IMPORTED_SWARM_MODE
        config.system_prompt_file = "templates/imported_swarms/agent_system_prompt.md"
        config.first_message_file = "templates/imported_swarms/agent_first_message.md"

    swarm_config = _load_swarm_config(config_path)
    runs_dir = repo_root / (args.runs_dir if config_path is None else "runs")
    experiment_dir = runs_dir / f"experiment_{config.experiment_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = _render_template(
        _load_template(repo_root / config.system_prompt_file),
        config.train_time_budget_seconds,
        config.slurm_enabled,
    )
    first_message_tmpl = _render_template(
        _load_template(repo_root / config.first_message_file),
        config.train_time_budget_seconds,
        config.slurm_enabled,
    )

    print(f"[launcher] Starting imported swarm experiment: {config.experiment_id}")
    print(
        f"[launcher] Agents: {len(config.agents)} | "
        f"Budget: {config.base_time_budget_minutes} min | "
        f"Train: {config.train_time_budget_seconds} s"
    )
    print(
        f"[launcher] Blackboard: {swarm_config.shared_memory_file} | "
        f"max_context: {swarm_config.max_context_entries}"
    )
    print(f"[launcher] Output directory: {experiment_dir}")

    run_imported_swarm_experiment(
        config=config,
        experiment_dir=experiment_dir,
        repo_root=repo_root,
        system_prompt=system_prompt,
        first_message_template=first_message_tmpl,
        swarm_config=swarm_config,
    )
    print(f"[launcher] Imported swarm experiment complete. Results: {experiment_dir}")


if __name__ == "__main__":
    main_imported_swarm()
