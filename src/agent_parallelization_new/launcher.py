"""Entry point: parse args, pick mode, run experiment."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from agent_parallelization_new.config import ExperimentConfig
from agent_parallelization_new.experiment_modes.parallel_shared_memory import run_parallel_shared_memory
from agent_parallelization_new.experiment_modes.parallel_two_agents import run_parallel_experiment
from agent_parallelization_new.experiment_modes.single_agent_memory import run_single_agent_memory
from agent_parallelization_new.experiment_modes.single_agent_double_budget import run_single_long_experiment
from agent_parallelization_new.outputs.reporter import write_final_comparison


def _repo_root() -> Path:
    return Path(__file__).parents[2]  # src/ → repo root


def _load_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text()


def _render_first_message(template: str, train_budget_seconds: int) -> str:
    train_min = max(1, train_budget_seconds // 60)
    return template.replace("{{TRAIN_TIME_BUDGET_MIN}}", str(train_min))


def _make_experiment_id(prefix: str = "exp") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main_parallel(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run parallel-agent experiment (Mode 1)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment.yaml. If provided, all other flags are ignored.")
    parser.add_argument("--time-budget", type=int, default=30, help="Budget per agent (minutes)")
    parser.add_argument("--train-budget", type=int, default=300, help="Budget per training run (seconds)")
    parser.add_argument("--n-agents", type=int, default=2, help="Number of parallel agents")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default="runs")
    args = parser.parse_args(argv)

    repo_root = _repo_root()

    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config), repo_root=str(repo_root))
    else:
        experiment_id = args.experiment_id or _make_experiment_id("parallel")
        config = ExperimentConfig.make_n_parallel(
            experiment_id=experiment_id,
            n_agents=args.n_agents,
            time_budget_minutes=args.time_budget,
            train_time_budget_seconds=args.train_budget,
            repo_root=str(repo_root),
        )

    runs_dir = repo_root / (args.runs_dir if not args.config else "runs")
    experiment_dir = runs_dir / f"experiment_{config.experiment_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = _load_template(repo_root / config.system_prompt_file)
    first_message_tmpl = _render_first_message(
        _load_template(repo_root / config.first_message_file),
        config.train_time_budget_seconds,
    )

    print(f"[launcher] Starting parallel experiment: {config.experiment_id}")
    print(f"[launcher] Agents: {len(config.agents)}  |  Budget: {config.base_time_budget_minutes} min  |  Train: {config.train_time_budget_seconds} s")
    print(f"[launcher] SLURM: partition={config.slurm_partition}  gres={config.slurm_gres}  time={config.slurm_time}")
    print(f"[launcher] Output directory: {experiment_dir}")

    run_parallel_experiment(
        config=config,
        experiment_dir=experiment_dir,
        repo_root=repo_root,
        system_prompt=system_prompt,
        first_message_template=first_message_tmpl,
    )
    print(f"[launcher] Parallel experiment complete. Results: {experiment_dir}")


def main_parallel_shared(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run parallel-agent experiment with shared memory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment.yaml. If provided, all other flags are ignored.")
    parser.add_argument("--time-budget", type=int, default=30, help="Budget per agent (minutes)")
    parser.add_argument("--train-budget", type=int, default=300, help="Budget per training run (seconds)")
    parser.add_argument("--n-agents", type=int, default=2, help="Number of parallel agents")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default="runs")
    args = parser.parse_args(argv)

    repo_root = _repo_root()

    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config), repo_root=str(repo_root))
    else:
        experiment_id = args.experiment_id or _make_experiment_id("parallel_shared")
        config = ExperimentConfig.make_n_parallel(
            experiment_id=experiment_id,
            n_agents=args.n_agents,
            time_budget_minutes=args.time_budget,
            train_time_budget_seconds=args.train_budget,
            repo_root=str(repo_root),
        )
        config.mode = "parallel_shared"
        for agent in config.agents:
            agent.use_shared_memory = True

    runs_dir = repo_root / (args.runs_dir if not args.config else "runs")
    experiment_dir = runs_dir / f"experiment_{config.experiment_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = _load_template(repo_root / config.system_prompt_file)
    first_message_tmpl = _render_first_message(
        _load_template(repo_root / config.first_message_file),
        config.train_time_budget_seconds,
    )

    print(f"[launcher] Starting parallel-shared experiment: {config.experiment_id}")
    print(f"[launcher] Agents: {len(config.agents)}  |  Budget: {config.base_time_budget_minutes} min  |  Train: {config.train_time_budget_seconds} s")
    print(f"[launcher] SLURM: partition={config.slurm_partition}  gres={config.slurm_gres}  time={config.slurm_time}")
    print(f"[launcher] Output directory: {experiment_dir}")

    run_parallel_shared_memory(
        config=config,
        experiment_dir=experiment_dir,
        repo_root=repo_root,
        system_prompt=system_prompt,
        first_message_template=first_message_tmpl,
    )
    print(f"[launcher] Parallel-shared experiment complete. Results: {experiment_dir}")


def main_single_long(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run single-agent-longer experiment (Mode 2)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment.yaml. If provided, all other flags are ignored.")
    parser.add_argument("--time-budget", type=int, default=30, help="Base budget T (minutes); agent gets 2T")
    parser.add_argument("--train-budget", type=int, default=300, help="Budget per training run (seconds)")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default="runs")
    args = parser.parse_args(argv)

    repo_root = _repo_root()

    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config), repo_root=str(repo_root))
    else:
        experiment_id = args.experiment_id or _make_experiment_id("single")
        config = ExperimentConfig.make_single_long(
            experiment_id=experiment_id,
            time_budget_minutes=args.time_budget,
            train_time_budget_seconds=args.train_budget,
            repo_root=str(repo_root),
        )

    runs_dir = repo_root / (args.runs_dir if not args.config else "runs")
    experiment_dir = runs_dir / f"experiment_{config.experiment_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = _load_template(repo_root / config.system_prompt_file)
    first_message_tmpl = _render_first_message(
        _load_template(repo_root / config.first_message_file),
        config.train_time_budget_seconds,
    )

    print(f"[launcher] Starting single-long experiment: {config.experiment_id}")
    print(f"[launcher] Output directory: {experiment_dir}")

    run_single_long_experiment(
        config=config,
        experiment_dir=experiment_dir,
        repo_root=repo_root,
        system_prompt=system_prompt,
        first_message_template=first_message_tmpl,
    )
    print(f"[launcher] Single-long experiment complete. Results: {experiment_dir}")


def main_single_memory(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run single-agent experiment with external memory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment.yaml. If provided, all other flags are ignored.")
    parser.add_argument("--time-budget", type=int, default=30, help="Base budget T (minutes); agent gets 2T")
    parser.add_argument("--train-budget", type=int, default=300, help="Budget per training run (seconds)")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default="runs")
    args = parser.parse_args(argv)

    repo_root = _repo_root()

    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config), repo_root=str(repo_root))
    else:
        experiment_id = args.experiment_id or _make_experiment_id("single_memory")
        config = ExperimentConfig.make_single_memory(
            experiment_id=experiment_id,
            time_budget_minutes=args.time_budget,
            train_time_budget_seconds=args.train_budget,
            repo_root=str(repo_root),
        )

    runs_dir = repo_root / (args.runs_dir if not args.config else "runs")
    experiment_dir = runs_dir / f"experiment_{config.experiment_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = _load_template(repo_root / config.system_prompt_file)
    first_message_tmpl = _render_first_message(
        _load_template(repo_root / config.first_message_file),
        config.train_time_budget_seconds,
    )

    print(f"[launcher] Starting single-memory experiment: {config.experiment_id}")
    print(f"[launcher] Output directory: {experiment_dir}")

    run_single_agent_memory(
        config=config,
        experiment_dir=experiment_dir,
        repo_root=repo_root,
        system_prompt=system_prompt,
        first_message_template=first_message_tmpl,
    )
    print(f"[launcher] Single-memory experiment complete. Results: {experiment_dir}")
