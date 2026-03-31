"""Experiment and agent configuration dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class AgentConfig:
    """Per-agent configuration."""
    agent_id: str
    time_budget_minutes: int = 30
    train_time_budget_seconds: int = 300
    cuda_device: str = "0"
    model: str = "claude-sonnet-4-6"
    temperature: Optional[float] = None
    system_prompt_file: str = "templates/agent_system_prompt.md"
    first_message_file: str = "templates/agent_first_message.md"

    @classmethod
    def from_json(cls, path: Path) -> "AgentConfig":
        data = json.loads(path.read_text())
        # Map old-style fields
        if "google_model" in data:
            data.pop("google_model", None)
        if "provider" in data:
            data.pop("provider", None)
        if "thinking" in data:
            data.pop("thinking", None)
        if "prompt_file" in data:
            data.pop("prompt_file", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    experiment_id: str
    mode: str  # "parallel" or "single_long"
    base_time_budget_minutes: int = 30
    train_time_budget_seconds: int = 300
    autoresearch_dir: str = "autoresearch"
    results_root: str = "results"
    agents: list[AgentConfig] = field(default_factory=list)
    repo_root: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def make_parallel(
        cls,
        experiment_id: str,
        time_budget_minutes: int,
        train_time_budget_seconds: int,
        repo_root: str,
    ) -> "ExperimentConfig":
        agents = [
            AgentConfig(
                agent_id="agent_0",
                time_budget_minutes=time_budget_minutes,
                train_time_budget_seconds=train_time_budget_seconds,
                cuda_device="0",
            ),
            AgentConfig(
                agent_id="agent_1",
                time_budget_minutes=time_budget_minutes,
                train_time_budget_seconds=train_time_budget_seconds,
                cuda_device="1",
            ),
        ]
        return cls(
            experiment_id=experiment_id,
            mode="parallel",
            base_time_budget_minutes=time_budget_minutes,
            train_time_budget_seconds=train_time_budget_seconds,
            repo_root=repo_root,
            agents=agents,
        )

    @classmethod
    def make_single_long(
        cls,
        experiment_id: str,
        time_budget_minutes: int,
        train_time_budget_seconds: int,
        repo_root: str,
    ) -> "ExperimentConfig":
        agents = [
            AgentConfig(
                agent_id="agent_0",
                time_budget_minutes=time_budget_minutes * 2,
                train_time_budget_seconds=train_time_budget_seconds,
                cuda_device="0",
            ),
        ]
        return cls(
            experiment_id=experiment_id,
            mode="single_long",
            base_time_budget_minutes=time_budget_minutes,
            train_time_budget_seconds=train_time_budget_seconds,
            repo_root=repo_root,
            agents=agents,
        )
