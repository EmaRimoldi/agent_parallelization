#!/usr/bin/env python3
"""Analyze per-cell cost variance and Jensen-gap proxies from pilot logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODE_DIR_BY_CELL = {
    "d00": "mode_single_long",
    "d10": "mode_single_memory",
    "d01": "mode_parallel",
    "d11": "mode_parallel_shared",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def chars4_proxy(turn: dict) -> float:
    return (
        float(turn.get("system_prompt_chars", 0))
        + float(turn.get("turn_msg_chars", 0))
        + float(turn.get("response_chars", 0))
    ) / 4.0


def token_value(turn: dict) -> float:
    observed = (turn.get("input_tokens") or 0) + (turn.get("output_tokens") or 0)
    return float(observed) if observed else chars4_proxy(turn)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def variance(values: list[float]) -> float:
    mu = mean(values)
    return sum((value - mu) ** 2 for value in values) / len(values)


def std(values: list[float]) -> float:
    return math.sqrt(variance(values))


def empirical_jensen_gap(values: list[float]) -> float:
    mu = mean(values)
    return math.log(mu) - mean([math.log(value) for value in values if value > 0])


def delta_method_gap(values: list[float]) -> float:
    mu = mean(values)
    return 0.5 * variance(values) / (mu ** 2) if mu > 0 else float("nan")


def collect_turn_metrics(experiment_dir: Path, cell: str) -> tuple[list[float], list[float]]:
    token_values = []
    wall_values = []
    mode_dir = experiment_dir / MODE_DIR_BY_CELL[cell]
    for agent_dir in sorted(mode_dir.glob("agent_*")):
        for row in load_jsonl(agent_dir / "results" / "turns.jsonl"):
            if row.get("wall_clock_seconds") is not None:
                wall_values.append(float(row["wall_clock_seconds"]))
            token_values.append(token_value(row))
    return token_values, wall_values


def summarize(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "variance": float("nan"),
            "std": float("nan"),
            "empirical_jensen_gap": float("nan"),
            "delta_method_gap": float("nan"),
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "variance": variance(values),
        "std": std(values),
        "empirical_jensen_gap": empirical_jensen_gap(values),
        "delta_method_gap": delta_method_gap(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    mapping = json.loads(args.mapping.read_text())

    summary = {}
    for cell, experiments in mapping.items():
        token_values = []
        wall_values = []
        for experiment in experiments:
            tokens, walls = collect_turn_metrics((repo_root / experiment).resolve(), cell)
            token_values.extend(tokens)
            wall_values.extend(walls)
        summary[cell] = {
            "token": summarize(token_values),
            "wall": summarize(wall_values),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
