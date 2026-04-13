#!/usr/bin/env python3
"""Analyze power calibration results from d00 vs d10 runs.

Computes effect sizes, mode diversity, and cost variance.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def collect_cell_data(pattern: str, repo_root: Path) -> dict:
    """Collect all training run val_bpb values and mode labels for a cell."""
    val_bpbs: list[float] = []
    mode_labels: list[str] = []
    accepted_modes: list[str] = []
    token_costs: list[float] = []
    wall_costs: list[float] = []

    for exp_dir in sorted(repo_root.glob(pattern)):
        for agent_dir in sorted(exp_dir.glob("mode_*/agent_*")):
            # Training runs
            for row in load_jsonl(agent_dir / "results" / "training_runs.jsonl"):
                val = row.get("val_bpb")
                if val is not None:
                    val_bpbs.append(float(val))

            # Mode labels
            for row in load_jsonl(agent_dir / "results" / "mode_labels.jsonl"):
                mode = row.get("mode")
                if mode:
                    mode_labels.append(str(mode))
                    if row.get("accepted") is True:
                        accepted_modes.append(str(mode))

            # Turn costs
            for row in load_jsonl(agent_dir / "results" / "turns.jsonl"):
                tokens = (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0)
                if tokens:
                    token_costs.append(float(tokens))
                wall = row.get("wall_clock_seconds")
                if wall is not None:
                    wall_costs.append(float(wall))

    return {
        "val_bpbs": val_bpbs,
        "mode_labels": mode_labels,
        "accepted_modes": accepted_modes,
        "token_costs": token_costs,
        "wall_costs": wall_costs,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def cohens_d(group1: list[float], group2: list[float]) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = sample_std(group1), sample_std(group2)
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled > 0 else float("nan")


def jensen_gap(values: list[float]) -> float:
    if not values or any(v <= 0 for v in values):
        return float("nan")
    return math.log(mean(values)) - mean([math.log(v) for v in values])


def analyze_cell(data: dict) -> dict:
    vals = data["val_bpbs"]
    mode_counts = Counter(data["mode_labels"])
    accepted_counts = Counter(data["accepted_modes"])
    modes_with_2plus = sum(1 for c in accepted_counts.values() if c >= 2)

    return {
        "n_runs": len(vals),
        "mean_val_bpb": mean(vals),
        "std_val_bpb": sample_std(vals),
        "min_val_bpb": min(vals) if vals else None,
        "max_val_bpb": max(vals) if vals else None,
        "n_mode_labels": len(data["mode_labels"]),
        "n_accepted_modes": len(data["accepted_modes"]),
        "distinct_modes": len(mode_counts),
        "distinct_accepted_modes": len(accepted_counts),
        "modes_with_2plus_accepted": modes_with_2plus,
        "mode_distribution": dict(mode_counts),
        "accepted_mode_distribution": dict(accepted_counts),
        "jensen_gap_token": jensen_gap(data["token_costs"]),
        "jensen_gap_wall": jensen_gap(data["wall_costs"]),
        "mean_token_cost": mean(data["token_costs"]),
        "mean_wall_cost": mean(data["wall_costs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--d00-pattern", default="runs/experiment_calibration_d00_rep*")
    parser.add_argument("--d10-pattern", default="runs/experiment_calibration_d10_rep*")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    d00_data = collect_cell_data(args.d00_pattern, repo_root)
    d10_data = collect_cell_data(args.d10_pattern, repo_root)

    d00_analysis = analyze_cell(d00_data)
    d10_analysis = analyze_cell(d10_data)

    d = cohens_d(d00_data["val_bpbs"], d10_data["val_bpbs"])

    result = {
        "d00": d00_analysis,
        "d10": d10_analysis,
        "cohens_d": d,
        "effect_interpretation": (
            "large" if abs(d) > 0.8 else
            "medium" if abs(d) > 0.5 else
            "small" if abs(d) > 0.3 else
            "negligible"
        ),
        "mode_diversity_ok": (
            d00_analysis["modes_with_2plus_accepted"] >= 3
            and d10_analysis["modes_with_2plus_accepted"] >= 3
        ),
        "sample_size_ok": d00_analysis["n_runs"] >= 50 and d10_analysis["n_runs"] >= 50,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Calibration Analysis:")
    print(f"  d00: n={d00_analysis['n_runs']}, "
          f"mean={d00_analysis['mean_val_bpb']:.4f}, "
          f"modes={d00_analysis['distinct_accepted_modes']}")
    print(f"  d10: n={d10_analysis['n_runs']}, "
          f"mean={d10_analysis['mean_val_bpb']:.4f}, "
          f"modes={d10_analysis['distinct_accepted_modes']}")
    print(f"  Cohen's d: {d:.3f} ({result['effect_interpretation']})")
    print(f"  Mode diversity OK: {result['mode_diversity_ok']}")
    print(f"  Sample size OK: {result['sample_size_ok']}")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
