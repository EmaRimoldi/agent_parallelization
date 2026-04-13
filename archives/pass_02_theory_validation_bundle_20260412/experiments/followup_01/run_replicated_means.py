#!/usr/bin/env python3
"""Repeat incumbent candidate evaluations for each pilot cell.

This script avoids best-of-N optimism by reevaluating one incumbent candidate per
replicate and aggregating repeated evaluations at the cell level.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


REPEAT_PLAN = [2, 2, 1]
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


def parse_val_bpb(text: str) -> float | None:
    for line in text.splitlines():
        if line.startswith("val_bpb:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def find_agent_dirs(experiment_dir: Path, cell: str) -> list[Path]:
    mode_dir = experiment_dir / MODE_DIR_BY_CELL[cell]
    return sorted(mode_dir.glob("agent_*"))


def reconstruct_train_py(workspace: Path, commit: str, dest: Path) -> None:
    content = subprocess.run(
        ["git", "show", f"{commit}:train.py"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dest.write_text(content)


def select_candidate(experiment_dir: Path, cell: str) -> dict:
    """Select one reproducible incumbent candidate for this replicate."""
    candidates: list[dict] = []

    for agent_dir in find_agent_dirs(experiment_dir, cell):
        workspace = agent_dir / "workspace"
        snapshots_dir = agent_dir / "snapshots"

        for meta_path in sorted(snapshots_dir.glob("step_*/metadata.json")):
            try:
                metadata = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            val = metadata.get("val_bpb_after")
            commit = metadata.get("git_commit")
            if val is None or not commit:
                continue
            snapshot_train = meta_path.parent / "train.py"
            candidates.append(
                {
                    "source": "snapshot",
                    "value": float(val),
                    "commit": str(commit),
                    "train_py_path": snapshot_train,
                    "workspace": workspace,
                    "agent_id": agent_dir.name,
                    "accepted": metadata.get("accepted"),
                    "step_index": metadata.get("step_index"),
                }
            )

        for row in load_jsonl(agent_dir / "results" / "training_runs.jsonl"):
            val = row.get("val_bpb")
            commit = row.get("commit")
            if val is None or not commit:
                continue
            candidates.append(
                {
                    "source": "training_run",
                    "value": float(val),
                    "commit": str(commit),
                    "train_py_path": None,
                    "workspace": workspace,
                    "agent_id": agent_dir.name,
                    "accepted": None,
                    "step_index": None,
                }
            )

    if not candidates:
        raise RuntimeError(f"No candidate found for {experiment_dir}")

    accepted = [candidate for candidate in candidates if candidate.get("accepted") is True]
    pool = accepted if accepted else candidates
    return min(pool, key=lambda candidate: candidate["value"])


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def run_candidate_repeats(
    repo_root: Path,
    output_root: Path,
    cell: str,
    experiment_dir: Path,
    repeat_count: int,
) -> dict:
    candidate = select_candidate(experiment_dir, cell)
    rep_name = experiment_dir.name
    eval_dir = output_root / "replicated_means" / cell / rep_name
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_py_dest = eval_dir / "train.py"
    if candidate["source"] == "snapshot" and candidate["train_py_path"] and Path(candidate["train_py_path"]).exists():
        shutil.copy2(candidate["train_py_path"], train_py_dest)
    else:
        reconstruct_train_py(Path(candidate["workspace"]), str(candidate["commit"]), train_py_dest)

    shutil.copy2(repo_root / "autoresearch" / "prepare.py", eval_dir / "prepare.py")
    data_src = repo_root / "autoresearch" / "data"
    data_dest = eval_dir / "data"
    if data_dest.exists() or data_dest.is_symlink():
        data_dest.unlink()
    data_dest.symlink_to(data_src)

    run_values: list[float] = []
    run_logs = []
    for index in range(1, repeat_count + 1):
        log_path = eval_dir / f"run_{index:02d}.log"
        result = subprocess.run(
            ["python", "train.py"],
            cwd=eval_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = (result.stdout or "") + ("\n[stderr]\n" + result.stderr if result.stderr else "")
        log_path.write_text(combined)
        value = parse_val_bpb(result.stdout or "")
        if value is None:
            raise RuntimeError(f"Missing val_bpb in {log_path}")
        run_values.append(value)
        run_logs.append({"run_index": index, "val_bpb": value, "log_path": str(log_path)})

    summary = {
        "cell": cell,
        "replicate": rep_name,
        "candidate_source": candidate["source"],
        "candidate_commit": candidate["commit"],
        "candidate_agent_id": candidate["agent_id"],
        "repeat_count": repeat_count,
        "values": run_values,
        "mean_val_bpb": mean(run_values),
        "std_val_bpb": std(run_values),
        "selected_candidate_pilot_val_bpb": candidate["value"],
        "logs": run_logs,
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    mapping = json.loads(args.mapping.read_text())

    all_results: dict[str, list[dict]] = {}
    cell_summaries: dict[str, dict] = {}

    for cell, experiments in mapping.items():
        cell_results = []
        for index, experiment in enumerate(experiments):
            repeat_count = REPEAT_PLAN[index] if index < len(REPEAT_PLAN) else 1
            summary = run_candidate_repeats(
                repo_root=repo_root,
                output_root=output_root,
                cell=cell,
                experiment_dir=(repo_root / experiment).resolve(),
                repeat_count=repeat_count,
            )
            cell_results.append(summary)

        flat_values = [value for entry in cell_results for value in entry["values"]]
        mu = mean(flat_values)
        sigma = std(flat_values)
        se = sigma / math.sqrt(len(flat_values)) if flat_values else float("nan")
        cell_summaries[cell] = {
            "n": len(flat_values),
            "mean_val_bpb": mu,
            "std_val_bpb": sigma,
            "se_val_bpb": se,
            "ci95_low": mu - 1.96 * se,
            "ci95_high": mu + 1.96 * se,
            "replicate_breakdown": cell_results,
        }
        all_results[cell] = cell_results

    output = {
        "repeat_plan": REPEAT_PLAN,
        "cells": cell_summaries,
    }
    output_path = output_root / "replicated_means_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
