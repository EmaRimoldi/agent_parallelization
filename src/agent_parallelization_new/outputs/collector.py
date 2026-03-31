"""Read per-agent output files after all agents finish."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from agent_parallelization_new.outputs.schema import AgentResult, ExperimentSummary, TrajectoryEntry


def collect_agent_result(
    agent_dir: Path,
    agent_id: str,
    experiment_id: str,
    mode: str,
) -> AgentResult:
    """Read all output files for one agent and build an AgentResult.

    Does not crash if agent produced no output (failed agents → failed=True).
    """
    result = AgentResult(
        agent_id=agent_id,
        experiment_id=experiment_id,
        mode=mode,
        workspace_path=str(agent_dir / "workspace"),
        results_path=str(agent_dir / "results"),
    )

    # Read metadata.json if present
    metadata_path = agent_dir / "results" / "metadata.json"
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            result.start_time = meta.get("start_time")
            result.end_time = meta.get("end_time")
            result.budget_seconds = meta.get("budget_seconds", 0)
            result.total_turns = meta.get("total_turns", 0)
        except Exception:
            pass

    # Read trajectory.jsonl (authoritative)
    traj_path = agent_dir / "results" / "trajectory.jsonl"
    if traj_path.exists():
        try:
            entries = []
            for line in traj_path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        entries.append(TrajectoryEntry.from_dict(json.loads(line)))
                    except Exception:
                        pass
            result.trajectory = entries
        except Exception:
            pass

    result.compute_derived()

    if not result.trajectory:
        result.failed = True
        result.failure_reason = "no trajectory entries found"

    return result


def collect_experiment(
    experiment_dir: Path,
    experiment_id: str,
    mode: str,
    agent_ids: list[str],
) -> ExperimentSummary:
    """Collect results for all agents in a mode directory.

    Never crashes — missing agents show up as failed rows.
    """
    summary = ExperimentSummary(experiment_id=experiment_id, mode=mode)

    mode_dir = experiment_dir / f"mode_{mode}"

    for agent_id in agent_ids:
        agent_dir = mode_dir / agent_id
        result = collect_agent_result(agent_dir, agent_id, experiment_id, mode)
        summary.agent_results.append(result)

    # Write aggregate outputs
    agg_dir = mode_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    combined_path = agg_dir / "combined_summary.json"
    combined_path.write_text(json.dumps(summary.to_dict(), indent=2))

    comparison_path = agg_dir / "comparison_table.csv"
    _write_comparison_csv(comparison_path, summary)

    return summary


def _write_comparison_csv(path: Path, summary: ExperimentSummary) -> None:
    rows = []
    for r in summary.agent_results:
        rows.append({
            "agent_id": r.agent_id,
            "best_val_bpb": r.best_val_bpb if r.best_val_bpb is not None else "",
            "first_val_bpb": r.first_val_bpb if r.first_val_bpb is not None else "",
            "improvement": r.improvement() if r.improvement() is not None else "",
            "total_runs": r.total_training_runs,
            "successful_runs": r.successful_training_runs,
            "failed": r.failed,
            "failure_reason": r.failure_reason,
            "total_turns": r.total_turns,
        })

    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
