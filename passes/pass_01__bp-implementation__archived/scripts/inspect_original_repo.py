#!/usr/bin/env python3
"""Utility: analyze original repo, print a machine-readable report.

Usage:
    python scripts/inspect_original_repo.py ~/projects/agent_parallelisation
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from agent_parallelization_new.compatibility.original_repo_adapter import (
    find_best_original_result,
    read_all_original_trajectories,
    ORIGINAL_BEST,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: inspect_original_repo.py <path_to_original_repo>")
        sys.exit(1)

    repo_path = Path(sys.argv[1]).expanduser().resolve()
    if not repo_path.exists():
        print(f"ERROR: path does not exist: {repo_path}")
        sys.exit(1)

    results_root = repo_path / "results"
    report = {
        "repo_path": str(repo_path),
        "results_root": str(results_root),
        "trajectories": {},
        "best_result": None,
        "trajectory_file_count": 0,
        "known_best": ORIGINAL_BEST,
    }

    # Count JSONL files
    if results_root.exists():
        jsonl_files = list(results_root.glob("trajectories/**/*.jsonl"))
        report["trajectory_file_count"] = len(jsonl_files)

        all_traj = read_all_original_trajectories(results_root)
        report["trajectories"] = {
            f"{run_id}/{agent_id}": [
                {"step": e.step, "val_bpb": e.val_bpb} for e in entries
            ]
            for (run_id, agent_id), entries in sorted(all_traj.items())
        }

        best = find_best_original_result(results_root)
        if best:
            report["best_result"] = {
                "run_id": best[0],
                "agent_id": best[1],
                "val_bpb": best[2],
            }

    print(json.dumps(report, indent=2))

    # Write markdown report
    doc_dir = Path(__file__).parents[1] / "docs"
    doc_dir.mkdir(exist_ok=True)
    md_path = doc_dir / "original_repo_reverse_engineering.md"

    lines = [
        "# Original Repo Reverse Engineering",
        "",
        f"Repo path: `{repo_path}`",
        f"Results root: `{results_root}`",
        f"Trajectory files found: {report['trajectory_file_count']}",
        "",
        "## Best result",
    ]
    if report["best_result"]:
        b = report["best_result"]
        lines.append(f"- Run: `{b['run_id']}` / Agent: `{b['agent_id']}`")
        lines.append(f"- val_bpb: `{b['val_bpb']}`")
    else:
        lines.append("No results found.")

    lines += [
        "",
        "## Known best (from build guide)",
        f"- val_bpb: `{ORIGINAL_BEST['val_bpb']}`",
        f"- Run: `{ORIGINAL_BEST['run_id']}` / Agent: `{ORIGINAL_BEST['agent_id']}`",
        f"- Snapshot: `{ORIGINAL_BEST['snapshot']}`",
        "",
        "## Hyperparameter changes",
    ]
    for k, v in ORIGINAL_BEST["hyperparameters"].items():
        lines.append(f"- `{k}` = `{v}`")

    lines += ["", "## All trajectories", ""]
    for key, entries in sorted(report["trajectories"].items()):
        best_bpb = min((e["val_bpb"] for e in entries), default=None)
        lines.append(f"### {key}")
        lines.append(f"- Entries: {len(entries)}, Best val_bpb: {best_bpb}")
        for e in entries[:5]:
            lines.append(f"  - step={e['step']} val_bpb={e['val_bpb']:.6f}")
        if len(entries) > 5:
            lines.append(f"  - ...({len(entries) - 5} more)")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nMarkdown report written to: {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
