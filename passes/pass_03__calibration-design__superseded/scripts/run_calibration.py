#!/usr/bin/env python3
"""Run power calibration experiments: d00 vs d10, multiple replicates.

Usage:
    python workflow/scripts/run_calibration.py --repo-root . --reps 5
    python workflow/scripts/run_calibration.py --repo-root . --reps 2 --cells d00  # quick test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CELL_CONFIGS = {
    "d00": {
        "config": "configs/experiment_d00.yaml",
        "mode": "single_long",
    },
    "d10": {
        "config": "configs/experiment_d10.yaml",
        "mode": "single_memory",
    },
    "d01": {
        "config": "configs/experiment_d01.yaml",
        "mode": "parallel",
    },
    "d11": {
        "config": "configs/experiment_d11.yaml",
        "mode": "parallel_shared",
    },
}


def run_experiment(
    repo_root: Path,
    cell: str,
    rep: int,
    config_path: str,
) -> dict:
    """Run a single calibration experiment."""
    experiment_id = f"calibration_{cell}_rep{rep}"
    print(f"\n{'='*60}")
    print(f"  Starting {experiment_id}")
    print(f"  Config: {config_path}")
    print(f"{'='*60}\n")

    # Use the launcher module directly via Python
    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.path.insert(0, str({str(repo_root / 'src')!r}))
from agent_parallelization_new.launcher import MODES
from agent_parallelization_new.config import ExperimentConfig
from pathlib import Path

config_path = Path({str(repo_root / config_path)!r})
repo_root = Path({str(repo_root)!r})
config = ExperimentConfig.from_yaml(config_path, repo_root=str(repo_root))
mode = config.mode

launcher_fn = MODES[mode]
launcher_fn(argv=[
    '--config', str(config_path),
    '--experiment-id', {experiment_id!r},
])
""",
    ]

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=False,  # stream output to console
        timeout=4800,  # 80 minutes hard limit
    )
    elapsed = time.time() - start

    experiment_dir = repo_root / "runs" / f"experiment_{experiment_id}"
    has_results = False
    n_runs = 0

    # Check for results
    for jsonl_path in experiment_dir.rglob("training_runs.jsonl"):
        has_results = True
        n_runs += sum(1 for line in jsonl_path.read_text().splitlines() if line.strip())

    return {
        "experiment_id": experiment_id,
        "cell": cell,
        "rep": rep,
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "has_results": has_results,
        "n_training_runs": n_runs,
        "experiment_dir": str(experiment_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration experiments")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=5, help="Number of replicates per cell")
    parser.add_argument("--cells", nargs="+", default=["d00", "d10"],
                        choices=["d00", "d10", "d01", "d11"], help="Which cells to run")
    parser.add_argument("--start-rep", type=int, default=1, help="Starting rep number (for resuming)")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    results = []

    print(f"Calibration Plan:")
    print(f"  Cells: {args.cells}")
    print(f"  Reps:  {args.start_rep} to {args.start_rep + args.reps - 1}")
    print(f"  Total experiments: {len(args.cells) * args.reps}")
    print()

    for cell in args.cells:
        cell_info = CELL_CONFIGS[cell]
        for rep in range(args.start_rep, args.start_rep + args.reps):
            # Check if already done
            experiment_id = f"calibration_{cell}_rep{rep}"
            experiment_dir = repo_root / "runs" / f"experiment_{experiment_id}"
            if any(experiment_dir.rglob("training_runs.jsonl")):
                print(f"  SKIP {experiment_id} — already has results")
                continue

            try:
                r = run_experiment(repo_root, cell, rep, cell_info["config"])
                results.append(r)
                print(f"\n  {experiment_id}: {'OK' if r['has_results'] else 'NO RESULTS'} "
                      f"({r['n_training_runs']} runs, {r['elapsed_seconds']:.0f}s)")
            except subprocess.TimeoutExpired:
                print(f"\n  {experiment_id}: TIMEOUT")
                results.append({
                    "experiment_id": experiment_id,
                    "cell": cell,
                    "rep": rep,
                    "returncode": -1,
                    "error": "timeout",
                })
            except Exception as e:
                print(f"\n  {experiment_id}: ERROR — {e}")
                results.append({
                    "experiment_id": experiment_id,
                    "cell": cell,
                    "rep": rep,
                    "returncode": -1,
                    "error": str(e),
                })

    # Save calibration run summary
    output_path = repo_root / "workflow" / "artifacts" / "calibration_runs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"Calibration complete.")
    total_runs = sum(r.get("n_training_runs", 0) for r in results)
    successful = sum(1 for r in results if r.get("has_results"))
    print(f"  Successful experiments: {successful}/{len(results)}")
    print(f"  Total training runs: {total_runs}")
    print(f"  Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
