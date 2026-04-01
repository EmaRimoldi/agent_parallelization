#!/usr/bin/env python3
"""CLI: run the merge phase on a completed parallel experiment.

Given an experiment directory produced by run_parallel_experiment.py, this
script:
  1. Gathers all agent snapshots, reasoning traces, and metrics
  2. Selects the best and most informative train.py variants
  3. Analyses per-parameter improvement correlations
  4. Produces a merged train.py candidate
  5. Optionally runs evaluation via SLURM
  6. Writes a merge report and comparison table

Usage:
    python scripts/run_merge_phase.py --experiment-dir runs/experiment_parallel_20260331_120000
    python scripts/run_merge_phase.py --experiment-dir runs/exp_... --evaluate
    python scripts/run_merge_phase.py --experiment-dir runs/exp_... --source-mode parallel
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from agent_parallelization_new.merger import MergeOrchestrator


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run merge phase on a parallel experiment.")
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Path to the experiment directory (contains mode_parallel/, config.json, etc.).",
    )
    parser.add_argument(
        "--source-mode", default="parallel",
        help="Which agent mode to read from (default: parallel).",
    )
    parser.add_argument(
        "--autoresearch-dir", default=None,
        help="Path to autoresearch/ dir. Defaults to <repo_root>/autoresearch.",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Attempt to evaluate the merged candidate via SLURM after producing it.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).parents[1]
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()

    if not experiment_dir.exists():
        print(f"[merge] Error: experiment directory not found: {experiment_dir}", file=sys.stderr)
        sys.exit(1)

    autoresearch_dir = (
        Path(args.autoresearch_dir).resolve()
        if args.autoresearch_dir
        else repo_root / "autoresearch"
    )

    print(f"[merge] Experiment:      {experiment_dir}")
    print(f"[merge] Source mode:     {args.source_mode}")
    print(f"[merge] Autoresearch:    {autoresearch_dir}")
    print(f"[merge] Run evaluation:  {args.evaluate}")

    merger = MergeOrchestrator(
        experiment_dir=experiment_dir,
        autoresearch_dir=autoresearch_dir,
        mode=args.source_mode,
    )
    results = merger.run(evaluate=args.evaluate)

    print("\n=== Merge Results ===")
    print(f"  Best individual agent:  {results.best_individual_agent}")
    print(f"  Best individual val_bpb: {results.best_individual_val_bpb}")
    print(f"  Merged val_bpb:          {results.merge_val_bpb}")
    print(f"  Merge won:               {results.merge_won}")
    print(f"  Delta val_bpb:           {results.delta_val_bpb}")
    print(f"\nMerge artifacts in: {experiment_dir}/mode_merge/")

    report_path = experiment_dir / "mode_merge" / "merge_report.txt"
    if report_path.exists():
        print("\n--- Merge Report ---")
        print(report_path.read_text())


if __name__ == "__main__":
    main()
