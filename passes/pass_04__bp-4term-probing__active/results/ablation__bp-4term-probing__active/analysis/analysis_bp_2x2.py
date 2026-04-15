#!/usr/bin/env python3
"""BP 2x2 Experiment Analysis Script.

Reads all training_runs.jsonl files from the four cells of the 2x2 design:
  d00 (single, no memory)    - 5 reps
  d10 (single, memory)       - 5 reps
  d01 (parallel, no memory)  - 3 reps
  d11 (parallel, memory)     - 2 reps (rep2 may be partial)

Computes per-cell summaries, 2x2 factorial effects, Cohen's d,
Jensen gap (R_alpha), and strategy diversity metrics.
"""

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/Users/emanuelerimoldi/Documents/GitHub/agent_parallelization")
RUNS_DIR = REPO_ROOT / "runs"
BASELINE_BPB = 0.925845

CELL_SPEC: Dict[str, Dict] = {
    "d00": {
        "label": "d00 (single, no mem)",
        "mode_dir": "mode_single_long",
        "reps": range(1, 6),
        "parallel": False,
    },
    "d10": {
        "label": "d10 (single, memory)",
        "mode_dir": "mode_single_memory",
        "reps": range(1, 6),
        "parallel": False,
    },
    "d01": {
        "label": "d01 (parallel, no mem)",
        "mode_dir": "mode_parallel",
        "reps": range(1, 4),
        "parallel": True,
    },
    "d11": {
        "label": "d11 (parallel, memory)",
        "mode_dir": "mode_parallel_shared",
        "reps": range(1, 4),  # rep3 might not exist
        "parallel": True,
    },
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    """Single training run record."""
    run_index: int
    turn: int
    experiment_id: str
    agent_id: str
    wall_seconds: float
    training_seconds: float
    val_bpb: float
    status: str
    hypothesis: str
    strategy_category: str
    baseline_candidate: bool
    protocol_mode: str
    memory_context_visible: bool
    memory_context_entries: int
    shared_memory_context_visible: bool
    shared_memory_context_entries: int
    promotion_decision: str
    is_reevaluation: bool
    evaluation_kind: str


@dataclass
class RepData:
    """All runs from one replication."""
    cell: str
    rep: int
    agents: Dict[str, List[RunRecord]] = field(default_factory=dict)

    @property
    def all_runs(self) -> List[RunRecord]:
        runs = []
        for agent_runs in self.agents.values():
            runs.extend(agent_runs)
        return runs


@dataclass
class CellData:
    """All replications for one cell."""
    cell: str
    reps: Dict[int, RepData] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[dict]:
    """Load a JSONL file, returning list of dicts."""
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_record(d: dict) -> RunRecord:
    """Parse a dict into a RunRecord."""
    return RunRecord(
        run_index=d.get("run_index", 0),
        turn=d.get("turn", 0),
        experiment_id=d.get("experiment_id", ""),
        agent_id=d.get("agent_id", ""),
        wall_seconds=d.get("wall_seconds", 0.0),
        training_seconds=d.get("training_seconds", 0.0),
        val_bpb=d.get("val_bpb", float("nan")),
        status=d.get("status", ""),
        hypothesis=d.get("hypothesis", ""),
        strategy_category=d.get("strategy_category", "other"),
        baseline_candidate=d.get("baseline_candidate", False),
        protocol_mode=d.get("protocol_mode", ""),
        memory_context_visible=d.get("memory_context_visible", False),
        memory_context_entries=d.get("memory_context_entries", 0),
        shared_memory_context_visible=d.get("shared_memory_context_visible", False),
        shared_memory_context_entries=d.get("shared_memory_context_entries", 0),
        promotion_decision=d.get("promotion_decision", ""),
        is_reevaluation=d.get("is_reevaluation", False),
        evaluation_kind=d.get("evaluation_kind", ""),
    )


def load_cell(cell_key: str, spec: dict) -> CellData:
    """Load all replications for a cell."""
    cell = CellData(cell=cell_key)
    for rep_num in spec["reps"]:
        rep_dir = RUNS_DIR / f"experiment_calibration_{cell_key}_rep{rep_num}"
        if not rep_dir.exists():
            continue
        mode_dir = rep_dir / spec["mode_dir"]
        if not mode_dir.exists():
            continue

        rep_data = RepData(cell=cell_key, rep=rep_num)

        if spec["parallel"]:
            # Look for agent_0, agent_1, ...
            for agent_dir in sorted(mode_dir.iterdir()):
                if agent_dir.is_dir() and agent_dir.name.startswith("agent_"):
                    jsonl_path = agent_dir / "results" / "training_runs.jsonl"
                    raw = load_jsonl(jsonl_path)
                    rep_data.agents[agent_dir.name] = [parse_record(r) for r in raw]
        else:
            # Single agent
            agent_dir = mode_dir / "agent_0"
            if agent_dir.exists():
                jsonl_path = agent_dir / "results" / "training_runs.jsonl"
                raw = load_jsonl(jsonl_path)
                rep_data.agents["agent_0"] = [parse_record(r) for r in raw]

        if rep_data.all_runs:
            cell.reps[rep_num] = rep_data

    return cell


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def mean(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def std(xs: List[float], ddof: int = 1) -> float:
    if len(xs) <= ddof:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)
    pooled_sd = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return float("nan")
    return (m1 - m2) / pooled_sd


def jensen_gap(xs: List[float]) -> float:
    """Jensen gap R_alpha = mean(x) - min(x).

    Measures cost of variance: how much worse the average run is
    compared to the best run. For optimization (lower is better),
    a larger gap means more wasted computation.
    """
    if not xs:
        return float("nan")
    return mean(xs) - min(xs)


# ---------------------------------------------------------------------------
# Per-cell analysis
# ---------------------------------------------------------------------------
def analyze_cell(cell: CellData) -> dict:
    """Compute per-cell summary statistics."""
    all_runs = []
    best_per_rep = {}
    runs_per_rep = {}
    strategies_per_agent: Dict[str, set] = defaultdict(set)
    strategy_counts: Dict[str, int] = defaultdict(int)

    for rep_num, rep_data in sorted(cell.reps.items()):
        rep_runs = rep_data.all_runs
        all_runs.extend(rep_runs)
        runs_per_rep[rep_num] = len(rep_runs)

        # Best val_bpb for this rep (lower is better)
        non_baseline_runs = [r for r in rep_runs if not r.baseline_candidate]
        all_val_bpb = [r.val_bpb for r in rep_runs if r.status == "success"]
        best_per_rep[rep_num] = min(all_val_bpb) if all_val_bpb else float("nan")

        # Strategy tracking per agent
        for agent_id, agent_runs in rep_data.agents.items():
            for r in agent_runs:
                if not r.baseline_candidate:
                    strategies_per_agent[agent_id].add(r.strategy_category)
                    strategy_counts[r.strategy_category] += 1

    # All val_bpb values (successful, non-baseline)
    all_val_bpb = [r.val_bpb for r in all_runs if r.status == "success"]
    non_baseline_val_bpb = [
        r.val_bpb for r in all_runs
        if r.status == "success" and not r.baseline_candidate
    ]
    all_wall = [r.wall_seconds for r in all_runs if r.status == "success"]
    all_training = [r.training_seconds for r in all_runs if r.status == "success"]

    # Success rate: runs that strictly beat baseline
    n_beat_baseline = sum(1 for v in non_baseline_val_bpb if v < BASELINE_BPB)
    success_rate = n_beat_baseline / len(non_baseline_val_bpb) if non_baseline_val_bpb else 0.0

    # Unique strategies per agent overlap (for parallel cells)
    all_strats = set()
    for s in strategies_per_agent.values():
        all_strats |= s
    if len(strategies_per_agent) > 1:
        agent_strat_sets = list(strategies_per_agent.values())
        overlap = agent_strat_sets[0]
        for s in agent_strat_sets[1:]:
            overlap = overlap & s
        overlap_count = len(overlap)
    else:
        overlap_count = None

    return {
        "total_runs": len(all_runs),
        "total_successful": len(all_val_bpb),
        "runs_per_rep": runs_per_rep,
        "best_per_rep": best_per_rep,
        "best_overall": min(all_val_bpb) if all_val_bpb else float("nan"),
        "n_beat_baseline": n_beat_baseline,
        "n_non_baseline": len(non_baseline_val_bpb),
        "success_rate": success_rate,
        "mean_val_bpb": mean(all_val_bpb),
        "std_val_bpb": std(all_val_bpb),
        "mean_non_baseline_bpb": mean(non_baseline_val_bpb),
        "std_non_baseline_bpb": std(non_baseline_val_bpb),
        "mean_wall_seconds": mean(all_wall),
        "std_wall_seconds": std(all_wall),
        "mean_training_seconds": mean(all_training),
        "std_training_seconds": std(all_training),
        "strategy_counts": dict(strategy_counts),
        "unique_strategies": sorted(all_strats),
        "n_unique_strategies": len(all_strats),
        "strategies_per_agent": {
            k: sorted(v) for k, v in strategies_per_agent.items()
        },
        "agent_strategy_overlap": overlap_count,
        "jensen_gap_all": jensen_gap(all_val_bpb),
        "jensen_gap_non_baseline": jensen_gap(non_baseline_val_bpb),
        "best_per_rep_values": best_per_rep,
        # For 2x2 comparison, use best-per-rep as the summary statistic
        "rep_bests": [v for _, v in sorted(best_per_rep.items())],
    }


# ---------------------------------------------------------------------------
# 2x2 Factorial Analysis
# ---------------------------------------------------------------------------
def factorial_analysis(results: Dict[str, dict]) -> dict:
    """Compute 2x2 factorial effects using best-per-rep values."""
    # Use best-per-rep as the outcome measure (lower is better)
    d00_bests = results["d00"]["rep_bests"]
    d10_bests = results["d10"]["rep_bests"]
    d01_bests = results["d01"]["rep_bests"]
    d11_bests = results["d11"]["rep_bests"]

    m00 = mean(d00_bests)
    m10 = mean(d10_bests)
    m01 = mean(d01_bests)
    m11 = mean(d11_bests)

    # Main effect of memory: average difference when memory is ON vs OFF
    # memory ON: d10, d11;  memory OFF: d00, d01
    memory_effect = 0.5 * ((m10 - m00) + (m11 - m01))

    # Main effect of parallelism: average difference when parallel is ON vs OFF
    # parallel ON: d01, d11;  parallel OFF: d00, d10
    parallel_effect = 0.5 * ((m01 - m00) + (m11 - m10))

    # Interaction: does memory's effect differ between single and parallel?
    interaction = (m11 - m01) - (m10 - m00)

    # Cohen's d for memory: pool all memory-on bests vs memory-off bests
    memory_on = d10_bests + d11_bests
    memory_off = d00_bests + d01_bests
    d_memory = cohens_d(memory_on, memory_off)

    # Cohen's d for parallelism
    parallel_on = d01_bests + d11_bests
    parallel_off = d00_bests + d10_bests
    d_parallel = cohens_d(parallel_on, parallel_off)

    # Jensen gap per cell using all non-baseline val_bpb
    jg = {}
    for cell_key in ["d00", "d10", "d01", "d11"]:
        jg[cell_key] = results[cell_key]["jensen_gap_non_baseline"]

    return {
        "cell_means_best": {"d00": m00, "d10": m10, "d01": m01, "d11": m11},
        "memory_effect": memory_effect,
        "parallel_effect": parallel_effect,
        "interaction": interaction,
        "cohens_d_memory": d_memory,
        "cohens_d_parallel": d_parallel,
        "jensen_gap": jg,
        # Raw data for reference
        "rep_bests": {
            "d00": d00_bests,
            "d10": d10_bests,
            "d01": d01_bests,
            "d11": d11_bests,
        },
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
SEPARATOR = "=" * 80
THIN_SEP = "-" * 80


def print_cell_summary(cell_key: str, spec: dict, stats: dict) -> None:
    """Print formatted summary for one cell."""
    print(f"\n{SEPARATOR}")
    print(f"  CELL: {spec['label'].upper()}")
    print(f"  Parallel: {'Yes' if spec['parallel'] else 'No'}  |  "
          f"Memory: {'Yes' if cell_key in ('d10', 'd11') else 'No'}")
    print(SEPARATOR)

    print(f"\n  Replications loaded: {len(stats['runs_per_rep'])}")
    print(f"  Total runs: {stats['total_runs']}  "
          f"(successful: {stats['total_successful']})")
    for rep, count in sorted(stats["runs_per_rep"].items()):
        best = stats["best_per_rep"][rep]
        marker = " *" if best < BASELINE_BPB else ""
        print(f"    rep{rep}: {count:3d} runs  |  best val_bpb = {best:.6f}{marker}")

    print(f"\n  --- Performance (val_bpb, lower is better) ---")
    print(f"  Best overall:        {stats['best_overall']:.6f}")
    print(f"  Baseline:            {BASELINE_BPB:.6f}")
    improvement = (BASELINE_BPB - stats["best_overall"]) / BASELINE_BPB * 100
    print(f"  Best improvement:    {improvement:+.4f}%")
    print(f"  Mean (all):          {stats['mean_val_bpb']:.6f}  "
          f"+/- {stats['std_val_bpb']:.6f}")
    print(f"  Mean (non-baseline): {stats['mean_non_baseline_bpb']:.6f}  "
          f"+/- {stats['std_non_baseline_bpb']:.6f}")
    print(f"  Mean of rep-bests:   {mean(stats['rep_bests']):.6f}  "
          f"+/- {std(stats['rep_bests']):.6f}")

    print(f"\n  --- Success Rate (beating baseline {BASELINE_BPB}) ---")
    print(f"  Runs beating baseline: {stats['n_beat_baseline']} / "
          f"{stats['n_non_baseline']}  "
          f"({stats['success_rate']*100:.1f}%)")

    print(f"\n  --- Timing ---")
    print(f"  Mean wall_seconds:      {stats['mean_wall_seconds']:.1f}  "
          f"+/- {stats['std_wall_seconds']:.1f}")
    print(f"  Mean training_seconds:  {stats['mean_training_seconds']:.1f}  "
          f"+/- {stats['std_training_seconds']:.1f}")

    print(f"\n  --- Strategy Diversity ---")
    print(f"  Unique strategies: {stats['n_unique_strategies']}  "
          f"{stats['unique_strategies']}")
    print(f"  Strategy counts:")
    for strat, count in sorted(stats["strategy_counts"].items(),
                                key=lambda x: -x[1]):
        print(f"    {strat:25s}: {count:3d}")
    if len(stats["strategies_per_agent"]) > 1:
        print(f"\n  Per-agent strategy sets:")
        for agent, strats in sorted(stats["strategies_per_agent"].items()):
            print(f"    {agent}: {strats}")
        print(f"  Inter-agent overlap: {stats['agent_strategy_overlap']} strategies")

    print(f"\n  --- Jensen Gap (cost of variance) ---")
    print(f"  R_alpha (all val_bpb):          {stats['jensen_gap_all']:.6f}")
    print(f"  R_alpha (non-baseline val_bpb): {stats['jensen_gap_non_baseline']:.6f}")


def print_factorial(fa: dict) -> None:
    """Print 2x2 factorial analysis."""
    print(f"\n\n{'#' * 80}")
    print(f"  2x2 FACTORIAL ANALYSIS  (outcome = mean of best-per-rep val_bpb)")
    print(f"{'#' * 80}")

    cm = fa["cell_means_best"]
    print(f"\n  Cell means (best val_bpb per rep, averaged):")
    print(f"  {'':20s} {'No Memory':>14s}  {'Memory':>14s}")
    print(f"  {'Single agent':20s} {cm['d00']:>14.6f}  {cm['d10']:>14.6f}")
    print(f"  {'Parallel agents':20s} {cm['d01']:>14.6f}  {cm['d11']:>14.6f}")

    print(f"\n  Rep-level best values:")
    for cell_key in ["d00", "d10", "d01", "d11"]:
        vals = fa["rep_bests"][cell_key]
        print(f"    {cell_key}: {[f'{v:.6f}' for v in vals]}")

    print(f"\n  {THIN_SEP}")
    print(f"  MAIN EFFECTS (negative = treatment helps, since lower bpb is better)")
    print(f"  {THIN_SEP}")
    print(f"  Memory effect:      {fa['memory_effect']:+.6f}  "
          f"(Cohen's d = {fa['cohens_d_memory']:+.3f})")
    print(f"  Parallelism effect: {fa['parallel_effect']:+.6f}  "
          f"(Cohen's d = {fa['cohens_d_parallel']:+.3f})")
    print(f"  Interaction (MxP):  {fa['interaction']:+.6f}")

    # Interpret
    print(f"\n  Interpretation:")
    me = fa["memory_effect"]
    pe = fa["parallel_effect"]
    ix = fa["interaction"]
    if me < 0:
        print(f"    Memory HELPS: reduces best val_bpb by {abs(me):.6f} on average")
    else:
        print(f"    Memory HURTS: increases best val_bpb by {abs(me):.6f} on average")
    if pe < 0:
        print(f"    Parallelism HELPS: reduces best val_bpb by {abs(pe):.6f} on average")
    else:
        print(f"    Parallelism HURTS: increases best val_bpb by {abs(pe):.6f} on average")
    if abs(ix) > 0.001:
        print(f"    Interaction is non-negligible: {ix:+.6f}")
        if ix < 0:
            print(f"      -> Memory + Parallelism together is BETTER than additive prediction")
        else:
            print(f"      -> Memory + Parallelism together is WORSE than additive prediction")
    else:
        print(f"    Interaction is negligible: {ix:+.6f}")

    print(f"\n  {THIN_SEP}")
    print(f"  JENSEN GAP R_alpha (non-baseline runs; lower = less wasted compute)")
    print(f"  {THIN_SEP}")
    for cell_key in ["d00", "d10", "d01", "d11"]:
        jg = fa["jensen_gap"][cell_key]
        print(f"    {cell_key}: {jg:.6f}")


def print_comparison_table(results: Dict[str, dict], fa: dict) -> None:
    """Print a compact comparison table."""
    print(f"\n\n{'#' * 80}")
    print(f"  COMPACT COMPARISON TABLE")
    print(f"{'#' * 80}")
    header = (
        f"  {'Cell':6s} | {'Reps':>4s} | {'Runs':>5s} | "
        f"{'Best':>10s} | {'Mean Best':>10s} | "
        f"{'Beat BL':>8s} | {'Strategies':>10s} | "
        f"{'Jens Gap':>10s} | {'Mean Wall(s)':>12s}"
    )
    print(f"\n{header}")
    print(f"  {'-'*6}-+-{'-'*4}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-"
          f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    for cell_key in ["d00", "d10", "d01", "d11"]:
        s = results[cell_key]
        n_reps = len(s["runs_per_rep"])
        total = s["total_runs"]
        best = s["best_overall"]
        mean_best = mean(s["rep_bests"])
        beat = f"{s['n_beat_baseline']}/{s['n_non_baseline']}"
        n_strat = s["n_unique_strategies"]
        jg = s["jensen_gap_non_baseline"]
        mwall = s["mean_wall_seconds"]
        print(
            f"  {cell_key:6s} | {n_reps:4d} | {total:5d} | "
            f"{best:10.6f} | {mean_best:10.6f} | "
            f"{beat:>8s} | {n_strat:10d} | "
            f"{jg:10.6f} | {mwall:12.1f}"
        )

    print(f"\n  Effects:")
    print(f"    Memory main effect:      {fa['memory_effect']:+.6f}  "
          f"(d={fa['cohens_d_memory']:+.3f})")
    print(f"    Parallelism main effect: {fa['parallel_effect']:+.6f}  "
          f"(d={fa['cohens_d_parallel']:+.3f})")
    print(f"    Interaction:             {fa['interaction']:+.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"{'#' * 80}")
    print(f"  BP 2x2 EXPERIMENT ANALYSIS")
    print(f"  Baseline val_bpb = {BASELINE_BPB}")
    print(f"  Data directory: {RUNS_DIR}")
    print(f"{'#' * 80}")

    # Load all cells
    cells: Dict[str, CellData] = {}
    results: Dict[str, dict] = {}
    for cell_key, spec in CELL_SPEC.items():
        cells[cell_key] = load_cell(cell_key, spec)
        results[cell_key] = analyze_cell(cells[cell_key])

    # Print per-cell summaries
    for cell_key, spec in CELL_SPEC.items():
        print_cell_summary(cell_key, spec, results[cell_key])

    # Factorial analysis
    fa = factorial_analysis(results)
    print_factorial(fa)

    # Compact comparison table
    print_comparison_table(results, fa)

    # Summary data quality note
    print(f"\n\n{'#' * 80}")
    print(f"  DATA QUALITY NOTES")
    print(f"{'#' * 80}")
    for cell_key in ["d00", "d10", "d01", "d11"]:
        n_reps = len(results[cell_key]["runs_per_rep"])
        expected = len(list(CELL_SPEC[cell_key]["reps"]))
        total = results[cell_key]["total_runs"]
        status = "OK" if n_reps == expected else f"PARTIAL ({n_reps}/{expected} reps)"
        print(f"  {cell_key}: {status}, {total} total runs across {n_reps} reps")
        for rep, count in sorted(results[cell_key]["runs_per_rep"].items()):
            if count <= 1:
                print(f"    WARNING: rep{rep} has only {count} run(s)")


if __name__ == "__main__":
    main()
