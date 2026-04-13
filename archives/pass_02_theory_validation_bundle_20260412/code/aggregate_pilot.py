"""Aggregate pilot repetition outputs into a summary report.

Usage:
    python scripts/aggregate_pilot.py \
        --mapping runs/pilot_mapping.json \
        --decomposition-glob 'results/decomposition_rep*.json' \
        --output-dir results
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CELL_ORDER = ["d00", "d10", "d01", "d11"]
DECOMP_CELLS = ["d10", "d01", "d11"]
TERM_ORDER = [
    "cost_term_token",
    "cost_term_wall",
    "phi",
    "G",
    "epsilon",
    "delta_token",
    "delta_wall",
]
BIN_ORDER = ["0-25%", "25-50%", "50-75%", "75-100%"]


def is_credible_training_run(row: dict) -> bool:
    """Filter out spurious watcher rows that do not look like real training attempts."""
    if row.get("val_bpb") is None:
        return False

    wall = row.get("wall_seconds")
    train = row.get("training_seconds")
    try:
        wall_f = float(wall) if wall is not None else None
    except (TypeError, ValueError):
        wall_f = None
    try:
        train_f = float(train) if train is not None else None
    except (TypeError, ValueError):
        train_f = None

    if train_f is not None and wall_f is not None:
        return wall_f >= max(30.0, 0.75 * train_f)
    if wall_f is not None:
        return wall_f >= 60.0
    return False


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_std(values: list[float]) -> float:
    return float(np.std(values, ddof=0)) if values else float("nan")


def bootstrap_mean_ci(
    values: list[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    samples = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_boot)
    ]
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def format_stat(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "nan"
    if math.isnan(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} +/- {std:.2f}"


def format_float(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def monotone_non_decreasing(values: list[float]) -> bool:
    if len(values) < 2:
        return True
    return all(curr >= prev for prev, curr in zip(values, values[1:]))


def context_values(context_bins: dict, cell: str) -> list[float]:
    bins = context_bins.get(cell, {})
    values = []
    for label in BIN_ORDER:
        value = bins.get(label)
        if value is not None:
            values.append(float(value))
    return values


def supports_h5(rep: dict) -> bool:
    """Operationalize H5 from the pilot summary requirement.

    We treat H5 as supported when d00 shows a monotone increase in token cost with
    context pressure, and d10 flattens that effect relative to d00.
    """
    d00_vals = context_values(rep.get("context_bins", {}), "d00")
    d10_vals = context_values(rep.get("context_bins", {}), "d10")
    if not d00_vals or not d10_vals:
        return False
    d00_span = d00_vals[-1] - d00_vals[0] if len(d00_vals) >= 2 else 0.0
    d10_span = d10_vals[-1] - d10_vals[0] if len(d10_vals) >= 2 else 0.0
    return monotone_non_decreasing(d00_vals) and d10_span <= d00_span


def load_experiment_metrics(experiment_dir: Path) -> dict:
    turns: list[dict] = []
    training_runs: list[dict] = []
    mode_labels: list[dict] = []
    metadata_rows: list[dict] = []

    for agent_dir in sorted(experiment_dir.glob("mode_*/agent_*")):
        agent_id = agent_dir.name
        agent_turns = load_jsonl(agent_dir / "results" / "turns.jsonl")
        for row in agent_turns:
            row["agent_id"] = agent_id
        turns.extend(agent_turns)

        agent_runs = load_jsonl(agent_dir / "results" / "training_runs.jsonl")
        for row in agent_runs:
            if is_credible_training_run(row):
                row["agent_id"] = agent_id
                training_runs.append(row)

        labels = load_jsonl(agent_dir / "results" / "mode_labels.jsonl")
        for row in labels:
            row["agent_id"] = agent_id
        mode_labels.extend(labels)

        meta_path = agent_dir / "results" / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            metadata["agent_id"] = agent_id
            metadata_rows.append(metadata)

    training_successes = [
        row for row in training_runs if row.get("val_bpb") is not None
    ]
    best_val_bpb = (
        min(float(row["val_bpb"]) for row in training_successes)
        if training_successes
        else float("nan")
    )
    total_tokens = float(
        sum(
            (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0)
            for row in turns
        )
    )
    mean_wall = safe_mean(
        [float(row["wall_seconds"]) for row in training_successes if row.get("wall_seconds") is not None]
    )

    accepted = [row for row in mode_labels if row.get("accepted")]
    counts = Counter(row.get("mode", "other") for row in accepted)
    total_accepted = sum(counts.values())
    mode_distribution = (
        {mode: count / total_accepted for mode, count in counts.items()}
        if total_accepted
        else {}
    )

    turns_sorted = sorted(
        turns,
        key=lambda row: (float(row.get("timestamp", 0.0)), row.get("agent_id", ""), int(row.get("turn", 0))),
    )
    runs_by_turn: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in training_successes:
        runs_by_turn[(row.get("agent_id", ""), int(row.get("turn", 0)))].append(
            float(row["val_bpb"])
        )

    best_so_far = float("inf")
    cumulative_tokens = 0.0
    best_curve = []
    for row in turns_sorted:
        cumulative_tokens += float(
            (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0)
        )
        key = (row.get("agent_id", ""), int(row.get("turn", 0)))
        if runs_by_turn.get(key):
            best_so_far = min(best_so_far, min(runs_by_turn[key]))
        if best_so_far != float("inf"):
            best_curve.append(
                {
                    "cumulative_tokens": cumulative_tokens,
                    "best_val_bpb": best_so_far,
                }
            )

    return {
        "experiment_dir": str(experiment_dir),
        "total_training_runs": len(training_successes),
        "best_val_bpb": best_val_bpb,
        "total_tokens_consumed": total_tokens,
        "mean_wall_clock_per_attempt": mean_wall,
        "total_turns": len(turns),
        "mode_distribution": mode_distribution,
        "best_so_far_curve": best_curve,
        "metadata": metadata_rows,
    }


def aggregate_decompositions(repetitions: list[dict]) -> dict:
    stats: dict[str, dict[str, dict]] = {}
    for cell in DECOMP_CELLS:
        stats[cell] = {}
        for term in TERM_ORDER:
            values = [
                float(rep["decompositions"][cell][term])
                for rep in repetitions
                if term in rep.get("decompositions", {}).get(cell, {})
            ]
            ci_low, ci_high = bootstrap_mean_ci(values)
            stats[cell][term] = {
                "values": values,
                "mean": safe_mean(values),
                "std": safe_std(values),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
    return stats


def aggregate_context_bins(repetitions: list[dict]) -> dict:
    aggregated = {}
    for cell in CELL_ORDER:
        bins = {}
        mean_values = []
        for label in BIN_ORDER:
            values = []
            for rep in repetitions:
                value = rep.get("context_bins", {}).get(cell, {}).get(label)
                if value is not None:
                    values.append(float(value))
            mean_value = safe_mean(values)
            bins[label] = mean_value if not math.isnan(mean_value) else None
            if not math.isnan(mean_value):
                mean_values.append(mean_value)
        aggregated[cell] = {
            "bins": bins,
            "monotone": monotone_non_decreasing(mean_values),
        }
    return aggregated


def aggregate_hypotheses(repetitions: list[dict]) -> dict:
    counts = {
        "H1": 0,
        "H2": 0,
        "H3": 0,
        "H4": 0,
        "H5": 0,
        "H6": 0,
    }
    for rep in repetitions:
        hypotheses = rep.get("hypotheses", {})
        counts["H1"] += int(bool(hypotheses.get("H1_holds")))
        counts["H2"] += int(bool(hypotheses.get("H2_holds")))
        counts["H3"] += int(bool(hypotheses.get("H3_holds")))
        counts["H4"] += int(bool(hypotheses.get("H4_epsilon_exceeds_log2")))
        counts["H5"] += int(supports_h5(rep))
        counts["H6"] += int(bool(hypotheses.get("H6_holds")))
    return counts


def aggregate_raw_metrics(mapping: dict[str, list[str]]) -> dict:
    results = {}
    for cell, experiment_dirs in mapping.items():
        experiments = [load_experiment_metrics(Path(path)) for path in experiment_dirs]
        results[cell] = {
            "experiments": experiments,
            "summary": {
                "total_training_runs": {
                    "values": [exp["total_training_runs"] for exp in experiments],
                    "mean": safe_mean([exp["total_training_runs"] for exp in experiments]),
                    "std": safe_std([exp["total_training_runs"] for exp in experiments]),
                },
                "best_val_bpb": {
                    "values": [exp["best_val_bpb"] for exp in experiments],
                    "mean": safe_mean([exp["best_val_bpb"] for exp in experiments]),
                    "std": safe_std([exp["best_val_bpb"] for exp in experiments]),
                },
                "total_tokens_consumed": {
                    "values": [exp["total_tokens_consumed"] for exp in experiments],
                    "mean": safe_mean([exp["total_tokens_consumed"] for exp in experiments]),
                    "std": safe_std([exp["total_tokens_consumed"] for exp in experiments]),
                },
                "mean_wall_clock_per_attempt": {
                    "values": [exp["mean_wall_clock_per_attempt"] for exp in experiments],
                    "mean": safe_mean([exp["mean_wall_clock_per_attempt"] for exp in experiments]),
                    "std": safe_std([exp["mean_wall_clock_per_attempt"] for exp in experiments]),
                },
            },
        }
    return results


def compute_negative_result(raw_metrics: dict) -> dict:
    xs = []
    ys = []
    for cell in CELL_ORDER:
        summary = raw_metrics.get(cell, {}).get("summary", {})
        x = summary.get("total_tokens_consumed", {}).get("mean")
        y = summary.get("best_val_bpb", {}).get("mean")
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            continue
        xs.append(float(x))
        ys.append(float(y))

    if len(xs) < 2:
        return {"r2": float("nan"), "negative_result": False}

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    coef = np.polyfit(x_arr, y_arr, deg=1)
    y_hat = coef[0] * x_arr + coef[1]
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "r2": r2,
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "negative_result": bool(not math.isnan(r2) and r2 > 0.9),
    }


def interp_step_curve(curve: list[dict], x_grid: np.ndarray) -> np.ndarray:
    if not curve:
        return np.full_like(x_grid, np.nan, dtype=float)
    xs = np.asarray([point["cumulative_tokens"] for point in curve], dtype=float)
    ys = np.asarray([point["best_val_bpb"] for point in curve], dtype=float)
    out = np.empty_like(x_grid, dtype=float)
    for index, x_value in enumerate(x_grid):
        eligible = np.where(xs <= x_value)[0]
        out[index] = ys[eligible[-1]] if len(eligible) else ys[0]
    return out


def maybe_generate_figures(
    raw_metrics: dict,
    decomp_stats: dict,
    context_summary: dict,
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figures_dir = output_dir / "figures" / "pass_01_pilot"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    colors = {
        "d00": "#264653",
        "d10": "#2a9d8f",
        "d01": "#e76f51",
        "d11": "#e9c46a",
    }

    plt.figure(figsize=(8, 5))
    for cell in CELL_ORDER:
        curves = [exp["best_so_far_curve"] for exp in raw_metrics[cell]["experiments"] if exp["best_so_far_curve"]]
        if not curves:
            continue
        max_tokens = max(curve[-1]["cumulative_tokens"] for curve in curves)
        x_grid = np.linspace(0, max_tokens, 100)
        y_stack = np.vstack([interp_step_curve(curve, x_grid) for curve in curves])
        y_mean = np.nanmean(y_stack, axis=0)
        plt.plot(x_grid, y_mean, label=cell, color=colors[cell], linewidth=2)
    plt.xlabel("Cumulative tokens consumed")
    plt.ylabel("Best val_bpb so far")
    plt.title("Best-so-far Curves by Cell")
    plt.legend()
    plt.tight_layout()
    curve_path = figures_dir / "best_so_far_curves.png"
    plt.savefig(curve_path, dpi=160)
    plt.close()
    generated.append(str(curve_path))

    plt.figure(figsize=(9, 5))
    x = np.arange(len(CELL_ORDER))
    width = 0.18
    for idx, bin_label in enumerate(BIN_ORDER):
        heights = []
        for cell in CELL_ORDER:
            value = context_summary[cell]["bins"].get(bin_label)
            heights.append(0.0 if value is None else value)
        plt.bar(x + idx * width, heights, width=width, label=bin_label)
    plt.xticks(x + 1.5 * width, CELL_ORDER)
    plt.ylabel("Mean kappa_token")
    plt.title("kappa_token by Context Bin")
    plt.legend()
    plt.tight_layout()
    context_path = figures_dir / "kappa_by_context_bin.png"
    plt.savefig(context_path, dpi=160)
    plt.close()
    generated.append(str(context_path))

    plt.figure(figsize=(8, 5))
    x = np.arange(len(DECOMP_CELLS))
    cost = [decomp_stats[cell]["cost_term_token"]["mean"] for cell in DECOMP_CELLS]
    phi = [decomp_stats[cell]["phi"]["mean"] for cell in DECOMP_CELLS]
    gain = [decomp_stats[cell]["G"]["mean"] for cell in DECOMP_CELLS]
    neg_eps = [-decomp_stats[cell]["epsilon"]["mean"] for cell in DECOMP_CELLS]
    plt.bar(x, cost, label="log(k0/k)_tok")
    plt.bar(x, phi, bottom=cost, label="phi")
    plt.bar(x, gain, bottom=np.asarray(cost) + np.asarray(phi), label="G")
    plt.bar(
        x,
        neg_eps,
        bottom=np.asarray(cost) + np.asarray(phi) + np.asarray(gain),
        label="-epsilon",
    )
    plt.xticks(x, DECOMP_CELLS)
    plt.ylabel("Contribution")
    plt.title("Token-side Decomposition Terms")
    plt.legend()
    plt.tight_layout()
    decomp_path = figures_dir / "decomposition_bar_chart.png"
    plt.savefig(decomp_path, dpi=160)
    plt.close()
    generated.append(str(decomp_path))

    all_modes = Counter()
    for cell in CELL_ORDER:
        for exp in raw_metrics[cell]["experiments"]:
            for mode, prob in exp["mode_distribution"].items():
                all_modes[mode] += prob
    if all_modes:
        plt.figure(figsize=(7, 4))
        labels = sorted(all_modes)
        values = [all_modes[label] for label in labels]
        plt.bar(labels, values, color="#577590")
        plt.ylabel("Aggregated accepted-mode mass")
        plt.title("Mode Distribution Across Pilot")
        plt.tight_layout()
        mode_path = figures_dir / "mode_distribution.png"
        plt.savefig(mode_path, dpi=160)
        plt.close()
        generated.append(str(mode_path))

    return generated


def build_summary_markdown(
    repetitions: list[dict],
    decomp_stats: dict,
    hypothesis_counts: dict,
    context_summary: dict,
    raw_metrics: dict,
    negative_result: dict,
    figure_paths: list[str],
) -> str:
    lines = [
        "# Pilot Summary",
        "",
        "## Decomposition Table (mean +/- std across 3 reps)",
        "",
        "| Cell | log(k0/k)_tok | log(k0/k)_wall | phi | G | -epsilon | delta_tok | delta_wall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in DECOMP_CELLS:
        lines.append(
            "| {cell} | {tok} | {wall} | {phi} | {gain} | {neg_eps} | {dtok} | {dwall} |".format(
                cell=cell,
                tok=format_stat(
                    decomp_stats[cell]["cost_term_token"]["mean"],
                    decomp_stats[cell]["cost_term_token"]["std"],
                ),
                wall=format_stat(
                    decomp_stats[cell]["cost_term_wall"]["mean"],
                    decomp_stats[cell]["cost_term_wall"]["std"],
                ),
                phi=format_stat(
                    decomp_stats[cell]["phi"]["mean"],
                    decomp_stats[cell]["phi"]["std"],
                ),
                gain=format_stat(
                    decomp_stats[cell]["G"]["mean"],
                    decomp_stats[cell]["G"]["std"],
                ),
                neg_eps=format_stat(
                    -decomp_stats[cell]["epsilon"]["mean"],
                    decomp_stats[cell]["epsilon"]["std"],
                ),
                dtok=format_stat(
                    decomp_stats[cell]["delta_token"]["mean"],
                    decomp_stats[cell]["delta_token"]["std"],
                ),
                dwall=format_stat(
                    decomp_stats[cell]["delta_wall"]["mean"],
                    decomp_stats[cell]["delta_wall"]["std"],
                ),
            )
        )

    lines.extend(
        [
            "",
            "Bootstrap 95% confidence intervals (mean of 1000 resamples):",
        ]
    )
    for cell in DECOMP_CELLS:
        lines.append(
            "- {cell}: tok [{tok_lo}, {tok_hi}], wall [{wall_lo}, {wall_hi}], G [{g_lo}, {g_hi}], -epsilon [{e_lo}, {e_hi}]".format(
                cell=cell,
                tok_lo=format_float(decomp_stats[cell]["cost_term_token"]["ci_low"]),
                tok_hi=format_float(decomp_stats[cell]["cost_term_token"]["ci_high"]),
                wall_lo=format_float(decomp_stats[cell]["cost_term_wall"]["ci_low"]),
                wall_hi=format_float(decomp_stats[cell]["cost_term_wall"]["ci_high"]),
                g_lo=format_float(decomp_stats[cell]["G"]["ci_low"]),
                g_hi=format_float(decomp_stats[cell]["G"]["ci_high"]),
                e_lo=format_float(-decomp_stats[cell]["epsilon"]["ci_high"]),
                e_hi=format_float(-decomp_stats[cell]["epsilon"]["ci_low"]),
            )
        )

    lines.extend(
        [
            "",
            "## Hypothesis Verdicts",
            "",
            f"- H1 (parallelism helps only wall-clock): {hypothesis_counts['H1']}/3 reps support",
            f"- H2 (memory helps both axes): {hypothesis_counts['H2']}/3 reps support",
            f"- H3 (shared memory lowers epsilon): {hypothesis_counts['H3']}/3 reps support",
            f"- H4 (parallelism sensitive to coordination): {hypothesis_counts['H4']}/3 reps support",
            f"- H5 (context pressure dominant): {hypothesis_counts['H5']}/3 reps support",
            f"- H6 (d11 dominates d00 on both axes): {hypothesis_counts['H6']}/3 reps support",
            "",
            "## Context Pressure Analysis (H5)",
            "",
            "| Cell | 0-25% | 25-50% | 50-75% | 75-100% | Monotone? |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for cell in CELL_ORDER:
        bins = context_summary[cell]["bins"]
        lines.append(
            "| {cell} | {b0} | {b1} | {b2} | {b3} | {mono} |".format(
                cell=cell,
                b0=format_float(bins.get("0-25%"), digits=0),
                b1=format_float(bins.get("25-50%"), digits=0),
                b2=format_float(bins.get("50-75%"), digits=0),
                b3=format_float(bins.get("75-100%"), digits=0),
                mono="yes" if context_summary[cell]["monotone"] else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Raw Metrics",
            "",
            "| Cell | Total training runs | Best val_bpb | Total tokens | Mean wall-clock / attempt (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cell in CELL_ORDER:
        summary = raw_metrics[cell]["summary"]
        lines.append(
            "| {cell} | {runs} | {best} | {tokens} | {wall} |".format(
                cell=cell,
                runs=format_stat(
                    summary["total_training_runs"]["mean"],
                    summary["total_training_runs"]["std"],
                ),
                best=format_stat(
                    summary["best_val_bpb"]["mean"],
                    summary["best_val_bpb"]["std"],
                ),
                tokens=format_stat(
                    summary["total_tokens_consumed"]["mean"],
                    summary["total_tokens_consumed"]["std"],
                ),
                wall=format_stat(
                    summary["mean_wall_clock_per_attempt"]["mean"],
                    summary["mean_wall_clock_per_attempt"]["std"],
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "H1: The wall-clock term for d01 determines whether parallelism helped by latency reduction alone; token-side gains near zero or negative indicate coordination overhead without extra search efficiency.",
            "H2: d10 is favorable when the memory table reduces token and wall costs simultaneously, implying better state compression than plain conversation accumulation.",
            "H3/H4: epsilon captures coordination mismatch. Lower epsilon in d11 than d01 supports useful shared-memory routing; high epsilon suggests the parallel cell is paying coordination tax.",
            "H5: The context-bin table tests whether token cost rises with context pressure in d00 and whether d10 flattens that curve. If it does, the memory mechanism is acting like a context compressor rather than extra baggage.",
            "H6: d11 only clearly dominates when both delta_wall and delta_token stay positive across reps. If that fails, the shared-memory benefits are still conditional on coordination quality or search diversity.",
            "",
            "## Negative Result Criterion",
            "",
            f"- Linear fit R^2(best_val_bpb ~ total_tokens): {format_float(negative_result.get('r2'))}",
            f"- Negative result criterion met (R^2 > 0.9): {'yes' if negative_result.get('negative_result') else 'no'}",
        ]
    )

    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for path in figure_paths:
            lines.append(f"- {path}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Repetitions aggregated: {len(repetitions)}",
            "- Cells use the Task 9 2x2 pilot mapping from runs/pilot_mapping.json.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("runs/pilot_mapping.json"),
        help="JSON file mapping d00/d10/d01/d11 to 3 experiment directories each.",
    )
    parser.add_argument(
        "--decomposition-glob",
        type=str,
        default="results/decomposition_rep*.json",
        help="Glob for repetition-level decomposition JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for pilot_summary.md, pilot_raw_data.json, and figures.",
    )
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text())
    rep_paths = sorted(Path(path) for path in glob.glob(args.decomposition_glob))
    if not rep_paths:
        raise FileNotFoundError(
            f"No decomposition files matched {args.decomposition_glob!r}"
        )

    repetitions = [json.loads(path.read_text()) for path in rep_paths]
    decomp_stats = aggregate_decompositions(repetitions)
    hypothesis_counts = aggregate_hypotheses(repetitions)
    context_summary = aggregate_context_bins(repetitions)
    raw_metrics = aggregate_raw_metrics(mapping)
    negative_result = compute_negative_result(raw_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = maybe_generate_figures(
        raw_metrics=raw_metrics,
        decomp_stats=decomp_stats,
        context_summary=context_summary,
        output_dir=args.output_dir,
    )

    summary_md = build_summary_markdown(
        repetitions=repetitions,
        decomp_stats=decomp_stats,
        hypothesis_counts=hypothesis_counts,
        context_summary=context_summary,
        raw_metrics=raw_metrics,
        negative_result=negative_result,
        figure_paths=figure_paths,
    )
    (args.output_dir / "pilot_summary.md").write_text(summary_md)

    raw_payload = {
        "mapping": mapping,
        "repetitions": repetitions,
        "decomposition_stats": decomp_stats,
        "hypothesis_counts": hypothesis_counts,
        "context_summary": context_summary,
        "raw_metrics": raw_metrics,
        "negative_result": negative_result,
        "figure_paths": figure_paths,
    }
    (args.output_dir / "pilot_raw_data.json").write_text(
        json.dumps(raw_payload, indent=2)
    )

    print(f"Wrote {args.output_dir / 'pilot_summary.md'}")
    print(f"Wrote {args.output_dir / 'pilot_raw_data.json'}")
    if figure_paths:
        print(f"Generated {len(figure_paths)} figure(s) under {args.output_dir / 'figures' / 'pass_01_pilot'}")


if __name__ == "__main__":
    main()
