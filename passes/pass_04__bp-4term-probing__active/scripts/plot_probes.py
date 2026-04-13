#!/usr/bin/env python3
"""Generate key figures for probe analysis.

Usage:
    python workflow/scripts/plot_probes.py --repo-root .
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_probe_runs(repo_root: Path, probe_id: str) -> list[dict]:
    """Load all training runs for a probe."""
    experiment_dir = repo_root / "runs" / f"experiment_probe_{probe_id}"
    if not experiment_dir.exists():
        return []
    runs = []
    for jf in experiment_dir.rglob("training_runs.jsonl"):
        agent_id = "agent_0"
        for p in str(jf).split("/"):
            if p.startswith("agent_"):
                agent_id = p
        for line in jf.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                d["_agent"] = d.get("agent_id", agent_id)
                runs.append(d)
            except json.JSONDecodeError:
                pass
    return runs


PROBES = {
    "P01": "parallel_homo",
    "P02": "parallel_diverse",
    "P03": "single_baseline",
    "P04": "single_30s",
    "P05": "single_memory*",
    "P06": "shared_diverse*",
    "P07": "shared_ext*",
    "P08": "memory_ext*",
    "P09": "diverse_ext",
    "P10": "homo_fixed",
    "P11": "single_ht",
    "P12": "shared_fixed",
    "P13": "dual_ht",
    "P14": "ht_memory",
    "P15": "seeded",
    "P16": "opt_baseline",
    "P17": "full_stack",
    "P18": "seeded_par",
}

CAT_COLORS = {
    "optimization": "#2ecc71",
    "regularization": "#e74c3c",
    "other": "#3498db",
    "data_pipeline": "#f39c12",
    "architecture": "#9b59b6",
    "memory_or_coordination": "#1abc9c",
    "unknown": "#95a5a6",
}


def plot_trajectories(repo_root: Path, output_dir: Path) -> None:
    """Plot bpb trajectories for all probes."""
    n_probes = len(PROBES)
    ncols = 6
    nrows = (n_probes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 4 * nrows), sharey=True)
    axes = axes.flatten()
    baseline_bpb = 0.925845

    for i, (pid, name) in enumerate(PROBES.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        runs = load_probe_runs(repo_root, pid)
        if not runs:
            ax.set_title(f"{pid}\n{name}\n(no data)", fontsize=9)
            ax.set_facecolor("#f0f0f0")
            continue

        agents = sorted(set(r["_agent"] for r in runs))
        for j, agent in enumerate(agents):
            agent_runs = sorted(
                [r for r in runs if r["_agent"] == agent],
                key=lambda r: r.get("run_index", 0),
            )
            bpbs = [r["val_bpb"] for r in agent_runs if r.get("val_bpb")]
            cats = [r.get("strategy_category", "unknown") for r in agent_runs
                    if r.get("val_bpb")]
            colors = [CAT_COLORS.get(c, "#95a5a6") for c in cats]

            x = list(range(1, len(bpbs) + 1))
            label = agent.replace("agent_", "a")
            ax.plot(x, bpbs, "-", alpha=0.4, linewidth=1, color=f"C{j}")
            ax.scatter(x, bpbs, c=colors, s=25, zorder=3, edgecolors="white",
                       linewidth=0.5)

        ax.axhline(baseline_bpb, color="red", linestyle="--", alpha=0.5,
                    linewidth=0.8)
        ax.set_title(f"{pid}\n{name}", fontsize=9)
        ax.set_xlabel("run", fontsize=8)
        if i % 5 == 0:
            ax.set_ylabel("val_bpb", fontsize=8)
        ax.tick_params(labelsize=7)

    # Remove empty subplots
    for i in range(len(PROBES), len(axes)):
        axes[i].set_visible(False)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=k)
                      for k, c in CAT_COLORS.items() if k != "unknown"]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=8,
               frameon=True, fancybox=True)

    fig.suptitle("Probe Trajectories: val_bpb per run (dashed = baseline 0.926)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = output_dir / "probe_trajectories.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_temperature_effect(repo_root: Path, output_dir: Path) -> None:
    """Plot P07 temperature comparison: agent_0 (temp=0.3) vs agent_1 (temp=1.2)."""
    runs = load_probe_runs(repo_root, "P07")
    if not runs:
        print("No P07 data, skipping temperature plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for agent, ax, temp_label in [
        ("agent_0", ax1, "temp=0.3 (conservative)"),
        ("agent_1", ax2, "temp=1.2 (exploratory)"),
    ]:
        agent_runs = sorted(
            [r for r in runs if r["_agent"] == agent],
            key=lambda r: r.get("run_index", 0),
        )
        bpbs = [r["val_bpb"] for r in agent_runs if r.get("val_bpb")]
        cats = [r.get("strategy_category", "unknown") for r in agent_runs
                if r.get("val_bpb")]
        colors = [CAT_COLORS.get(c, "#95a5a6") for c in cats]

        x = list(range(1, len(bpbs) + 1))
        ax.plot(x, bpbs, "-o", alpha=0.6, color="gray", markersize=4, linewidth=1)
        ax.scatter(x, bpbs, c=colors, s=60, zorder=3, edgecolors="black",
                   linewidth=0.5)

        ax.axhline(0.925845, color="red", linestyle="--", alpha=0.5, linewidth=1,
                    label="baseline")
        best_idx = bpbs.index(min(bpbs)) + 1
        ax.annotate(f"best={min(bpbs):.4f}", xy=(best_idx, min(bpbs)),
                    xytext=(best_idx, min(bpbs) - 0.03), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="green"),
                    color="green", fontweight="bold")

        ax.set_title(f"{temp_label}\n{len(bpbs)} runs in 30 min", fontsize=11)
        ax.set_xlabel("run index", fontsize=10)
        ax.set_ylabel("val_bpb", fontsize=10)
        ax.legend(fontsize=8)

    legend_patches = [mpatches.Patch(color=c, label=k)
                      for k, c in CAT_COLORS.items()
                      if k in {"optimization", "regularization", "other"}]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9)

    fig.suptitle("P07: Temperature Controls Iteration Speed\n"
                 "(Same budget, same task — 5x more runs with temp=1.2)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    path = output_dir / "probe_P07_temperature_effect.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_best_bpb_comparison(repo_root: Path, output_dir: Path) -> None:
    """Bar chart of best bpb per probe."""
    data = {}
    for pid in PROBES:
        runs = load_probe_runs(repo_root, pid)
        if runs:
            bpbs = [r["val_bpb"] for r in runs if r.get("val_bpb")]
            if bpbs:
                data[pid] = {
                    "best": min(bpbs),
                    "n_runs": len(bpbs),
                    "name": PROBES[pid],
                }

    if not data:
        print("No data for bar chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    pids = sorted(data.keys())
    bests = [data[p]["best"] for p in pids]
    n_runs = [data[p]["n_runs"] for p in pids]
    labels = [f"{p}\n{data[p]['name']}\n({data[p]['n_runs']} runs)" for p in pids]

    bars = ax.bar(range(len(pids)), bests, color=["#3498db" if b > 0.925845
                  else "#2ecc71" for b in bests], edgecolor="white", linewidth=0.5)

    ax.axhline(0.925845, color="red", linestyle="--", alpha=0.7, linewidth=1,
               label="Phase 03 baseline")
    ax.set_xticks(range(len(pids)))
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("Best val_bpb (lower is better)", fontsize=10)
    ax.set_title("Best val_bpb per Probe\n(green = below Phase 03 baseline)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0.85, max(bests) + 0.05)

    for i, (b, n) in enumerate(zip(bests, n_runs)):
        ax.text(i, b + 0.005, f"{b:.4f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = output_dir / "probe_best_bpb_comparison.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_degradation_comparison(repo_root: Path, output_dir: Path) -> None:
    """Compare degradation trajectories: P07 (success) vs P08/P11 (failure)."""
    probes_to_plot = {
        "P07-agent_1": ("P07", "agent_1", "temp=1.2, no memory (SUCCESS)"),
        "P08-agent_0": ("P08", "agent_0", "temp=default, memory broken"),
        "P11-agent_0": ("P11", "agent_0", "temp=1.2, no memory (LONG)"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    baseline = 0.925845

    for ax, (label, (pid, agent_filter, title)) in zip(axes, probes_to_plot.items()):
        runs = load_probe_runs(repo_root, pid)
        if not runs:
            ax.set_title(f"{pid}\n(no data)")
            continue
        agent_runs = sorted(
            [r for r in runs if r["_agent"] == agent_filter],
            key=lambda r: r.get("run_index", 0),
        )
        bpbs = [r["val_bpb"] for r in agent_runs if r.get("val_bpb")]
        cats = [r.get("strategy_category", "unknown") for r in agent_runs
                if r.get("val_bpb")]
        colors = [CAT_COLORS.get(c, "#95a5a6") for c in cats]

        x = list(range(1, len(bpbs) + 1))
        ax.plot(x, bpbs, "-", alpha=0.4, color="gray", linewidth=1)
        ax.scatter(x, bpbs, c=colors, s=50, zorder=3, edgecolors="black",
                   linewidth=0.5)

        ax.axhline(baseline, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.fill_between(x, [baseline] * len(x), bpbs,
                        where=[b > baseline for b in bpbs],
                        alpha=0.1, color="red", label="above baseline")
        ax.fill_between(x, [baseline] * len(x), bpbs,
                        where=[b <= baseline for b in bpbs],
                        alpha=0.1, color="green", label="below baseline")

        if bpbs:
            best_idx = bpbs.index(min(bpbs))
            ax.annotate(f"{min(bpbs):.4f}", xy=(best_idx + 1, min(bpbs)),
                        xytext=(best_idx + 1, min(bpbs) - 0.05),
                        fontsize=8, color="green", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="green"))

        ax.set_title(f"{pid}: {title}\n{len(bpbs)} runs", fontsize=10)
        ax.set_xlabel("run index", fontsize=9)
        ax.set_ylabel("val_bpb", fontsize=9)

    fig.suptitle("G without ε: High Generation Rate without Memory Correction\n"
                 "Red fill = above baseline (degradation), Green fill = below baseline (improvement)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    path = output_dir / "probe_degradation_comparison.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_memory_effect(repo_root: Path, output_dir: Path) -> None:
    """Plot Wave 3 memory effect: P11 (no memory) vs P14 (memory) both at temp=1.2."""
    data = {}
    for pid in ("P11", "P14"):
        runs = load_probe_runs(repo_root, pid)
        if runs:
            agent_runs = sorted(
                [r for r in runs if r.get("val_bpb")],
                key=lambda r: r.get("run_index", 0),
            )
            data[pid] = [r["val_bpb"] for r in agent_runs]

    if not data or len(data) < 2:
        print("Need both P11 and P14 data for memory effect plot, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    baseline = 0.925845

    for ax, pid, title in [
        (ax1, "P11", "temp=1.2, NO memory"),
        (ax2, "P14", "temp=1.2, WITH memory"),
    ]:
        bpbs = data.get(pid, [])
        if not bpbs:
            ax.set_title(f"{pid}: {title}\n(no data)")
            continue
        x = list(range(1, len(bpbs) + 1))
        ax.plot(x, bpbs, "-o", alpha=0.6, color="gray", markersize=4, linewidth=1)
        ax.axhline(baseline, color="red", linestyle="--", alpha=0.5, linewidth=1)

        # Color-code improvement
        for i, b in enumerate(bpbs):
            color = "#2ecc71" if b < baseline else "#e74c3c"
            ax.scatter([i + 1], [b], c=color, s=60, zorder=3, edgecolors="black",
                       linewidth=0.5)

        ax.set_title(f"{pid}: {title}\n{len(bpbs)} runs, best={min(bpbs):.4f}", fontsize=11)
        ax.set_xlabel("run index", fontsize=10)
        ax.set_ylabel("val_bpb", fontsize=10)

    fig.suptitle("Memory Effect at High Temperature (ε term in BP framework)\n"
                 "Green = below baseline, Red = above baseline",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    path = output_dir / "probe_memory_effect.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main() -> None:
    repo_root = (
        Path(sys.argv[sys.argv.index("--repo-root") + 1]).resolve()
        if "--repo-root" in sys.argv
        else Path.cwd()
    )
    output_dir = repo_root / "results" / "figures" / "phase_04_probes"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectories(repo_root, output_dir)
    plot_temperature_effect(repo_root, output_dir)
    plot_best_bpb_comparison(repo_root, output_dir)
    plot_degradation_comparison(repo_root, output_dir)
    plot_memory_effect(repo_root, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
