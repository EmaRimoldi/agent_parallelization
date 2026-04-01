"""
analyze_runs.py — Analysis pipeline for agent parallelisation experiment results.

Repository structure assumed (verified from runs/):
  runs/<experiment_id>/
    config.json
    mode_parallel/
      agent_<N>/
        results/
          trajectory.jsonl     ← {step, val_bpb} per training run
          metadata.json        ← summary (best_val_bpb, total_training_runs, …)
        snapshots/
          step_NNN/
            train.py           ← frozen snapshot of train.py at this step
            metadata.json      ← git_message, val_bpb_after, accepted, …
        workspace/
          train.py.baseline    ← original unmodified train.py

Usage:
    python scripts/analyze_runs.py                           # latest experiment
    python scripts/analyze_runs.py --runs-dir runs/          # all experiments
    python scripts/analyze_runs.py --experiment exp_20260401_013535
    python scripts/analyze_runs.py --output-dir my_analysis/
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Run / agent discovery
# ---------------------------------------------------------------------------

def discover_experiments(runs_dir: Path) -> list[Path]:
    """Return sorted list of valid experiment directories under runs_dir."""
    experiments = sorted(
        p for p in runs_dir.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )
    return experiments


def discover_agents(experiment_dir: Path) -> list[Path]:
    """Return sorted list of agent directories inside an experiment."""
    mode_parallel = experiment_dir / "mode_parallel"
    if not mode_parallel.exists():
        return []
    agents = sorted(
        p for p in mode_parallel.iterdir()
        if p.is_dir() and p.name.startswith("agent_")
    )
    return agents


# ---------------------------------------------------------------------------
# 2. Metric extraction
# ---------------------------------------------------------------------------

def load_trajectory(agent_dir: Path) -> pd.DataFrame:
    """Load loss-over-step trajectory. Returns DataFrame with [step, val_bpb]."""
    traj_path = agent_dir / "results" / "trajectory.jsonl"
    if not traj_path.exists():
        return pd.DataFrame(columns=["step", "val_bpb"])
    rows = []
    with traj_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame(columns=["step", "val_bpb"])
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["val_bpb"])
    df = df.sort_values("step").reset_index(drop=True)
    return df


def load_agent_metadata(agent_dir: Path) -> dict:
    """Load agent-level summary metadata."""
    meta_path = agent_dir / "results" / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# 3. Snapshot / diff extraction
# ---------------------------------------------------------------------------

def load_snapshots(agent_dir: Path) -> list[dict]:
    """
    Load all snapshots for an agent, sorted by step index.
    Each entry: {step_index, git_message, val_bpb_after, accepted, train_py_text}
    """
    snapshots_dir = agent_dir / "snapshots"
    if not snapshots_dir.exists():
        return []

    entries = []
    for step_dir in sorted(snapshots_dir.iterdir()):
        if not step_dir.is_dir():
            continue
        meta_path = step_dir / "metadata.json"
        train_py_path = step_dir / "train.py"
        if not meta_path.exists() or not train_py_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        train_text = train_py_path.read_text(errors="replace")
        entries.append({
            "step_index": meta.get("step_index", -1),
            "git_message": meta.get("git_message", ""),
            "val_bpb_before": meta.get("val_bpb_before"),
            "val_bpb_after": meta.get("val_bpb_after"),
            "accepted": meta.get("accepted"),
            "hypothesis": meta.get("hypothesis", ""),
            "train_py_text": train_text,
        })
    entries.sort(key=lambda x: x["step_index"])
    return entries


def load_baseline(agent_dir: Path) -> Optional[str]:
    """Load the baseline train.py text."""
    baseline_path = agent_dir / "workspace" / "train.py.baseline"
    if not baseline_path.exists():
        return None
    return baseline_path.read_text(errors="replace")


# ---------------------------------------------------------------------------
# 4. Parameter-level diff extraction
# ---------------------------------------------------------------------------

# Matches top-level UPPERCASE parameter assignments, e.g. MATRIX_LR = 0.04
PARAM_RE = re.compile(
    r"^(?P<name>[A-Z][A-Z0-9_]+)\s*=\s*(?P<value>.+?)(?:\s*#.*)?$",
    re.MULTILINE,
)


def extract_params(text: str) -> dict[str, str]:
    """Extract top-level uppercase parameter assignments from a Python file."""
    params: dict[str, str] = {}
    for m in PARAM_RE.finditer(text):
        params[m.group("name")] = m.group("value").strip()
    return params


def diff_params(before: str, after: str) -> dict[str, tuple[str, str]]:
    """
    Return {param_name: (old_value, new_value)} for parameters that changed.
    """
    p_before = extract_params(before)
    p_after = extract_params(after)
    changed = {}
    all_keys = set(p_before) | set(p_after)
    for k in all_keys:
        v_before = p_before.get(k)
        v_after = p_after.get(k)
        if v_before != v_after:
            changed[k] = (v_before, v_after)
    return changed


def count_line_changes(before: str, after: str) -> tuple[int, int]:
    """Return (lines_added, lines_removed) between two text versions."""
    diff = list(difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm=""
    ))
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    return added, removed


def build_modification_table(
    agent_id: str,
    snapshots: list[dict],
    baseline_text: Optional[str],
) -> pd.DataFrame:
    """
    For each snapshot step, compute what changed relative to the *previous* snapshot
    (or to baseline for step 0). Returns a per-step DataFrame.
    """
    rows = []
    prev_text = baseline_text

    for snap in snapshots:
        curr_text = snap["train_py_text"]
        if prev_text is None:
            # No baseline available — can't compute diff for step 0
            rows.append({
                "agent_id": agent_id,
                "step_index": snap["step_index"],
                "git_message": snap["git_message"],
                "val_bpb_after": snap["val_bpb_after"],
                "accepted": snap["accepted"],
                "params_changed": None,
                "param_names": "",
                "lines_added": None,
                "lines_removed": None,
            })
            prev_text = curr_text
            continue

        param_diffs = diff_params(prev_text, curr_text)
        lines_added, lines_removed = count_line_changes(prev_text, curr_text)

        rows.append({
            "agent_id": agent_id,
            "step_index": snap["step_index"],
            "git_message": snap["git_message"],
            "val_bpb_after": snap["val_bpb_after"],
            "accepted": snap["accepted"],
            "params_changed": len(param_diffs),
            "param_names": ", ".join(sorted(param_diffs.keys())),
            "lines_added": lines_added,
            "lines_removed": lines_removed,
        })
        prev_text = curr_text

    return pd.DataFrame(rows)


def build_param_frequency_table(
    agent_id: str,
    snapshots: list[dict],
    baseline_text: Optional[str],
) -> pd.DataFrame:
    """Count how often each parameter was changed by this agent."""
    param_counts: dict[str, int] = {}
    prev_text = baseline_text
    for snap in snapshots:
        curr_text = snap["train_py_text"]
        if prev_text is not None:
            diffs = diff_params(prev_text, curr_text)
            for param in diffs:
                param_counts[param] = param_counts.get(param, 0) + 1
        prev_text = curr_text
    rows = [
        {"agent_id": agent_id, "param": k, "times_changed": v}
        for k, v in sorted(param_counts.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["agent_id", "param", "times_changed"]
    )


def build_param_impact_table(
    agent_id: str,
    snapshots: list[dict],
    baseline_text: Optional[str],
) -> pd.DataFrame:
    """
    For each snapshot step where both val_bpb_before and val_bpb_after are
    available, identify which parameters changed and record the signed loss
    delta (val_bpb_after - val_bpb_before). Lower val_bpb is better, so a
    negative delta means improvement.

    Returns one row per (agent, step, parameter) with columns:
        agent_id, step_index, param, delta_bpb, accepted
    """
    rows = []
    prev_text = baseline_text

    for snap in snapshots:
        curr_text = snap["train_py_text"]
        bpb_before = snap.get("val_bpb_before")  # stored directly in snapshot metadata
        bpb_after = snap.get("val_bpb_after")

        # Skip steps where we can't compute a meaningful delta
        # (bpb_before=0.0 is a sentinel used in agent_1/step_000)
        if (
            prev_text is not None
            and bpb_before is not None
            and bpb_before != 0.0
            and bpb_after is not None
        ):
            delta = bpb_after - bpb_before
            param_diffs = diff_params(prev_text, curr_text)
            for param in param_diffs:
                rows.append({
                    "agent_id": agent_id,
                    "step_index": snap["step_index"],
                    "param": param,
                    "delta_bpb": delta,
                    "accepted": snap.get("accepted"),
                })

        prev_text = curr_text

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["agent_id", "step_index", "param", "delta_bpb", "accepted"]
    )


# ---------------------------------------------------------------------------
# 5. Parameter impact heatmap
# ---------------------------------------------------------------------------

def plot_param_impact_heatmap(
    all_impact_tables: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """
    Heatmap: parameters × agents, color = mean signed delta_bpb.

    - Blue  (negative delta) = parameter change was associated with improvement.
    - Red   (positive delta) = parameter change was associated with worsening.
    - White = no data for that (agent, parameter) pair.
    - Parameters touched only by one agent are labelled with ★.
    - Cell annotations show: mean delta (top) and change count (bottom).

    A companion bar chart (right panel) shows the mean delta averaged across
    all agents, giving a quick ranking of which parameters actually help.
    """
    if not all_impact_tables or all(df.empty for df in all_impact_tables.values()):
        print("  [warn] No impact data to plot.")
        return

    combined = pd.concat(list(all_impact_tables.values()), ignore_index=True)
    if combined.empty:
        print("  [warn] No impact data to plot.")
        return

    agents = sorted(combined["agent_id"].unique())
    params = sorted(combined["param"].unique())

    # Build pivot: mean delta per (param, agent)
    pivot_mean = combined.pivot_table(
        index="param", columns="agent_id", values="delta_bpb",
        aggfunc="mean",
    ).reindex(index=params, columns=agents)

    # Build pivot: count per (param, agent) — for annotations
    pivot_count = combined.pivot_table(
        index="param", columns="agent_id", values="delta_bpb",
        aggfunc="count",
    ).reindex(index=params, columns=agents).fillna(0).astype(int)

    # Identify parameters touched by only one agent
    agent_sets: dict[str, set] = {
        agent: set(combined[combined["agent_id"] == agent]["param"].unique())
        for agent in agents
    }
    all_params_set = set(params)
    exclusive: dict[str, set] = {}
    for agent in agents:
        others = all_params_set - agent_sets[agent]
        # exclusive to this agent = touched by this agent but not by any other
        exclusive[agent] = agent_sets[agent] - set.union(
            *(agent_sets[a] for a in agents if a != agent), set()
        )
    exclusive_params = set.union(*exclusive.values()) if exclusive else set()

    # Sort parameters by mean delta (averaged across agents, most improvement first)
    row_means = pivot_mean.mean(axis=1, skipna=True)
    sorted_params = row_means.sort_values().index.tolist()  # most improvement (most negative) on top
    pivot_mean = pivot_mean.reindex(sorted_params)
    pivot_count = pivot_count.reindex(sorted_params)

    # Compute symmetric color scale
    vmax = pivot_mean.abs().max().max()
    if np.isnan(vmax) or vmax == 0:
        vmax = 0.01
    vmin = -vmax

    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2,
        figsize=(5 + 2 * len(agents), max(5, 0.55 * len(sorted_params) + 2)),
        gridspec_kw={"width_ratios": [len(agents) * 2, 2]},
    )
    fig.suptitle(
        "Parameter Impact on val_bpb\n"
        "(mean signed Δval_bpb per change; blue = improvement, red = worsening)",
        fontsize=12, fontweight="bold",
    )

    # --- heatmap ---
    import matplotlib.colors as mcolors
    cmap = plt.get_cmap("RdBu")  # blue=negative (improvement), red=positive (worsening)
    mat = pivot_mean.values.astype(float)

    im = ax_heat.imshow(
        mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
    )

    # Axes labels
    ax_heat.set_xticks(range(len(agents)))
    ax_heat.set_xticklabels(agents, fontsize=10)
    ax_heat.set_yticks(range(len(sorted_params)))

    # Y-axis tick labels: add ★ for exclusive params
    ylabels = []
    for p in sorted_params:
        if p in exclusive_params:
            owners = [a for a in agents if p in exclusive.get(a, set())]
            ylabels.append(f"{p}  ★({owners[0]})" if owners else f"{p}  ★")
        else:
            ylabels.append(p)
    ax_heat.set_yticklabels(ylabels, fontsize=9)

    # Cell annotations: mean delta + count
    for i, param in enumerate(sorted_params):
        for j, agent in enumerate(agents):
            mean_val = pivot_mean.loc[param, agent] if agent in pivot_mean.columns else np.nan
            cnt = pivot_count.loc[param, agent] if agent in pivot_count.columns else 0
            if np.isnan(mean_val):
                ax_heat.text(j, i, "—", ha="center", va="center", fontsize=9, color="#aaaaaa")
            else:
                sign = "−" if mean_val < 0 else "+"
                mag = abs(mean_val)
                text_color = "white" if abs(mean_val) > vmax * 0.55 else "black"
                ax_heat.text(
                    j, i,
                    f"{sign}{mag:.4f}\n(n={cnt})",
                    ha="center", va="center", fontsize=7.5, color=text_color,
                    linespacing=1.4,
                )

    # Grid lines between cells
    ax_heat.set_xticks(np.arange(-0.5, len(agents)), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(sorted_params)), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.5)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    plt.colorbar(im, ax=ax_heat, label="mean Δval_bpb", shrink=0.7)

    # --- companion bar chart: overall mean delta per parameter ---
    overall_mean = pivot_mean.mean(axis=1, skipna=True).reindex(sorted_params)
    colors_bar = ["#2166ac" if v < 0 else "#d73027" for v in overall_mean]
    # Mark exclusive params
    edge_colors = ["black" if p in exclusive_params else "none" for p in sorted_params]

    ax_bar.barh(
        range(len(sorted_params)), overall_mean.values,
        color=colors_bar, edgecolor=edge_colors, linewidth=1.2,
    )
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_yticks(range(len(sorted_params)))
    ax_bar.set_yticklabels([])   # shared y-axis with heatmap — labels already on left
    ax_bar.set_xlabel("Mean Δval_bpb\n(all agents combined)", fontsize=9)
    ax_bar.set_title("Overall\nimpact", fontsize=9)
    ax_bar.grid(True, axis="x", alpha=0.3)

    # Legend
    legend_handles = [
        mpatches.Patch(color="#2166ac", label="improvement (Δ < 0)"),
        mpatches.Patch(color="#d73027", label="worsening  (Δ > 0)"),
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=1.2,
                       label="★ exclusive to one agent"),
    ]
    ax_heat.legend(handles=legend_handles, fontsize=8,
                   loc="lower right", bbox_to_anchor=(1.0, -0.02),
                   framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# 7. Loss-over-step visualization
# ---------------------------------------------------------------------------

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_loss_curves(
    trajectories: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """
    Plot val_bpb vs training step for all agents.
    Produces a combined figure + a multi-panel figure.
    """
    n = len(trajectories)
    if n == 0:
        print("  [warn] No trajectories to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Loss (val_bpb) per Agent", fontsize=13, fontweight="bold")

    # --- left panel: combined ---
    ax_combined = axes[0]
    ax_combined.set_title("Combined")
    ax_combined.set_xlabel("Training step")
    ax_combined.set_ylabel("val_bpb (lower is better)")

    for idx, (agent_id, df) in enumerate(sorted(trajectories.items())):
        if df.empty:
            continue
        color = COLORS[idx % len(COLORS)]
        ax_combined.plot(
            df["step"], df["val_bpb"],
            marker=".", linewidth=1.5, color=color, label=agent_id, alpha=0.85,
        )
        best_idx = df["val_bpb"].idxmin()
        best_step = df.loc[best_idx, "step"]
        best_loss = df.loc[best_idx, "val_bpb"]
        ax_combined.scatter(
            [best_step], [best_loss],
            color=color, s=120, zorder=5, marker="*",
        )
        ax_combined.annotate(
            f" {best_loss:.5f}",
            xy=(best_step, best_loss),
            fontsize=8, color=color, va="center",
        )

    ax_combined.legend(fontsize=9)
    ax_combined.grid(True, alpha=0.3)

    # --- right panel: multi-panel ---
    ax_multi = axes[1]
    ax_multi.axis("off")

    # Build subplots grid
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    # Replace right panel with a proper subplot grid
    fig2, panel_axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )
    fig2.suptitle("Training Loss per Agent (individual panels)", fontsize=13, fontweight="bold")

    for idx, (agent_id, df) in enumerate(sorted(trajectories.items())):
        row, col = divmod(idx, ncols)
        ax = panel_axes[row][col]
        ax.set_title(agent_id, fontsize=10)
        ax.set_xlabel("Training step")
        ax.set_ylabel("val_bpb")

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        color = COLORS[idx % len(COLORS)]
        ax.plot(df["step"], df["val_bpb"], marker=".", linewidth=1.5, color=color, alpha=0.85)

        best_idx = df["val_bpb"].idxmin()
        best_step = df.loc[best_idx, "step"]
        best_loss = df.loc[best_idx, "val_bpb"]

        ax.scatter([best_step], [best_loss], color="red", s=150, zorder=5, marker="*", label=f"Best: {best_loss:.5f}")
        ax.axhline(best_loss, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        panel_axes[row][col].axis("off")

    # Fix layout and remove empty right panel from fig
    axes[1].remove()
    fig.tight_layout()
    # Save combined figure
    combined_path = output_path.parent / "loss_curves_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {combined_path}")

    fig2.tight_layout()
    fig2.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# 6. Best-loss summary table
# ---------------------------------------------------------------------------

def build_loss_summary(
    trajectories: dict[str, pd.DataFrame],
    agent_metadata: dict[str, dict],
) -> pd.DataFrame:
    """Build summary table with one row per agent."""
    rows = []
    for agent_id, df in sorted(trajectories.items()):
        meta = agent_metadata.get(agent_id, {})
        if df.empty:
            rows.append({
                "agent_id": agent_id,
                "total_steps": 0,
                "best_val_bpb": None,
                "best_step": None,
                "final_val_bpb": None,
                "total_training_runs": meta.get("total_training_runs"),
                "model": meta.get("model"),
            })
            continue
        best_idx = df["val_bpb"].idxmin()
        rows.append({
            "agent_id": agent_id,
            "total_steps": len(df),
            "best_val_bpb": df.loc[best_idx, "val_bpb"],
            "best_step": int(df.loc[best_idx, "step"]),
            "final_val_bpb": df.iloc[-1]["val_bpb"],
            "total_training_runs": meta.get("total_training_runs"),
            "model": meta.get("model"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Modification visualizations
# ---------------------------------------------------------------------------

def plot_modification_summary(
    all_mod_tables: dict[str, pd.DataFrame],
    all_param_freq: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """
    Produce a 2-row figure:
      Row 1: per-agent bar chart of total line changes (added/removed) per step
      Row 2: heatmap / bar chart of parameter change frequency across agents
    """
    n_agents = len(all_mod_tables)
    if n_agents == 0:
        print("  [warn] No modification data to plot.")
        return

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("train.py Modifications per Agent", fontsize=13, fontweight="bold")

    # --- top section: line changes per step for each agent ---
    for idx, (agent_id, df) in enumerate(sorted(all_mod_tables.items())):
        ax = fig.add_subplot(2, n_agents, idx + 1)
        ax.set_title(f"{agent_id}\nline changes per step", fontsize=9)
        ax.set_xlabel("Step index")
        ax.set_ylabel("Lines")

        if df.empty or df["lines_added"].isna().all():
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        valid = df.dropna(subset=["lines_added", "lines_removed"])
        steps = valid["step_index"].tolist()
        added = valid["lines_added"].tolist()
        removed = [-v for v in valid["lines_removed"].tolist()]

        # Color bars by accepted status
        colors_add = [
            "#2ca02c" if a else "#d62728" if a is False else "#7f7f7f"
            for a in valid["accepted"].tolist()
        ]

        ax.bar(steps, added, color=colors_add, alpha=0.75, label="added")
        ax.bar(steps, removed, color="#aec7e8", alpha=0.6, label="removed")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(steps)
        ax.set_xticklabels([str(s) for s in steps], fontsize=7, rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

        # Legend for accepted/rejected
        patches = [
            mpatches.Patch(color="#2ca02c", label="accepted"),
            mpatches.Patch(color="#d62728", label="rejected"),
            mpatches.Patch(color="#7f7f7f", label="unknown"),
        ]
        ax.legend(handles=patches, fontsize=7, loc="upper right")

    # --- bottom section: parameter change frequency ---
    # Combine all param freq tables
    combined_freq = pd.concat(list(all_param_freq.values()), ignore_index=True)

    if not combined_freq.empty:
        # Pivot: rows=params, cols=agents
        pivot = combined_freq.pivot_table(
            index="param", columns="agent_id", values="times_changed", fill_value=0
        )
        pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)

        ax_bottom = fig.add_subplot(2, 1, 2)
        ax_bottom.set_title("Parameter change frequency by agent", fontsize=10)

        x = np.arange(len(pivot))
        bar_width = 0.8 / len(pivot.columns)
        for i, col in enumerate(pivot.columns):
            offset = (i - len(pivot.columns) / 2 + 0.5) * bar_width
            color = COLORS[i % len(COLORS)]
            ax_bottom.bar(
                x + offset, pivot[col], width=bar_width,
                label=col, color=color, alpha=0.85,
            )

        ax_bottom.set_xticks(x)
        ax_bottom.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right", fontsize=9)
        ax_bottom.set_ylabel("Times changed")
        ax_bottom.legend(fontsize=9)
        ax_bottom.grid(True, axis="y", alpha=0.3)
    else:
        ax_bottom = fig.add_subplot(2, 1, 2)
        ax_bottom.text(0.5, 0.5, "No parameter-level data", ha="center", va="center",
                       transform=ax_bottom.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_modification_trajectory(
    all_mod_tables: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """
    Plot accepted/rejected trajectory with val_bpb overlay per agent.
    X-axis = step index, Y-axis = val_bpb_after (when available).
    Green markers = accepted, red = rejected.
    """
    n_agents = len(all_mod_tables)
    if n_agents == 0:
        return

    fig, axes = plt.subplots(1, n_agents, figsize=(7 * n_agents, 5), squeeze=False)
    fig.suptitle("Modification trajectory (accepted / rejected) per agent",
                 fontsize=13, fontweight="bold")

    for idx, (agent_id, df) in enumerate(sorted(all_mod_tables.items())):
        ax = axes[0][idx]
        ax.set_title(agent_id, fontsize=10)
        ax.set_xlabel("Step index")
        ax.set_ylabel("val_bpb_after")

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        valid = df.dropna(subset=["val_bpb_after"])
        if valid.empty:
            ax.text(0.5, 0.5, "No val_bpb data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ax.plot(valid["step_index"], valid["val_bpb_after"],
                color="steelblue", linewidth=1.2, alpha=0.6, zorder=1)

        accepted = valid[valid["accepted"] == True]   # noqa: E712
        rejected = valid[valid["accepted"] == False]   # noqa: E712

        if not accepted.empty:
            ax.scatter(accepted["step_index"], accepted["val_bpb_after"],
                       color="#2ca02c", s=80, zorder=3, label="accepted", marker="o")
        if not rejected.empty:
            ax.scatter(rejected["step_index"], rejected["val_bpb_after"],
                       color="#d62728", s=80, zorder=3, label="rejected", marker="x")

        # Annotate git messages briefly (first word or param name)
        for _, row in valid.iterrows():
            short = row["git_message"].split()[0] if row["git_message"] else ""
            ax.annotate(short, xy=(row["step_index"], row["val_bpb_after"]),
                        fontsize=6, rotation=45, alpha=0.7)

        best_idx = valid["val_bpb_after"].idxmin()
        best_step = valid.loc[best_idx, "step_index"]
        best_bpb = valid.loc[best_idx, "val_bpb_after"]
        ax.scatter([best_step], [best_bpb], color="gold", s=200, zorder=5,
                   marker="*", label=f"Best {best_bpb:.5f}")
        ax.axhline(best_bpb, color="gold", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# 9. Report generation
# ---------------------------------------------------------------------------

def write_markdown_report(
    experiment_id: str,
    loss_summary: pd.DataFrame,
    all_mod_tables: dict[str, pd.DataFrame],
    all_param_freq: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Write a human-readable markdown summary report."""
    lines = [
        f"# Experiment Analysis Report: {experiment_id}",
        "",
        "## Loss Summary",
        "",
        loss_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Modification Summary per Agent",
        "",
    ]

    for agent_id, df in sorted(all_mod_tables.items()):
        lines.append(f"### {agent_id}")
        lines.append("")
        if df.empty:
            lines.append("_No snapshot data available._")
        else:
            show_cols = ["step_index", "git_message", "val_bpb_after", "accepted",
                         "params_changed", "param_names", "lines_added", "lines_removed"]
            show_cols = [c for c in show_cols if c in df.columns]
            lines.append(df[show_cols].to_markdown(index=False, floatfmt=".6f"))
        lines.append("")

    lines += [
        "## Parameter Change Frequency",
        "",
    ]
    combined_freq = pd.concat(list(all_param_freq.values()), ignore_index=True)
    if not combined_freq.empty:
        pivot = combined_freq.pivot_table(
            index="param", columns="agent_id", values="times_changed", fill_value=0
        ).reset_index()
        lines.append(pivot.to_markdown(index=False))
    else:
        lines.append("_No parameter-level data._")

    lines += [
        "",
        "## Notes",
        "",
        "- `val_bpb` = validation bits-per-byte (lower is better).",
        "- Modification diffs are computed between consecutive snapshots.",
        "- Parameter detection uses `^[A-Z][A-Z0-9_]+ =` regex in train.py.",
        "- Steps where `accepted = None` are pending / incomplete runs.",
    ]

    output_path.write_text("\n".join(lines))
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# 10. Main orchestration
# ---------------------------------------------------------------------------

def analyze_experiment(experiment_dir: Path, output_dir: Path) -> None:
    """Full analysis pipeline for a single experiment directory."""
    experiment_id = experiment_dir.name
    print(f"\n=== Analysing experiment: {experiment_id} ===")

    output_dir.mkdir(parents=True, exist_ok=True)

    agents = discover_agents(experiment_dir)
    if not agents:
        print(f"  [warn] No agents found in {experiment_dir}")
        return
    print(f"  Found {len(agents)} agents: {[a.name for a in agents]}")

    # --- collect data ---
    trajectories: dict[str, pd.DataFrame] = {}
    agent_metadata: dict[str, dict] = {}
    all_mod_tables: dict[str, pd.DataFrame] = {}
    all_param_freq: dict[str, pd.DataFrame] = {}
    all_impact_tables: dict[str, pd.DataFrame] = {}

    for agent_dir in agents:
        agent_id = agent_dir.name
        traj = load_trajectory(agent_dir)
        meta = load_agent_metadata(agent_dir)
        snapshots = load_snapshots(agent_dir)
        baseline = load_baseline(agent_dir)

        trajectories[agent_id] = traj
        agent_metadata[agent_id] = meta

        mod_table = build_modification_table(agent_id, snapshots, baseline)
        param_freq = build_param_frequency_table(agent_id, snapshots, baseline)
        impact_table = build_param_impact_table(agent_id, snapshots, baseline)
        all_mod_tables[agent_id] = mod_table
        all_param_freq[agent_id] = param_freq
        all_impact_tables[agent_id] = impact_table

        print(f"  {agent_id}: {len(traj)} trajectory steps, "
              f"{len(snapshots)} snapshots, "
              f"{len(param_freq)} params touched")

    # --- plots ---
    print("  Generating plots …")
    plot_loss_curves(
        trajectories,
        output_dir / "loss_curves.png",
    )
    plot_modification_summary(
        all_mod_tables,
        all_param_freq,
        output_dir / "train_py_modifications.png",
    )
    plot_modification_trajectory(
        all_mod_tables,
        output_dir / "modification_trajectory.png",
    )
    plot_param_impact_heatmap(
        all_impact_tables,
        output_dir / "param_impact_heatmap.png",
    )

    # --- CSV outputs ---
    print("  Writing CSV summaries …")
    loss_summary = build_loss_summary(trajectories, agent_metadata)
    loss_summary.to_csv(output_dir / "best_loss_summary.csv", index=False)
    print(f"  Saved: {output_dir / 'best_loss_summary.csv'}")

    combined_mod = pd.concat(list(all_mod_tables.values()), ignore_index=True)
    combined_mod.to_csv(output_dir / "train_py_modification_summary.csv", index=False)
    print(f"  Saved: {output_dir / 'train_py_modification_summary.csv'}")

    combined_freq = pd.concat(list(all_param_freq.values()), ignore_index=True)
    combined_freq.to_csv(output_dir / "param_change_frequency.csv", index=False)
    print(f"  Saved: {output_dir / 'param_change_frequency.csv'}")

    combined_impact = pd.concat(list(all_impact_tables.values()), ignore_index=True)
    combined_impact.to_csv(output_dir / "param_impact_by_agent.csv", index=False)
    print(f"  Saved: {output_dir / 'param_impact_by_agent.csv'}")

    # --- Markdown report ---
    print("  Writing markdown report …")
    write_markdown_report(
        experiment_id,
        loss_summary,
        all_mod_tables,
        all_param_freq,
        output_dir / "report.md",
    )

    print(f"  Done. Outputs in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse agent parallelisation experiment runs.")
    parser.add_argument(
        "--runs-dir", type=Path,
        default=Path(__file__).parent.parent / "runs",
        help="Directory containing experiment folders (default: runs/)",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Specific experiment ID to analyse (default: latest).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Analyse all experiments found in --runs-dir.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: analysis/<experiment_id>/).",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    if not runs_dir.exists():
        print(f"Error: runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    experiments = discover_experiments(runs_dir)
    if not experiments:
        print(f"No experiments found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    if args.all:
        target_experiments = experiments
    elif args.experiment:
        # Find by partial or full name
        matches = [e for e in experiments if args.experiment in e.name]
        if not matches:
            print(f"No experiment matching '{args.experiment}' found.", file=sys.stderr)
            sys.exit(1)
        target_experiments = matches
    else:
        # Default: latest experiment
        target_experiments = [experiments[-1]]

    for exp_dir in target_experiments:
        out_dir = args.output_dir if args.output_dir else (
            Path(__file__).parent.parent / "analysis" / exp_dir.name
        )
        analyze_experiment(exp_dir, out_dir)


if __name__ == "__main__":
    main()
