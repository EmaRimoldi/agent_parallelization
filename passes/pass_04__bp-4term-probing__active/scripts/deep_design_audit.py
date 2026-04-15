#!/usr/bin/env python3
"""Deep design audit of BP 2×2 experiment.

Analyzes 5 potential confounds in the experimental design:
1. CPU contention (parallel cells share hardware)
2. Agent homogeneity (identical LLM → identical strategies)
3. Memory anchoring vs routing (memory hurts instead of helping)
4. Task ceiling (CIFAR-10/585 steps too constrained)
5. Budget sufficiency (run-9 wall vs available iterations)

Outputs figures to results/figures/pass_03_design_audit/
Prints comprehensive statistical results to stdout.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "runs"
FIG_DIR = REPO_ROOT / "results" / "figures" / "pass_03_design_audit"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_VAL_BPB = 0.925845
CELLS = ["d00", "d10", "d01", "d11"]
CELL_LABELS = {
    "d00": "d00\n(single, no mem)",
    "d10": "d10\n(single, memory)",
    "d01": "d01\n(parallel, no sharing)",
    "d11": "d11\n(parallel, shared)",
}
CELL_COLORS = {
    "d00": "#2196F3",
    "d10": "#FF9800",
    "d01": "#4CAF50",
    "d11": "#E91E63",
}
MAX_REPS = {"d00": 5, "d10": 5, "d01": 3, "d11": 3}

# ── Style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ── Data Loading ───────────────────────────────────────────────────────

def load_all_runs() -> list[dict]:
    """Load all training_runs.jsonl entries across all cells and reps."""
    all_runs = []
    for cell in CELLS:
        for rep in range(1, MAX_REPS[cell] + 1):
            base = RUNS_DIR / f"experiment_calibration_{cell}_rep{rep}"
            if not base.exists():
                continue
            for jf in sorted(base.rglob("training_runs.jsonl")):
                agent_id = "agent_0"
                parts = str(jf).split("/")
                for i, p in enumerate(parts):
                    if p.startswith("agent_"):
                        agent_id = p
                        break
                for line in jf.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        d["cell"] = cell
                        d["rep"] = rep
                        # Prefer JSONL agent_id field; fall back to dir path
                        d["agent"] = d.get("agent_id", agent_id)
                        d["rep_key"] = f"{cell}_rep{rep}"
                        all_runs.append(d)
                    except json.JSONDecodeError:
                        pass
    return all_runs


def load_trace_data() -> list[dict]:
    """Load all reasoning trace entries."""
    all_traces = []
    for cell in CELLS:
        for rep in range(1, MAX_REPS[cell] + 1):
            base = RUNS_DIR / f"experiment_calibration_{cell}_rep{rep}"
            if not base.exists():
                continue
            for tf in sorted(base.rglob("trace.jsonl")):
                agent_id = "agent_0"
                parts = str(tf).split("/")
                for p in parts:
                    if p.startswith("agent_"):
                        agent_id = p
                        break
                for line in tf.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        d["cell"] = cell
                        d["rep"] = rep
                        d["agent"] = agent_id
                        all_traces.append(d)
                    except json.JSONDecodeError:
                        pass
    return all_traces


# ── Descriptive Statistics ─────────────────────────────────────────────

def print_descriptive(runs: list[dict]) -> dict:
    """Print and return per-cell descriptive statistics."""
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)

    cell_stats = {}
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        if not cell_runs:
            continue

        # Exclude baseline runs (first run per agent per rep)
        non_baseline = [r for r in cell_runs if not r.get("baseline_candidate", False)]
        all_vbpb = [r["val_bpb"] for r in cell_runs if r.get("val_bpb") is not None]
        nb_vbpb = [r["val_bpb"] for r in non_baseline if r.get("val_bpb") is not None]
        wall = [r["wall_seconds"] for r in cell_runs if r.get("wall_seconds")]
        train = [r["training_seconds"] for r in cell_runs if r.get("training_seconds")]

        # Best per rep
        reps_seen = sorted(set(r["rep"] for r in cell_runs))
        best_per_rep = []
        for rep in reps_seen:
            rep_runs = [r for r in cell_runs if r["rep"] == rep]
            rep_vbpb = [r["val_bpb"] for r in rep_runs if r.get("val_bpb") is not None]
            if rep_vbpb:
                best_per_rep.append(min(rep_vbpb))

        # Success rate (non-baseline runs that beat baseline)
        successes = [v for v in nb_vbpb if v < BASELINE_VAL_BPB]
        success_rate = len(successes) / len(nb_vbpb) if nb_vbpb else 0

        cell_stats[cell] = {
            "n_runs": len(cell_runs),
            "n_reps": len(reps_seen),
            "n_non_baseline": len(nb_vbpb),
            "mean_vbpb": np.mean(all_vbpb) if all_vbpb else None,
            "std_vbpb": np.std(all_vbpb, ddof=1) if len(all_vbpb) > 1 else None,
            "best_vbpb": min(all_vbpb) if all_vbpb else None,
            "best_per_rep": best_per_rep,
            "mean_best": np.mean(best_per_rep) if best_per_rep else None,
            "std_best": np.std(best_per_rep, ddof=1) if len(best_per_rep) > 1 else None,
            "success_rate": success_rate,
            "n_successes": len(successes),
            "mean_wall": np.mean(wall) if wall else None,
            "std_wall": np.std(wall, ddof=1) if len(wall) > 1 else None,
            "median_wall": np.median(wall) if wall else None,
            "mean_train": np.mean(train) if train else None,
            "std_train": np.std(train, ddof=1) if len(train) > 1 else None,
            "median_train": np.median(train) if train else None,
            "wall_values": wall,
            "train_values": train,
            "all_vbpb": all_vbpb,
            "nb_vbpb": nb_vbpb,
        }

        print(f"\n--- {cell} ({len(reps_seen)} reps, {len(cell_runs)} runs) ---")
        print(f"  val_bpb: mean={np.mean(all_vbpb):.6f} ± {np.std(all_vbpb, ddof=1):.6f}")
        print(f"  best:    {min(all_vbpb):.6f}")
        print(f"  best/rep: {[f'{v:.6f}' for v in best_per_rep]}")
        print(f"  mean best/rep: {np.mean(best_per_rep):.6f} ± {np.std(best_per_rep, ddof=1):.6f}" if len(best_per_rep) > 1 else "")
        print(f"  success rate: {len(successes)}/{len(nb_vbpb)} = {success_rate:.1%}")
        print(f"  wall_sec:  mean={np.mean(wall):.1f} ± {np.std(wall, ddof=1):.1f}, median={np.median(wall):.1f}" if wall else "  wall_sec: N/A")
        print(f"  train_sec: mean={np.mean(train):.1f} ± {np.std(train, ddof=1):.1f}, median={np.median(train):.1f}" if train else "  train_sec: N/A")

    return cell_stats


# ── Analysis 1: CPU Contention ─────────────────────────────────────────

def analyze_cpu_contention(runs: list[dict], cell_stats: dict) -> dict:
    """Test whether parallel cells have inflated training times."""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: CPU CONTENTION CONFOUND")
    print("=" * 80)
    print("H0: Training time per run is the same across cells")
    print("H1: Parallel cells (d01, d11) have higher training time due to CPU sharing")

    results = {}

    # Kruskal-Wallis across all 4 cells
    groups = []
    labels = []
    for cell in CELLS:
        vals = cell_stats.get(cell, {}).get("train_values", [])
        if vals:
            groups.append(vals)
            labels.append(cell)

    if len(groups) >= 2:
        H_stat, kw_p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis test: H={H_stat:.3f}, p={kw_p:.6f}")
        results["kruskal_wallis"] = {"H": H_stat, "p": kw_p}

    # Pairwise Mann-Whitney U tests
    print("\nPairwise Mann-Whitney U (training_seconds):")
    pairs = [("d00", "d01"), ("d00", "d11"), ("d10", "d01"), ("d10", "d11"),
             ("d00", "d10"), ("d01", "d11")]
    for c1, c2 in pairs:
        v1 = cell_stats.get(c1, {}).get("train_values", [])
        v2 = cell_stats.get(c2, {}).get("train_values", [])
        if v1 and v2:
            U, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            # Effect size r = Z / sqrt(N)
            z = stats.norm.ppf(p / 2) if p < 1 else 0
            r = abs(z) / np.sqrt(len(v1) + len(v2))
            print(f"  {c1} vs {c2}: U={U:.0f}, p={p:.6f}, r={r:.3f}, "
                  f"medians={np.median(v1):.1f} vs {np.median(v2):.1f}")
            results[f"{c1}_vs_{c2}"] = {"U": U, "p": p, "r": r}

    # Overhead ratio
    single_train = []
    parallel_train = []
    for cell in ["d00", "d10"]:
        single_train.extend(cell_stats.get(cell, {}).get("train_values", []))
    for cell in ["d01", "d11"]:
        parallel_train.extend(cell_stats.get(cell, {}).get("train_values", []))

    if single_train and parallel_train:
        overhead = np.median(parallel_train) / np.median(single_train)
        print(f"\nOverhead ratio (median parallel / median single): {overhead:.2f}x")
        print(f"  Single cells median: {np.median(single_train):.1f}s")
        print(f"  Parallel cells median: {np.median(parallel_train):.1f}s")
        results["overhead_ratio"] = overhead

    return results


def figure_cpu_contention(runs: list[dict], cell_stats: dict):
    """Figure 1: CPU contention evidence."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Training time distributions
    ax = axes[0]
    data = []
    positions = []
    colors = []
    for i, cell in enumerate(CELLS):
        vals = cell_stats.get(cell, {}).get("train_values", [])
        if vals:
            data.append(vals)
            positions.append(i)
            colors.append(CELL_COLORS[cell])

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], fontsize=8)
    ax.set_ylabel("Training time (seconds)")
    ax.set_title("A: Training Time per Run")
    ax.axhline(60, color="gray", linestyle="--", alpha=0.5, label="Budget (60s)")
    ax.legend(fontsize=7)

    # Panel B: Wall time distributions
    ax = axes[1]
    data = []
    for cell in CELLS:
        vals = cell_stats.get(cell, {}).get("wall_values", [])
        data.append(vals if vals else [])

    bp = ax.boxplot(data, positions=range(len(CELLS)), widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], [CELL_COLORS[c] for c in CELLS]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], fontsize=8)
    ax.set_ylabel("Wall-clock time (seconds)")
    ax.set_title("B: Wall-Clock Time per Run")

    # Panel C: Per-rep training time mean ± std
    ax = axes[2]
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))
        rep_means = []
        rep_stds = []
        for rep in reps:
            rep_train = [r["training_seconds"] for r in cell_runs
                         if r["rep"] == rep and r.get("training_seconds")]
            if rep_train:
                rep_means.append(np.mean(rep_train))
                rep_stds.append(np.std(rep_train, ddof=1) if len(rep_train) > 1 else 0)
        if rep_means:
            x = range(1, len(rep_means) + 1)
            ax.errorbar(x, rep_means, yerr=rep_stds, marker="o", capsize=3,
                        label=cell, color=CELL_COLORS[cell], linewidth=1.5)

    ax.set_xlabel("Replicate")
    ax.set_ylabel("Mean training time (s)")
    ax.set_title("C: Training Time by Replicate")
    ax.legend(fontsize=7)
    ax.axhline(60, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure-01-cpu-contention.png")
    fig.savefig(FIG_DIR / "figure-01-cpu-contention.pdf")
    plt.close(fig)
    print(f"\n  Saved: figure-01-cpu-contention.{{png,pdf}}")


# ── Analysis 2: Agent Homogeneity ──────────────────────────────────────

def analyze_agent_homogeneity(runs: list[dict], traces: list[dict]) -> dict:
    """Test whether parallel agents explore different strategies."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: AGENT HOMOGENEITY")
    print("=" * 80)
    print("H0: Parallel agents explore different strategy spaces")
    print("H1: Same-model agents converge on identical strategies")

    results = {}

    # Strategy categories per agent per rep (parallel cells only)
    for cell in ["d01", "d11"]:
        print(f"\n--- {cell} ---")
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))

        for rep in reps:
            rep_runs = [r for r in cell_runs if r["rep"] == rep]
            a0_cats = [r.get("strategy_category", "unknown")
                       for r in rep_runs if r["agent"] == "agent_0"
                       and not r.get("baseline_candidate", False)]
            a1_cats = [r.get("strategy_category", "unknown")
                       for r in rep_runs if r["agent"] == "agent_1"
                       and not r.get("baseline_candidate", False)]

            a0_set = set(a0_cats)
            a1_set = set(a1_cats)
            overlap = a0_set & a1_set
            union = a0_set | a1_set
            jaccard = len(overlap) / len(union) if union else 0

            print(f"  rep{rep}: agent_0 cats={sorted(a0_set)}, agent_1 cats={sorted(a1_set)}")
            print(f"         Jaccard similarity: {jaccard:.2f} (overlap={len(overlap)}/{len(union)})")

            results[f"{cell}_rep{rep}_jaccard"] = jaccard

            # Hypothesis-level comparison (from descriptions)
            a0_hyp = [r.get("hypothesis", r.get("git_message", ""))
                      for r in rep_runs if r["agent"] == "agent_0"
                      and not r.get("baseline_candidate", False)]
            a1_hyp = [r.get("hypothesis", r.get("git_message", ""))
                      for r in rep_runs if r["agent"] == "agent_1"
                      and not r.get("baseline_candidate", False)]

            # Extract keywords
            def extract_keywords(hyps):
                kw = set()
                for h in hyps:
                    h_lower = h.lower()
                    for term in ["dropout", "learning rate", "lr", "channels",
                                 "batch", "depth", "weight decay", "augment",
                                 "embed", "optimizer", "adam", "sgd", "momentum",
                                 "layer", "norm", "activation", "gelu", "relu",
                                 "steps", "warmup", "scheduler", "cosine"]:
                        if term in h_lower:
                            kw.add(term)
                return kw

            a0_kw = extract_keywords(a0_hyp)
            a1_kw = extract_keywords(a1_hyp)
            kw_overlap = a0_kw & a1_kw
            kw_union = a0_kw | a1_kw
            kw_jaccard = len(kw_overlap) / len(kw_union) if kw_union else 0
            print(f"         Keyword Jaccard: {kw_jaccard:.2f} "
                  f"(shared={sorted(kw_overlap)}, "
                  f"a0-only={sorted(a0_kw - a1_kw)}, "
                  f"a1-only={sorted(a1_kw - a0_kw)})")

            results[f"{cell}_rep{rep}_kw_jaccard"] = kw_jaccard

    # Compare strategy diversity across cells
    print("\n--- Strategy diversity per cell ---")
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell
                     and not r.get("baseline_candidate", False)]
        cats = [r.get("strategy_category", "unknown") for r in cell_runs]
        cat_counts = Counter(cats)
        n_unique = len(cat_counts)
        entropy = -sum((c / len(cats)) * np.log2(c / len(cats))
                       for c in cat_counts.values()) if cats else 0
        print(f"  {cell}: {n_unique} categories, entropy={entropy:.3f}, "
              f"dist={dict(cat_counts)}")
        results[f"{cell}_n_categories"] = n_unique
        results[f"{cell}_entropy"] = entropy

    return results


def figure_agent_homogeneity(runs: list[dict]):
    """Figure 2: Agent homogeneity evidence."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Strategy category distribution per cell
    ax = axes[0]
    all_cats = sorted(set(r.get("strategy_category", "unknown")
                          for r in runs if not r.get("baseline_candidate", False)))
    x = np.arange(len(all_cats))
    width = 0.2
    for i, cell in enumerate(CELLS):
        cell_runs = [r for r in runs if r["cell"] == cell
                     and not r.get("baseline_candidate", False)]
        cats = [r.get("strategy_category", "unknown") for r in cell_runs]
        counts = Counter(cats)
        total = sum(counts.values())
        fracs = [counts.get(c, 0) / total if total else 0 for c in all_cats]
        ax.bar(x + i * width, fracs, width, label=cell,
               color=CELL_COLORS[cell], alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([c[:12] for c in all_cats], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Fraction of runs")
    ax.set_title("A: Strategy Category Distribution")
    ax.legend(fontsize=7)

    # Panel B: Per-agent strategy overlap in d01
    ax = axes[1]
    d01_runs = [r for r in runs if r["cell"] == "d01"
                and not r.get("baseline_candidate", False)]
    reps = sorted(set(r["rep"] for r in d01_runs))
    for rep in reps:
        rep_runs = [r for r in d01_runs if r["rep"] == rep]
        a0_cats = Counter(r.get("strategy_category", "unknown")
                          for r in rep_runs if r["agent"] == "agent_0")
        a1_cats = Counter(r.get("strategy_category", "unknown")
                          for r in rep_runs if r["agent"] == "agent_1")

        cats = sorted(set(list(a0_cats.keys()) + list(a1_cats.keys())))
        y = np.arange(len(cats))
        a0_vals = [a0_cats.get(c, 0) for c in cats]
        a1_vals = [-a1_cats.get(c, 0) for c in cats]

        ax.barh(y + (rep - 1) * 0.25, a0_vals, 0.2,
                color="#2196F3", alpha=0.6, label=f"rep{rep} a0" if rep == 1 else "")
        ax.barh(y + (rep - 1) * 0.25, a1_vals, 0.2,
                color="#FF9800", alpha=0.6, label=f"rep{rep} a1" if rep == 1 else "")

    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels([c[:15] for c in cats], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("← agent_1 | agent_0 →")
    ax.set_title("B: d01 Per-Agent Strategies")
    ax.legend(fontsize=7)

    # Panel C: Per-agent overlap in d11
    ax = axes[2]
    d11_runs = [r for r in runs if r["cell"] == "d11"
                and not r.get("baseline_candidate", False)]
    reps = sorted(set(r["rep"] for r in d11_runs))
    if reps:
        for rep in reps:
            rep_runs = [r for r in d11_runs if r["rep"] == rep]
            a0_cats = Counter(r.get("strategy_category", "unknown")
                              for r in rep_runs if r["agent"] == "agent_0")
            a1_cats = Counter(r.get("strategy_category", "unknown")
                              for r in rep_runs if r["agent"] == "agent_1")

            cats = sorted(set(list(a0_cats.keys()) + list(a1_cats.keys())))
            y = np.arange(len(cats))
            a0_vals = [a0_cats.get(c, 0) for c in cats]
            a1_vals = [-a1_cats.get(c, 0) for c in cats]

            ax.barh(y + (rep - 1) * 0.25, a0_vals, 0.2,
                    color="#4CAF50", alpha=0.6, label=f"rep{rep} a0" if rep == 1 else "")
            ax.barh(y + (rep - 1) * 0.25, a1_vals, 0.2,
                    color="#E91E63", alpha=0.6, label=f"rep{rep} a1" if rep == 1 else "")

        ax.set_yticks(np.arange(len(cats)))
        ax.set_yticklabels([c[:15] for c in cats], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("← agent_1 | agent_0 →")
    ax.set_title("C: d11 Per-Agent Strategies")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure-02-agent-homogeneity.png")
    fig.savefig(FIG_DIR / "figure-02-agent-homogeneity.pdf")
    plt.close(fig)
    print(f"\n  Saved: figure-02-agent-homogeneity.{{png,pdf}}")


# ── Analysis 3: Memory Anchoring ──────────────────────────────────────

def analyze_memory_anchoring(runs: list[dict]) -> dict:
    """Test whether memory enables routing or causes anchoring."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: MEMORY ANCHORING vs ROUTING")
    print("=" * 80)
    print("H0: Memory improves strategy selection (routing)")
    print("H1: Memory anchors agents to suboptimal strategies (sunk-cost bias)")

    results = {}

    # Compare strategy diversity within session (early vs late runs)
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell
                     and not r.get("baseline_candidate", False)]
        if not cell_runs:
            continue

        # Split into early (first half) and late (second half) per rep
        reps = sorted(set(r["rep"] for r in cell_runs))
        early_cats = []
        late_cats = []
        for rep in reps:
            rep_runs = sorted(
                [r for r in cell_runs if r["rep"] == rep],
                key=lambda r: r.get("run_index", 0)
            )
            mid = len(rep_runs) // 2
            early_cats.extend(r.get("strategy_category", "unknown")
                              for r in rep_runs[:mid])
            late_cats.extend(r.get("strategy_category", "unknown")
                             for r in rep_runs[mid:])

        early_counts = Counter(early_cats)
        late_counts = Counter(late_cats)
        all_cats_set = sorted(set(list(early_counts.keys()) + list(late_counts.keys())))

        # Chi-squared test for strategy shift
        observed_early = [early_counts.get(c, 0) for c in all_cats_set]
        observed_late = [late_counts.get(c, 0) for c in all_cats_set]

        if sum(observed_early) > 0 and sum(observed_late) > 0:
            # Only test if at least 2 non-zero categories
            contingency = np.array([observed_early, observed_late])
            # Remove zero columns
            mask = contingency.sum(axis=0) > 0
            contingency = contingency[:, mask]
            if contingency.shape[1] >= 2:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                print(f"\n  {cell}: Early vs Late strategy shift: chi2={chi2:.3f}, "
                      f"p={p:.4f}, dof={dof}")
                results[f"{cell}_chi2_shift"] = {"chi2": chi2, "p": p}
            else:
                print(f"\n  {cell}: Too few categories for chi2 test")

    # Memory depth vs performance correlation (d10 and d11)
    print("\n--- Memory depth vs val_bpb (d10, d11) ---")
    for cell in ["d10", "d11"]:
        cell_runs = [r for r in runs if r["cell"] == cell
                     and not r.get("baseline_candidate", False)]
        mem_entries = [r.get("memory_context_entries", 0) + r.get("shared_memory_context_entries", 0)
                       for r in cell_runs]
        vbpb = [r["val_bpb"] for r in cell_runs if r.get("val_bpb") is not None]
        mem_entries = mem_entries[:len(vbpb)]

        if len(vbpb) > 2:
            r_val, p_val = stats.spearmanr(mem_entries, vbpb)
            print(f"  {cell}: Spearman r={r_val:.4f}, p={p_val:.4f}, n={len(vbpb)}")
            results[f"{cell}_mem_corr"] = {"r": r_val, "p": p_val}

    # Consecutive same-strategy streaks (anchoring signal)
    print("\n--- Consecutive same-strategy streaks ---")
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))
        all_streaks = []
        for rep in reps:
            rep_runs = sorted(
                [r for r in cell_runs if r["rep"] == rep],
                key=lambda r: r.get("run_index", 0)
            )
            # For parallel cells, check per-agent
            agents = sorted(set(r["agent"] for r in rep_runs))
            for agent in agents:
                agent_runs = [r for r in rep_runs if r["agent"] == agent]
                cats = [r.get("strategy_category", "unknown") for r in agent_runs]
                # Count max consecutive same category
                if cats:
                    streak = 1
                    max_streak = 1
                    for i in range(1, len(cats)):
                        if cats[i] == cats[i - 1]:
                            streak += 1
                            max_streak = max(max_streak, streak)
                        else:
                            streak = 1
                    all_streaks.append(max_streak)

        if all_streaks:
            print(f"  {cell}: mean max streak={np.mean(all_streaks):.1f}, "
                  f"max={max(all_streaks)}, streaks={all_streaks}")
            results[f"{cell}_mean_streak"] = np.mean(all_streaks)

    return results


def figure_memory_anchoring(runs: list[dict]):
    """Figure 3: Memory anchoring evidence."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Strategy switch rate over run index
    ax = axes[0]
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))
        switch_rates = defaultdict(list)

        for rep in reps:
            rep_runs = sorted(
                [r for r in cell_runs if r["rep"] == rep],
                key=lambda r: r.get("run_index", 0)
            )
            agents = sorted(set(r["agent"] for r in rep_runs))
            for agent in agents:
                agent_runs = [r for r in rep_runs if r["agent"] == agent]
                cats = [r.get("strategy_category", "unknown") for r in agent_runs]
                for i in range(1, len(cats)):
                    switched = 1 if cats[i] != cats[i - 1] else 0
                    switch_rates[i + 1].append(switched)

        if switch_rates:
            idxs = sorted(switch_rates.keys())
            means = [np.mean(switch_rates[i]) for i in idxs]
            ax.plot(idxs, means, marker="o", markersize=3,
                    label=cell, color=CELL_COLORS[cell], linewidth=1.5)

    ax.set_xlabel("Run index")
    ax.set_ylabel("Strategy switch probability")
    ax.set_title("A: Strategy Switching Over Time")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)

    # Panel B: Memory depth vs val_bpb scatter
    ax = axes[1]
    for cell in ["d10", "d11"]:
        cell_runs = [r for r in runs if r["cell"] == cell
                     and not r.get("baseline_candidate", False)]
        mem = [r.get("memory_context_entries", 0) + r.get("shared_memory_context_entries", 0)
               for r in cell_runs]
        vbpb = [r.get("val_bpb", None) for r in cell_runs]
        valid = [(m, v) for m, v in zip(mem, vbpb) if v is not None]
        if valid:
            ms, vs = zip(*valid)
            ax.scatter(ms, vs, alpha=0.5, s=20, label=cell,
                       color=CELL_COLORS[cell], edgecolors="none")

    ax.axhline(BASELINE_VAL_BPB, color="gray", linestyle="--", alpha=0.5,
               label="Baseline")
    ax.set_xlabel("Memory context entries")
    ax.set_ylabel("val_bpb")
    ax.set_title("B: Memory Depth vs Performance")
    ax.legend(fontsize=7)

    # Panel C: Cumulative unique strategies over run count
    ax = axes[2]
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))
        all_curves = []

        for rep in reps:
            rep_runs = sorted(
                [r for r in cell_runs if r["rep"] == rep],
                key=lambda r: r.get("run_index", 0)
            )
            seen = set()
            curve = []
            for r in rep_runs:
                cat = r.get("strategy_category", "unknown")
                seen.add(cat)
                curve.append(len(seen))
            all_curves.append(curve)

        # Average curves (pad to max length)
        if all_curves:
            max_len = max(len(c) for c in all_curves)
            padded = [c + [c[-1]] * (max_len - len(c)) for c in all_curves]
            mean_curve = np.mean(padded, axis=0)
            ax.plot(range(1, max_len + 1), mean_curve,
                    label=cell, color=CELL_COLORS[cell], linewidth=1.5)

    ax.set_xlabel("Cumulative runs")
    ax.set_ylabel("Unique strategy categories seen")
    ax.set_title("C: Exploration Breadth Over Time")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure-03-memory-anchoring.png")
    fig.savefig(FIG_DIR / "figure-03-memory-anchoring.pdf")
    plt.close(fig)
    print(f"\n  Saved: figure-03-memory-anchoring.{{png,pdf}}")


# ── Analysis 4: Task Ceiling ──────────────────────────────────────────

def analyze_task_ceiling(runs: list[dict], cell_stats: dict) -> dict:
    """Test whether the task is too constrained for meaningful optimization."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: TASK CEILING EFFECT")
    print("=" * 80)
    print("H0: The search space has sufficient room for diverse improvements")
    print("H1: CIFAR-10/585 steps is near-optimal, leaving no room for agents")

    results = {}

    # Distribution of val_bpb relative to baseline
    all_vbpb = [r["val_bpb"] for r in runs if r.get("val_bpb") is not None]
    improvements = [v for v in all_vbpb if v < BASELINE_VAL_BPB]
    degradations = [v for v in all_vbpb if v > BASELINE_VAL_BPB]

    print(f"\nOverall distribution (N={len(all_vbpb)}):")
    print(f"  Improvements (< baseline): {len(improvements)} ({len(improvements)/len(all_vbpb):.1%})")
    print(f"  At baseline: {len(all_vbpb) - len(improvements) - len(degradations)}")
    print(f"  Degradations (> baseline): {len(degradations)} ({len(degradations)/len(all_vbpb):.1%})")

    if improvements:
        print(f"  Improvement range: [{min(improvements):.6f}, {max(improvements):.6f}]")
        print(f"  Max improvement delta: {BASELINE_VAL_BPB - min(improvements):.6f}")
        print(f"  Mean improvement delta: {BASELINE_VAL_BPB - np.mean(improvements):.6f}")

    results["n_improvements"] = len(improvements)
    results["n_degradations"] = len(degradations)
    results["improvement_rate"] = len(improvements) / len(all_vbpb) if all_vbpb else 0

    # Successful strategy analysis
    print("\n--- What works? Strategy categories of successful runs ---")
    success_cats = Counter()
    fail_cats = Counter()
    for r in runs:
        if r.get("baseline_candidate", False):
            continue
        cat = r.get("strategy_category", "unknown")
        if r.get("val_bpb") is not None and r["val_bpb"] < BASELINE_VAL_BPB:
            success_cats[cat] += 1
        elif r.get("val_bpb") is not None:
            fail_cats[cat] += 1

    all_cats_set = sorted(set(list(success_cats.keys()) + list(fail_cats.keys())))
    print(f"  {'Category':<20} {'Success':>8} {'Fail':>8} {'Win Rate':>10}")
    print(f"  {'-'*48}")
    for cat in all_cats_set:
        s = success_cats.get(cat, 0)
        f = fail_cats.get(cat, 0)
        wr = s / (s + f) if (s + f) > 0 else 0
        print(f"  {cat:<20} {s:>8} {f:>8} {wr:>9.1%}")
        results[f"winrate_{cat}"] = wr

    # Improvement magnitude by cell
    print("\n--- Improvement magnitude by cell ---")
    for cell in CELLS:
        cell_improvements = [r["val_bpb"] for r in runs
                             if r["cell"] == cell
                             and r.get("val_bpb") is not None
                             and r["val_bpb"] < BASELINE_VAL_BPB
                             and not r.get("baseline_candidate", False)]
        if cell_improvements:
            deltas = [BASELINE_VAL_BPB - v for v in cell_improvements]
            print(f"  {cell}: n={len(cell_improvements)}, "
                  f"mean_delta={np.mean(deltas):.6f}, "
                  f"max_delta={max(deltas):.6f}")
        else:
            print(f"  {cell}: NO improvements")

    return results


def figure_task_ceiling(runs: list[dict]):
    """Figure 4: Task ceiling evidence."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: val_bpb distribution with baseline reference
    ax = axes[0]
    for cell in CELLS:
        vbpb = [r["val_bpb"] for r in runs
                if r["cell"] == cell and r.get("val_bpb") is not None
                and not r.get("baseline_candidate", False)]
        if vbpb:
            ax.hist(vbpb, bins=20, alpha=0.5, label=cell,
                    color=CELL_COLORS[cell], density=True)

    ax.axvline(BASELINE_VAL_BPB, color="red", linestyle="--", linewidth=2,
               label=f"Baseline ({BASELINE_VAL_BPB:.3f})")
    ax.set_xlabel("val_bpb")
    ax.set_ylabel("Density")
    ax.set_title("A: val_bpb Distribution (lower is better)")
    ax.legend(fontsize=7)

    # Panel B: Improvement fraction by cell
    ax = axes[1]
    cells_data = []
    for cell in CELLS:
        nb = [r for r in runs if r["cell"] == cell
              and not r.get("baseline_candidate", False)
              and r.get("val_bpb") is not None]
        imp = [r for r in nb if r["val_bpb"] < BASELINE_VAL_BPB]
        cells_data.append((cell, len(imp), len(nb)))

    x = range(len(cells_data))
    imp_fracs = [d[1] / d[2] if d[2] > 0 else 0 for d in cells_data]
    bars = ax.bar(x, imp_fracs, color=[CELL_COLORS[d[0]] for d in cells_data], alpha=0.8)
    for bar, d in zip(bars, cells_data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{d[1]}/{d[2]}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[d[0]] for d in cells_data], fontsize=8)
    ax.set_ylabel("Fraction of runs beating baseline")
    ax.set_title("B: Success Rate by Cell")

    # Panel C: Strategy win rates (cross-cell)
    ax = axes[2]
    success_cats = Counter()
    fail_cats = Counter()
    for r in runs:
        if r.get("baseline_candidate", False):
            continue
        cat = r.get("strategy_category", "unknown")
        if r.get("val_bpb") is not None and r["val_bpb"] < BASELINE_VAL_BPB:
            success_cats[cat] += 1
        elif r.get("val_bpb") is not None:
            fail_cats[cat] += 1

    all_cats = sorted(set(list(success_cats.keys()) + list(fail_cats.keys())))
    y = np.arange(len(all_cats))
    s_vals = [success_cats.get(c, 0) for c in all_cats]
    f_vals = [fail_cats.get(c, 0) for c in all_cats]

    ax.barh(y, s_vals, 0.4, label="Beat baseline", color="#4CAF50", alpha=0.8)
    ax.barh(y + 0.4, f_vals, 0.4, label="Worse/equal", color="#F44336", alpha=0.8)
    ax.set_yticks(y + 0.2)
    ax.set_yticklabels(all_cats, fontsize=8)
    ax.set_xlabel("Number of runs")
    ax.set_title("C: Strategy Win/Lose Counts")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure-04-task-ceiling.png")
    fig.savefig(FIG_DIR / "figure-04-task-ceiling.pdf")
    plt.close(fig)
    print(f"\n  Saved: figure-04-task-ceiling.{{png,pdf}}")


# ── Analysis 5: Budget Sufficiency ────────────────────────────────────

def analyze_budget_sufficiency(runs: list[dict]) -> dict:
    """Test whether budget is sufficient for improvements to manifest."""
    print("\n" + "=" * 80)
    print("ANALYSIS 5: BUDGET SUFFICIENCY (Run-9 Wall)")
    print("=" * 80)
    print("H0: Improvements can occur at any point in the session")
    print("H1: A minimum exploration threshold exists before improvements occur")

    results = {}

    # First improvement run index per rep
    print("\n--- First improvement run index per rep ---")
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))

        first_imp_indices = []
        for rep in reps:
            rep_runs = sorted(
                [r for r in cell_runs if r["rep"] == rep],
                key=lambda r: r.get("run_index", 0)
            )
            found = False
            for r in rep_runs:
                if (not r.get("baseline_candidate", False)
                        and r.get("val_bpb") is not None
                        and r["val_bpb"] < BASELINE_VAL_BPB):
                    first_imp_indices.append(r.get("run_index", 0))
                    found = True
                    break
            if not found:
                first_imp_indices.append(None)

        print(f"  {cell}: first improvement at run indices: {first_imp_indices}")
        valid = [i for i in first_imp_indices if i is not None]
        if valid:
            print(f"       mean={np.mean(valid):.1f}, min={min(valid)}, max={max(valid)}")
            results[f"{cell}_mean_first_imp"] = np.mean(valid)
            results[f"{cell}_reps_with_improvement"] = len(valid)
        results[f"{cell}_reps_without_improvement"] = sum(
            1 for i in first_imp_indices if i is None
        )

    # Runs available per agent per rep
    print("\n--- Runs available per agent (effective budget) ---")
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))
        runs_per_agent = []
        for rep in reps:
            rep_runs = [r for r in cell_runs if r["rep"] == rep]
            agents = sorted(set(r["agent"] for r in rep_runs))
            for agent in agents:
                n = sum(1 for r in rep_runs if r["agent"] == agent)
                runs_per_agent.append(n)

        print(f"  {cell}: runs/agent = {runs_per_agent}, "
              f"mean={np.mean(runs_per_agent):.1f}")
        results[f"{cell}_mean_runs_per_agent"] = np.mean(runs_per_agent)

    # Probability of improvement as function of session length
    print("\n--- Session length vs improvement probability ---")
    all_session_lengths = []
    all_improved = []
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))
        for rep in reps:
            rep_runs = [r for r in cell_runs if r["rep"] == rep]
            n_runs = len(rep_runs)
            best = min((r["val_bpb"] for r in rep_runs if r.get("val_bpb")),
                       default=BASELINE_VAL_BPB)
            improved = 1 if best < BASELINE_VAL_BPB else 0
            all_session_lengths.append(n_runs)
            all_improved.append(improved)

    if all_session_lengths:
        r_val, p_val = stats.pointbiserialr(all_improved, all_session_lengths)
        print(f"  Point-biserial r={r_val:.4f}, p={p_val:.4f}")
        results["session_length_corr"] = {"r": r_val, "p": p_val}

    return results


def figure_budget_sufficiency(runs: list[dict]):
    """Figure 5: Budget sufficiency evidence."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Best-so-far trajectories per rep
    ax = axes[0]
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))
        for rep in reps:
            rep_runs = sorted(
                [r for r in cell_runs if r["rep"] == rep],
                key=lambda r: r.get("run_index", 0)
            )
            vbpb = [r["val_bpb"] for r in rep_runs if r.get("val_bpb") is not None]
            if vbpb:
                best_so_far = np.minimum.accumulate(vbpb)
                ax.plot(range(1, len(best_so_far) + 1), best_so_far,
                        color=CELL_COLORS[cell], alpha=0.5, linewidth=1)

    ax.axhline(BASELINE_VAL_BPB, color="red", linestyle="--", alpha=0.7,
               label="Baseline")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Best val_bpb so far")
    ax.set_title("A: Optimization Trajectories (all reps)")
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=CELL_COLORS[c], lw=2, label=c)
                       for c in CELLS]
    legend_elements.append(Line2D([0], [0], color="red", lw=1, ls="--",
                                  label="Baseline"))
    ax.legend(handles=legend_elements, fontsize=7)

    # Panel B: First improvement distribution
    ax = axes[1]
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))
        first_imps = []
        no_imp_count = 0
        for rep in reps:
            rep_runs = sorted(
                [r for r in cell_runs if r["rep"] == rep],
                key=lambda r: r.get("run_index", 0)
            )
            found = False
            for r in rep_runs:
                if (not r.get("baseline_candidate", False)
                        and r.get("val_bpb") is not None
                        and r["val_bpb"] < BASELINE_VAL_BPB):
                    first_imps.append(r.get("run_index", 0))
                    found = True
                    break
            if not found:
                no_imp_count += 1

        if first_imps:
            ax.scatter(first_imps, [cell] * len(first_imps),
                       color=CELL_COLORS[cell], s=80, zorder=3,
                       label=f"{cell} (improved)")
        if no_imp_count > 0:
            ax.scatter([0] * no_imp_count, [cell] * no_imp_count,
                       color=CELL_COLORS[cell], s=80, marker="x",
                       zorder=3, label=f"{cell} (no improvement)")

    ax.axvline(9, color="gray", linestyle=":", alpha=0.7, label="Run-9 wall")
    ax.set_xlabel("Run index of first improvement")
    ax.set_title("B: First Improvement Timing")
    ax.legend(fontsize=6, loc="upper right")

    # Panel C: Runs per agent vs improvement
    ax = axes[2]
    for cell in CELLS:
        cell_runs = [r for r in runs if r["cell"] == cell]
        reps = sorted(set(r["rep"] for r in cell_runs))
        for rep in reps:
            rep_runs = [r for r in cell_runs if r["rep"] == rep]
            agents = sorted(set(r["agent"] for r in rep_runs))
            total_runs = len(rep_runs)
            best = min((r["val_bpb"] for r in rep_runs if r.get("val_bpb")),
                       default=BASELINE_VAL_BPB)
            improved = best < BASELINE_VAL_BPB
            marker = "o" if improved else "x"
            ax.scatter(total_runs, best, color=CELL_COLORS[cell],
                       marker=marker, s=60, alpha=0.7)

    ax.axhline(BASELINE_VAL_BPB, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Total runs in session")
    ax.set_ylabel("Best val_bpb")
    ax.set_title("C: Session Length vs Best Outcome")
    legend_elements = [Line2D([0], [0], color=CELL_COLORS[c], lw=0, marker="o",
                              markersize=6, label=c) for c in CELLS]
    legend_elements.append(Line2D([0], [0], color="gray", lw=0, marker="o",
                                  markersize=6, label="Improved"))
    legend_elements.append(Line2D([0], [0], color="gray", lw=0, marker="x",
                                  markersize=6, label="No improvement"))
    ax.legend(handles=legend_elements, fontsize=6)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure-05-budget-sufficiency.png")
    fig.savefig(FIG_DIR / "figure-05-budget-sufficiency.pdf")
    plt.close(fig)
    print(f"\n  Saved: figure-05-budget-sufficiency.{{png,pdf}}")


# ── Summary Figure ────────────────────────────────────────────────────

def figure_2x2_summary(runs: list[dict], cell_stats: dict):
    """Figure 6: 2×2 factorial summary."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Best val_bpb per rep
    ax = axes[0, 0]
    for i, cell in enumerate(CELLS):
        bests = cell_stats.get(cell, {}).get("best_per_rep", [])
        if bests:
            x = [i] * len(bests)
            ax.scatter(x, bests, color=CELL_COLORS[cell], s=60, zorder=3)
            ax.plot([i - 0.2, i + 0.2], [np.mean(bests)] * 2,
                    color=CELL_COLORS[cell], linewidth=2)

    ax.axhline(BASELINE_VAL_BPB, color="red", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], fontsize=8)
    ax.set_ylabel("Best val_bpb per rep")
    ax.set_title("A: Best-of-Rep Performance")

    # Panel B: Success rate
    ax = axes[0, 1]
    rates = []
    for cell in CELLS:
        nb = [r for r in runs if r["cell"] == cell
              and not r.get("baseline_candidate", False)
              and r.get("val_bpb") is not None]
        imp = [r for r in nb if r["val_bpb"] < BASELINE_VAL_BPB]
        rates.append(len(imp) / len(nb) if nb else 0)

    bars = ax.bar(range(len(CELLS)), rates,
                  color=[CELL_COLORS[c] for c in CELLS], alpha=0.8)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], fontsize=8)
    ax.set_ylabel("Fraction beating baseline")
    ax.set_title("B: Per-Run Success Rate")

    # Panel C: 2×2 interaction plot (mean best per rep)
    ax = axes[1, 0]
    means = {}
    for cell in CELLS:
        bests = cell_stats.get(cell, {}).get("best_per_rep", [])
        means[cell] = np.mean(bests) if bests else BASELINE_VAL_BPB

    # Memory axis: no_mem=[d00, d01], mem=[d10, d11]
    # Parallelism axis: single=[d00, d10], parallel=[d01, d11]
    ax.plot([0, 1], [means["d00"], means["d10"]], "o-",
            color="#2196F3", linewidth=2, markersize=8, label="Single agent")
    ax.plot([0, 1], [means["d01"], means["d11"]], "s-",
            color="#4CAF50", linewidth=2, markersize=8, label="Parallel agents")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No memory", "Memory"])
    ax.set_ylabel("Mean best val_bpb per rep")
    ax.set_title("C: 2×2 Interaction Plot")
    ax.legend(fontsize=8)
    ax.axhline(BASELINE_VAL_BPB, color="red", linestyle="--", alpha=0.3)

    # Panel D: Jensen gap comparison
    ax = axes[1, 1]
    jensen_gaps = []
    for cell in CELLS:
        wall = cell_stats.get(cell, {}).get("wall_values", [])
        if wall:
            wall_arr = np.array(wall)
            log_mean = np.log(np.mean(wall_arr))
            mean_log = np.mean(np.log(wall_arr))
            R = log_mean - mean_log
            jensen_gaps.append(R)
        else:
            jensen_gaps.append(0)

    bars = ax.bar(range(len(CELLS)), jensen_gaps,
                  color=[CELL_COLORS[c] for c in CELLS], alpha=0.8)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], fontsize=8)
    ax.set_ylabel("Jensen gap R_α")
    ax.set_title("D: Cost Variance (Jensen Gap)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure-06-2x2-summary.png")
    fig.savefig(FIG_DIR / "figure-06-2x2-summary.pdf")
    plt.close(fig)
    print(f"\n  Saved: figure-06-2x2-summary.{{png,pdf}}")


# ── 2×2 Factorial ANOVA ──────────────────────────────────────────────

def analyze_factorial(cell_stats: dict) -> dict:
    """Compute 2×2 factorial effects and interaction."""
    print("\n" + "=" * 80)
    print("2×2 FACTORIAL ANALYSIS")
    print("=" * 80)

    results = {}

    # Mean of best-per-rep for each cell
    means = {}
    for cell in CELLS:
        bests = cell_stats.get(cell, {}).get("best_per_rep", [])
        means[cell] = np.mean(bests) if bests else BASELINE_VAL_BPB
        print(f"  {cell} mean best/rep: {means[cell]:.6f} (n={len(bests)})")

    # Main effects (note: lower val_bpb is better, so positive = worse)
    memory_effect = ((means["d10"] - means["d00"]) + (means["d11"] - means["d01"])) / 2
    parallel_effect = ((means["d01"] - means["d00"]) + (means["d11"] - means["d10"])) / 2
    interaction = (means["d11"] - means["d01"]) - (means["d10"] - means["d00"])

    print(f"\nMain effect of MEMORY: {memory_effect:+.6f} (positive = hurts)")
    print(f"Main effect of PARALLELISM: {parallel_effect:+.6f} (positive = hurts)")
    print(f"Interaction (M×P): {interaction:+.6f}")

    results["memory_effect"] = memory_effect
    results["parallel_effect"] = parallel_effect
    results["interaction"] = interaction

    # Cohen's d for each main effect
    # Memory: pooled from d10+d11 vs d00+d01
    mem_yes = cell_stats.get("d10", {}).get("best_per_rep", []) + \
              cell_stats.get("d11", {}).get("best_per_rep", [])
    mem_no = cell_stats.get("d00", {}).get("best_per_rep", []) + \
             cell_stats.get("d01", {}).get("best_per_rep", [])
    if mem_yes and mem_no and len(mem_yes) > 1 and len(mem_no) > 1:
        pooled_std = np.sqrt(
            ((len(mem_yes) - 1) * np.var(mem_yes, ddof=1) +
             (len(mem_no) - 1) * np.var(mem_no, ddof=1)) /
            (len(mem_yes) + len(mem_no) - 2)
        )
        d_mem = (np.mean(mem_yes) - np.mean(mem_no)) / pooled_std if pooled_std > 0 else 0
        print(f"Cohen's d (memory): {d_mem:+.3f}")
        results["cohens_d_memory"] = d_mem

    par_yes = cell_stats.get("d01", {}).get("best_per_rep", []) + \
              cell_stats.get("d11", {}).get("best_per_rep", [])
    par_no = cell_stats.get("d00", {}).get("best_per_rep", []) + \
             cell_stats.get("d10", {}).get("best_per_rep", [])
    if par_yes and par_no and len(par_yes) > 1 and len(par_no) > 1:
        pooled_std = np.sqrt(
            ((len(par_yes) - 1) * np.var(par_yes, ddof=1) +
             (len(par_no) - 1) * np.var(par_no, ddof=1)) /
            (len(par_yes) + len(par_no) - 2)
        )
        d_par = (np.mean(par_yes) - np.mean(par_no)) / pooled_std if pooled_std > 0 else 0
        print(f"Cohen's d (parallelism): {d_par:+.3f}")
        results["cohens_d_parallelism"] = d_par

    # Permutation test for interaction
    all_bests = []
    labels_mem = []
    labels_par = []
    for cell in CELLS:
        bests = cell_stats.get(cell, {}).get("best_per_rep", [])
        for b in bests:
            all_bests.append(b)
            labels_mem.append(1 if cell in ["d10", "d11"] else 0)
            labels_par.append(1 if cell in ["d01", "d11"] else 0)

    if len(all_bests) >= 8:
        observed_interaction = interaction
        n_perm = 10000
        count = 0
        rng = np.random.RandomState(42)
        all_bests_arr = np.array(all_bests)
        for _ in range(n_perm):
            perm = rng.permutation(len(all_bests_arr))
            perm_bests = all_bests_arr[perm]
            # Recompute interaction
            perm_means = {}
            idx = 0
            for cell in CELLS:
                n = len(cell_stats.get(cell, {}).get("best_per_rep", []))
                perm_means[cell] = np.mean(perm_bests[idx:idx + n]) if n > 0 else BASELINE_VAL_BPB
                idx += n
            perm_int = ((perm_means["d11"] - perm_means["d01"]) -
                        (perm_means["d10"] - perm_means["d00"]))
            if abs(perm_int) >= abs(observed_interaction):
                count += 1

        perm_p = count / n_perm
        print(f"Permutation test for interaction: p={perm_p:.4f} ({n_perm} permutations)")
        results["interaction_perm_p"] = perm_p

    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("BP 2×2 EXPERIMENT — DEEP DESIGN AUDIT")
    print("=" * 80)

    # Load data
    runs = load_all_runs()
    traces = load_trace_data()
    print(f"\nLoaded {len(runs)} runs, {len(traces)} trace entries")

    # Descriptive
    cell_stats = print_descriptive(runs)

    # 5 confound analyses
    cpu_results = analyze_cpu_contention(runs, cell_stats)
    figure_cpu_contention(runs, cell_stats)

    homo_results = analyze_agent_homogeneity(runs, traces)
    figure_agent_homogeneity(runs)

    anchor_results = analyze_memory_anchoring(runs)
    figure_memory_anchoring(runs)

    ceiling_results = analyze_task_ceiling(runs, cell_stats)
    figure_task_ceiling(runs)

    budget_results = analyze_budget_sufficiency(runs)
    figure_budget_sufficiency(runs)

    # 2×2 factorial
    factorial_results = analyze_factorial(cell_stats)
    figure_2x2_summary(runs, cell_stats)

    # Save all results
    all_results = {
        "n_total_runs": len(runs),
        "cell_summary": {cell: {
            "n_runs": s["n_runs"],
            "n_reps": s["n_reps"],
            "mean_vbpb": s["mean_vbpb"],
            "best_vbpb": s["best_vbpb"],
            "mean_best": s["mean_best"],
            "success_rate": s["success_rate"],
            "mean_wall": s["mean_wall"],
            "mean_train": s["mean_train"],
        } for cell, s in cell_stats.items()},
        "cpu_contention": cpu_results,
        "agent_homogeneity": homo_results,
        "memory_anchoring": anchor_results,
        "task_ceiling": ceiling_results,
        "budget_sufficiency": budget_results,
        "factorial": factorial_results,
    }

    output_path = REPO_ROOT / "workflow" / "artifacts" / "design_audit_results.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str) + "\n")
    print(f"\n\nResults saved to: {output_path}")
    print(f"Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
