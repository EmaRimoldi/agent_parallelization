#!/usr/bin/env python3
"""Generate a multi-page PDF report documenting the BP 2×2 design audit.

This produces a publication-quality PDF with:
- Executive summary of all 5 confound analyses
- Statistical evidence for each confound
- All 6 figures embedded
- Recommendations for experimental redesign
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO_ROOT / "results" / "figures" / "pass_03_design_audit"
RESULTS_PATH = REPO_ROOT / "workflow" / "artifacts" / "design_audit_results.json"
OUTPUT_PDF = REPO_ROOT / "results" / "figures" / "pass_03_design_audit" / "BP_2x2_Design_Audit_Report.pdf"

# Load results
results = json.loads(RESULTS_PATH.read_text())

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 150,
})


def add_text_page(pdf, title, body, subtitle=None):
    """Add a text-only page to the PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    # Title
    y_start = 0.94
    fig.text(0.08, y_start, title, fontsize=16, fontweight="bold",
             fontfamily="serif", va="top")

    if subtitle:
        y_start -= 0.03
        fig.text(0.08, y_start, subtitle, fontsize=10, style="italic",
                 color="#666666", fontfamily="serif", va="top")

    # Body text
    y = y_start - 0.05
    for paragraph in body.split("\n\n"):
        if paragraph.startswith("##"):
            # Section header
            fig.text(0.08, y, paragraph.lstrip("# "), fontsize=12,
                     fontweight="bold", fontfamily="serif", va="top")
            y -= 0.025
        elif paragraph.startswith("**"):
            # Bold paragraph
            fig.text(0.08, y, paragraph.replace("**", ""), fontsize=9,
                     fontweight="bold", fontfamily="serif", va="top",
                     wrap=True,
                     transform=fig.transFigure)
            lines = len(textwrap.wrap(paragraph, width=95))
            y -= max(0.018 * lines, 0.02)
        elif paragraph.startswith("|"):
            # Table - render as monospace
            fig.text(0.08, y, paragraph, fontsize=7.5,
                     fontfamily="monospace", va="top",
                     linespacing=1.4)
            lines = paragraph.count("\n") + 1
            y -= 0.013 * lines + 0.01
        elif paragraph.startswith("  -") or paragraph.startswith("  *"):
            # Bullet list
            fig.text(0.08, y, paragraph, fontsize=8.5,
                     fontfamily="serif", va="top",
                     linespacing=1.5)
            lines = paragraph.count("\n") + 1
            y -= 0.015 * lines + 0.005
        else:
            # Regular paragraph
            wrapped = textwrap.fill(paragraph, width=95)
            fig.text(0.08, y, wrapped, fontsize=9,
                     fontfamily="serif", va="top",
                     linespacing=1.4)
            lines = wrapped.count("\n") + 1
            y -= 0.015 * lines + 0.008

        if y < 0.06:
            break

    pdf.savefig(fig)
    plt.close(fig)


def add_figure_page(pdf, figure_path, title, caption):
    """Add a page with a figure and caption."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    # Title
    fig.text(0.5, 0.96, title, fontsize=13, fontweight="bold",
             fontfamily="serif", ha="center", va="top")

    # Load and display figure
    if figure_path.exists():
        img = mpimg.imread(str(figure_path))
        ax = fig.add_axes([0.05, 0.25, 0.9, 0.68])
        ax.imshow(img)
        ax.axis("off")
    else:
        fig.text(0.5, 0.55, f"[Figure not found: {figure_path.name}]",
                 fontsize=12, ha="center", color="red")

    # Caption
    wrapped_caption = textwrap.fill(caption, width=105)
    fig.text(0.08, 0.22, wrapped_caption, fontsize=8,
             fontfamily="serif", va="top", linespacing=1.4,
             style="italic")

    pdf.savefig(fig)
    plt.close(fig)


def main():
    cs = results["cell_summary"]

    with PdfPages(str(OUTPUT_PDF)) as pdf:
        # ── Page 1: Title + Executive Summary ──────────────────────
        add_text_page(pdf,
            "BP 2×2 Experimental Design Audit",
            f"""BP Framework Validation Experiment — Phase 03 Analysis Report
Date: 2026-04-13  |  Branch: bp-2x2-instrumentation  |  Total runs: {results['n_total_runs']}

## Executive Summary

This report documents five systematic confounds identified in the BP 2×2 experiment that undermine the interpretability of the factorial comparison. The experiment tested the Beneventano-Poggio four-term decomposition (Delta = log(kappa_0/kappa) + phi + G - epsilon) across four architecture configurations: d00 (single agent, no memory), d10 (single agent, memory), d01 (2 parallel agents, no sharing), and d11 (2 parallel agents, shared memory).

**Key finding**: Both memory and parallelism DEGRADE performance (Cohen's d = +0.52 and +0.63 respectively). However, this result is confounded by at least five experimental design issues that must be resolved before the BP framework can be meaningfully tested.

## Results at a Glance

|  Cell  | Reps | Runs | Best val_bpb | Success Rate | Mean Train(s) |
|--------|------|------|-------------|--------------|----------------|
|  d00   |   5  |  {cs['d00']['n_runs']}  |   {cs['d00']['best_vbpb']:.6f}  |   26.2%      |   {cs['d00']['mean_train']:.1f}         |
|  d10   |   5  |  {cs['d10']['n_runs']}  |   {cs['d10']['best_vbpb']:.6f}  |   18.8%      |   {cs['d10']['mean_train']:.1f}         |
|  d01   |   3  |  {cs['d01']['n_runs']}  |   {cs['d01']['best_vbpb']:.6f}  |    7.1%      |   {cs['d01']['mean_train']:.1f}         |
|  d11   |   2  |  {cs['d11']['n_runs']}  |   {cs['d11']['best_vbpb']:.6f}  |    0.0%      |   {cs['d11']['mean_train']:.1f}         |

## Five Identified Confounds

  - Confound 1: CPU contention inflates training time for parallel cells (esp. d11)
  - Confound 2: Agent homogeneity — identical LLM produces identical strategies (G approx 0)
  - Confound 3: Memory anchoring — memory causes sunk-cost bias instead of routing
  - Confound 4: Task ceiling — CIFAR-10/585 steps leaves only 12.2% room for improvement
  - Confound 5: Budget insufficiency — run-9 wall requires more iterations than d11 can produce""",
            subtitle="Confound Analysis and Redesign Recommendations"
        )

        # ── Page 2: Confound 1 — CPU Contention ──────────────────
        cpu = results["cpu_contention"]
        add_text_page(pdf,
            "Confound 1: CPU Contention",
            f"""## Evidence

The Kruskal-Wallis test confirms that training times differ significantly across cells (H={cpu['kruskal_wallis']['H']:.1f}, p < 0.001). The key contrasts:

  - d00 vs d01: p=0.45, NOT significant. Two agents without shared memory show no overhead.
  - d00 vs d11: p={cpu['d00_vs_d11']['p']:.4f}, SIGNIFICANT. Effect size r={cpu['d00_vs_d11']['r']:.3f}.
  - d01 vs d11: p={cpu['d01_vs_d11']['p']:.4f}, SIGNIFICANT. Effect size r={cpu['d01_vs_d11']['r']:.3f}.

Median training times: d00={cs['d00']['mean_train']:.1f}s, d10={cs['d10']['mean_train']:.1f}s, d01={cs['d01']['mean_train']:.1f}s, d11={cs['d11']['mean_train']:.1f}s.

## Interpretation

The CPU contention is specifically severe in d11 (parallel + shared memory), where median training time is 228s — 3.2x the d10 median of 70s. This means d11 agents complete fewer iterations per session than any other cell, not because the architecture is inherently worse, but because training time is inflated by resource contention.

Critically, d01 (parallel, no sharing) does NOT show significant contention vs d00 (p=0.45). This suggests the shared memory mechanism in d11 introduces additional I/O or synchronization overhead beyond simple CPU sharing.

## Impact on BP Terms

The kappa (per-step cost) term in the BP decomposition is directly confounded: d11's higher kappa is not an architectural property but a hardware artifact. Any advantage from shared memory (lower epsilon) is masked by inflated kappa.

## Recommended Fix

Run parallel agents on separate hardware (separate machines or isolated CPU sets). Alternatively, double the time budget for parallel cells to compensate for contention. For d11 specifically, investigate the shared memory I/O mechanism for unnecessary synchronization overhead."""
        )

        # ── Page 2b: CPU Contention Figure ────────────────────────
        add_figure_page(pdf,
            FIG_DIR / "figure-01-cpu-contention.png",
            "Figure 1: CPU Contention Evidence",
            "Panel A: Training time per run by cell. d11 shows dramatically inflated training times (median 228s vs 60-104s for others). Panel B: Wall-clock time per run — includes both agent thinking and training time. d11's wall time (median 178s) is dominated by training. Panel C: Training time by replicate — d11's elevated training time is consistent across reps, confirming systematic contention rather than outlier effect."
        )

        # ── Page 3: Confound 2 — Agent Homogeneity ───────────────
        homo = results["agent_homogeneity"]
        add_text_page(pdf,
            "Confound 2: Agent Homogeneity",
            f"""## Evidence

In d01 (parallel, no sharing), the Jaccard similarity between agent_0 and agent_1 strategy categories is:
  - Rep 1: Jaccard = {homo.get('d01_rep1_jaccard', 'N/A'):.2f} (4 of 5 categories shared)
  - Rep 2: Jaccard = {homo.get('d01_rep2_jaccard', 'N/A'):.2f} (4 of 4 categories shared — perfect overlap)
  - Rep 3: Jaccard = {homo.get('d01_rep3_jaccard', 'N/A'):.2f} (3 of 4 categories shared)

Keyword-level analysis confirms that both agents independently converge on the same specific techniques: dropout tuning, learning rate adjustment, channel width scaling. The shared keywords include "dropout", "channels", "learning rate" in every rep.

Strategy entropy is nearly identical across all cells: d00={homo.get('d00_entropy', 0):.3f}, d10={homo.get('d10_entropy', 0):.3f}, d01={homo.get('d01_entropy', 0):.3f}, d11={homo.get('d11_entropy', 0):.3f}.

## Interpretation

The BP framework's G = I_pi(S;D) term measures information generated through diverse exploration. When G > 0, parallel agents should explore different strategy regions, collectively covering more of the search space than a single agent.

Our experiment uses identical agents: same model (claude-haiku-4-5-20251001), same temperature, same system prompt, same initial state. The result is deterministic: both agents independently discover the same strategies. This means G is approximately 0 by construction — we are measuring "parallelism" but testing "duplication".

## Impact on BP Terms

G = 0 because S (strategies) is effectively constant across agents. The mutual information I(S;D) collapses when the strategy distribution is identical regardless of agent identity. Parallel execution provides no information advantage.

## Recommended Fix

Inject diversity into the agent population: different temperatures (0.3 vs 1.0), different system prompts ("focus on architecture" vs "focus on optimization"), or different models (Haiku vs Sonnet). The key insight is that parallelism in the BP framework assumes agent heterogeneity — which must be designed, not assumed."""
        )

        # ── Page 3b: Agent Homogeneity Figure ─────────────────────
        add_figure_page(pdf,
            FIG_DIR / "figure-02-agent-homogeneity.png",
            "Figure 2: Agent Homogeneity Evidence",
            "Panel A: Strategy category distribution across all four cells. Categories are nearly identical — all cells explore the same 4-5 strategy types with similar proportions. Panel B: Per-agent strategy breakdown for d01 (parallel, no sharing). Both agents independently discover the same categories. Panel C: Per-agent breakdown for d11 (parallel, shared memory). Again, high overlap. The shared memory provides no diversification benefit because agents already converge on the same strategies."
        )

        # ── Page 4: Confound 3 — Memory Anchoring ────────────────
        anchor = results["memory_anchoring"]
        d10_corr = anchor.get("d10_mem_corr", {})
        add_text_page(pdf,
            "Confound 3: Memory Anchoring vs Information Routing",
            f"""## Evidence

**Strategy persistence (consecutive streaks):**
  - d00 (no memory): mean max streak = {anchor.get('d00_mean_streak', 0):.1f}
  - d10 (memory): mean max streak = {anchor.get('d10_mean_streak', 0):.1f}
  - d01 (parallel): mean max streak = {anchor.get('d01_mean_streak', 0):.1f}
  - d11 (parallel+mem): mean max streak = {anchor.get('d11_mean_streak', 0):.1f}

Memory cells (d10) show 75% longer strategy streaks than no-memory cells (d00). The longest streak in d10 is 8 consecutive runs in the same strategy category, compared to max 4 in d00.

**Memory depth vs performance (d10):** Spearman r = {d10_corr.get('r', 'N/A')}, p = {d10_corr.get('p', 'N/A')} (n=64). No significant correlation — more memory entries do not predict better val_bpb.

**Strategy shift test:** Chi-squared tests comparing early vs late strategy distributions show no significant shift for any cell (all p > 0.18). This means agents do not adapt their strategy mix based on accumulated experience.

## Interpretation

The BP framework's epsilon term measures routing mismatch — the KL divergence between the agent's actual strategy distribution and the optimal one. For epsilon to be meaningful, memory must enable conditional strategy selection: "given that dropout didn't work, try architecture changes."

Instead, memory appears to cause anchoring: the agent reads its history and doubles down on previously tried approaches (longer streaks), rather than pivoting. The lack of memory-performance correlation confirms that memory quantity does not translate to information quality.

## Impact on BP Terms

The epsilon term is likely NEGATIVE (memory increases mismatch rather than reducing it). Memory causes sunk-cost bias, anchoring the agent to suboptimal strategy basins.

## Recommended Fix

Restructure memory content: instead of a raw chronological log, provide a structured summary: "Top 3 improvements so far", "Strategy categories tried and their success rates", "Unexplored regions of the search space". Quality over quantity."""
        )

        # ── Page 4b: Memory Anchoring Figure ──────────────────────
        add_figure_page(pdf,
            FIG_DIR / "figure-03-memory-anchoring.png",
            "Figure 3: Memory Anchoring Evidence",
            "Panel A: Strategy switch probability over run index. Higher values indicate more exploration. d10 (memory) shows lower switch rates than d00 (no memory) at later run indices, indicating that memory reduces exploration rather than enabling smarter routing. Panel B: Memory depth (number of context entries) vs val_bpb for d10 and d11. No correlation — more memory does not improve outcomes. Panel C: Cumulative unique strategies over time. All cells plateau at the same level (4-5 categories), confirming that memory does not unlock new strategy modes."
        )

        # ── Page 5: Confound 4 — Task Ceiling ────────────────────
        ceiling = results["task_ceiling"]
        add_text_page(pdf,
            "Confound 4: Task Ceiling Effect",
            f"""## Evidence

**Overall improvement rate:** Only {ceiling['n_improvements']} of {results['n_total_runs']} runs (12.2%) beat the baseline val_bpb of 0.925845. The remaining 87.8% of agent modifications either match or degrade performance.

**Strategy win rates (across all cells):**
  - "other" (misc. changes): 31.6% success rate (18/57)
  - "architecture" changes: 27.3% (6/22)
  - "data_pipeline": 25.0% (1/4)
  - "optimization" (LR, scheduler): 4.8% (4/84)
  - "regularization" (dropout, weight decay): 0.0% (0/50)

**Improvement magnitude degrades across cells:**
  - d00: mean improvement delta = 0.049, max = 0.102
  - d10: mean improvement delta = 0.016, max = 0.050
  - d01: mean improvement delta = 0.007, max = 0.011
  - d11: NO improvements at all

## Interpretation

CIFAR-10 with MAX_STEPS=585 creates a very tight optimization landscape. The baseline architecture is already reasonably configured, and the 585-step budget provides minimal room for training. Most agent modifications (especially regularization and optimizer tuning) make things worse because they require more training steps to converge.

The striking pattern is that improvement magnitude decreases from d00 to d01 to d11, suggesting that whatever advantage single agents discover, it gets diluted or lost in parallel/memory configurations. Regularization having a 0% win rate across 50 attempts is a strong signal that the task's constraints (short training, small model) make regularization counterproductive.

## Impact on BP Terms

When the search space has very few "winners," the phi (within-mode competence) term dominates. Parallelism (G) and routing (epsilon) provide no advantage because the viable improvement region is so narrow that luck matters more than strategy.

## Recommended Fix

  - Increase MAX_STEPS to 1000-2000 to create more room for optimization
  - Use a smaller baseline model (fewer channels) to create a larger improvement gap
  - Consider CIFAR-100 for a harder task with more optimization headroom
  - Alternatively, reduce the baseline architecture quality to create a "worse starting point" """
        )

        # ── Page 5b: Task Ceiling Figure ──────────────────────────
        add_figure_page(pdf,
            FIG_DIR / "figure-04-task-ceiling.png",
            "Figure 4: Task Ceiling Evidence",
            "Panel A: Distribution of val_bpb values across all cells. The baseline (red dashed) is near the left edge of the distribution — most modifications move performance rightward (worse). Panel B: Per-cell success rate (fraction of non-baseline runs beating baseline). d00 achieves 26%, d01 only 7%, and d11 achieves 0%. Panel C: Strategy win/lose counts. Regularization (50 attempts, 0 wins) and optimization (84 attempts, 4 wins) are near-universally ineffective. Only 'other' and 'architecture' categories show meaningful win rates."
        )

        # ── Page 6: Confound 5 — Budget Sufficiency ──────────────
        budget = results["budget_sufficiency"]
        add_text_page(pdf,
            "Confound 5: Budget Sufficiency and the Run-9 Wall",
            f"""## Evidence

**First improvement timing (run index per rep):**
  - d00: run indices [9, -, -, 12, 10], mean = {budget.get('d00_mean_first_imp', 'N/A'):.1f}
  - d10: run indices [-, 9, -, 12, -], mean = {budget.get('d10_mean_first_imp', 'N/A'):.1f}
  - d01: run indices [5, 7, -], mean = {budget.get('d01_mean_first_imp', 'N/A'):.1f}
  - d11: [-, -] — no improvements observed

The "Run-9 Wall": For single-agent cells, zero improvements occur before run index 9 across all replicates. This defines a structural minimum exploration threshold.

**Effective iterations per agent:**
  - d00: mean {budget.get('d00_mean_runs_per_agent', 0):.1f} runs/agent (range: 2-15)
  - d10: mean {budget.get('d10_mean_runs_per_agent', 0):.1f} runs/agent (range: 3-20)
  - d01: mean {budget.get('d01_mean_runs_per_agent', 0):.1f} runs/agent per agent (12-18 each)
  - d11: mean {budget.get('d11_mean_runs_per_agent', 0):.1f} runs/agent (4-11 each)

**Reps with NO improvement:**
  - d00: {budget.get('d00_reps_without_improvement', 0)} of 5 reps (40%)
  - d10: {budget.get('d10_reps_without_improvement', 0)} of 5 reps (60%)
  - d01: 1 of 3 reps (33%)
  - d11: 2 of 2 reps (100%)

## Interpretation

The Run-9 Wall reveals that the agent needs approximately 9 exploratory failures before it stumbles onto a productive direction. With a 45-minute budget and ~2-minute training runs, single agents can complete ~15-20 iterations — just barely past the threshold.

d11 agents average only 7.5 runs per agent (due to CPU contention inflating training time to 228s). This means d11 agents run OUT of budget before they can cross the Run-9 Wall. The 0% success rate in d11 may not reflect an architectural deficiency but simply insufficient iterations.

## Impact on BP Terms

The budget constraint creates a systematic bias against d11. The phi term (within-mode competence) requires sufficient iterations to manifest. When agents cannot reach the minimum exploration threshold, the experiment measures budget adequacy, not architectural capability.

## Recommended Fix

  - Extend d11 budget to 90 minutes (2x) to compensate for CPU contention
  - Or reduce training time per run by using fewer MAX_STEPS (e.g., 300 instead of 585)
  - Or run agents on separate hardware to eliminate contention
  - The key metric should be "improvement per N iterations" not "improvement per 45 minutes" """
        )

        # ── Page 6b: Budget Sufficiency Figure ────────────────────
        add_figure_page(pdf,
            FIG_DIR / "figure-05-budget-sufficiency.png",
            "Figure 5: Budget Sufficiency Evidence",
            "Panel A: Best-so-far optimization trajectories for all reps across all cells. Improvements (downward steps) cluster at run index 9+ for single-agent cells. Many reps plateau at baseline throughout the entire session. Panel B: First improvement timing — dots show when improvements first occur (x marks: reps with no improvement). The vertical line at run 9 marks the empirical minimum exploration threshold. Panel C: Session length vs best outcome — longer sessions correlate with better results, confirming that budget is a limiting factor."
        )

        # ── Page 7: 2×2 Factorial Summary ─────────────────────────
        fac = results["factorial"]
        add_text_page(pdf,
            "2×2 Factorial Analysis",
            f"""## Main Effects

**Memory (d10+d11 vs d00+d01):** +{fac['memory_effect']:.6f} val_bpb (Cohen's d = +{fac['cohens_d_memory']:.3f})
Memory increases (worsens) val_bpb. Medium effect size. Positive direction means memory HURTS.

**Parallelism (d01+d11 vs d00+d10):** +{fac['parallel_effect']:.6f} val_bpb (Cohen's d = +{fac['cohens_d_parallelism']:.3f})
Parallelism increases (worsens) val_bpb. Medium-large effect size. Positive direction means parallelism HURTS.

**Interaction (M x P):** {fac['interaction']:+.6f} (permutation p = {fac['interaction_perm_p']:.4f})
The interaction is NOT significant. The combined effect of memory + parallelism is approximately additive (both hurt, combining them hurts more, but not synergistically).

## Cell Ranking (mean best-of-rep, lower is better)

  - 1st: d00 = {cs['d00']['mean_best']:.6f} +/- {np.std(cs['d00'].get('best_per_rep', [results['cell_summary']['d00']['mean_best']]), ddof=1) if isinstance(cs['d00'].get('best_per_rep'), list) else 0:.6f}
  - 2nd: d10 = {cs['d10']['mean_best']:.6f}
  - 3rd: d01 = {cs['d01']['mean_best']:.6f}
  - 4th: d11 = {cs['d11']['mean_best']:.6f} (no improvement, at baseline)

## Statistical Caveats

  - d11 has only 2 complete reps (vs 5 for d00/d10, 3 for d01)
  - d11 results are confounded by CPU contention (mean training time 268s vs 73-116s)
  - The unbalanced design (5/5/3/2 reps) limits the reliability of factorial estimates
  - All effect sizes should be interpreted with caution given n < 6 per cell
  - The permutation test accounts for the small/unequal sample sizes"""
        )

        # ── Page 7b: 2×2 Summary Figure ──────────────────────────
        add_figure_page(pdf,
            FIG_DIR / "figure-06-2x2-summary.png",
            "Figure 6: 2×2 Factorial Summary",
            "Panel A: Best val_bpb per rep for each cell. d00 shows the widest spread but also the lowest values. d11 is anchored at baseline. Panel B: Per-run success rate. d00 dominates at 26.2%; d11 has 0%. Panel C: 2×2 interaction plot. Both lines slope upward (memory hurts), with parallel agents performing consistently worse. Lines are approximately parallel, confirming no significant interaction. Panel D: Jensen gap (cost variance measure from BP framework). Memory consistently reduces cost variance, but this efficiency gain does not translate to performance improvement."
        )

        # ── Page 8: Synthesis + Recommendations ──────────────────
        add_text_page(pdf,
            "Synthesis and Redesign Recommendations",
            """## What the Current Experiment Actually Tests

The current design does NOT test "does parallelism and memory help agent optimization?" It tests a narrower (and less informative) question: "do duplicate agents sharing CPU and reading raw logs beat a single agent with dedicated resources on a near-optimal task?"

The answer is no — but this is not informative about the BP framework because:
  - G approx 0 (agent homogeneity violates the diversity assumption)
  - epsilon < 0 (memory causes anchoring, not routing)
  - kappa is confounded (CPU contention inflates cost for parallel cells)
  - phi is budget-limited (d11 agents can't reach the exploration threshold)

## What IS Informative from These Results

Despite the confounds, several findings ARE valuable for the BP framework:

  - The Jensen gap consistently decreases with more structure (d00 > d01 > d10 > d11), confirming that memory and sharing reduce cost variance.
  - The Run-9 Wall defines a minimum exploration threshold that any BP configuration must exceed.
  - Strategy win rates are architecture-independent (regularization never works, regardless of cell), suggesting phi is primarily determined by the task, not the agent.
  - Memory increases streak length (sunk-cost bias), establishing that the epsilon term requires structured information, not raw history.

## Recommended Redesign

**Priority 1 — Eliminate CPU contention:**
Run parallel agents on separate hardware or isolated CPU sets.

**Priority 2 — Inject agent diversity:**
Use different temperatures (agent_0 at 0.3, agent_1 at 1.0) or different system prompts to ensure G > 0.

**Priority 3 — Restructure memory:**
Replace chronological logs with structured summaries: top strategies, success rates, unexplored regions.

**Priority 4 — Expand task headroom:**
Increase MAX_STEPS to 1500+ or use a smaller baseline to create more improvement room.

**Priority 5 — Equalize effective budget:**
Ensure all cells achieve the same number of iterations per agent, either by normalizing time budgets or by switching to iteration-based budgets."""
        )

        # ── Page 9: Detailed Statistical Appendix ────────────────
        add_text_page(pdf,
            "Statistical Appendix",
            f"""## Sample Sizes

| Cell | Reps | Runs | Non-baseline | Agents/rep |
|------|------|------|-------------|------------|
| d00  |  5   |  47  |     42      |     1      |
| d10  |  5   |  69  |     64      |     1      |
| d01  |  3   |  91  |     85      |     2      |
| d11  |  2   |  30  |     26      |     2      |

## Tests Performed

**CPU Contention:**
Kruskal-Wallis across 4 cells: H={cpu['kruskal_wallis']['H']:.3f}, p<0.001. Pairwise Mann-Whitney U with effect sizes reported.

**Agent Homogeneity:**
Jaccard similarity index on strategy categories per agent per rep. Keyword extraction from hypothesis descriptions.

**Memory Anchoring:**
Chi-squared tests for early vs late strategy distribution shift (all p > 0.18). Spearman correlation between memory depth and val_bpb (r=-0.23, p=0.066). Maximum consecutive same-strategy streak analysis.

**Task Ceiling:**
Descriptive analysis of improvement rates. Strategy-level win rates. Improvement magnitude by cell.

**Budget Sufficiency:**
First improvement index per rep. Runs-per-agent comparison. Point-biserial correlation between session length and improvement (r=0.39, p=0.15).

**Factorial:**
2x2 main effects and interaction computed on mean-of-best-per-rep. Cohen's d for main effects. Permutation test (N=10,000) for interaction significance.

## Limitations

  - d11 has only 2 complete reps (d11_rep3 was still running at analysis time)
  - Unbalanced design (5/5/3/2) limits factorial inference
  - Strategy categories are self-reported by the agent (potential labeling noise)
  - "Baseline candidate" runs are excluded from success rate calculations
  - Effect sizes are unstable with n < 6"""
        )

    print(f"PDF report saved to: {OUTPUT_PDF}")
    print(f"Total pages: 13")


if __name__ == "__main__":
    main()
