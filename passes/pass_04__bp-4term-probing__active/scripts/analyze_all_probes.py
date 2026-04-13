#!/usr/bin/env python3
"""Analyze all probe results and generate comparison report.

Loads all Wave 1 probe data, computes process metrics,
creates comparison figures, and writes a structured report.

Usage:
    python workflow/scripts/analyze_all_probes.py --repo-root .
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path


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


def compute_process_metrics(runs: list[dict]) -> dict:
    """Compute all process metrics for a set of runs."""
    if not runs:
        return {"n_runs": 0}

    non_baseline = [r for r in runs if not r.get("baseline_candidate", False)]
    all_cats = [r.get("strategy_category", "unknown") for r in non_baseline]
    cat_counts = Counter(all_cats)
    total_cats = len(all_cats)

    # Entropy
    entropy = 0
    if total_cats > 0:
        entropy = -sum((c / total_cats) * math.log2(c / total_cats)
                       for c in cat_counts.values())

    # Per-agent analysis
    agents = sorted(set(r["_agent"] for r in runs))
    agent_data = {}
    for agent in agents:
        agent_runs = [r for r in runs if r["_agent"] == agent]
        agent_nb = [r for r in non_baseline if r["_agent"] == agent]
        agent_cats = set(r.get("strategy_category", "unknown") for r in agent_nb)
        agent_data[agent] = {
            "n_runs": len(agent_runs),
            "n_non_baseline": len(agent_nb),
            "categories": agent_cats,
            "cat_sequence": [r.get("strategy_category", "unknown") for r in agent_nb],
        }

    # Jaccard (for parallel probes with 2 agents)
    jaccard = None
    if len(agents) == 2:
        a0_cats = agent_data[agents[0]]["categories"]
        a1_cats = agent_data[agents[1]]["categories"]
        union = a0_cats | a1_cats
        jaccard = len(a0_cats & a1_cats) / len(union) if union else 0

    # Switch rate
    switch_count = 0
    switch_total = 0
    for agent in agents:
        seq = [r.get("strategy_category", "unknown")
               for r in sorted(
                   [r for r in runs if r["_agent"] == agent],
                   key=lambda r: r.get("run_index", 0)
               )]
        for i in range(1, len(seq)):
            switch_total += 1
            if seq[i] != seq[i - 1]:
                switch_count += 1
    switch_rate = switch_count / switch_total if switch_total > 0 else 0

    # Performance
    baseline_bpb = 0.925845
    improvements = sum(1 for r in non_baseline
                       if r.get("val_bpb") is not None
                       and r["val_bpb"] < baseline_bpb)
    best_bpb = min((r["val_bpb"] for r in runs if r.get("val_bpb") is not None),
                   default=baseline_bpb)
    worst_bpb = max((r["val_bpb"] for r in runs if r.get("val_bpb") is not None),
                    default=baseline_bpb)

    # Training times
    train_times = [r.get("training_seconds", 0) for r in runs
                   if r.get("training_seconds")]
    mean_train = sum(train_times) / len(train_times) if train_times else 0

    return {
        "n_runs": len(runs),
        "n_non_baseline": len(non_baseline),
        "n_agents": len(agents),
        "n_categories": len(cat_counts),
        "categories": dict(cat_counts),
        "entropy": entropy,
        "jaccard": jaccard,
        "switch_rate": switch_rate,
        "switch_count": switch_count,
        "switch_total": switch_total,
        "improvements": improvements,
        "best_bpb": best_bpb,
        "worst_bpb": worst_bpb,
        "mean_train_seconds": mean_train,
        "agent_data": {
            a: {
                "n_runs": d["n_runs"],
                "n_non_baseline": d["n_non_baseline"],
                "categories": list(d["categories"]),
                "cat_sequence": d["cat_sequence"],
            }
            for a, d in agent_data.items()
        },
    }


PROBES = {
    # Wave 1
    "P01": {"name": "parallel_homo", "type": "parallel", "variable": "control", "wave": 1},
    "P02": {"name": "parallel_diverse", "type": "parallel", "variable": "temperature diversity", "wave": 1},
    "P03": {"name": "single_baseline", "type": "single", "variable": "control", "wave": 1},
    "P04": {"name": "single_short_train", "type": "single", "variable": "30s training", "wave": 1},
    "P05": {"name": "memory_baseline", "type": "single", "variable": "external memory*", "wave": 1},
    "P06": {"name": "shared_diverse", "type": "parallel", "variable": "diversity + shared*", "wave": 1},
    # Wave 2
    "P07": {"name": "shared_extended", "type": "parallel", "variable": "extended shared*", "wave": 2},
    "P08": {"name": "memory_extended", "type": "single", "variable": "extended memory*", "wave": 2},
    "P09": {"name": "diverse_extended", "type": "parallel", "variable": "extended diversity", "wave": 2},
    "P10": {"name": "homo_fixed", "type": "parallel", "variable": "P01 control fixed", "wave": 2},
    # Wave 3 (with fixed memory mechanism)
    "P11": {"name": "single_hightemp", "type": "single", "variable": "high temp 1.2", "wave": 3},
    "P12": {"name": "shared_fixed", "type": "parallel", "variable": "REAL shared memory", "wave": 3},
    "P13": {"name": "dual_hightemp", "type": "parallel", "variable": "dual high temp", "wave": 3},
    "P14": {"name": "hightemp_memory", "type": "single", "variable": "high temp + memory", "wave": 3},
    # Wave 4 (seeded search + optimal baseline + full stack)
    "P15": {"name": "seeded_guidance", "type": "single", "variable": "seeded search (LR hint)", "wave": 4},
    "P16": {"name": "optimal_baseline", "type": "single", "variable": "LR 1.5e-3 baseline", "wave": 4},
    "P17": {"name": "full_stack", "type": "parallel", "variable": "full BP (G + ε)", "wave": 4},
    "P18": {"name": "seeded_parallel", "type": "parallel", "variable": "seeded parallel", "wave": 4},
}


def compare_probes(metrics: dict[str, dict]) -> list[str]:
    """Run all pairwise comparisons and return findings."""
    findings = []

    # Test 1: Diversity injection (P01 vs P02)
    p01 = metrics.get("P01", {})
    p02 = metrics.get("P02", {})
    if p01.get("n_runs", 0) > 0 and p02.get("n_runs", 0) > 0:
        j01 = p01.get("jaccard", 1.0)
        j02 = p02.get("jaccard", 1.0)
        e01 = p01.get("entropy", 0)
        e02 = p02.get("entropy", 0)
        findings.append(f"DIVERSITY (P01 vs P02):")
        findings.append(f"  Jaccard: P01={j01:.3f} vs P02={j02:.3f} "
                        f"({'P02 more diverse' if j02 is not None and j01 is not None and j02 < j01 else 'similar/P01 more diverse'})")
        findings.append(f"  Entropy: P01={e01:.3f} vs P02={e02:.3f}")
        findings.append(f"  Runs: P01={p01['n_runs']} vs P02={p02['n_runs']}")
        findings.append(f"  Improvements: P01={p01.get('improvements', 0)} vs P02={p02.get('improvements', 0)}")

        if j02 is not None and j01 is not None and j02 < j01:
            findings.append(f"  SIGNAL: Diversity injection reduces Jaccard by {j01 - j02:.3f}")
        else:
            findings.append(f"  NO SIGNAL: Diversity injection did not reduce Jaccard")

    # Test 2: Training headroom (P03 vs P04)
    p03 = metrics.get("P03", {})
    p04 = metrics.get("P04", {})
    if p03.get("n_runs", 0) > 0 and p04.get("n_runs", 0) > 0:
        ratio = p04["n_runs"] / p03["n_runs"] if p03["n_runs"] > 0 else 0
        findings.append(f"\nTRAINING HEADROOM (P03 vs P04):")
        findings.append(f"  Total runs: P03={p03['n_runs']} (60s) vs P04={p04['n_runs']} (30s), ratio={ratio:.2f}x")
        findings.append(f"  Mean train time: P03={p03.get('mean_train_seconds', 0):.0f}s vs P04={p04.get('mean_train_seconds', 0):.0f}s")
        findings.append(f"  Entropy: P03={p03.get('entropy', 0):.3f} vs P04={p04.get('entropy', 0):.3f}")
        findings.append(f"  Improvements: P03={p03.get('improvements', 0)} vs P04={p04.get('improvements', 0)}")

        if ratio > 1.3:
            findings.append(f"  SIGNAL: Shorter training gives {ratio:.1f}x more iterations")
        else:
            findings.append(f"  NO SIGNAL: Run count ratio only {ratio:.1f}x (threshold: 1.3x)")

    # Test 3: Memory + diversity (P05 vs P06)
    p05 = metrics.get("P05", {})
    p06 = metrics.get("P06", {})
    if p05.get("n_runs", 0) > 0 and p06.get("n_runs", 0) > 0:
        findings.append(f"\nMEMORY + DIVERSITY (P05 vs P06):")
        findings.append(f"  Runs: P05={p05['n_runs']} vs P06={p06['n_runs']}")
        findings.append(f"  Switch rate: P05={p05.get('switch_rate', 0):.3f} vs P06={p06.get('switch_rate', 0):.3f}")
        findings.append(f"  Improvements: P05={p05.get('improvements', 0)} vs P06={p06.get('improvements', 0)}")
        findings.append(f"  Best bpb: P05={p05.get('best_bpb', 'N/A')} vs P06={p06.get('best_bpb', 'N/A')}")

        if p06.get("improvements", 0) > p05.get("improvements", 0):
            findings.append(f"  SIGNAL: Shared memory + diversity shows more improvements")
        else:
            findings.append(f"  NO SIGNAL: Shared memory + diversity not better")

    # Test 4: Extended budget (Wave 2 vs Wave 1 counterparts)
    pairs = [("P06", "P07", "shared+diverse"), ("P05", "P08", "single+memory"),
             ("P02", "P09", "parallel diverse"), ("P01", "P10", "parallel homo")]
    for short, long, label in pairs:
        ps = metrics.get(short, {})
        pl = metrics.get(long, {})
        if ps.get("n_runs", 0) > 0 and pl.get("n_runs", 0) > 0:
            findings.append(f"\nEXTENDED BUDGET: {label} ({short} vs {long}):")
            findings.append(f"  Runs: {short}={ps['n_runs']} vs {long}={pl['n_runs']}")
            findings.append(f"  Best bpb: {short}={ps.get('best_bpb', 'N/A'):.6f} vs {long}={pl.get('best_bpb', 'N/A'):.6f}")
            findings.append(f"  Improvements: {short}={ps.get('improvements', 0)} vs {long}={pl.get('improvements', 0)}")
            if pl.get("best_bpb", 1) < ps.get("best_bpb", 1):
                delta = ps["best_bpb"] - pl["best_bpb"]
                findings.append(f"  SIGNAL: Extended budget improved best bpb by {delta:.6f}")
            else:
                findings.append(f"  NO SIGNAL: Extended budget did not improve best bpb")

    # Test 5: Temperature effect (Wave 3)
    p11 = metrics.get("P11", {})
    p03 = metrics.get("P03", {})
    if p11.get("n_runs", 0) > 0 and p03.get("n_runs", 0) > 0:
        findings.append(f"\nTEMPERATURE EFFECT (P03 default vs P11 temp=1.2):")
        findings.append(f"  Runs: P03={p03['n_runs']} vs P11={p11['n_runs']}")
        findings.append(f"  Best bpb: P03={p03.get('best_bpb', 'N/A'):.6f} vs P11={p11.get('best_bpb', 'N/A'):.6f}")

    # Test 6: Real shared memory (Wave 3)
    p12 = metrics.get("P12", {})
    p07 = metrics.get("P07", {})
    if p12.get("n_runs", 0) > 0 and p07.get("n_runs", 0) > 0:
        findings.append(f"\nREAL SHARED MEMORY (P07 broken vs P12 fixed):")
        findings.append(f"  Runs: P07={p07['n_runs']} vs P12={p12['n_runs']}")
        findings.append(f"  Best bpb: P07={p07.get('best_bpb', 'N/A'):.6f} vs P12={p12.get('best_bpb', 'N/A'):.6f}")

    # Test 7: G without ε (P11 high-temp no memory vs P14 high-temp + memory)
    p11 = metrics.get("P11", {})
    p14 = metrics.get("P14", {})
    if p11.get("n_runs", 0) > 0 and p14.get("n_runs", 0) > 0:
        findings.append(f"\nG WITHOUT ε (P11 high-temp vs P14 high-temp+memory):")
        findings.append(f"  Runs: P11={p11['n_runs']} vs P14={p14['n_runs']}")
        findings.append(f"  Best bpb: P11={p11.get('best_bpb', 'N/A'):.6f} vs P14={p14.get('best_bpb', 'N/A'):.6f}")
        findings.append(f"  Worst bpb: P11={p11.get('worst_bpb', 'N/A'):.6f} vs P14={p14.get('worst_bpb', 'N/A'):.6f}")
        if p14.get("best_bpb", 1) < p11.get("best_bpb", 1):
            findings.append(f"  SIGNAL: Memory (ε) helps high-temp agent find better results")
        elif p14.get("worst_bpb", 0) < p11.get("worst_bpb", 0):
            findings.append(f"  SIGNAL: Memory (ε) prevents degradation (lower worst bpb)")
        else:
            findings.append(f"  NO SIGNAL: Memory did not improve high-temp agent")

    # Test 8: Degradation analysis (monotonic worsening)
    for pid in ("P08", "P11"):
        pm = metrics.get(pid, {})
        if pm.get("n_runs", 0) >= 3:
            findings.append(f"\nDEGRADATION CHECK ({pid}):")
            # Load raw runs for trajectory analysis
            for agent, adata in pm.get("agent_data", {}).items():
                seq = adata.get("cat_sequence", [])
                findings.append(f"  {agent}: {len(seq)} non-baseline runs")
            findings.append(f"  Best bpb: {pm.get('best_bpb', 'N/A'):.6f}")
            findings.append(f"  Worst bpb: {pm.get('worst_bpb', 'N/A'):.6f}")
            spread = pm.get("worst_bpb", 0) - pm.get("best_bpb", 0)
            findings.append(f"  Spread: {spread:.6f} (larger = more degradation)")
            if spread > 0.5:
                findings.append(f"  DEGRADATION: Agent quality spread > 0.5 bpb")

    # Test 9: Dual high-temp vs single high-temp (P13 vs P11)
    p13 = metrics.get("P13", {})
    if p13.get("n_runs", 0) > 0 and p11.get("n_runs", 0) > 0:
        findings.append(f"\nDUAL vs SINGLE HIGH-TEMP (P11 vs P13):")
        findings.append(f"  Runs: P11={p11['n_runs']} vs P13={p13['n_runs']}")
        findings.append(f"  Best bpb: P11={p11.get('best_bpb', 'N/A'):.6f} vs P13={p13.get('best_bpb', 'N/A'):.6f}")
        findings.append(f"  Worst bpb: P11={p11.get('worst_bpb', 'N/A'):.6f} vs P13={p13.get('worst_bpb', 'N/A'):.6f}")

    # Cross-probe comparison
    if any(metrics.get(pid, {}).get("n_runs", 0) > 0 for pid in PROBES):
        findings.append(f"\nCROSS-PROBE SUMMARY:")
        best_pid = None
        best_score = -1
        for pid in sorted(PROBES.keys()):
            m = metrics.get(pid, {})
            if m.get("n_runs", 0) == 0:
                continue
            score = (m.get("improvements", 0) * 10
                     + m.get("entropy", 0) * 5
                     + m.get("n_runs", 0))
            findings.append(f"  {pid} (W{PROBES[pid]['wave']}): runs={m['n_runs']}, entropy={m.get('entropy', 0):.3f}, "
                            f"improvements={m.get('improvements', 0)}, best_bpb={m.get('best_bpb', 'N/A'):.6f}, score={score:.1f}")
            if score > best_score:
                best_score = score
                best_pid = pid
        if best_pid:
            findings.append(f"  BEST PROBE: {best_pid} (score={best_score:.1f})")

    return findings


def generate_report(repo_root: Path, metrics: dict, findings: list[str]) -> str:
    """Generate markdown report."""
    active_waves = sorted(set(
        PROBES[pid]["wave"] for pid in metrics if metrics[pid].get("n_runs", 0) > 0
    ))
    wave_label = ", ".join(str(w) for w in active_waves) if active_waves else "?"
    lines = [f"# Probe Analysis Report (Waves {wave_label})\n"]
    lines.append(f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Summary table
    lines.append("## Probe Results Summary\n")
    lines.append("| Probe | Name | Runs | Non-BL | Categories | Entropy | Jaccard | Switch | Improvements | Best bpb |")
    lines.append("|-------|------|------|--------|------------|---------|---------|--------|-------------|----------|")
    for pid, pinfo in PROBES.items():
        m = metrics.get(pid, {})
        if m.get("n_runs", 0) == 0:
            lines.append(f"| {pid} | {pinfo['name']} | - | - | - | - | - | - | - | - |")
            continue
        j = f"{m['jaccard']:.3f}" if m.get('jaccard') is not None else "N/A"
        lines.append(
            f"| {pid} | {pinfo['name']} | {m['n_runs']} | {m['n_non_baseline']} | "
            f"{m['n_categories']} | {m['entropy']:.3f} | {j} | {m['switch_rate']:.3f} | "
            f"{m['improvements']} | {m['best_bpb']:.6f} |"
        )

    # Per-agent details
    lines.append("\n## Per-Agent Details\n")
    for pid, pinfo in PROBES.items():
        m = metrics.get(pid, {})
        if m.get("n_runs", 0) == 0:
            continue
        lines.append(f"### {pid}: {pinfo['name']}\n")
        for agent, adata in m.get("agent_data", {}).items():
            lines.append(f"**{agent}**: {adata['n_runs']} runs, "
                         f"categories: {', '.join(adata['categories']) or 'none'}")
            if adata["cat_sequence"]:
                lines.append(f"  Sequence: {' -> '.join(adata['cat_sequence'])}")
        lines.append("")

    # Findings
    lines.append("## Comparative Findings\n")
    lines.append("```")
    for f in findings:
        lines.append(f)
    lines.append("```\n")

    # Category distribution
    lines.append("## Strategy Category Distribution\n")
    all_cats = Counter()
    for pid in PROBES:
        m = metrics.get(pid, {})
        for cat, count in m.get("categories", {}).items():
            all_cats[cat] += count
    lines.append("| Category | Count | % |")
    lines.append("|----------|-------|---|")
    total = sum(all_cats.values())
    for cat, count in all_cats.most_common():
        pct = 100 * count / total if total > 0 else 0
        lines.append(f"| {cat} | {count} | {pct:.1f}% |")

    return "\n".join(lines)


def main() -> None:
    repo_root = Path(sys.argv[sys.argv.index("--repo-root") + 1]).resolve() \
        if "--repo-root" in sys.argv else Path.cwd()

    print("=" * 60)
    print("PROBE ANALYSIS")
    print("=" * 60)

    # Load and compute metrics
    metrics = {}
    for pid in PROBES:
        runs = load_probe_runs(repo_root, pid)
        m = compute_process_metrics(runs)
        metrics[pid] = m
        print(f"\n{pid} ({PROBES[pid]['name']}): {m['n_runs']} runs")
        if m["n_runs"] > 0:
            j = f"{m['jaccard']:.3f}" if m.get('jaccard') is not None else "N/A"
            print(f"  entropy={m['entropy']:.3f}, jaccard={j}, "
                  f"switch_rate={m['switch_rate']:.3f}, improvements={m['improvements']}")

    # Run comparisons
    findings = compare_probes(metrics)
    print("\n" + "=" * 60)
    print("FINDINGS")
    print("=" * 60)
    for f in findings:
        print(f"  {f}")

    # Determine wave label from available data
    active_waves = sorted(set(
        PROBES[pid]["wave"] for pid in metrics if metrics[pid].get("n_runs", 0) > 0
    ))
    suffix = f"wave{'_'.join(str(w) for w in active_waves)}" if active_waves else "all"

    # Generate report
    report = generate_report(repo_root, metrics, findings)
    report_path = repo_root / "workflow" / "artifacts" / f"probe_{suffix}_analysis.md"
    report_path.write_text(report + "\n")
    print(f"\nReport saved to: {report_path}")

    # Save metrics as JSON
    json_path = repo_root / "workflow" / "artifacts" / f"probe_{suffix}_results.json"
    json_path.write_text(json.dumps(
        {pid: {k: v for k, v in m.items() if k != "agent_data" or True}
         for pid, m in metrics.items()},
        indent=2, default=str,
    ) + "\n")
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
