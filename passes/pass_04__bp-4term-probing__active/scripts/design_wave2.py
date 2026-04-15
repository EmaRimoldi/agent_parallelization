#!/usr/bin/env python3
"""Auto-design Wave 2 probes based on Wave 1 results.

Reads Wave 1 results, identifies which variables showed signal,
creates new configs and adds them to the WAVES dict.
"""

from __future__ import annotations

import json
import math
import sys
import yaml
from collections import Counter
from pathlib import Path


def load_probe_data(repo_root: Path, probe_id: str) -> dict:
    """Load all training runs for a probe."""
    experiment_dir = repo_root / "runs" / f"experiment_probe_{probe_id}"
    if not experiment_dir.exists():
        return {"runs": [], "n_runs": 0}

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

    return {"runs": runs, "n_runs": len(runs)}


def compute_metrics(data: dict) -> dict:
    """Compute process metrics for a probe."""
    runs = data["runs"]
    if not runs:
        return {"n_runs": 0}

    non_baseline = [r for r in runs if not r.get("baseline_candidate", False)]
    cats = [r.get("strategy_category", "unknown") for r in non_baseline]
    cat_counts = Counter(cats)
    total = len(cats)
    entropy = -sum((c / total) * math.log2(c / total)
                   for c in cat_counts.values()) if total > 0 else 0

    # Per-agent analysis
    agents = sorted(set(r["_agent"] for r in runs))
    agent_cats = {}
    for agent in agents:
        agent_runs = [r for r in non_baseline if r["_agent"] == agent]
        agent_cats[agent] = set(r.get("strategy_category", "unknown")
                                for r in agent_runs)

    jaccard = None
    if len(agents) == 2:
        a0 = agent_cats.get(agents[0], set())
        a1 = agent_cats.get(agents[1], set())
        union = a0 | a1
        jaccard = len(a0 & a1) / len(union) if union else 0

    # Switch rate
    switch_count = 0
    switch_total = 0
    for agent in agents:
        agent_runs = sorted(
            [r for r in runs if r["_agent"] == agent],
            key=lambda r: r.get("run_index", 0)
        )
        seq = [r.get("strategy_category", "unknown") for r in agent_runs]
        for i in range(1, len(seq)):
            switch_total += 1
            if seq[i] != seq[i - 1]:
                switch_count += 1

    switch_rate = switch_count / switch_total if switch_total > 0 else 0

    # Improvements
    baseline = 0.925845
    improvements = sum(1 for r in non_baseline
                       if r.get("val_bpb") is not None
                       and r["val_bpb"] < baseline)

    best_vbpb = min((r["val_bpb"] for r in runs if r.get("val_bpb") is not None),
                    default=baseline)

    train_times = [r.get("training_seconds", 0) for r in runs
                   if r.get("training_seconds")]
    mean_train = sum(train_times) / len(train_times) if train_times else 0

    return {
        "n_runs": len(runs),
        "n_non_baseline": len(non_baseline),
        "n_categories": len(cat_counts),
        "entropy": entropy,
        "jaccard": jaccard,
        "switch_rate": switch_rate,
        "improvements": improvements,
        "best_vbpb": best_vbpb,
        "mean_train_seconds": mean_train,
        "categories": dict(cat_counts),
    }


def write_config(path: Path, mode: str, n_agents: int,
                 time_budget: int, train_budget: int,
                 overrides: list, shared_memory: bool = False,
                 external_memory: bool = False,
                 comment: str = "") -> None:
    """Write a probe config YAML."""
    config = {
        "experiment": {
            "id": None,
            "mode": mode,
            "runs_dir": "runs",
        },
        "agents": {
            "n": n_agents,
            "model": "claude-haiku-4-5-20251001",
            "time_budget_minutes": time_budget,
            "train_time_budget_seconds": train_budget,
            "temperature": None,
            "cuda_devices": None,
            "use_external_memory": external_memory,
            "use_shared_memory": shared_memory,
            "overrides": overrides,
        },
        "slurm": {
            "enabled": False,
            "partition": "pi_tpoggio",
            "gres": "gpu:1",
            "time": "00:10:00",
        },
        "templates": {
            "system_prompt": "templates/agent_system_prompt.md",
            "first_message": "templates/agent_first_message.md",
        },
    }

    header = f"# {comment}\n" if comment else ""
    path.write_text(header + yaml.dump(config, default_flow_style=False))


def main() -> None:
    repo_root = Path(sys.argv[sys.argv.index("--repo-root") + 1]).resolve() \
        if "--repo-root" in sys.argv else Path.cwd()

    print("=" * 60)
    print("WAVE 2 AUTO-DESIGN")
    print("=" * 60)

    # Load Wave 1 results
    w1_results_path = repo_root / "workflow" / "artifacts" / "probe_wave1_results.json"
    if not w1_results_path.exists():
        print("Wave 1 results not found. Computing from probe data...")

    # Compute metrics for all Wave 1 probes
    metrics = {}
    for pid in ["P01", "P02", "P03", "P04", "P05", "P06"]:
        data = load_probe_data(repo_root, pid)
        m = compute_metrics(data)
        metrics[pid] = m
        print(f"\n{pid}: {m['n_runs']} runs, entropy={m.get('entropy', 0):.3f}, "
              f"jaccard={m.get('jaccard', 'N/A')}, switch_rate={m.get('switch_rate', 0):.2f}, "
              f"improvements={m.get('improvements', 0)}, best={m.get('best_vbpb', 'N/A')}")

    # Decision logic
    wave2_probes = []
    decisions = []

    # Test 1: Did diversity help? (P01 vs P02)
    p01 = metrics.get("P01", {})
    p02 = metrics.get("P02", {})
    if p01.get("n_runs", 0) > 0 and p02.get("n_runs", 0) > 0:
        j01 = p01.get("jaccard", 1.0)
        j02 = p02.get("jaccard", 1.0)
        e01 = p01.get("entropy", 0)
        e02 = p02.get("entropy", 0)

        if j02 is not None and j01 is not None and j02 < j01:
            decisions.append(f"DIVERSITY HELPS: P02 Jaccard={j02:.2f} < P01 Jaccard={j01:.2f}")
            # Scale up diversity: longer budget + diverse temps
            write_config(
                repo_root / "configs" / "probe_P07_diverse_extended.yaml",
                mode="parallel", n_agents=2, time_budget=30, train_budget=60,
                overrides=[
                    {"agent_id": "agent_0", "temperature": 0.3},
                    {"agent_id": "agent_1", "temperature": 1.2},
                ],
                comment="P07: Diverse parallel, extended 30 min budget"
            )
            wave2_probes.append({
                "probe_id": "P07", "name": "diverse_extended",
                "config": "configs/probe_P07_diverse_extended.yaml",
                "description": "Diverse parallel, 30 min (2x budget)",
                "variable": "extended budget with diversity",
            })
        else:
            decisions.append(f"DIVERSITY NEUTRAL/NEGATIVE: P02 Jaccard={j02}, P01 Jaccard={j01}")
            # Try even more diversity: 3 agents with different temps
            write_config(
                repo_root / "configs" / "probe_P07_3agents_diverse.yaml",
                mode="parallel", n_agents=3, time_budget=15, train_budget=60,
                overrides=[
                    {"agent_id": "agent_0", "temperature": 0.3},
                    {"agent_id": "agent_1", "temperature": None},
                    {"agent_id": "agent_2", "temperature": 1.2},
                ],
                comment="P07: 3 diverse agents (conservative, neutral, exploratory)"
            )
            wave2_probes.append({
                "probe_id": "P07", "name": "3agents_diverse",
                "config": "configs/probe_P07_3agents_diverse.yaml",
                "description": "3 agents with different temps",
                "variable": "agent count + diversity",
            })

    # Test 2: Did shorter training help? (P03 vs P04)
    p03 = metrics.get("P03", {})
    p04 = metrics.get("P04", {})
    if p03.get("n_runs", 0) > 0 and p04.get("n_runs", 0) > 0:
        ratio = p04["n_runs"] / p03["n_runs"] if p03["n_runs"] > 0 else 0
        if ratio > 1.3:
            decisions.append(f"SHORT TRAINING HELPS: P04 {p04['n_runs']} runs vs P03 {p03['n_runs']} ({ratio:.1f}x)")
            # Combine short training + diversity
            write_config(
                repo_root / "configs" / "probe_P08_diverse_short.yaml",
                mode="parallel", n_agents=2, time_budget=20, train_budget=30,
                overrides=[
                    {"agent_id": "agent_0", "temperature": 0.3},
                    {"agent_id": "agent_1", "temperature": 1.2},
                ],
                comment="P08: Diverse parallel + short training (30s)"
            )
            wave2_probes.append({
                "probe_id": "P08", "name": "diverse_short",
                "config": "configs/probe_P08_diverse_short.yaml",
                "description": "Diverse parallel + short training (30s), 20 min",
                "variable": "diversity + short training combined",
            })
        else:
            decisions.append(f"SHORT TRAINING NEUTRAL: ratio={ratio:.1f}x")
            # Try even shorter training
            write_config(
                repo_root / "configs" / "probe_P08_very_short.yaml",
                mode="single_long", n_agents=1, time_budget=15, train_budget=15,
                overrides=[],
                comment="P08: Very short training (15s) for maximum iterations"
            )
            wave2_probes.append({
                "probe_id": "P08", "name": "very_short_train",
                "config": "configs/probe_P08_very_short.yaml",
                "description": "Single agent, 15s training, max iterations",
                "variable": "very short training",
            })

    # Test 3: Did shared memory + diversity help? (P05 vs P06)
    p05 = metrics.get("P05", {})
    p06 = metrics.get("P06", {})
    if p05.get("n_runs", 0) > 0 and p06.get("n_runs", 0) > 0:
        if p06.get("improvements", 0) > p05.get("improvements", 0):
            decisions.append(f"SHARED+DIVERSE HELPS: P06 {p06['improvements']} improvements vs P05 {p05['improvements']}")
            # Scale up shared + diverse
            write_config(
                repo_root / "configs" / "probe_P09_shared_diverse_ext.yaml",
                mode="parallel_shared", n_agents=2, time_budget=30, train_budget=60,
                overrides=[
                    {"agent_id": "agent_0", "temperature": 0.3},
                    {"agent_id": "agent_1", "temperature": 1.2},
                ],
                shared_memory=True,
                comment="P09: Shared memory + diverse, extended 30 min"
            )
            wave2_probes.append({
                "probe_id": "P09", "name": "shared_diverse_ext",
                "config": "configs/probe_P09_shared_diverse_ext.yaml",
                "description": "Shared memory + diverse, 30 min",
                "variable": "shared + diversity extended",
            })
        else:
            decisions.append(f"SHARED+DIVERSE NEUTRAL: P06 {p06.get('improvements',0)} vs P05 {p05.get('improvements',0)}")
            # Try memory with single agent but exploratory
            write_config(
                repo_root / "configs" / "probe_P09_memory_exploratory.yaml",
                mode="single_memory", n_agents=1, time_budget=20, train_budget=60,
                overrides=[{"agent_id": "agent_0", "temperature": 1.2}],
                external_memory=True,
                comment="P09: Single agent + memory + exploratory temp"
            )
            wave2_probes.append({
                "probe_id": "P09", "name": "memory_exploratory",
                "config": "configs/probe_P09_memory_exploratory.yaml",
                "description": "Single + memory + exploratory temp, 20 min",
                "variable": "memory + exploratory style",
            })

    # Always include a "best-so-far replication" — take the config that showed
    # the most improvements and run it again with longer budget
    best_probe = max(
        [pid for pid in ["P01", "P02", "P03", "P04", "P05", "P06"]
         if metrics.get(pid, {}).get("n_runs", 0) > 0],
        key=lambda pid: (metrics[pid].get("improvements", 0),
                         -metrics[pid].get("best_vbpb", 999)),
        default=None
    )
    if best_probe:
        decisions.append(f"BEST WAVE 1 PROBE: {best_probe} ({metrics[best_probe].get('improvements', 0)} improvements)")
        # Replicate with extended budget
        src_config = repo_root / "configs" / f"probe_{best_probe}_*.yaml"
        import glob
        src_files = glob.glob(str(src_config))
        if src_files:
            with open(src_files[0]) as f:
                content = f.read()
            # Modify budget to 30 min
            content = content.replace("time_budget_minutes: 15",
                                      "time_budget_minutes: 30")
            config_path = repo_root / "configs" / "probe_P10_best_replicate.yaml"
            config_path.write_text(f"# P10: Replication of {best_probe} with 30 min budget\n" + content)
            wave2_probes.append({
                "probe_id": "P10", "name": f"replicate_{best_probe}",
                "config": f"configs/probe_P10_best_replicate.yaml",
                "description": f"Replication of {best_probe} with 30 min budget",
                "variable": "budget extension of best",
            })

    # Print decisions
    print("\n" + "=" * 60)
    print("DECISIONS")
    print("=" * 60)
    for d in decisions:
        print(f"  → {d}")

    print(f"\nWave 2 probes: {[p['probe_id'] for p in wave2_probes]}")

    # Update run_probes.py WAVES dict by writing a wave2 manifest
    manifest = {
        "wave": 2,
        "probes": wave2_probes,
        "decisions": decisions,
        "wave1_metrics": {k: {kk: vv for kk, vv in v.items()
                              if not isinstance(vv, (list, dict)) or kk == "categories"}
                         for k, v in metrics.items()},
    }
    manifest_path = repo_root / "workflow" / "artifacts" / "wave2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\nManifest saved to: {manifest_path}")

    # Inject Wave 2 into run_probes.py
    probes_script = repo_root / "workflow" / "scripts" / "run_probes.py"
    content = probes_script.read_text()

    # Build Wave 2 definition string
    wave2_str = "WAVE_2 = " + json.dumps(wave2_probes, indent=4) + "\n"

    if "WAVE_2 = " in content:
        # Replace existing Wave 2
        import re
        content = re.sub(r"WAVE_2 = \[.*?\]\n", wave2_str, content, flags=re.DOTALL)
    else:
        # Insert before WAVES dict
        content = content.replace(
            "WAVES = {1: WAVE_1}",
            wave2_str + "\nWAVES = {1: WAVE_1, 2: WAVE_2}"
        )

    probes_script.write_text(content)
    print("Updated run_probes.py with Wave 2 definition")


if __name__ == "__main__":
    main()
