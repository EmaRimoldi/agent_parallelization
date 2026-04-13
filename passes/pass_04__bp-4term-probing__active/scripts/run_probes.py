#!/usr/bin/env python3
"""Run Phase 04 rapid probing experiments sequentially.

Executes probe experiments one at a time (no CPU contention),
analyzes after each probe, and writes a running log.

Usage:
    python workflow/scripts/run_probes.py --repo-root . --wave 1
    python workflow/scripts/run_probes.py --repo-root . --wave 1 --probes P01 P02
    python workflow/scripts/run_probes.py --repo-root . --wave 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Wave definitions ───────────────────────────────────────────────────

WAVE_1 = [
    {
        "probe_id": "P01",
        "name": "parallel_homo",
        "config": "configs/probe_P01_parallel_homo.yaml",
        "description": "Parallel homogeneous (control for diversity)",
        "variable": "none (control)",
    },
    {
        "probe_id": "P02",
        "name": "parallel_diverse",
        "config": "configs/probe_P02_parallel_diverse.yaml",
        "description": "Parallel diverse (temp 0.3 vs 1.2)",
        "variable": "temperature diversity",
    },
    {
        "probe_id": "P03",
        "name": "single_baseline",
        "config": "configs/probe_P03_single_baseline.yaml",
        "description": "Single agent baseline",
        "variable": "none (control)",
    },
    {
        "probe_id": "P04",
        "name": "single_short_train",
        "config": "configs/probe_P04_single_short_train.yaml",
        "description": "Single agent, 30s training (task headroom)",
        "variable": "train_time_budget 30s vs 60s",
    },
    {
        "probe_id": "P05",
        "name": "memory_baseline",
        "config": "configs/probe_P05_memory_baseline.yaml",
        "description": "Single agent with memory",
        "variable": "none (control for memory)",
    },
    {
        "probe_id": "P06",
        "name": "shared_diverse",
        "config": "configs/probe_P06_shared_diverse.yaml",
        "description": "Parallel diverse + shared memory",
        "variable": "diversity + shared memory",
    },
]

WAVE_2 = [
    {
        "probe_id": "P07",
        "name": "shared_extended",
        "config": "configs/probe_P07_shared_extended.yaml",
        "description": "Shared memory + diverse, 30 min (best Wave 1 improvement rate)",
        "variable": "extended budget for shared+diverse",
    },
    {
        "probe_id": "P08",
        "name": "memory_extended",
        "config": "configs/probe_P08_memory_extended.yaml",
        "description": "Single + memory, 30 min (test memory degradation curve)",
        "variable": "extended budget for memory",
    },
    {
        "probe_id": "P09",
        "name": "diverse_extended",
        "config": "configs/probe_P09_diverse_extended.yaml",
        "description": "Parallel diverse, 30 min (test diversity compounding)",
        "variable": "extended budget for diversity",
    },
    {
        "probe_id": "P10",
        "name": "homo_fixed",
        "config": "configs/probe_P10_homo_fixed.yaml",
        "description": "Parallel homogeneous, 60s training (comparable P01 control)",
        "variable": "P01 control with fixed template",
    },
]

WAVE_3 = [
    {
        "probe_id": "P11",
        "name": "single_hightemp",
        "config": "configs/probe_P11_single_hightemp.yaml",
        "description": "Single agent, temp=1.2, 45 min (test high-temp compounding)",
        "variable": "high temperature + long budget",
    },
    {
        "probe_id": "P12",
        "name": "shared_fixed",
        "config": "configs/probe_P12_shared_fixed.yaml",
        "description": "Parallel shared + diverse, 45 min (FIRST REAL shared memory test)",
        "variable": "fixed shared memory mechanism",
    },
    {
        "probe_id": "P13",
        "name": "dual_hightemp",
        "config": "configs/probe_P13_dual_hightemp.yaml",
        "description": "Parallel diverse, both high temp (1.0/1.2), 45 min",
        "variable": "dual high temperature",
    },
    {
        "probe_id": "P14",
        "name": "hightemp_memory",
        "config": "configs/probe_P14_hightemp_memory.yaml",
        "description": "Single + memory, temp=1.2, 30 min (test memory + high temp)",
        "variable": "high temperature overcoming memory anchoring",
    },
]

WAVE_4 = [
    {
        "probe_id": "P15",
        "name": "seeded_guidance",
        "config": "configs/probe_P15_seeded_guidance.yaml",
        "description": "Single, temp=1.2, 45 min + LR hint in first message",
        "variable": "seeded search (explicit LR guidance)",
        "first_message_extra": (
            "\n\nIMPORTANT PRIOR KNOWLEDGE: In previous experiments on this exact task, "
            "the single most productive change was increasing LEARNING_RATE from 1e-3 to "
            "1.5e-3. This produced the best result ever (val_bpb=0.906). Start with that "
            "change, then explore further. Avoid large LR jumps (>3e-3) — they consistently "
            "make things worse."
        ),
    },
    {
        "probe_id": "P16",
        "name": "optimal_baseline",
        "config": "configs/probe_P16_optimal_baseline.yaml",
        "description": "Single, temp=0.5, 45 min, LR=1.5e-3 in template",
        "variable": "raised baseline (LR 1.5e-3 default)",
        "train_py_patches": {"LEARNING_RATE = 1e-3": "LEARNING_RATE = 1.5e-3"},
    },
    {
        "probe_id": "P17",
        "name": "full_stack",
        "config": "configs/probe_P17_full_stack.yaml",
        "description": "Parallel shared + private memory + diverse, 45 min",
        "variable": "full BP framework (G + ε)",
    },
    {
        "probe_id": "P18",
        "name": "seeded_parallel",
        "config": "configs/probe_P18_seeded_parallel.yaml",
        "description": "Parallel diverse, temp=1.0/1.2, 45 min + LR hint",
        "variable": "seeded parallel search",
        "first_message_extra": (
            "\n\nIMPORTANT PRIOR KNOWLEDGE: In previous experiments on this exact task, "
            "the single most productive change was increasing LEARNING_RATE from 1e-3 to "
            "1.5e-3. This produced the best result ever (val_bpb=0.906). Start with that "
            "change, then explore further. Avoid large LR jumps (>3e-3) — they consistently "
            "make things worse."
        ),
    },
]

WAVES = {1: WAVE_1, 2: WAVE_2, 3: WAVE_3, 4: WAVE_4}


def wait_for_clear_runway(repo_root: Path) -> None:
    """Wait until no other calibration/probe experiments are running."""
    print("Checking for running experiments...")
    while True:
        result = subprocess.run(
            ["pgrep", "-f", "run_calibration"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # No running calibration
            break
        print(f"  Waiting for background experiments to finish... "
              f"(PIDs: {result.stdout.strip()})")
        time.sleep(60)

    print("Runway clear. Starting probes.")


def _apply_train_py_patches(repo_root: Path, patches: dict[str, str]) -> str | None:
    """Apply patches to autoresearch/train.py, return original content for restore."""
    train_py = repo_root / "autoresearch" / "train.py"
    if not train_py.exists() or not patches:
        return None
    original = train_py.read_text()
    patched = original
    for old, new in patches.items():
        patched = patched.replace(old, new)
    if patched != original:
        train_py.write_text(patched)
        print(f"  Applied {len(patches)} train.py patch(es)")
        return original
    return None


def _restore_train_py(repo_root: Path, original: str | None) -> None:
    """Restore original autoresearch/train.py content."""
    if original is None:
        return
    train_py = repo_root / "autoresearch" / "train.py"
    train_py.write_text(original)
    print(f"  Restored original train.py")


def run_single_probe(
    repo_root: Path,
    probe: dict,
) -> dict:
    """Run a single probe experiment."""
    probe_id = probe["probe_id"]
    experiment_id = f"probe_{probe_id}"
    config_path = probe["config"]

    print(f"\n{'='*60}")
    print(f"  PROBE {probe_id}: {probe['description']}")
    print(f"  Variable: {probe['variable']}")
    print(f"  Config: {config_path}")
    if probe.get("first_message_extra"):
        print(f"  First message extra: YES ({len(probe['first_message_extra'])} chars)")
    if probe.get("train_py_patches"):
        print(f"  Train.py patches: {list(probe['train_py_patches'].keys())}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    # Check if already done
    experiment_dir = repo_root / "runs" / f"experiment_{experiment_id}"
    if any(experiment_dir.rglob("training_runs.jsonl")):
        existing_runs = sum(
            1 for jf in experiment_dir.rglob("training_runs.jsonl")
            for line in jf.read_text().splitlines() if line.strip()
        )
        if existing_runs > 0:
            print(f"  SKIP {experiment_id} — already has {existing_runs} results")
            return {
                "probe_id": probe_id,
                "experiment_id": experiment_id,
                "status": "skipped",
                "n_runs": existing_runs,
            }

    # Apply train.py patches (restored after probe completes)
    train_py_original = _apply_train_py_patches(
        repo_root, probe.get("train_py_patches", {})
    )

    # Apply first_message_extra by appending to the first message template
    first_msg_original = None
    if probe.get("first_message_extra"):
        first_msg_path = repo_root / "templates" / "agent_first_message.md"
        if first_msg_path.exists():
            first_msg_original = first_msg_path.read_text()
            first_msg_path.write_text(first_msg_original + probe["first_message_extra"])
            print(f"  Injected first_message_extra into template")

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

    def _restore_all():
        _restore_train_py(repo_root, train_py_original)
        if first_msg_original is not None:
            first_msg_path = repo_root / "templates" / "agent_first_message.md"
            first_msg_path.write_text(first_msg_original)
            print(f"  Restored original first_message template")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=False,
            timeout=3600,  # 1 hour hard limit
        )
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        _restore_all()
        print(f"\n  TIMEOUT after {elapsed:.0f}s")
        return {
            "probe_id": probe_id,
            "experiment_id": experiment_id,
            "status": "timeout",
            "elapsed_seconds": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - start
        _restore_all()
        print(f"\n  ERROR: {e}")
        return {
            "probe_id": probe_id,
            "experiment_id": experiment_id,
            "status": "error",
            "error": str(e),
            "elapsed_seconds": round(elapsed, 1),
        }

    # Restore patched files
    _restore_all()

    # Collect results
    experiment_dir = repo_root / "runs" / f"experiment_{experiment_id}"
    n_runs = 0
    best_vbpb = 999
    for jf in experiment_dir.rglob("training_runs.jsonl"):
        for line in jf.read_text().splitlines():
            if not line.strip():
                continue
            n_runs += 1
            try:
                d = json.loads(line)
                v = d.get("val_bpb")
                if v is not None and v < best_vbpb:
                    best_vbpb = v
            except json.JSONDecodeError:
                pass

    status = "ok" if n_runs > 0 else "no_results"
    print(f"\n  {experiment_id}: {status} — {n_runs} runs, "
          f"best={best_vbpb:.6f}, elapsed={elapsed:.0f}s")

    return {
        "probe_id": probe_id,
        "experiment_id": experiment_id,
        "name": probe["name"],
        "description": probe["description"],
        "variable": probe["variable"],
        "status": status,
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "n_runs": n_runs,
        "best_val_bpb": best_vbpb if best_vbpb < 999 else None,
    }


def quick_analyze(repo_root: Path, probe_results: list[dict]) -> str:
    """Quick analysis of probe results for the log."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("QUICK ANALYSIS")
    lines.append("=" * 60)

    baseline_vbpb = 0.925845

    for pr in probe_results:
        if pr["status"] in ("skipped", "timeout", "error"):
            lines.append(f"\n{pr['probe_id']}: {pr['status']}")
            continue

        experiment_dir = repo_root / "runs" / f"experiment_{pr['experiment_id']}"
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

        if not runs:
            lines.append(f"\n{pr['probe_id']}: no data")
            continue

        # Strategy analysis
        non_baseline = [r for r in runs if not r.get("baseline_candidate", False)]
        cats = [r.get("strategy_category", "unknown") for r in non_baseline]
        from collections import Counter
        cat_counts = Counter(cats)
        n_unique = len(cat_counts)

        import math
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

        # Jaccard (for parallel probes)
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
            agent_cat_seq = [r.get("strategy_category", "unknown") for r in agent_runs]
            for i in range(1, len(agent_cat_seq)):
                switch_total += 1
                if agent_cat_seq[i] != agent_cat_seq[i - 1]:
                    switch_count += 1
        switch_rate = switch_count / switch_total if switch_total > 0 else 0

        # Improvements
        improvements = [r for r in non_baseline
                        if r.get("val_bpb") is not None
                        and r["val_bpb"] < baseline_vbpb]

        # Training time
        train_times = [r.get("training_seconds", 0) for r in runs
                       if r.get("training_seconds")]
        mean_train = sum(train_times) / len(train_times) if train_times else 0

        lines.append(f"\n{pr['probe_id']} ({pr['name']}):")
        lines.append(f"  Runs: {len(runs)} ({len(non_baseline)} non-baseline)")
        lines.append(f"  Strategy categories: {n_unique}, entropy={entropy:.3f}")
        lines.append(f"  Distribution: {dict(cat_counts)}")
        if jaccard is not None:
            lines.append(f"  Agent Jaccard similarity: {jaccard:.2f}")
            for agent in agents:
                lines.append(f"    {agent}: {sorted(agent_cats.get(agent, set()))}")
        lines.append(f"  Switch rate: {switch_rate:.2f} ({switch_count}/{switch_total})")
        lines.append(f"  Improvements: {len(improvements)}/{len(non_baseline)}")
        lines.append(f"  Best val_bpb: {pr.get('best_val_bpb', 'N/A')}")
        lines.append(f"  Mean training time: {mean_train:.1f}s")
        lines.append(f"  Elapsed: {pr.get('elapsed_seconds', 0):.0f}s")

    # Pair comparisons
    lines.append("\n" + "-" * 40)
    lines.append("PAIRED COMPARISONS")
    lines.append("-" * 40)

    def get_result(pid):
        for pr in probe_results:
            if pr["probe_id"] == pid:
                return pr
        return None

    # P01 vs P02 (diversity)
    p01 = get_result("P01")
    p02 = get_result("P02")
    if p01 and p02 and p01["status"] == "ok" and p02["status"] == "ok":
        lines.append(f"\nDiversity test (P01 homo vs P02 diverse):")
        lines.append(f"  P01 runs={p01['n_runs']}, P02 runs={p02['n_runs']}")
        lines.append(f"  → Check agent Jaccard above: lower = more diverse")

    # P03 vs P04 (training budget)
    p03 = get_result("P03")
    p04 = get_result("P04")
    if p03 and p04 and p03["status"] == "ok" and p04["status"] == "ok":
        lines.append(f"\nTraining budget test (P03 60s vs P04 30s):")
        lines.append(f"  P03 runs={p03['n_runs']}, P04 runs={p04['n_runs']}")
        ratio = p04["n_runs"] / p03["n_runs"] if p03["n_runs"] > 0 else 0
        lines.append(f"  Iteration ratio: {ratio:.2f}x")

    # P05 vs P06 (memory + diversity)
    p05 = get_result("P05")
    p06 = get_result("P06")
    if p05 and p06 and p05["status"] == "ok" and p06["status"] == "ok":
        lines.append(f"\nMemory + diversity test (P05 single_mem vs P06 shared_diverse):")
        lines.append(f"  P05 runs={p05['n_runs']}, P06 runs={p06['n_runs']}")

    return "\n".join(lines)


def write_log(repo_root: Path, wave: int, probe_results: list[dict],
              analysis: str) -> None:
    """Write the execution log."""
    log_path = repo_root / "workflow" / "logs" / f"probe_wave{wave}_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Probe Wave {wave} Execution Log")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n## Probe Results\n")
    lines.append("| Probe | Name | Status | Runs | Best val_bpb | Time |")
    lines.append("|-------|------|--------|------|-------------|------|")
    for pr in probe_results:
        vbpb = f"{pr.get('best_val_bpb', 'N/A'):.6f}" if pr.get("best_val_bpb") else "N/A"
        elapsed = f"{pr.get('elapsed_seconds', 0):.0f}s"
        lines.append(f"| {pr['probe_id']} | {pr.get('name', '')} | "
                      f"{pr['status']} | {pr.get('n_runs', '-')} | {vbpb} | {elapsed} |")

    lines.append(f"\n## Analysis\n```\n{analysis}\n```")

    # Save raw results as JSON too
    json_path = repo_root / "workflow" / "artifacts" / f"probe_wave{wave}_results.json"
    json_path.write_text(json.dumps(probe_results, indent=2, default=str) + "\n")

    log_path.write_text("\n".join(lines) + "\n")
    print(f"\nLog written to: {log_path}")
    print(f"Results written to: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probe experiments")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--wave", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--probes", nargs="+", default=None,
                        help="Run specific probes (e.g., P01 P02)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for other experiments to finish")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    # Get wave probes
    wave_probes = WAVES.get(args.wave, [])
    if not wave_probes:
        print(f"Wave {args.wave} not defined yet. Available: {list(WAVES.keys())}")
        sys.exit(1)

    # Filter to specific probes if requested
    if args.probes:
        wave_probes = [p for p in wave_probes if p["probe_id"] in args.probes]
        if not wave_probes:
            print(f"No matching probes found for: {args.probes}")
            sys.exit(1)

    print(f"Phase 04 Probe Wave {args.wave}")
    print(f"  Probes: {[p['probe_id'] for p in wave_probes]}")
    print(f"  Total: {len(wave_probes)} experiments")
    print()

    # Wait for clear runway
    if not args.no_wait:
        wait_for_clear_runway(repo_root)

    # Run probes sequentially
    all_results = []
    for probe in wave_probes:
        result = run_single_probe(repo_root, probe)
        all_results.append(result)

        # Quick status after each probe
        print(f"\n  Completed {len(all_results)}/{len(wave_probes)} probes")

    # Analyze and log
    analysis = quick_analyze(repo_root, all_results)
    print(analysis)
    write_log(repo_root, args.wave, all_results, analysis)

    print(f"\n{'='*60}")
    print(f"Wave {args.wave} complete.")
    print(f"  Successful: {sum(1 for r in all_results if r['status'] == 'ok')}/{len(all_results)}")
    print(f"  Total runs: {sum(r.get('n_runs', 0) for r in all_results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
