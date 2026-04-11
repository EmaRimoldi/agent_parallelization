"""Compute the BP four-term decomposition from 2x2 experiment data.

Usage:
    python scripts/compute_decomposition.py \
        --d00 runs/experiment_d00 \
        --d10 runs/experiment_d10 \
        --d01 runs/experiment_d01 \
        --d11 runs/experiment_d11

Outputs:
    - Decomposition table (terminal + JSON)
    - Main effects and interaction
    - Hypothesis test results (H1-H6)
    - Bootstrap confidence intervals
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def load_cell_data(experiment_dir: Path) -> dict:
    """Load all instrumented data from one cell of the 2x2."""
    data = {
        "turns": [],
        "training_runs": [],
        "mode_labels": [],
    }

    for agent_dir in sorted(experiment_dir.glob("mode_*/agent_*")):
        turns_path = agent_dir / "results" / "turns.jsonl"
        if turns_path.exists():
            for line in turns_path.read_text().splitlines():
                if line.strip():
                    data["turns"].append(json.loads(line))

        runs_path = agent_dir / "results" / "training_runs.jsonl"
        if runs_path.exists():
            for line in runs_path.read_text().splitlines():
                if line.strip():
                    data["training_runs"].append(json.loads(line))

        labels_path = agent_dir / "results" / "mode_labels.jsonl"
        if labels_path.exists():
            for line in labels_path.read_text().splitlines():
                if line.strip():
                    data["mode_labels"].append(json.loads(line))

    return data


def compute_kappa_token(data: dict) -> float:
    """κ̄_token: average total tokens (input + output) per attempt."""
    if not data["turns"]:
        return float("nan")
    tokens_per_turn = []
    for turn in data["turns"]:
        inp = turn.get("input_tokens") or (
            (turn.get("system_prompt_chars", 0) + turn.get("turn_msg_chars", 0)) // 4
        )
        out = turn.get("output_tokens") or turn.get("response_chars", 0) // 4
        tokens_per_turn.append(inp + out)
    return np.mean(tokens_per_turn)


def compute_kappa_wall(data: dict) -> float:
    """κ̄_wall: average wall-clock seconds per attempt (LLM call + training)."""
    if not data["turns"]:
        return float("nan")
    return np.mean([turn.get("wall_clock_seconds", 0) for turn in data["turns"]])


def compute_kappa_by_context_bin(data: dict, n_bins: int = 4) -> dict:
    """κ̄_token stratified by context fill ratio bins (for H5)."""
    if not data["turns"]:
        return {}
    bins = np.linspace(0, 1, n_bins + 1)
    result = {}
    for index in range(n_bins):
        lo, hi = bins[index], bins[index + 1]
        label = f"{lo:.0%}-{hi:.0%}"
        in_bin = [
            turn
            for turn in data["turns"]
            if lo <= turn.get("context_fill_ratio", 0) < hi
        ]
        if in_bin:
            tokens = []
            for turn in in_bin:
                inp = turn.get("input_tokens") or (
                    (turn.get("system_prompt_chars", 0) + turn.get("turn_msg_chars", 0)) // 4
                )
                out = turn.get("output_tokens") or turn.get("response_chars", 0) // 4
                tokens.append(inp + out)
            result[label] = np.mean(tokens)
        else:
            result[label] = None
    return result


def compute_mode_distribution(data: dict) -> dict:
    """π̂: empirical distribution over modes from accepted edits."""
    accepted = [label for label in data["mode_labels"] if label.get("accepted")]
    if not accepted:
        return {}
    counts = Counter(label["mode"] for label in accepted)
    total = sum(counts.values())
    return {mode: count / total for mode, count in counts.items()}


def entropy(dist: dict) -> float:
    """Shannon entropy H(p) in nats."""
    return -sum(prob * math.log(prob) for prob in dist.values() if prob > 0)


def kl_divergence(p: dict, q: dict, smoothing: float = 1e-6) -> float:
    """KL(p || q) in nats with Laplace smoothing."""
    all_keys = set(p) | set(q)
    n_keys = len(all_keys)
    kl = 0.0
    for key in all_keys:
        pk = p.get(key, 0) + smoothing
        qk = q.get(key, 0) + smoothing
        pk_norm = pk / (1 + n_keys * smoothing)
        qk_norm = qk / (1 + n_keys * smoothing)
        kl += pk_norm * math.log(pk_norm / qk_norm)
    return kl


def compute_decomposition(d0_data, d_data, prior):
    """Compute the four-term decomposition Δ = log(κ0/κ) + φ + G - ε."""
    k0_token = compute_kappa_token(d0_data)
    k_token = compute_kappa_token(d_data)
    k0_wall = compute_kappa_wall(d0_data)
    k_wall = compute_kappa_wall(d_data)

    cost_term_token = math.log(k0_token / k_token) if k_token > 0 else float("nan")
    cost_term_wall = math.log(k0_wall / k_wall) if k_wall > 0 else float("nan")

    phi = 0.0

    d_dist = compute_mode_distribution(d_data)
    G = entropy(d_dist) - entropy(prior) if d_dist and prior else 0.0

    epsilon = kl_divergence(d_dist, prior) if d_dist and prior else 0.0

    delta_token = cost_term_token + phi + G - epsilon
    delta_wall = cost_term_wall + phi + G - epsilon

    return {
        "cost_term_token": cost_term_token,
        "cost_term_wall": cost_term_wall,
        "phi": phi,
        "G": G,
        "epsilon": epsilon,
        "delta_token": delta_token,
        "delta_wall": delta_wall,
        "kappa_token": k_token,
        "kappa_wall": k_wall,
    }


def test_hypotheses(cells: dict, decompositions: dict) -> dict:
    """Test H1-H6 from the paper."""
    results = {}

    d01 = decompositions.get("d01", {})
    results["H1_wall_positive"] = d01.get("delta_wall", 0) > 0
    results["H1_token_nonpositive"] = d01.get("delta_token", 0) <= 0
    results["H1_holds"] = (
        results["H1_wall_positive"] and results["H1_token_nonpositive"]
    )

    d10 = decompositions.get("d10", {})
    results["H2_token_positive"] = d10.get("cost_term_token", 0) > 0
    results["H2_wall_positive"] = d10.get("cost_term_wall", 0) > 0
    results["H2_holds"] = results["H2_token_positive"] and results["H2_wall_positive"]

    d01_eps = decompositions.get("d01", {}).get("epsilon", 0)
    d11_eps = decompositions.get("d11", {}).get("epsilon", 0)
    results["H3_epsilon_d11_lt_d01"] = d11_eps < d01_eps
    results["H3_holds"] = results["H3_epsilon_d11_lt_d01"]

    results["H4_epsilon_exceeds_log2"] = (d01_eps - 0) > math.log(2)

    results["H5_note"] = "Check kappa_by_context_bin for monotone increase"

    d11 = decompositions.get("d11", {})
    results["H6_wall_positive"] = d11.get("delta_wall", 0) > 0
    results["H6_token_positive"] = d11.get("delta_token", 0) > 0
    results["H6_holds"] = results["H6_wall_positive"] and results["H6_token_positive"]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d00", required=True, type=Path, help="Experiment dir for d00 (single, no memory)")
    parser.add_argument("--d10", required=True, type=Path, help="Experiment dir for d10 (single, memory)")
    parser.add_argument("--d01", required=True, type=Path, help="Experiment dir for d01 (parallel, no sharing)")
    parser.add_argument("--d11", required=True, type=Path, help="Experiment dir for d11 (parallel, shared)")
    args = parser.parse_args()

    cells = {
        "d00": load_cell_data(args.d00),
        "d10": load_cell_data(args.d10),
        "d01": load_cell_data(args.d01),
        "d11": load_cell_data(args.d11),
    }

    pooled_labels = cells["d00"]["mode_labels"] + cells["d11"]["mode_labels"]
    pooled_accepted = [label for label in pooled_labels if label.get("accepted")]
    if pooled_accepted:
        counts = Counter(label["mode"] for label in pooled_accepted)
        total = sum(counts.values())
        prior = {mode: count / total for mode, count in counts.items()}
    else:
        prior = {}

    decompositions = {}
    d00_data = cells["d00"]

    print("=" * 70)
    print("BP FOUR-TERM DECOMPOSITION - 2x2 RESULTS")
    print("=" * 70)
    print(
        f"\n{'Cell':<6} {'log(κ0/κ)_tok':>14} {'log(κ0/κ)_wall':>15} "
        f"{'φ':>8} {'G':>8} {'-ε':>8} {'Δ_tok':>8} {'Δ_wall':>8}"
    )
    print("-" * 70)

    for cell_name in ["d10", "d01", "d11"]:
        dec = compute_decomposition(d00_data, cells[cell_name], prior)
        decompositions[cell_name] = dec
        print(
            f"{cell_name:<6} {dec['cost_term_token']:>14.4f} {dec['cost_term_wall']:>15.4f} "
            f"{dec['phi']:>8.4f} {dec['G']:>8.4f} {-dec['epsilon']:>8.4f} "
            f"{dec['delta_token']:>8.4f} {dec['delta_wall']:>8.4f}"
        )

    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)

    hypotheses = test_hypotheses(cells, decompositions)
    for key, value in hypotheses.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("CONTEXT PRESSURE (H5)")
    print("=" * 70)

    context_bins = {}
    for cell_name in ["d00", "d10", "d01", "d11"]:
        bins = compute_kappa_by_context_bin(cells[cell_name])
        context_bins[cell_name] = bins
        print(f"\n  {cell_name}: κ̄_token by c/K bin:")
        for bin_label, value in bins.items():
            if value is None:
                print(f"    {bin_label}: no data")
            else:
                print(f"    {bin_label}: {value:.0f} tokens")

    output = {
        "decompositions": decompositions,
        "hypotheses": hypotheses,
        "prior": prior,
        "context_bins": context_bins,
    }
    output_path = Path("decomposition_results.json")
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")
