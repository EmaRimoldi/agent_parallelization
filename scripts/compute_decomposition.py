"""Compute a structurally aligned BP-style decomposition from 2x2 experiment data.

Usage:
    python scripts/compute_decomposition.py \
        --d00 runs/experiment_d00 \
        --d10 runs/experiment_d10 \
        --d01 runs/experiment_d01 \
        --d11 runs/experiment_d11 \
        --output decomposition_results.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


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
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_cell_data(experiment_dir: Path) -> dict:
    """Load all instrumented data from one cell of the 2x2."""
    data = {
        "turns": [],
        "training_runs": [],
        "mode_labels": [],
    }

    for agent_dir in sorted(experiment_dir.glob("mode_*/agent_*")):
        for row in load_jsonl(agent_dir / "results" / "turns.jsonl"):
            data["turns"].append(row)

        for row in load_jsonl(agent_dir / "results" / "training_runs.jsonl"):
            if is_credible_training_run(row):
                data["training_runs"].append(row)

        for row in load_jsonl(agent_dir / "results" / "mode_labels.jsonl"):
            data["mode_labels"].append(row)

    return data


def normalize_counter(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def chars4_proxy(turn: dict) -> float:
    return (
        float(turn.get("system_prompt_chars", 0))
        + float(turn.get("turn_msg_chars", 0))
        + float(turn.get("response_chars", 0))
    ) / 4.0


def estimate_token_proxy_calibration(turns: list[dict]) -> dict[str, float | int]:
    """Estimate how close chars//4 is to observed tokens on turns that have both."""
    ratios: list[float] = []
    turns_with_observed_tokens = 0

    for turn in turns:
        observed = (turn.get("input_tokens") or 0) + (turn.get("output_tokens") or 0)
        proxy = chars4_proxy(turn)
        if observed and proxy > 0:
            turns_with_observed_tokens += 1
            ratios.append(float(observed) / proxy)

    calibration_factor = float(np.median(ratios)) if ratios else 1.0
    return {
        "calibration_factor": calibration_factor,
        "turns_with_observed_tokens": turns_with_observed_tokens,
    }


def estimate_turn_tokens(turn: dict, calibration_factor: float) -> float:
    observed = (turn.get("input_tokens") or 0) + (turn.get("output_tokens") or 0)
    if observed:
        return float(observed)
    return chars4_proxy(turn) * calibration_factor


def compute_kappa_token_summary(data: dict) -> dict[str, float | int]:
    """κ̄_token plus variance summary, using observed tokens when available."""
    turns = data["turns"]
    if not turns:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "calibration_factor": 1.0,
            "turn_count": 0,
        }

    calibration = estimate_token_proxy_calibration(turns)
    factor = float(calibration["calibration_factor"])
    token_values = [estimate_turn_tokens(turn, factor) for turn in turns]
    return {
        "mean": safe_mean(token_values),
        "std": safe_std(token_values),
        "calibration_factor": factor,
        "turn_count": len(token_values),
        "turns_with_observed_tokens": int(calibration["turns_with_observed_tokens"]),
    }


def compute_kappa_wall_summary(data: dict) -> dict[str, float | int]:
    """κ̄_wall plus variance summary from turn-level wall-clock."""
    values = [
        float(turn["wall_clock_seconds"])
        for turn in data["turns"]
        if turn.get("wall_clock_seconds") is not None
    ]
    return {
        "mean": safe_mean(values),
        "std": safe_std(values),
        "turn_count": len(values),
    }


def compute_kappa_by_context_bin(data: dict, n_bins: int = 4) -> dict:
    """κ̄_token stratified by context fill ratio bins (for H5)."""
    if not data["turns"]:
        return {}
    calibration = estimate_token_proxy_calibration(data["turns"])
    factor = float(calibration["calibration_factor"])
    bins = np.linspace(0, 1, n_bins + 1)
    result = {}
    for index in range(n_bins):
        lo, hi = bins[index], bins[index + 1]
        label = f"{lo:.0%}-{hi:.0%}"
        in_bin = [
            turn
            for turn in data["turns"]
            if lo <= float(turn.get("context_fill_ratio", 0.0)) < hi
        ]
        if in_bin:
            values = [estimate_turn_tokens(turn, factor) for turn in in_bin]
            result[label] = safe_mean(values)
        else:
            result[label] = None
    return result


def compute_mode_distribution(
    labels: list[dict],
    *,
    accepted_only: bool,
) -> dict[str, float]:
    rows = labels
    if accepted_only:
        rows = [row for row in labels if row.get("accepted") is True]
    counts = Counter(str(row.get("mode", "other")) for row in rows if row.get("mode"))
    return normalize_counter(counts)


def smoothed_kl_divergence(p: dict[str, float], q: dict[str, float], smoothing: float = 1e-6) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return float("nan")

    p_mass = sum(p.get(key, 0.0) for key in keys)
    q_mass = sum(q.get(key, 0.0) for key in keys)
    if p_mass <= 0 or q_mass <= 0:
        return float("nan")

    p_total = p_mass + smoothing * len(keys)
    q_total = q_mass + smoothing * len(keys)

    kl = 0.0
    for key in keys:
        pk = (p.get(key, 0.0) + smoothing) / p_total
        qk = (q.get(key, 0.0) + smoothing) / q_total
        kl += pk * math.log(pk / qk)
    return kl


def compute_global_prior(cells: dict[str, dict]) -> dict[str, float]:
    accepted = Counter()
    for cell_data in cells.values():
        for row in cell_data["mode_labels"]:
            if row.get("accepted") is True and row.get("mode"):
                accepted[str(row["mode"])] += 1
    if accepted:
        return normalize_counter(accepted)

    proposals = Counter()
    for cell_data in cells.values():
        for row in cell_data["mode_labels"]:
            if row.get("mode"):
                proposals[str(row["mode"])] += 1
    return normalize_counter(proposals)


def compute_first_success_steps(data: dict) -> dict[str, float]:
    first_steps: dict[str, float] = {}
    for row in data["mode_labels"]:
        if row.get("accepted") is not True:
            continue
        mode = row.get("mode")
        step = row.get("step")
        if mode is None:
            continue
        try:
            step_value = float(step) + 1.0
        except (TypeError, ValueError):
            continue
        key = str(mode)
        current = first_steps.get(key)
        if current is None or step_value < current:
            first_steps[key] = step_value
    return first_steps


def compute_phi_estimate(d0_data: dict, d_data: dict, global_prior: dict[str, float]) -> tuple[float, dict]:
    """Proxy for φ using mode-conditional attempts-to-first-success."""
    baseline_steps = compute_first_success_steps(d0_data)
    design_steps = compute_first_success_steps(d_data)
    overlap = sorted(set(baseline_steps) & set(design_steps))
    if not overlap:
        return float("nan"), {"status": "insufficient_mode_overlap", "overlap_modes": []}

    weights = {
        mode: global_prior.get(mode, 0.0)
        for mode in overlap
    }
    weight_total = sum(weights.values())
    if weight_total <= 0:
        weights = {mode: 1.0 / len(overlap) for mode in overlap}
    else:
        weights = {mode: value / weight_total for mode, value in weights.items()}

    contributions = {
        mode: math.log(baseline_steps[mode] / design_steps[mode])
        for mode in overlap
        if baseline_steps[mode] > 0 and design_steps[mode] > 0
    }
    if not contributions:
        return float("nan"), {"status": "nonpositive_steps", "overlap_modes": overlap}

    phi = sum(weights[mode] * contributions[mode] for mode in contributions)
    return phi, {
        "status": "ok",
        "overlap_modes": overlap,
        "weights": weights,
        "per_mode_log_ratio": contributions,
        "baseline_first_success_step": baseline_steps,
        "design_first_success_step": design_steps,
    }


def compute_information_generation(
    d_data: dict,
    global_prior: dict[str, float],
) -> tuple[float, dict]:
    """Per-design pointwise mutual-information contribution KL(pi_D || pi_global)."""
    posterior = compute_mode_distribution(d_data["mode_labels"], accepted_only=True)
    if not posterior or not global_prior:
        return float("nan"), {"status": "missing_posterior_or_prior"}
    value = smoothed_kl_divergence(posterior, global_prior)
    return value, {
        "status": "ok",
        "posterior_mode_distribution": posterior,
        "global_prior_mode_distribution": global_prior,
        "estimator": "pointwise_mi_contribution",
    }


def compute_routing_mismatch(d_data: dict) -> tuple[float, dict]:
    """Proxy epsilon = KL(pi_D || q_D) with proposal vs accepted mode distributions."""
    posterior = compute_mode_distribution(d_data["mode_labels"], accepted_only=True)
    routing = compute_mode_distribution(d_data["mode_labels"], accepted_only=False)
    if not posterior:
        return float("nan"), {"status": "missing_posterior"}
    if not routing:
        return float("nan"), {"status": "missing_routing_distribution"}
    value = smoothed_kl_divergence(posterior, routing)
    return value, {
        "status": "ok",
        "posterior_mode_distribution": posterior,
        "routing_mode_distribution": routing,
        "routing_proxy": "all proposed mode labels",
    }


def combine_terms(*values: float) -> float:
    return sum(values) if all(is_finite_number(value) for value in values) else float("nan")


def compute_decomposition(d0_data: dict, d_data: dict, global_prior: dict[str, float]) -> dict:
    """Compute a structurally aligned BP-style decomposition."""
    token0 = compute_kappa_token_summary(d0_data)
    token_d = compute_kappa_token_summary(d_data)
    wall0 = compute_kappa_wall_summary(d0_data)
    wall_d = compute_kappa_wall_summary(d_data)

    k0_token = float(token0["mean"])
    k_token = float(token_d["mean"])
    k0_wall = float(wall0["mean"])
    k_wall = float(wall_d["mean"])

    cost_term_token = (
        math.log(k0_token / k_token)
        if is_finite_number(k0_token) and is_finite_number(k_token) and k0_token > 0 and k_token > 0
        else float("nan")
    )
    cost_term_wall = (
        math.log(k0_wall / k_wall)
        if is_finite_number(k0_wall) and is_finite_number(k_wall) and k0_wall > 0 and k_wall > 0
        else float("nan")
    )

    phi, phi_details = compute_phi_estimate(d0_data, d_data, global_prior)
    G, G_details = compute_information_generation(d_data, global_prior)
    epsilon, epsilon_details = compute_routing_mismatch(d_data)

    delta_token = combine_terms(cost_term_token, phi, G, -epsilon if is_finite_number(epsilon) else float("nan"))
    delta_wall = combine_terms(cost_term_wall, phi, G, -epsilon if is_finite_number(epsilon) else float("nan"))

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
        "kappa_token_std": float(token_d["std"]),
        "kappa_wall_std": float(wall_d["std"]),
        "token_calibration_factor": float(token_d["calibration_factor"]),
        "posterior_mode_distribution": G_details.get("posterior_mode_distribution"),
        "routing_mode_distribution": epsilon_details.get("routing_mode_distribution"),
        "phi_details": phi_details,
        "G_details": G_details,
        "epsilon_details": epsilon_details,
    }


def test_hypotheses(cells: dict, decompositions: dict) -> dict:
    """Test H1-H6 from the paper with NaN-safe handling."""
    results = {}

    d01 = decompositions.get("d01", {})
    d01_delta_wall = d01.get("delta_wall")
    d01_delta_token = d01.get("delta_token")
    results["H1_wall_positive"] = is_finite_number(d01_delta_wall) and d01_delta_wall > 0
    results["H1_token_nonpositive"] = is_finite_number(d01_delta_token) and d01_delta_token <= 0
    results["H1_holds"] = results["H1_wall_positive"] and results["H1_token_nonpositive"]

    d10 = decompositions.get("d10", {})
    d10_cost_tok = d10.get("cost_term_token")
    d10_cost_wall = d10.get("cost_term_wall")
    results["H2_token_positive"] = is_finite_number(d10_cost_tok) and d10_cost_tok > 0
    results["H2_wall_positive"] = is_finite_number(d10_cost_wall) and d10_cost_wall > 0
    results["H2_holds"] = results["H2_token_positive"] and results["H2_wall_positive"]

    d01_eps = decompositions.get("d01", {}).get("epsilon")
    d11_eps = decompositions.get("d11", {}).get("epsilon")
    results["H3_epsilon_d11_lt_d01"] = (
        is_finite_number(d11_eps) and is_finite_number(d01_eps) and d11_eps < d01_eps
    )
    results["H3_holds"] = results["H3_epsilon_d11_lt_d01"]

    results["H4_epsilon_exceeds_log2"] = (
        is_finite_number(d01_eps) and d01_eps > math.log(2)
    )

    results["H5_note"] = "Inspect context_bins and token calibration for monotone context-pressure growth."

    d11 = decompositions.get("d11", {})
    d11_delta_wall = d11.get("delta_wall")
    d11_delta_token = d11.get("delta_token")
    results["H6_wall_positive"] = is_finite_number(d11_delta_wall) and d11_delta_wall > 0
    results["H6_token_positive"] = is_finite_number(d11_delta_token) and d11_delta_token > 0
    results["H6_holds"] = results["H6_wall_positive"] and results["H6_token_positive"]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d00", required=True, type=Path, help="Experiment dir for d00 (single, no memory)")
    parser.add_argument("--d10", required=True, type=Path, help="Experiment dir for d10 (single, memory)")
    parser.add_argument("--d01", required=True, type=Path, help="Experiment dir for d01 (parallel, no sharing)")
    parser.add_argument("--d11", required=True, type=Path, help="Experiment dir for d11 (parallel, shared)")
    parser.add_argument("--output", type=Path, default=Path("decomposition_results.json"))
    args = parser.parse_args()

    cells = {
        "d00": load_cell_data(args.d00),
        "d10": load_cell_data(args.d10),
        "d01": load_cell_data(args.d01),
        "d11": load_cell_data(args.d11),
    }

    global_prior = compute_global_prior(cells)
    decompositions = {}
    d00_data = cells["d00"]

    print("=" * 78)
    print("BP-STYLE DECOMPOSITION - 2x2 RESULTS")
    print("=" * 78)
    print(
        f"\n{'Cell':<6} {'log(κ0/κ)_tok':>14} {'log(κ0/κ)_wall':>15} "
        f"{'phi':>8} {'G':>8} {'-eps':>8} {'Δ_tok':>8} {'Δ_wall':>8}"
    )
    print("-" * 78)

    for cell_name in ["d10", "d01", "d11"]:
        dec = compute_decomposition(d00_data, cells[cell_name], global_prior)
        decompositions[cell_name] = dec
        print(
            f"{cell_name:<6} {dec['cost_term_token']:>14.4f} {dec['cost_term_wall']:>15.4f} "
            f"{dec['phi']:>8.4f} {dec['G']:>8.4f} "
            f"{(-dec['epsilon']) if is_finite_number(dec['epsilon']) else float('nan'):>8.4f} "
            f"{dec['delta_token']:>8.4f} {dec['delta_wall']:>8.4f}"
        )

    print("\n" + "=" * 78)
    print("HYPOTHESIS TESTS")
    print("=" * 78)
    hypotheses = test_hypotheses(cells, decompositions)
    for key, value in hypotheses.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 78)
    print("CONTEXT PRESSURE (H5)")
    print("=" * 78)
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
        "global_prior": global_prior,
        "context_bins": context_bins,
    }
    with open(args.output, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nFull results saved to {args.output}")
