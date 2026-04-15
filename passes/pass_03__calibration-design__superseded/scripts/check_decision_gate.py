#!/usr/bin/env python3
"""Evaluate the Phase 02b decision gate criteria from calibration analysis.

Reads calibration_analysis.json and prints the recommended decision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path,
                        help="Path to calibration_analysis.json")
    args = parser.parse_args()

    data = json.loads(args.analysis.read_text())

    d = data.get("cohens_d", float("nan"))
    d00_modes = data.get("d00", {}).get("modes_with_2plus_accepted", 0)
    d10_modes = data.get("d10", {}).get("modes_with_2plus_accepted", 0)
    d00_n = data.get("d00", {}).get("n_runs", 0)
    d10_n = data.get("d10", {}).get("n_runs", 0)
    min_modes = min(d00_modes, d10_modes)
    min_n = min(d00_n, d10_n)

    print(f"Decision Gate Evaluation:")
    print(f"  Cohen's d:       {d:.3f}")
    print(f"  Min mode count:  {min_modes} (d00={d00_modes}, d10={d10_modes})")
    print(f"  Min sample size: {min_n} (d00={d00_n}, d10={d10_n})")
    print()

    if math.isnan(d):
        decision = "extend_budget"
        reason = "Cohen's d is NaN — insufficient data"
    elif abs(d) <= 0.1 and min_modes <= 1:
        decision = "structured_search"
        reason = (f"Negligible effect (d={d:.3f}) AND degenerate modes "
                  f"({min_modes} modes) — free-form search is not working")
    elif abs(d) <= 0.3 and min_n >= 50:
        decision = "escalate_cifar100"
        reason = (f"Small effect (d={d:.3f}) with adequate sample size "
                  f"(n≥{min_n}) — task is too simple")
    elif abs(d) > 0.3 and min_modes >= 3 and min_n >= 50:
        decision = "proceed"
        reason = (f"Adequate effect (d={d:.3f}), mode diversity "
                  f"({min_modes} modes), and sample size (n≥{min_n})")
    elif abs(d) > 0.3 and (min_modes < 3 or min_n < 30):
        decision = "extend_budget"
        reason = (f"Effect detected (d={d:.3f}) but insufficient "
                  f"modes ({min_modes}) or samples ({min_n})")
    else:
        decision = "extend_budget"
        reason = "Borderline criteria — extending budget is safest"

    print(f"  DECISION: {decision}")
    print(f"  REASON:   {reason}")
    print()
    print(f"To record: python workflow/run.py decide {decision}")


if __name__ == "__main__":
    main()
