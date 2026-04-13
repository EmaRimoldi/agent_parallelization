#!/usr/bin/env python3
"""Verify that train.py produces deterministic results.

Runs the baseline train.py N times and checks that all val_bpb values are identical.
Exit code 0 = deterministic (PASS), 1 = non-deterministic (FAIL).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AUTORESEARCH_DIR = REPO_ROOT / "autoresearch"
N_RUNS = 5
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def parse_val_bpb(stdout: str) -> float | None:
    for line in stdout.splitlines():
        if line.strip().startswith("val_bpb:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                return None
    return None


def run_training(run_index: int) -> dict:
    start = time.time()
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=str(AUTORESEARCH_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - start
    val_bpb = parse_val_bpb(result.stdout)

    return {
        "run_index": run_index,
        "val_bpb": val_bpb,
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "stdout_tail": result.stdout[-500:] if result.stdout else "",
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def main() -> int:
    print(f"Determinism Verification: running train.py {N_RUNS} times")
    print(f"Working directory: {AUTORESEARCH_DIR}")
    print(f"{'=' * 60}")

    results = []
    for i in range(1, N_RUNS + 1):
        print(f"\n  Run {i}/{N_RUNS}...", end=" ", flush=True)
        try:
            r = run_training(i)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            r = {"run_index": i, "val_bpb": None, "returncode": -1,
                 "elapsed_seconds": 600, "error": "timeout"}
        results.append(r)

        if r["val_bpb"] is not None:
            print(f"val_bpb = {r['val_bpb']:.6f}  ({r['elapsed_seconds']:.1f}s)")
        else:
            print(f"FAILED (returncode={r['returncode']})")

    # Save raw results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "determinism_verification.json"
    output_file.write_text(json.dumps(results, indent=2) + "\n")

    # Analyze
    values = [r["val_bpb"] for r in results if r["val_bpb"] is not None]
    print(f"\n{'=' * 60}")

    if len(values) < N_RUNS:
        failed = N_RUNS - len(values)
        print(f"FAIL: {failed} of {N_RUNS} runs did not produce val_bpb")
        return 1

    unique = set(values)
    if len(unique) == 1:
        print(f"PASS: All {N_RUNS} runs produced identical val_bpb = {values[0]:.6f}")
        print(f"Results saved to: {output_file}")
        return 0

    spread = max(values) - min(values)
    print(f"FAIL: Got {len(unique)} distinct values")
    print(f"  Values: {[f'{v:.6f}' for v in values]}")
    print(f"  Range:  {spread:.6f}")
    print(f"  Mean:   {sum(values) / len(values):.6f}")
    print(f"Results saved to: {output_file}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
