"""Generate run_training.sh and check_training.sh for each agent workspace.

These scripts are called by the Claude Code sub-agent via its Bash tool,
exactly as in the original OpenClaw-based system.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def generate_run_training_sh(workspace: Path, train_budget_seconds: int) -> Path:
    """Write run_training.sh into workspace. Returns path to the script."""
    uv_bin = _find_bin("uv")
    path_additions = _path_additions()

    script = f"""#!/bin/bash
export PATH="{path_additions}:$PATH"
cd "{workspace}"
# Kill any previous training still running
pkill -f "uv run train.py" 2>/dev/null || true
sleep 1
# Start training in background — agent polls run.log for results
nohup {uv_bin} run train.py > run.log 2>&1 &
echo "Training started (PID=$!, budget={train_budget_seconds}s). Poll with: ./check_training.sh"
"""
    out = workspace / "run_training.sh"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


def generate_check_training_sh(workspace: Path) -> Path:
    """Write check_training.sh into workspace. Returns path to the script."""
    script = f"""#!/bin/bash
cd "{workspace}"
if pgrep -f "uv run train.py" > /dev/null 2>&1; then
  echo "TRAINING RUNNING"
  tail -c 300 run.log 2>/dev/null | grep -oP 'step \\d+.*' | tail -1
else
  echo "TRAINING DONE"
  grep "^val_bpb:\\|^peak_vram_mb:" run.log 2>/dev/null || echo "No results (check run.log for errors)"
fi
"""
    out = workspace / "check_training.sh"
    out.write_text(script)
    out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return out


def _find_bin(name: str) -> str:
    """Find a binary, falling back to ~/.local/bin/<name>."""
    import shutil
    found = shutil.which(name)
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / name
    if fallback.exists():
        return str(fallback)
    return name


def _path_additions() -> str:
    """Return extra PATH entries needed for uv/python3."""
    additions = [
        str(Path.home() / ".local" / "bin"),
        str(Path.home() / "miniforge3" / "bin"),
    ]
    return ":".join(additions)
