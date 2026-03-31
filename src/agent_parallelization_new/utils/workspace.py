"""Git worktree creation and teardown for isolated agent workspaces."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from agent_parallelization_new.compatibility.training_harness import (
    generate_check_training_sh,
    generate_run_training_sh,
)


class WorkspaceError(Exception):
    pass


def create_workspace(
    autoresearch_dir: Path,
    workspace_path: Path,
    branch_name: str,
    train_budget_seconds: int,
    run_id: str,
    agent_id: str,
    results_root: Path,
) -> Path:
    """Create an isolated git worktree for one agent.

    Steps:
    1. Create branch in autoresearch if not exists
    2. Create git worktree at workspace_path
    3. Copy train.py.baseline
    4. Symlink .venv and data/
    5. Generate run_training.sh and check_training.sh
    6. Create results subdirectory structure

    Returns workspace_path.
    """
    autoresearch_dir = autoresearch_dir.resolve()
    workspace_path = workspace_path.resolve()

    _ensure_branch(autoresearch_dir, branch_name)
    _create_worktree(autoresearch_dir, workspace_path, branch_name)
    _save_baseline(workspace_path)
    _symlink_shared(autoresearch_dir, workspace_path)

    # Generate training wrapper scripts
    generate_run_training_sh(workspace_path, train_budget_seconds)
    generate_check_training_sh(workspace_path)

    # Set up per-agent results directory
    agent_results = results_root / "trajectories" / run_id
    agent_results.mkdir(parents=True, exist_ok=True)
    snap_dir = results_root / "snapshots" / run_id / agent_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = results_root / "weights" / run_id / agent_id
    weights_dir.mkdir(parents=True, exist_ok=True)

    return workspace_path


def destroy_workspace(autoresearch_dir: Path, workspace_path: Path) -> None:
    """Remove a git worktree and its directory."""
    autoresearch_dir = autoresearch_dir.resolve()
    workspace_path = workspace_path.resolve()
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(workspace_path)],
            cwd=autoresearch_dir,
            check=False,
            capture_output=True,
        )
    except Exception:
        pass
    if workspace_path.exists():
        shutil.rmtree(workspace_path, ignore_errors=True)


def _ensure_branch(autoresearch_dir: Path, branch_name: str) -> None:
    result = subprocess.run(
        ["git", "show-ref", "--quiet", f"refs/heads/{branch_name}"],
        cwd=autoresearch_dir,
        capture_output=True,
    )
    if result.returncode != 0:
        subprocess.run(
            ["git", "branch", branch_name, "HEAD"],
            cwd=autoresearch_dir,
            check=True,
            capture_output=True,
        )


def _create_worktree(
    autoresearch_dir: Path, workspace_path: Path, branch_name: str
) -> None:
    if workspace_path.exists():
        return
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "worktree", "add", str(workspace_path), branch_name],
        cwd=autoresearch_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise WorkspaceError(
            f"Failed to create worktree at {workspace_path}: {result.stderr}"
        )


def _save_baseline(workspace_path: Path) -> None:
    train_py = workspace_path / "train.py"
    baseline = workspace_path / "train.py.baseline"
    if train_py.exists() and not baseline.exists():
        shutil.copy2(train_py, baseline)


def _symlink_shared(autoresearch_dir: Path, workspace_path: Path) -> None:
    venv_src = autoresearch_dir / ".venv"
    venv_dst = workspace_path / ".venv"
    if venv_src.exists() and not venv_dst.exists():
        venv_dst.symlink_to(venv_src)

    data_src = autoresearch_dir / "data"
    data_dst = workspace_path / "data"
    if data_src.exists() and not data_dst.exists():
        data_dst.symlink_to(data_src)
