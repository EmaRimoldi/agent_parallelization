"""Workspace helper for the imported agents-swarms blackboard mode."""

from __future__ import annotations

import shutil
from pathlib import Path


def install_imported_swarm_tools(
    *,
    repo_root: Path,
    workspace_path: Path,
    swarm_memory_path: Path,
) -> None:
    """Install the imported swarm protocol files into one agent workspace.

    The current repository's normal workspace builder creates the git-backed
    workspace and training scripts. This helper only adds the imported
    blackboard-facing files after that setup is complete.
    """
    package_dir = Path(__file__).resolve().parent
    src_root = package_dir.parents[1]
    template_dir = repo_root / "templates" / "imported_swarms"

    coordinator_src = package_dir / "coordinator.py"
    coordinator_dst = workspace_path / "coordinator.py"
    shutil.copy2(coordinator_src, coordinator_dst)

    coordinator_local_src = package_dir / "coordinator_local.py"
    coordinator_local_dst = workspace_path / "coordinator_local.py"
    shutil.copy2(coordinator_local_src, coordinator_local_dst)

    for name in ("collab.md", "program.md"):
        src = template_dir / name
        if src.exists():
            shutil.copy2(src, workspace_path / name)

    env_file = workspace_path / ".swarm_env"
    env_file.write_text(
        "\n".join(
            [
                f"SWARM_MEMORY_PATH={swarm_memory_path}",
                f"AGENT_PARALLELIZATION_SRC={src_root}",
            ]
        )
        + "\n"
    )

