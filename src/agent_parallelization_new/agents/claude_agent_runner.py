"""Run a Claude Code sub-agent session.

Replaces run_single_agent.sh + OpenClaw invocation entirely.
Uses `claude --print` (non-interactive) with session continuation to run
a multi-turn agent loop that manages its own time budget.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_parallelization_new.agents.base import AgentRunner
from agent_parallelization_new.budgeting import BudgetTracker
from agent_parallelization_new.config import AgentConfig


class ClaudeAgentRunner(AgentRunner):
    """Runs a Claude Code sub-agent via the `claude` CLI.

    The agent is invoked in a loop:
    - First turn: full first_message with context
    - Subsequent turns: "Continue. ~N min remaining. Keep experimenting."
    - Budget clock starts after first successful turn

    Failure modes handled:
    - API errors: retry with exponential backoff (no time refund)
    - Rate limits: backoff aggressively (no time refund)
    - No-reply turns: track count, rotate session after MAX_NOREPLY
    - Budget exceeded: break loop
    - Startup timeout: exit(2)
    """

    MAX_NOREPLY = 5
    MIN_TURN_INTERVAL_SEC = 5
    INITIAL_BACKOFF_SEC = 5
    MAX_BACKOFF_SEC = 60
    FIRST_TURN_TIMEOUT_SEC = 900  # 15 min: start_gpu_worker + SLURM queue + compile + first run
    MAX_TURN_TIMEOUT_SEC = 900    # 15 min per subsequent turn

    @staticmethod
    def _temperature_directive(temperature: Optional[float]) -> str:
        """Return a system-prompt suffix that approximates the requested temperature.

        The claude CLI exposes no --temperature flag.  We approximate the
        effect via instructional wording:
          temp ≥ 1.0  → exploratory / high-variance search style
          temp < 0.5  → conservative / low-variance search style
          otherwise   → no modification
        """
        if temperature is None:
            return ""
        if temperature >= 1.0:
            return (
                "\n\n[SEARCH STYLE: Be creative and exploratory. "
                "Prefer bold, diverse changes over incremental refinement. "
                "Try unconventional hyperparameter combinations that you would not "
                "normally attempt. High variance in search is desirable.]"
            )
        if temperature < 0.5:
            return (
                "\n\n[SEARCH STYLE: Be conservative and methodical. "
                "Make only small, well-motivated incremental changes. "
                "Exploit the best-known region before exploring new directions. "
                "Low variance and high reliability are desirable.]"
            )
        return ""

    def run(
        self,
        run_id: str,
        experiment_id: str,
        system_prompt: str,
        first_message: str,
    ) -> None:
        """Run the agent loop until budget expires. Writes metadata.json at end."""
        config = self.config

        # Inject temperature-approximate behaviour into system prompt
        # (claude CLI has no --temperature flag; we use instructional wording)
        effective_system_prompt = system_prompt + self._temperature_directive(
            config.temperature
        )

        budget = BudgetTracker(
            wall_clock_budget_seconds=config.time_budget_minutes * 60,
            train_time_budget_seconds=config.train_time_budget_seconds,
            startup_deadline_seconds=config.time_budget_minutes * 60 + 300,  # full budget + 5 min buffer
        )

        # Unique session ID (never shared between agents)
        session_id = f"{experiment_id}-{config.agent_id}-{int(time.time())}-{os.getpid()}"

        env = self._build_env(run_id, experiment_id)
        session_log = self.logs_dir / "run_agent.log"

        start_time = datetime.now(timezone.utc).isoformat()
        total_turns = 0
        backoff = self.INITIAL_BACKOFF_SEC
        noreply_count = 0
        first_turn = True

        with open(session_log, "w") as log_fh:
            log_fh.write(f"[{config.agent_id}] Session starting: {session_id}\n")
            log_fh.flush()

            # Background thread: watches for gpu_allocated_at and starts budget clock
            # from the moment the GPU is allocated, not after the first claude turn.
            _stop_watcher = threading.Event()
            threading.Thread(
                target=self._watch_gpu_allocation,
                args=(budget, log_fh, _stop_watcher),
                daemon=True,
            ).start()

            # Background thread: logs workspace events (trigger, result, val_bpb)
            threading.Thread(
                target=self._watch_workspace_events,
                args=(log_fh, _stop_watcher),
                daemon=True,
            ).start()

            while True:
                # Hard wall-clock cap
                if budget.startup_expired():
                    msg = f"[{config.agent_id}] ABORT: no successful turn within startup deadline.\n"
                    log_fh.write(msg)
                    sys.stderr.write(msg)
                    break

                if budget.should_stop():
                    log_fh.write(f"[{config.agent_id}] Budget expired — stopping.\n")
                    break

                # Build turn message
                if first_turn:
                    turn_msg = first_message
                    # First turn can include GPU queue wait + compile + first training run,
                    # so give it the full remaining budget rather than a fixed cap.
                    turn_timeout = max(self.FIRST_TURN_TIMEOUT_SEC, budget.remaining_seconds())
                else:
                    mins_left = budget.remaining_minutes()
                    turn_msg = (
                        f"Continue the research. ~{mins_left} min remaining in budget. "
                        f"Keep modifying train.py and running experiments to improve val_bpb. "
                        f"Do NOT stop until time runs out."
                    )
                    remaining = budget.remaining_seconds()
                    turn_timeout = min(int(remaining), self.MAX_TURN_TIMEOUT_SEC)

                log_fh.write(
                    f"[{config.agent_id}] Turn {total_turns} starting "
                    f"({'first turn' if first_turn else f'~{budget.remaining_minutes()} min remaining'}).\n"
                )
                log_fh.flush()
                turn_start = time.monotonic()
                exit_code, output = self._run_turn(
                    turn_msg=turn_msg,
                    session_id=session_id,
                    system_prompt=effective_system_prompt,
                    timeout_seconds=turn_timeout,
                    env=env,
                )
                turn_elapsed = time.monotonic() - turn_start

                log_fh.write(
                    f"[{config.agent_id}] Turn {total_turns}: exit={exit_code} elapsed={turn_elapsed:.1f}s\n"
                )
                if output:
                    log_fh.write(output[:2000] + ("\n...(truncated)\n" if len(output) > 2000 else "\n"))
                log_fh.flush()

                is_noreply = "No reply from agent" in output or (not output.strip() and exit_code == 0)
                is_ratelimit = "rate limit" in output.lower() or "rate_limit" in output.lower()
                is_error = exit_code != 0

                if is_error:
                    log_fh.write(f"[{config.agent_id}] Error turn, retrying in {backoff}s...\n")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF_SEC)
                elif is_ratelimit:
                    log_fh.write(f"[{config.agent_id}] Rate limit, backing off {backoff}s...\n")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF_SEC)
                elif is_noreply:
                    noreply_count += 1
                    log_fh.write(f"[{config.agent_id}] No-reply turn #{noreply_count}/{self.MAX_NOREPLY}\n")
                    if noreply_count >= self.MAX_NOREPLY:
                        noreply_count = 0
                        session_id = f"{experiment_id}-{config.agent_id}-{int(time.time())}-{os.getpid()}"
                        first_turn = True
                        log_fh.write(f"[{config.agent_id}] Session rotated to {session_id}\n")
                    _enforce_min_interval(turn_elapsed, self.MIN_TURN_INTERVAL_SEC)
                else:
                    # Successful turn
                    backoff = self.INITIAL_BACKOFF_SEC
                    noreply_count = 0
                    total_turns += 1

                    # Fallback: start clock after first turn if gpu_allocated_at never appeared
                    if not budget.budget_started():
                        budget.start_budget_clock()
                        log_fh.write(
                            f"[{config.agent_id}] Budget clock started (fallback, no gpu_allocated_at) — "
                            f"{budget.wall_clock_budget_seconds}s remaining.\n"
                        )
                    first_turn = False
                    _enforce_min_interval(turn_elapsed, self.MIN_TURN_INTERVAL_SEC)

        _stop_watcher.set()

        end_time = datetime.now(timezone.utc).isoformat()
        self._write_metadata(
            run_id=run_id,
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=end_time,
            total_turns=total_turns,
            budget_seconds=config.time_budget_minutes * 60,
        )

    def _watch_gpu_allocation(
        self,
        budget: BudgetTracker,
        log_fh,
        stop_event: threading.Event,
    ) -> None:
        """Background thread: start budget clock when gpu_allocated_at appears."""
        marker = self.workspace / "gpu_allocated_at"
        while not stop_event.is_set():
            if not budget.budget_started() and marker.exists():
                budget.start_budget_clock()
                ts = marker.read_text().strip()
                log_fh.write(
                    f"[{self.config.agent_id}] GPU allocated at {ts} — "
                    f"budget clock started ({budget.wall_clock_budget_seconds}s).\n"
                )
                log_fh.flush()
                return
            stop_event.wait(2)

    def _watch_workspace_events(
        self,
        log_fh,
        stop_event: threading.Event,
    ) -> None:
        """Background thread: log key workspace file events as they happen."""
        ws = self.workspace
        agent_id = self.config.agent_id

        trigger = ws / "run.trigger"
        result = ws / "run.result"
        train_out = ws / "logs" / "train_current.out"

        trigger_seen = False
        result_seen = False
        run_count = 0

        while not stop_event.is_set():
            # Trigger appeared → training started
            if not trigger_seen and trigger.exists():
                trigger_seen = True
                result_seen = False
                run_count += 1
                log_fh.write(f"[{agent_id}] Training run #{run_count} started (trigger dropped).\n")
                log_fh.flush()

            # Result appeared → training finished
            if trigger_seen and not result_seen and result.exists():
                result_seen = True
                trigger_seen = False
                # Read val_bpb from result or train_current.out
                val_bpb = None
                try:
                    for src in (result, train_out):
                        if src.exists():
                            for line in src.read_text().splitlines():
                                if line.startswith("val_bpb:"):
                                    val_bpb = line.split(":", 1)[1].strip()
                                    break
                        if val_bpb:
                            break
                except OSError:
                    pass
                if val_bpb:
                    log_fh.write(f"[{agent_id}] Training run #{run_count} done — val_bpb: {val_bpb}\n")
                else:
                    content = ""
                    try:
                        content = result.read_text().strip().splitlines()[0] if result.exists() else "no result"
                    except OSError:
                        pass
                    log_fh.write(f"[{agent_id}] Training run #{run_count} done — {content}\n")
                log_fh.flush()

            stop_event.wait(2)

    def _run_turn(
        self,
        turn_msg: str,
        session_id: str,
        system_prompt: str,
        timeout_seconds: int,
        env: dict,
    ) -> tuple[int, str]:
        """Invoke `claude --print` for one turn. Returns (exit_code, output)."""
        cmd = [
            "claude",
            "--print",
            "--output-format", "text",
            "--dangerously-skip-permissions",
        ]

        # Pass system prompt if provided — Claude Code CLI supports --system-prompt flag
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]

        # Prompt is a positional argument (not --message) in claude CLI ≥2.x
        cmd += [turn_msg]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            return result.returncode, output
        except subprocess.TimeoutExpired:
            return -1, f"[timeout after {timeout_seconds}s]"
        except FileNotFoundError:
            return -2, "[claude CLI not found in PATH]"
        except Exception as e:
            return -3, f"[exception: {e}]"

    def _build_env(self, run_id: str, experiment_id: str) -> dict:
        """Build environment variables for the agent subprocess."""
        env = os.environ.copy()
        env["RUN_ID"] = run_id
        env["AGENT_ID"] = self.config.agent_id
        env["RESULTS_ROOT"] = str(self.results_dir)
        env["AUTOSEARCH_TIME_BUDGET"] = str(self.config.train_time_budget_seconds)
        env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_device
        env["EXPERIMENT_ID"] = experiment_id
        # Ensure uv/python3 are findable
        extra_path = ":".join([
            str(Path.home() / ".local" / "bin"),
            str(Path.home() / "miniforge3" / "bin"),
        ])
        env["PATH"] = extra_path + ":" + env.get("PATH", "")
        return env

    def _write_metadata(
        self,
        run_id: str,
        experiment_id: str,
        start_time: str,
        end_time: str,
        total_turns: int,
        budget_seconds: int,
    ) -> None:
        """Write metadata.json to results dir."""
        metadata = {
            "agent_id": self.config.agent_id,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "start_time": start_time,
            "end_time": end_time,
            "total_turns": total_turns,
            "budget_seconds": budget_seconds,
            "model": self.config.model,
        }
        meta_path = self.results_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        # Count training runs from trajectory.jsonl
        traj_path = self.results_dir / "trajectory.jsonl"
        if traj_path.exists():
            lines = [l for l in traj_path.read_text().splitlines() if l.strip()]
            metadata["total_training_runs"] = len(lines)
            if lines:
                import json as _json
                bpbs = [_json.loads(l).get("val_bpb") for l in lines if l.strip()]
                bpbs = [b for b in bpbs if b is not None]
                if bpbs:
                    metadata["best_val_bpb"] = min(bpbs)
            meta_path.write_text(json.dumps(metadata, indent=2))


def _enforce_min_interval(elapsed: float, min_interval: float) -> None:
    """Sleep to ensure at least min_interval seconds between turns."""
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
