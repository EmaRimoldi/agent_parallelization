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
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_parallelization_new.agents.base import AgentRunner
from agent_parallelization_new.budgeting import BudgetTracker
from agent_parallelization_new.config import AgentConfig
from agent_parallelization_new.utils.log_parser import parse_training_seconds, parse_val_bpb


def _ts() -> str:
    """Return current local time as ISO-8601 string with second precision."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(log_fh, msg: str) -> None:
    """Write a timestamped system-event line and flush."""
    log_fh.write(f"[{_ts()}] {msg}\n")
    log_fh.flush()


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
    HEARTBEAT_INTERVAL_SEC = 30   # log "still alive" every 30s during a turn

    @staticmethod
    def _temperature_directive(temperature: Optional[float]) -> str:
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

        effective_system_prompt = system_prompt + self._temperature_directive(
            config.temperature
        )

        self._active_proc: Optional[subprocess.Popen] = None

        budget = BudgetTracker(
            wall_clock_budget_seconds=config.time_budget_minutes * 60,
            train_time_budget_seconds=config.train_time_budget_seconds,
            startup_deadline_seconds=config.time_budget_minutes * 60 + 300,
        )

        session_id = f"{experiment_id}-{config.agent_id}-{int(time.time())}-{os.getpid()}"
        env = self._build_env(run_id, experiment_id)
        session_log = self.logs_dir / "run_agent.log"

        start_time = datetime.now(timezone.utc).isoformat()
        self.turn_count = 0
        self.turns_log_path = self.results_dir / "turns.jsonl"
        self.turns_log_path.write_text("")
        self.training_runs_log_path = self.results_dir / "training_runs.jsonl"
        self.training_runs_log_path.write_text("")
        self._cumulative_chars = 0
        self._turn_records: list[dict] = []
        self._training_run_count = 0
        backoff = self.INITIAL_BACKOFF_SEC
        noreply_count = 0
        first_turn = True

        with open(session_log, "w") as log_fh:
            _log(log_fh, f"[{config.agent_id}] Session starting: {session_id}")

            _stop_watcher = threading.Event()
            _observed_val_bpbs: list[float] = []

            # GPU allocation watcher — starts budget clock
            threading.Thread(
                target=self._watch_gpu_allocation,
                args=(budget, log_fh, _stop_watcher),
                daemon=True,
            ).start()

            # Workspace event watcher — training trigger/result/file changes
            threading.Thread(
                target=self._watch_workspace_events,
                args=(log_fh, _stop_watcher, _observed_val_bpbs),
                daemon=True,
            ).start()

            while True:
                if budget.startup_expired():
                    msg = f"[{config.agent_id}] ABORT: no successful turn within startup deadline."
                    _log(log_fh, msg)
                    sys.stderr.write(msg + "\n")
                    break

                if budget.should_stop():
                    _log(log_fh, f"[{config.agent_id}] Budget expired — stopping.")
                    break

                if first_turn:
                    turn_msg = first_message
                    turn_timeout = max(self.FIRST_TURN_TIMEOUT_SEC, int(budget.remaining_seconds()))
                    _log(log_fh, f"[{config.agent_id}] Turn {self.turn_count} starting (first turn).")
                else:
                    mins_left = budget.remaining_minutes()
                    secs_left = int(budget.remaining_seconds())
                    # Expected wall time per run: train_budget + ~90s compile/eval overhead
                    run_wall_sec = config.train_time_budget_seconds + 90
                    run_wall_min = round(run_wall_sec / 60)
                    if secs_left < run_wall_sec + 60:
                        time_guidance = (
                            f"WARNING: only ~{mins_left} min left — NOT ENOUGH for another "
                            f"training run (~{run_wall_min} min each). "
                            f"Do NOT start a new run. Instead review results.tsv, "
                            f"ensure the best result is committed, and stop."
                        )
                    else:
                        runs_remaining = secs_left // run_wall_sec
                        time_guidance = (
                            f"Each training run takes ~{run_wall_min} min. "
                            f"You can fit approximately {runs_remaining} more run(s)."
                        )
                    turn_msg = (
                        f"Continue the research. ~{mins_left} min remaining in budget. "
                        f"{time_guidance} "
                        f"Keep modifying train.py and running experiments to improve val_bpb."
                    )
                    if self.config.use_shared_memory:
                        memory = self._build_shared_memory_context()
                        if memory:
                            turn_msg = f"{memory}\n\n---\n\n{turn_msg}"
                    elif self.config.use_external_memory:
                        memory = self._build_memory_context()
                        if memory:
                            turn_msg = f"{memory}\n\n---\n\n{turn_msg}"
                    turn_timeout = min(secs_left, self.MAX_TURN_TIMEOUT_SEC)
                    _log(log_fh, f"[{config.agent_id}] Turn {self.turn_count} starting (~{mins_left} min remaining).")

                turn_start = time.monotonic()

                # Heartbeat thread: logs "still alive" every HEARTBEAT_INTERVAL_SEC
                _turn_done = threading.Event()
                threading.Thread(
                    target=self._heartbeat,
                    args=(config.agent_id, self.turn_count, turn_start, _turn_done, log_fh),
                    daemon=True,
                ).start()

                exit_code, output, usage = self._run_turn(
                    turn_msg=turn_msg,
                    session_id=session_id,
                    system_prompt=effective_system_prompt,
                    timeout_seconds=turn_timeout,
                    env=env,
                    log_fh=log_fh,
                )
                _turn_done.set()
                turn_elapsed = time.monotonic() - turn_start

                system_prompt_chars = len(effective_system_prompt) if effective_system_prompt else 0
                turn_msg_chars = len(turn_msg)
                response_chars = len(output)

                input_tokens = _coerce_token_count(usage.get("input_tokens"))
                if input_tokens is None:
                    input_tokens = (system_prompt_chars + turn_msg_chars) // 4

                output_tokens = _coerce_token_count(usage.get("output_tokens"))
                if output_tokens is None:
                    output_tokens = response_chars // 4

                self._cumulative_chars += system_prompt_chars + turn_msg_chars + response_chars
                turn_record = {
                    "turn": self.turn_count,
                    "timestamp": time.time(),
                    "model": self.config.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "system_prompt_chars": system_prompt_chars,
                    "turn_msg_chars": turn_msg_chars,
                    "response_chars": response_chars,
                    "context_fill_ratio": self._estimate_context_fill(),
                    "wall_clock_seconds": turn_elapsed,
                }
                self._turn_records.append(turn_record)
                with open(self.turns_log_path, "a") as turns_fh:
                    turns_fh.write(json.dumps(turn_record) + "\n")

                _log(log_fh,
                    f"[{config.agent_id}] Turn {self.turn_count} finished: exit={exit_code} elapsed={turn_elapsed:.1f}s")
                if output:
                    log_fh.write(output[:2000] + ("\n...(truncated)\n" if len(output) > 2000 else "\n"))
                    log_fh.flush()

                is_noreply = "No reply from agent" in output or (not output.strip() and exit_code == 0)
                is_ratelimit = "rate limit" in output.lower() or "rate_limit" in output.lower()
                is_error = exit_code != 0

                if is_error:
                    _log(log_fh, f"[{config.agent_id}] Error turn, retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF_SEC)
                elif is_ratelimit:
                    _log(log_fh, f"[{config.agent_id}] Rate limit, backing off {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF_SEC)
                elif is_noreply:
                    noreply_count += 1
                    _log(log_fh, f"[{config.agent_id}] No-reply turn #{noreply_count}/{self.MAX_NOREPLY}")
                    if noreply_count >= self.MAX_NOREPLY:
                        noreply_count = 0
                        session_id = f"{experiment_id}-{config.agent_id}-{int(time.time())}-{os.getpid()}"
                        first_turn = True
                        _log(log_fh, f"[{config.agent_id}] Session rotated to {session_id}")
                    _enforce_min_interval(turn_elapsed, self.MIN_TURN_INTERVAL_SEC)
                else:
                    backoff = self.INITIAL_BACKOFF_SEC
                    noreply_count = 0
                    self.turn_count += 1

                    if not budget.budget_started():
                        budget.start_budget_clock()
                        _log(log_fh,
                            f"[{config.agent_id}] Budget clock started (fallback, no gpu_allocated_at) — "
                            f"{budget.wall_clock_budget_seconds}s remaining.")
                    first_turn = False
                    _enforce_min_interval(turn_elapsed, self.MIN_TURN_INTERVAL_SEC)

        _stop_watcher.set()

        end_time = datetime.now(timezone.utc).isoformat()
        self._write_metadata(
            run_id=run_id,
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=end_time,
            total_turns=self.turn_count,
            budget_seconds=config.time_budget_minutes * 60,
            observed_val_bpbs=_observed_val_bpbs,
        )

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _watch_gpu_allocation(
        self,
        budget: BudgetTracker,
        log_fh,
        stop_event: threading.Event,
    ) -> None:
        """Start budget clock when gpu_allocated_at appears."""
        marker = self.workspace / "gpu_allocated_at"
        while not stop_event.is_set():
            if not budget.budget_started() and marker.exists():
                budget.start_budget_clock()
                ts = marker.read_text().strip()
                _log(log_fh,
                    f"[{self.config.agent_id}] GPU allocated at {ts} — "
                    f"budget clock started ({budget.wall_clock_budget_seconds}s).")
                return
            stop_event.wait(2)

    def _watch_workspace_events(
        self,
        log_fh,
        stop_event: threading.Event,
        observed_val_bpbs: list,
    ) -> None:
        """Log key workspace file events: trigger, result, train.py edits, results.tsv rows."""
        ws = self.workspace
        agent_id = self.config.agent_id

        trigger = ws / "run.trigger"
        result = ws / "run.result"
        train_out = ws / "logs" / "train_current.out"
        train_py = ws / "train.py"
        results_tsv = ws / "results" / "results.tsv"
        trace_path = self.agent_dir / "reasoning" / "trace.jsonl"

        trigger_seen = False
        result_seen = False
        run_count = 0
        run_wall_start: Optional[float] = None
        train_py_mtime: Optional[float] = None
        results_tsv_lines = 0
        train_out_lines = 0
        shared_logged_steps: set[int] = set()

        while not stop_event.is_set():
            # train.py modified → log diff vs baseline so we see what changed
            try:
                mtime = train_py.stat().st_mtime if train_py.exists() else None
                if mtime is not None and mtime != train_py_mtime:
                    if train_py_mtime is not None:
                        _log(log_fh, f"[{agent_id}] train.py modified.")
                        _log_train_diff(train_py, log_fh, agent_id)
                    train_py_mtime = mtime
            except OSError:
                pass

            if self.config.use_shared_memory and trace_path.exists():
                try:
                    for raw_line in trace_path.read_text().splitlines():
                        if not raw_line.strip():
                            continue
                        entry = json.loads(raw_line)
                        step = entry.get("step_index")
                        if not isinstance(step, int) or step in shared_logged_steps:
                            continue
                        accepted = entry.get("accepted")
                        val_bpb = entry.get("val_bpb_after")
                        if accepted is None and val_bpb is None:
                            continue
                        self._append_shared_log(
                            step=step,
                            hypothesis=str(entry.get("hypothesis", "")),
                            val_bpb=val_bpb,
                            accepted=bool(accepted),
                        )
                        shared_logged_steps.add(step)
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    pass

            # results.tsv new row → agent logged a result
            try:
                if results_tsv.exists():
                    lines = [l for l in results_tsv.read_text().splitlines() if l.strip()]
                    if len(lines) > results_tsv_lines:
                        for row in lines[results_tsv_lines:]:
                            if not row.startswith("commit"):  # skip header
                                _log(log_fh, f"[{agent_id}] results.tsv: {row}")
                        results_tsv_lines = len(lines)
            except OSError:
                pass

            # run.trigger appeared → training started
            if not trigger_seen and trigger.exists():
                trigger_seen = True
                result_seen = False
                run_count += 1
                run_wall_start = time.time()
                train_out_lines = 0  # reset stream cursor for new run
                _log(log_fh, f"[{agent_id}] Training run #{run_count} started.")

            # stream new lines from train_current.out while a run is active
            if trigger_seen and not result_seen:
                try:
                    if train_out.exists():
                        all_lines = train_out.read_text().splitlines()
                        new_lines = all_lines[train_out_lines:]
                        for line in new_lines:
                            log_fh.write(f"[{_ts()}] [{agent_id}][training] {line}\n")
                        if new_lines:
                            log_fh.flush()
                        train_out_lines = len(all_lines)
                except OSError:
                    pass

            # run.result appeared → training finished
            if trigger_seen and not result_seen and result.exists():
                result_seen = True
                trigger_seen = False
                finished_at = time.time()
                wall_seconds = finished_at - run_wall_start if run_wall_start else None
                elapsed = f"{wall_seconds:.0f}s" if wall_seconds is not None else "?s"
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

                parsed_val_bpb = parse_val_bpb(train_out) if train_out.exists() else None
                if parsed_val_bpb is None and val_bpb is not None:
                    try:
                        parsed_val_bpb = float(val_bpb)
                    except ValueError:
                        parsed_val_bpb = None
                training_seconds = (
                    parse_training_seconds(train_out) if train_out.exists() else None
                )
                self._training_run_count += 1
                training_run_record = {
                    "run_index": self._training_run_count,
                    "turn": self.turn_count,
                    "started_at": run_wall_start,
                    "finished_at": finished_at,
                    "wall_seconds": wall_seconds,
                    "training_seconds": training_seconds,
                    "val_bpb": parsed_val_bpb,
                    "status": "success" if parsed_val_bpb is not None else "crash",
                }
                with open(self.training_runs_log_path, "a") as runs_fh:
                    runs_fh.write(json.dumps(training_run_record) + "\n")

                if val_bpb:
                    try:
                        observed_val_bpbs.append(float(val_bpb))
                    except ValueError:
                        pass
                    _log(log_fh, f"[{agent_id}] Training run #{run_count} done — val_bpb: {val_bpb} (elapsed: {elapsed})")
                else:
                    status = ""
                    try:
                        status = result.read_text().strip().splitlines()[0] if result.exists() else "no result"
                    except OSError:
                        pass
                    _log(log_fh, f"[{agent_id}] Training run #{run_count} done — {status} (elapsed: {elapsed})")
                    _dump_slurm_failure_logs(ws, agent_id, run_count, log_fh)

            stop_event.wait(2)

    def _heartbeat(
        self,
        agent_id: str,
        turn_num: int,
        turn_start: float,
        done_event: threading.Event,
        log_fh,
    ) -> None:
        """Log 'still alive' every HEARTBEAT_INTERVAL_SEC during a turn."""
        while not done_event.wait(self.HEARTBEAT_INTERVAL_SEC):
            elapsed = time.monotonic() - turn_start
            _log(log_fh, f"[{agent_id}] Turn {turn_num} still running ({elapsed:.0f}s elapsed).")

    # ------------------------------------------------------------------
    # Core turn execution
    # ------------------------------------------------------------------

    def _parse_claude_output(self, raw_stdout: str) -> tuple[str, dict]:
        """Parse JSON output from claude CLI. Returns (text_response, usage_dict)."""
        try:
            data = json.loads(raw_stdout)
        except json.JSONDecodeError:
            return raw_stdout, {}

        if not isinstance(data, dict):
            return raw_stdout, {}

        text = data.get("result", raw_stdout)
        usage = data.get("usage", {})
        return str(text), usage if isinstance(usage, dict) else {}

    def _run_turn(
        self,
        turn_msg: str,
        session_id: str,
        system_prompt: str,
        timeout_seconds: int,
        env: dict,
        log_fh,
    ) -> tuple[int, str, dict]:
        """Invoke `claude --print` for one turn, streaming output to log in real-time."""
        cmd = [
            "claude",
            "--print",
            "--output-format", "json",
            "--dangerously-skip-permissions",
        ]
        if self.config.model:
            cmd += ["--model", self.config.model]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]
        cmd += [turn_msg]

        output_lines: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._active_proc = proc

            # Stream stdout in real-time
            def _stream_stdout():
                for line in proc.stdout:
                    output_lines.append(line)
                    log_fh.write(f"  {line}" if not line.startswith("[") else line)
                    log_fh.flush()

            stdout_thread = threading.Thread(target=_stream_stdout, daemon=True)
            stdout_thread.start()

            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return -1, f"[timeout after {timeout_seconds}s]", {}

            stdout_thread.join(timeout=5)
            stderr = proc.stderr.read()
            self._active_proc = None
            output, usage = self._parse_claude_output("".join(output_lines))
            if stderr:
                output += "\n[stderr]\n" + stderr
            return proc.returncode, output, usage

        except FileNotFoundError:
            return -2, "[claude CLI not found in PATH]", {}
        except Exception as e:
            return -3, f"[exception: {e}]", {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_context_fill(self) -> float:
        """Estimate context fill ratio c/K from cumulative character count."""
        estimated_tokens = self._cumulative_chars / 4
        return min(estimated_tokens / 200_000, 1.0)

    def _build_memory_context(self) -> str:
        """Build a compact experiment log from the agent's private trace."""
        trace_path = self.agent_dir / "reasoning" / "trace.jsonl"
        if not trace_path.exists():
            return ""

        lines = [
            "# Experiment Log",
            "| # | change | bpb | Δ | best |",
            "|---|--------|-----|---|------|",
        ]
        best_bpb = float("inf")

        for raw_line in trace_path.read_text().splitlines():
            if not raw_line.strip():
                continue
            try:
                entry = json.loads(raw_line)
                step = entry.get("step_index", entry.get("step", "?"))
                hypothesis = str(entry.get("hypothesis", "?"))[:40]
                bpb = entry.get("val_bpb_after")
                if bpb is None:
                    continue
                bpb_val = float(bpb)
                prev = entry.get("val_bpb_before")
                delta = (
                    f"{bpb_val - float(prev):+.4f}"
                    if prev not in (None, "")
                    else "—"
                )
                is_best = "✓" if bpb_val < best_bpb else ""
                if bpb_val < best_bpb:
                    best_bpb = bpb_val
                lines.append(
                    f"| {step} | {hypothesis} | {bpb_val:.4f} | {delta} | {is_best} |"
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        return "\n".join(lines) if len(lines) > 3 else ""

    def _build_shared_memory_context(self) -> str:
        """Build a compact experiment log across all agents."""
        shared_path = self.workspace / "shared_results_log.jsonl"
        if not shared_path.exists():
            return ""

        lines = [
            "# Shared Experiment Log (all agents)",
            "| agent | # | change | bpb | kept |",
            "|-------|---|--------|-----|------|",
        ]

        for raw_line in shared_path.read_text().splitlines():
            if not raw_line.strip():
                continue
            try:
                entry = json.loads(raw_line)
                agent = str(entry.get("agent_id", "?"))[-4:]
                step = entry.get("step", "?")
                hypothesis = str(entry.get("hypothesis", "?"))[:35]
                bpb = entry.get("val_bpb")
                accepted = "✓" if entry.get("accepted") else "✗"
                if bpb is not None:
                    lines.append(
                        f"| {agent} | {step} | {hypothesis} | {float(bpb):.4f} | {accepted} |"
                    )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        return "\n".join(lines) if len(lines) > 3 else ""

    def _append_shared_log(self, step, hypothesis, val_bpb, accepted) -> None:
        """Append one completed experiment result to the shared JSONL log."""
        shared_path = self.workspace / "shared_results_log.jsonl"
        if not shared_path.exists():
            return

        record = json.dumps(
            {
                "agent_id": self.config.agent_id,
                "step": step,
                "hypothesis": str(hypothesis)[:60],
                "val_bpb": val_bpb,
                "accepted": accepted,
                "timestamp": time.time(),
            }
        )
        with open(shared_path, "a") as shared_fh:
            fcntl.flock(shared_fh, fcntl.LOCK_EX)
            shared_fh.write(record + "\n")
            fcntl.flock(shared_fh, fcntl.LOCK_UN)

    def _build_env(self, run_id: str, experiment_id: str) -> dict:
        env = os.environ.copy()
        env["RUN_ID"] = run_id
        env["AGENT_ID"] = self.config.agent_id
        env["RESULTS_ROOT"] = str(self.results_dir)
        env["AUTOSEARCH_TIME_BUDGET"] = str(self.config.train_time_budget_seconds)
        env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_device
        env["EXPERIMENT_ID"] = experiment_id
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
        observed_val_bpbs: list | None = None,
    ) -> None:
        metadata = {
            "agent_id": self.config.agent_id,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "start_time": start_time,
            "end_time": end_time,
            "total_turns": total_turns,
            "budget_seconds": budget_seconds,
            "model": self.config.model,
            "total_input_tokens": sum(
                record.get("input_tokens") or 0 for record in getattr(self, "_turn_records", [])
            ),
            "total_output_tokens": sum(
                record.get("output_tokens") or 0 for record in getattr(self, "_turn_records", [])
            ),
            "avg_context_fill": (
                sum(record.get("context_fill_ratio", 0.0) for record in getattr(self, "_turn_records", []))
                / len(getattr(self, "_turn_records", []))
                if getattr(self, "_turn_records", [])
                else 0.0
            ),
            "final_context_fill": (
                getattr(self, "_turn_records", [])[-1].get("context_fill_ratio", 0.0)
                if getattr(self, "_turn_records", [])
                else 0.0
            ),
        }
        meta_path = self.results_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        # Build trajectory.jsonl — primary source: observed val_bpb values captured
        # by the workspace watcher. Fallback: workspace/results/results.tsv.
        traj_path = self.results_dir / "trajectory.jsonl"
        traj_bpbs: list[float] = list(observed_val_bpbs) if observed_val_bpbs else []
        if not traj_bpbs:
            results_tsv = self.workspace / "results" / "results.tsv"
            if results_tsv.exists():
                for row in results_tsv.read_text().splitlines():
                    if not row.strip() or row.startswith("commit"):
                        continue
                    parts = row.split("\t")
                    if len(parts) >= 2:
                        try:
                            traj_bpbs.append(float(parts[1]))
                        except ValueError:
                            pass
        if traj_bpbs:
            traj_lines = [json.dumps({"step": i, "val_bpb": v}) for i, v in enumerate(traj_bpbs)]
            traj_path.write_text("\n".join(traj_lines) + "\n")

        if traj_path.exists():
            lines = [l for l in traj_path.read_text().splitlines() if l.strip()]
            metadata["total_training_runs"] = len(lines)
            if lines:
                bpbs = [json.loads(l).get("val_bpb") for l in lines if l.strip()]
                bpbs = [b for b in bpbs if b is not None]
                if bpbs:
                    metadata["best_val_bpb"] = min(bpbs)
            meta_path.write_text(json.dumps(metadata, indent=2))


def _enforce_min_interval(elapsed: float, min_interval: float) -> None:
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)


def _coerce_token_count(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _log_train_diff(train_py: Path, log_fh, agent_id: str, max_lines: int = 40) -> None:
    """Log a unified diff of train.py vs train.py.baseline."""
    import difflib
    baseline = train_py.parent / "train.py.baseline"
    if not baseline.exists():
        return
    try:
        old = baseline.read_text().splitlines()
        new = train_py.read_text().splitlines()
        diff = list(difflib.unified_diff(old, new, fromfile="train.py.baseline", tofile="train.py", lineterm="", n=2))
        if not diff:
            return
        log_fh.write(f"[{agent_id}] --- train.py diff vs baseline ({len(diff)} lines) ---\n")
        for line in diff[:max_lines]:
            log_fh.write(f"[{agent_id}]   {line}\n")
        if len(diff) > max_lines:
            log_fh.write(f"[{agent_id}]   ... ({len(diff) - max_lines} more lines truncated)\n")
        log_fh.write(f"[{agent_id}] --- end diff ---\n")
    except OSError:
        pass


def _dump_slurm_failure_logs(
    workspace: Path,
    agent_id: str,
    run_count: int,
    log_fh,
    tail_lines: int = 50,
) -> None:
    """Append SLURM training logs to the agent log on training failure.

    Dumps (up to tail_lines each):
    - workspace/logs/train_current.out  — stdout of the failing train.py run
    - workspace/logs/worker_*.err       — stderr of the SLURM worker job
    """
    logs_dir = workspace / "logs"

    train_out = logs_dir / "train_current.out"
    if train_out.exists():
        try:
            lines = train_out.read_text().splitlines()
            tail = lines[-tail_lines:]
            log_fh.write(f"[{agent_id}] --- train_current.out (last {len(tail)} lines) ---\n")
            for line in tail:
                log_fh.write(f"[{agent_id}]   {line}\n")
            log_fh.write(f"[{agent_id}] --- end train_current.out ---\n")
        except OSError:
            pass

    try:
        err_files = sorted(logs_dir.glob("worker_*.err"))
        if err_files:
            latest_err = err_files[-1]
            lines = latest_err.read_text().splitlines()
            if lines:
                tail = lines[-tail_lines:]
                log_fh.write(f"[{agent_id}] --- {latest_err.name} (last {len(tail)} lines) ---\n")
                for line in tail:
                    log_fh.write(f"[{agent_id}]   {line}\n")
                log_fh.write(f"[{agent_id}] --- end {latest_err.name} ---\n")
    except OSError:
        pass

    log_fh.flush()
