#!/usr/bin/env python3
"""Autonomous research workflow orchestrator.

Usage:
    python workflow/run.py status          Show current state and next phase
    python workflow/run.py next            Render and display the next phase prompt
    python workflow/run.py complete        Mark current phase as completed
    python workflow/run.py fail            Mark current phase as failed (triggers retry/branch)
    python workflow/run.py decide <choice> Record a branching decision
    python workflow/run.py log <message>   Append a message to the phase log
    python workflow/run.py measure <json>  Record measured quantities into state
    python workflow/run.py reset           Reset state to initial (with backup)
    python workflow/run.py render [phase]  Render a specific phase prompt without advancing
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import textwrap
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent
PHASES_DIR = WORKFLOW_DIR / "phases"
ARTIFACTS_DIR = WORKFLOW_DIR / "artifacts"
LOGS_DIR = WORKFLOW_DIR / "logs"
PROMPTS_DIR = WORKFLOW_DIR / "prompts"
STATE_FILE = WORKFLOW_DIR / "state.json"
MANIFEST_FILE = WORKFLOW_DIR / "phases.json"
REPO_ROOT = WORKFLOW_DIR.parent

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return _default_state()


def save_state(state: dict) -> None:
    state["updated_at"] = _now()
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def _default_state() -> dict:
    return {
        "current_phase": "00_overview",
        "completed": [],
        "failed": [],
        "skipped": [],
        "decisions": {},
        "measurements": {},
        "artifacts": {},
        "created_at": _now(),
        "updated_at": _now(),
        "retry_counts": {},
        "branch": "main",
    }


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Manifest (phase graph)
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    return json.loads(MANIFEST_FILE.read_text())


def get_phase(manifest: dict, phase_id: str) -> dict | None:
    return manifest.get(phase_id)


def resolve_next(manifest: dict, phase_id: str, decision: str | None = None) -> str | None:
    phase = manifest.get(phase_id, {})
    if decision and "branches" in phase:
        return phase["branches"].get(decision)
    return phase.get("next")


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_prompt(manifest: dict, state: dict, phase_id: str | None = None) -> str:
    pid = phase_id or state["current_phase"]
    phase = manifest.get(pid)
    if not phase:
        return f"ERROR: Unknown phase '{pid}'"

    md_path = PHASES_DIR / phase["file"]
    if not md_path.exists():
        return f"ERROR: Phase file not found: {md_path}"

    body = md_path.read_text()

    # Build context header
    context_lines = [
        "<!-- AUTO-GENERATED PROMPT — DO NOT EDIT -->",
        f"<!-- Phase: {pid} | Branch: {state['branch']} "
        f"| Generated: {_now()} -->",
        "",
        "## Injected Context",
        "",
        f"- **Current phase**: `{pid}`",
        f"- **Branch**: `{state['branch']}`",
        f"- **Completed phases**: {', '.join(state['completed']) or 'none'}",
        f"- **Repo root**: `{REPO_ROOT}`",
        f"- **Workflow dir**: `{WORKFLOW_DIR}`",
    ]

    if state["decisions"]:
        context_lines.append(f"- **Decisions so far**: "
                             f"{json.dumps(state['decisions'])}")
    if state["measurements"]:
        context_lines.append("")
        context_lines.append("### Key Measurements")
        context_lines.append("```json")
        context_lines.append(json.dumps(state["measurements"], indent=2))
        context_lines.append("```")
    if state["artifacts"]:
        context_lines.append("")
        context_lines.append("### Artifacts from Prior Phases")
        for k, v in state["artifacts"].items():
            context_lines.append(f"- `{k}`: `{v}`")

    context_lines.append("")
    context_lines.append("---")
    context_lines.append("")

    return "\n".join(context_lines) + body


def save_rendered_prompt(rendered: str, phase_id: str) -> Path:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PROMPTS_DIR / f"{phase_id}_prompt.md"
    out.write_text(rendered)
    return out


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def append_log(phase_id: str, message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{phase_id}.log"
    timestamp = _now()
    with open(log_file, "a") as fh:
        fh.write(f"[{timestamp}] {message}\n")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_status(state: dict, manifest: dict) -> None:
    pid = state["current_phase"]
    phase = manifest.get(pid, {})

    print(f"=== Workflow Status ===")
    print(f"  Current phase : {pid} — {phase.get('title', '?')}")
    print(f"  Branch        : {state['branch']}")
    print(f"  Completed     : {len(state['completed'])} phases")
    print(f"  Failed        : {len(state['failed'])} phases")
    print(f"  Last updated  : {state['updated_at']}")

    if state["completed"]:
        print(f"\n  Completed phases:")
        for c in state["completed"]:
            t = manifest.get(c, {}).get("title", "")
            print(f"    [{c}] {t}")

    if state["decisions"]:
        print(f"\n  Decisions:")
        for k, v in state["decisions"].items():
            print(f"    {k}: {v}")

    if state["measurements"]:
        print(f"\n  Measurements:")
        for k, v in state["measurements"].items():
            print(f"    {k}: {v}")

    # Show what comes next
    phase_type = phase.get("type", "unknown")
    print(f"\n  Next action: execute phase '{pid}' (type: {phase_type})")
    if "branches" in phase:
        print(f"  Branches: {list(phase['branches'].keys())}")
    elif "next" in phase:
        print(f"  Then: {phase['next']}")


def cmd_next(state: dict, manifest: dict) -> None:
    pid = state["current_phase"]
    phase = manifest.get(pid)
    if not phase:
        print(f"ERROR: Unknown phase '{pid}'")
        sys.exit(1)

    rendered = render_prompt(manifest, state, pid)
    out_path = save_rendered_prompt(rendered, pid)
    append_log(pid, "Phase prompt rendered")

    print(rendered)
    print(f"\n{'=' * 60}")
    print(f"Rendered prompt saved to: {out_path}")
    print(f"Phase type: {phase.get('type', 'unknown')}")
    print(f"\nAfter completing this phase, run:")
    print(f"  python workflow/run.py complete")
    if "branches" in phase:
        branches = list(phase["branches"].keys())
        print(f"\nOr, if this is a decision point:")
        print(f"  python workflow/run.py decide <{' | '.join(branches)}>")


def cmd_complete(state: dict, manifest: dict, output_file: str | None,
                 measurements_json: str | None) -> None:
    pid = state["current_phase"]
    phase = manifest.get(pid)
    if not phase:
        print(f"ERROR: Unknown phase '{pid}'")
        sys.exit(1)

    # Record output artifact
    if output_file:
        output_path = Path(output_file)
        if output_path.exists():
            dest = ARTIFACTS_DIR / f"{pid}_output{output_path.suffix}"
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, dest)
            state["artifacts"][pid] = str(dest)
            append_log(pid, f"Artifact saved: {dest}")
        else:
            state["artifacts"][pid] = output_file
            append_log(pid, f"Artifact path recorded: {output_file}")

    # Record measurements
    if measurements_json:
        try:
            m = json.loads(measurements_json)
            state["measurements"].update(m)
            append_log(pid, f"Measurements recorded: {measurements_json}")
        except json.JSONDecodeError:
            print(f"WARNING: Could not parse measurements JSON: {measurements_json}")

    # Mark completed
    state["completed"].append(pid)
    append_log(pid, "Phase completed")

    # Advance to next
    next_id = resolve_next(manifest, pid)
    if next_id:
        state["current_phase"] = next_id
        append_log(pid, f"Advancing to: {next_id}")
        print(f"Phase '{pid}' completed.")
        print(f"Next phase: '{next_id}' — {manifest.get(next_id, {}).get('title', '?')}")
    else:
        state["current_phase"] = "__done__"
        print(f"Phase '{pid}' completed.")
        print("Workflow finished. All phases complete.")

    save_state(state)


def cmd_fail(state: dict, manifest: dict, reason: str | None) -> None:
    pid = state["current_phase"]
    phase = manifest.get(pid, {})
    retries = state["retry_counts"].get(pid, 0)

    state["retry_counts"][pid] = retries + 1
    msg = reason or "unspecified"
    append_log(pid, f"Phase failed (attempt {retries + 1}): {msg}")

    if retries + 1 >= MAX_RETRIES:
        state["failed"].append(pid)
        # Check for failure branch
        fail_target = phase.get("on_fail")
        if fail_target:
            state["current_phase"] = fail_target
            print(f"Phase '{pid}' failed after {retries + 1} attempts. "
                  f"Branching to: {fail_target}")
        else:
            state["current_phase"] = "__failed__"
            print(f"Phase '{pid}' failed after {retries + 1} attempts. "
                  f"No fallback branch. Workflow halted.")
    else:
        print(f"Phase '{pid}' failed (attempt {retries + 1}/{MAX_RETRIES}). "
              f"Retry by running: python workflow/run.py next")

    save_state(state)


def cmd_decide(state: dict, manifest: dict, choice: str) -> None:
    pid = state["current_phase"]
    phase = manifest.get(pid, {})
    branches = phase.get("branches", {})

    if choice not in branches:
        print(f"ERROR: Invalid choice '{choice}' for phase '{pid}'.")
        print(f"Valid choices: {list(branches.keys())}")
        sys.exit(1)

    state["decisions"][pid] = choice
    state["completed"].append(pid)
    next_id = branches[choice]
    state["current_phase"] = next_id
    append_log(pid, f"Decision: {choice} → {next_id}")

    # Update branch tracking
    if choice == "escalate_cifar100":
        state["branch"] = "cifar100"
    elif choice == "structured_search":
        state["branch"] = "structured_search"

    save_state(state)
    print(f"Decision recorded: '{choice}' at phase '{pid}'.")
    print(f"Next phase: '{next_id}' — {manifest.get(next_id, {}).get('title', '?')}")


def cmd_log(state: dict, message: str) -> None:
    pid = state["current_phase"]
    append_log(pid, message)
    print(f"Logged to {pid}: {message}")


def cmd_measure(state: dict, measurements_json: str) -> None:
    try:
        m = json.loads(measurements_json)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON: {measurements_json}")
        sys.exit(1)
    state["measurements"].update(m)
    save_state(state)
    pid = state["current_phase"]
    append_log(pid, f"Measurements: {measurements_json}")
    print(f"Recorded: {json.dumps(m, indent=2)}")


def cmd_render(state: dict, manifest: dict, phase_id: str | None) -> None:
    pid = phase_id or state["current_phase"]
    rendered = render_prompt(manifest, state, pid)
    print(rendered)


def cmd_reset(state: dict) -> None:
    if STATE_FILE.exists():
        backup = STATE_FILE.with_suffix(f".{_now().replace(':', '-')}.bak.json")
        shutil.copy2(STATE_FILE, backup)
        print(f"Backed up state to: {backup}")
    new_state = _default_state()
    save_state(new_state)
    print("State reset to initial.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous research workflow orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Commands:
              status              Show current state and next phase
              next                Render and display the next phase prompt
              complete            Mark current phase as completed
              fail                Mark current phase as failed
              decide <choice>     Record a branching decision
              log <message>       Append to phase log
              measure <json>      Record measurements into state
              render [--phase X]  Render a phase prompt without advancing
              reset               Reset to initial state (with backup)
        """),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show current state")
    sub.add_parser("next", help="Render next phase prompt")

    p_complete = sub.add_parser("complete", help="Mark phase completed")
    p_complete.add_argument("--output", type=str, default=None,
                            help="Path to output artifact file")
    p_complete.add_argument("--measurements", type=str, default=None,
                            help="JSON string of measured quantities")

    p_fail = sub.add_parser("fail", help="Mark phase as failed")
    p_fail.add_argument("--reason", type=str, default=None)

    p_decide = sub.add_parser("decide", help="Record decision")
    p_decide.add_argument("choice", type=str)

    p_log = sub.add_parser("log", help="Append log message")
    p_log.add_argument("message", type=str)

    p_measure = sub.add_parser("measure", help="Record measurements")
    p_measure.add_argument("json_str", type=str, metavar="JSON")

    p_render = sub.add_parser("render", help="Render a phase prompt")
    p_render.add_argument("--phase", type=str, default=None)

    sub.add_parser("reset", help="Reset state to initial")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Ensure directories exist
    for d in [ARTIFACTS_DIR, LOGS_DIR, PROMPTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    state = load_state()
    manifest = load_manifest()

    if args.command == "status":
        cmd_status(state, manifest)
    elif args.command == "next":
        cmd_next(state, manifest)
    elif args.command == "complete":
        cmd_complete(state, manifest, args.output, args.measurements)
    elif args.command == "fail":
        cmd_fail(state, manifest, args.reason)
    elif args.command == "decide":
        cmd_decide(state, manifest, args.choice)
    elif args.command == "log":
        cmd_log(state, args.message)
    elif args.command == "measure":
        cmd_measure(state, args.json_str)
    elif args.command == "render":
        cmd_render(state, manifest, args.phase)
    elif args.command == "reset":
        cmd_reset(state)


if __name__ == "__main__":
    main()
