#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_tasks.sh — Unattended sequential task runner for Claude Code / Codex
#
# Usage:
#   chmod +x run_tasks.sh
#   ./run_tasks.sh                    # runs all tasks in order
#   ./run_tasks.sh --from 3           # resume from task 3
#   ./run_tasks.sh --dry-run          # print prompts without executing
#   ./run_tasks.sh --agent codex      # use codex instead of claude
#
# Prerequisites:
#   - claude CLI (or codex CLI) installed and authenticated
#   - Repository cloned and on the correct branch
#   - IMPLEMENTATION_GUIDE.md and CPU_SUBSTRATE_GUIDE.md in repo root
#
# What it does:
#   1. Reads prompt files from ai_task_passes/pass_01_bp_implementation/
#      (task_01.md, task_02.md, ...)
#   2. For each task, invokes the agent CLI in the repo directory
#   3. Waits for completion
#   4. Logs stdout/stderr to ai_task_passes/pass_01_bp_implementation/logs/task_runner/task_XX.log
#   5. Checks exit code — stops on failure unless --continue-on-error
#   6. Commits changes after each successful task
#   7. Moves to next task
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_DIR="$ROOT_DIR/ai_task_passes/pass_01_bp_implementation"
LOGS_DIR="$ROOT_DIR/ai_task_passes/pass_01_bp_implementation/logs/task_runner"
AGENT="claude"
START_FROM=1
DRY_RUN=false
CONTINUE_ON_ERROR=false
MAX_RETRIES=1

# ─── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --from)       START_FROM="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --agent)      AGENT="$2"; shift 2 ;;
        --continue-on-error) CONTINUE_ON_ERROR=true; shift ;;
        --retries)    MAX_RETRIES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Setup ───────────────────────────────────────────────────────────
mkdir -p "$LOGS_DIR"
SUMMARY_LOG="$LOGS_DIR/summary.log"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() {
    echo "[$(timestamp)] $*" | tee -a "$SUMMARY_LOG"
}

# ─── Discover tasks ─────────────────────────────────────────────────
TASK_FILES=($(ls "$TASKS_DIR"/task_*.md 2>/dev/null | sort))
TOTAL_TASKS=${#TASK_FILES[@]}

if [[ $TOTAL_TASKS -eq 0 ]]; then
    echo "ERROR: No task files found in $TASKS_DIR/"
    echo "Expected files like: $TASKS_DIR/task_01.md, $TASKS_DIR/task_02.md, ..."
    exit 1
fi

log "═══════════════════════════════════════════════════════════"
log "Task runner started: $TOTAL_TASKS tasks found, starting from $START_FROM"
log "Agent: $AGENT | Dry run: $DRY_RUN | Continue on error: $CONTINUE_ON_ERROR"
log "═══════════════════════════════════════════════════════════"

# ─── Build agent command ─────────────────────────────────────────────
run_agent() {
    local prompt="$1"
    local logfile="$2"

    case "$AGENT" in
        claude)
            claude --print \
                --dangerously-skip-permissions \
                --output-format text \
                "$prompt" \
                > "$logfile" 2>&1
            ;;
        codex)
            # OpenAI Codex CLI
            codex --approval-mode full-auto \
                "$prompt" \
                > "$logfile" 2>&1
            ;;
        *)
            echo "Unknown agent: $AGENT"
            exit 1
            ;;
    esac
}

# ─── Main loop ───────────────────────────────────────────────────────
PASSED=0
FAILED=0

for TASK_FILE in "${TASK_FILES[@]}"; do
    # Extract task number from filename (task_01.md -> 1)
    TASK_NUM=$(basename "$TASK_FILE" | sed 's/task_0*\([0-9]*\)\.md/\1/')

    # Skip tasks before START_FROM
    if [[ $TASK_NUM -lt $START_FROM ]]; then
        continue
    fi

    TASK_NAME=$(head -1 "$TASK_FILE" | sed 's/^#* *//')
    TASK_LOG="$LOGS_DIR/task_$(printf '%02d' $TASK_NUM).log"
    PROMPT=$(cat "$TASK_FILE")

    log "───────────────────────────────────────────────────────"
    log "TASK $TASK_NUM/$TOTAL_TASKS: $TASK_NAME"
    log "  File: $TASK_FILE"
    log "  Log:  $TASK_LOG"

    if $DRY_RUN; then
        log "  [DRY RUN] Would execute prompt (${#PROMPT} chars)"
        echo "$PROMPT" | head -5
        echo "  ..."
        continue
    fi

    # Retry loop
    ATTEMPT=0
    SUCCESS=false

    while [[ $ATTEMPT -lt $MAX_RETRIES ]]; do
        ATTEMPT=$((ATTEMPT + 1))
        log "  Attempt $ATTEMPT/$MAX_RETRIES — starting at $(timestamp)"
        START_TIME=$(date +%s)

        if run_agent "$PROMPT" "$TASK_LOG"; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            log "  ✓ Completed in ${ELAPSED}s"
            SUCCESS=true
            break
        else
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            EXIT_CODE=$?
            log "  ✗ Failed (exit code $EXIT_CODE) after ${ELAPSED}s"
            # Save failed log with attempt suffix
            cp "$TASK_LOG" "${TASK_LOG%.log}_attempt${ATTEMPT}.log"
        fi
    done

    if $SUCCESS; then
        PASSED=$((PASSED + 1))

        # Auto-commit after successful task
        if git diff --quiet && git diff --staged --quiet; then
            log "  No changes to commit"
        else
            git add -A
            git commit -m "Task $TASK_NUM: $TASK_NAME [automated]" --no-verify 2>/dev/null || true
            log "  Committed changes"
        fi
    else
        FAILED=$((FAILED + 1))
        log "  ✗ FAILED after $MAX_RETRIES attempts"

        if ! $CONTINUE_ON_ERROR; then
            log ""
            log "Stopping. Resume with: ./run_tasks.sh --from $TASK_NUM"
            log "Check log: $TASK_LOG"
            break
        fi
    fi
done

# ─── Summary ─────────────────────────────────────────────────────────
log ""
log "═══════════════════════════════════════════════════════════"
log "SUMMARY: $PASSED passed, $FAILED failed, $TOTAL_TASKS total"
log "═══════════════════════════════════════════════════════════"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
