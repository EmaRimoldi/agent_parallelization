#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="$ROOT_DIR/ai_task_passes/pass_02_theory_formalization"
STATE_DIR="$TASK_DIR/.runner_state"
INDEX_FILE="$STATE_DIR/current_index"
DONE_LOG="$STATE_DIR/completed.log"

usage() {
    cat <<'EOF'
Usage:
  ./ai_task_passes/run_pass_02_theory_formalization.sh start
  ./ai_task_passes/run_pass_02_theory_formalization.sh status
  ./ai_task_passes/run_pass_02_theory_formalization.sh list
  ./ai_task_passes/run_pass_02_theory_formalization.sh current
  ./ai_task_passes/run_pass_02_theory_formalization.sh show
  ./ai_task_passes/run_pass_02_theory_formalization.sh complete "note"
  ./ai_task_passes/run_pass_02_theory_formalization.sh reset --force

Behavior:
  - keeps a pointer to the current task Markdown file
  - marks a task complete only when `complete` is called
  - after completion, automatically advances and prints the next task
EOF
}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

discover_tasks() {
    TASK_FILES=()
    while IFS= read -r file; do
        TASK_FILES+=("$file")
    done < <(find "$TASK_DIR" -maxdepth 1 -type f -name 'task_*.md' | sort)
    if [[ ${#TASK_FILES[@]} -eq 0 ]]; then
        echo "ERROR: no task files found in $TASK_DIR" >&2
        exit 1
    fi
}

ensure_state() {
    discover_tasks
    mkdir -p "$STATE_DIR"
    if [[ ! -f "$INDEX_FILE" ]]; then
        echo "0" > "$INDEX_FILE"
    fi
    touch "$DONE_LOG"
}

read_index() {
    ensure_state
    local idx
    idx="$(cat "$INDEX_FILE")"
    if ! [[ "$idx" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid state index: $idx" >&2
        exit 1
    fi
    echo "$idx"
}

task_count() {
    ensure_state
    echo "${#TASK_FILES[@]}"
}

load_runtime_state() {
    ensure_state
    CURRENT_INDEX="$(cat "$INDEX_FILE")"
    if ! [[ "$CURRENT_INDEX" =~ ^[0-9]+$ ]]; then
        echo "ERROR: invalid state index: $CURRENT_INDEX" >&2
        exit 1
    fi
    TASK_TOTAL="${#TASK_FILES[@]}"
}

show_task_by_index() {
    local idx="$1"
    ensure_state
    local total="${#TASK_FILES[@]}"
    if (( idx < 0 || idx >= total )); then
        echo "No task at index $idx"
        return 1
    fi
    local file="${TASK_FILES[$idx]}"
    echo "============================================================"
    echo "Task $((idx + 1)) / $total"
    echo "File: $file"
    echo "============================================================"
    cat "$file"
}

cmd_start() {
    load_runtime_state
    if (( CURRENT_INDEX >= TASK_TOTAL )); then
        echo "All tasks are already complete."
        return 0
    fi
    show_task_by_index "$CURRENT_INDEX"
}

cmd_status() {
    load_runtime_state
    local completed="$CURRENT_INDEX"
    echo "Task directory: $TASK_DIR"
    echo "Completed: $completed / $TASK_TOTAL"
    if (( CURRENT_INDEX >= TASK_TOTAL )); then
        echo "Current: none (all tasks complete)"
    else
        echo "Current: ${TASK_FILES[$CURRENT_INDEX]}"
    fi
    echo "Done log: $DONE_LOG"
}

cmd_list() {
    load_runtime_state
    local i
    for (( i = 0; i < ${#TASK_FILES[@]}; i++ )); do
        if (( i < CURRENT_INDEX )); then
            echo "[done]    ${TASK_FILES[$i]}"
        elif (( i == CURRENT_INDEX )); then
            echo "[current] ${TASK_FILES[$i]}"
        else
            echo "[pending] ${TASK_FILES[$i]}"
        fi
    done
}

cmd_current() {
    load_runtime_state
    if (( CURRENT_INDEX >= TASK_TOTAL )); then
        echo "All tasks complete"
        return 0
    fi
    echo "${TASK_FILES[$CURRENT_INDEX]}"
}

cmd_show() {
    load_runtime_state
    if (( CURRENT_INDEX >= TASK_TOTAL )); then
        echo "All tasks complete"
        return 0
    fi
    show_task_by_index "$CURRENT_INDEX"
}

cmd_complete() {
    local note="${*:-}"
    load_runtime_state
    if (( CURRENT_INDEX >= TASK_TOTAL )); then
        echo "All tasks are already complete."
        return 0
    fi

    local file="${TASK_FILES[$CURRENT_INDEX]}"
    printf '%s\t%s\t%s\n' "$(timestamp)" "$file" "$note" >> "$DONE_LOG"
    local next=$((CURRENT_INDEX + 1))
    echo "$next" > "$INDEX_FILE"

    echo "Marked complete: $file"
    if [[ -n "$note" ]]; then
        echo "Note: $note"
    fi

    if (( next >= TASK_TOTAL )); then
        echo
        echo "All tasks are now complete."
        return 0
    fi

    echo
    echo "Advancing to next task..."
    echo
    show_task_by_index "$next"
}

cmd_reset() {
    if [[ "${1:-}" != "--force" ]]; then
        echo "Refusing to reset without --force"
        exit 1
    fi
    rm -rf "$STATE_DIR"
    ensure_state
    echo "Runner state reset."
    cmd_status
}

main() {
    local cmd="${1:-}"
    shift || true
    case "$cmd" in
        start) cmd_start "$@" ;;
        status) cmd_status "$@" ;;
        list) cmd_list "$@" ;;
        current) cmd_current "$@" ;;
        show) cmd_show "$@" ;;
        complete) cmd_complete "$@" ;;
        reset) cmd_reset "$@" ;;
        ""|-h|--help|help) usage ;;
        *)
            echo "Unknown command: $cmd" >&2
            usage
            exit 1
            ;;
    esac
}

main "$@"
