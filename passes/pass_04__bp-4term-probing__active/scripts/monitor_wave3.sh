#!/usr/bin/env bash
# Quick Wave 3 status check
REPO="/Users/emanuelerimoldi/Documents/GitHub/agent_parallelization"
echo "=== Wave 3 Status @ $(date) ==="
echo ""
for P in P11 P12 P13 P14; do
    DIR="$REPO/runs/experiment_probe_${P}"
    if [ ! -d "$DIR" ]; then
        echo "$P: NOT STARTED"
        continue
    fi
    # Count runs from training_runs.jsonl files
    TOTAL=0
    BEST=999
    for JSONL in $(find "$DIR" -name "training_runs.jsonl" 2>/dev/null); do
        AGENT=$(echo "$JSONL" | grep -o "agent_[0-9]*" | head -1)
        COUNT=$(python3 -c "
import json
runs = [json.loads(l) for l in open('$JSONL') if l.strip()]
bpbs = [r['val_bpb'] for r in runs if r.get('val_bpb')]
print(f'{len(runs)}|{min(bpbs) if bpbs else 999:.6f}')
" 2>/dev/null)
        N=$(echo "$COUNT" | cut -d'|' -f1)
        B=$(echo "$COUNT" | cut -d'|' -f2)
        TOTAL=$((TOTAL + N))
        if python3 -c "exit(0 if $B < $BEST else 1)" 2>/dev/null; then
            BEST=$B
        fi
        echo "  $P $AGENT: $N runs, best=$B"
    done
    echo "  $P TOTAL: $TOTAL runs, best=$BEST"
    echo ""
done
# Check if runner is still alive
if ps -p $(cat /tmp/wave3_pid 2>/dev/null || echo 17937) > /dev/null 2>&1; then
    echo "Runner: ALIVE (PID 17937)"
else
    echo "Runner: STOPPED"
fi
