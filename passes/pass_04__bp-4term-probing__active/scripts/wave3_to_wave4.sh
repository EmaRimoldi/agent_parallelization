#!/usr/bin/env bash
# Monitors Wave 3 completion, runs analysis, then launches Wave 4.
# Usage: bash workflow/scripts/wave3_to_wave4.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[$(date)] Waiting for Wave 3 runner (PID 17937) to finish..."

# Wait for the Wave 3 runner process to finish
while kill -0 17937 2>/dev/null; do
    # Check P11-P14 status
    for pid in P11 P12 P13 P14; do
        dir="runs/experiment_probe_${pid}"
        if [ -d "$dir" ]; then
            count=$(find "$dir" -name "training_runs.jsonl" -exec cat {} \; 2>/dev/null | grep -c "val_bpb" || true)
            echo -n "  ${pid}:${count}runs "
        else
            echo -n "  ${pid}:-- "
        fi
    done
    echo "  [$(date +%H:%M)]"
    sleep 60
done

echo ""
echo "[$(date)] Wave 3 runner finished!"
echo ""

# Run analysis
echo "[$(date)] Running Wave 3 analysis..."
python3 workflow/scripts/analyze_all_probes.py --repo-root . 2>&1 | tee workflow/logs/probe_wave3_analysis.log

echo "[$(date)] Generating Wave 3 figures..."
python3 workflow/scripts/plot_probes.py --repo-root . 2>&1

echo ""
echo "[$(date)] Launching Wave 4..."
python3 workflow/scripts/run_probes.py --repo-root . --wave 4 --no-wait 2>&1 | tee workflow/logs/probe_wave4_full.log

echo ""
echo "[$(date)] Wave 4 complete! Running final analysis..."
python3 workflow/scripts/analyze_all_probes.py --repo-root . 2>&1 | tee workflow/logs/probe_wave4_analysis.log
python3 workflow/scripts/plot_probes.py --repo-root . 2>&1

echo "[$(date)] All done. Check results in results/figures/phase_04_probes/"
