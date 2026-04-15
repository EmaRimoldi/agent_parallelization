#!/bin/bash
# Full exploration loop: Wave 1 → analyze → Wave 2 → analyze → Wave 3
# This script runs all waves sequentially and continues iterating.
# Each wave builds on the previous one's results.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "  BP 2x2 Full Exploration Loop"
echo "  Started: $(date)"
echo "  Repo: $REPO_ROOT"
echo "================================================================"

# Wave 1: Signal detection (6 probes × 15 min = ~90 min)
echo ""
echo ">>> WAVE 1: Signal Detection"
echo ">>> Started: $(date)"
python workflow/scripts/run_probes.py --repo-root . --wave 1 --no-wait 2>&1 | tee workflow/logs/probe_wave1_full.log

echo ""
echo ">>> WAVE 1 COMPLETE: $(date)"
echo ">>> Analyzing results..."

# Auto-analyze Wave 1 and design Wave 2
python workflow/scripts/design_wave2.py --repo-root . 2>&1 | tee workflow/logs/wave2_design.log

# Wave 2: Focused follow-up (designed by Wave 1 analysis)
echo ""
echo ">>> WAVE 2: Focused Follow-up"
echo ">>> Started: $(date)"
python workflow/scripts/run_probes.py --repo-root . --wave 2 --no-wait 2>&1 | tee workflow/logs/probe_wave2_full.log

echo ""
echo ">>> WAVE 2 COMPLETE: $(date)"

# Final analysis
python workflow/scripts/analyze_all_probes.py --repo-root . 2>&1 | tee workflow/logs/final_probe_analysis.log

echo ""
echo "================================================================"
echo "  EXPLORATION COMPLETE: $(date)"
echo "================================================================"
