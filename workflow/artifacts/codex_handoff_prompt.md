# Codex Handoff Prompt — BP 2×2 Instrumentation Experiment

**Generated**: 2026-04-13
**Branch**: `bp-2x2-instrumentation`
**Repo**: `/Users/emanuelerimoldi/Documents/GitHub/agent_parallelization`

---

## Your Role

You are a proactive research engineer embedded in an ongoing empirical study. Your job is to:
1. Deeply understand what is being tested and why
2. Identify the current experimental state and what needs to happen next
3. Prepare analysis infrastructure and parallel workstreams
4. Propose extensions, alternatives, and refinements to the experimental plan

Do NOT wait for instructions — act as an autonomous collaborator who reads the full context, plans ahead, and executes.

---

## Project Overview

We are empirically validating the **Beneventano-Poggio (BP) four-term decomposition** applied to autonomous coding agents:

```
Δ = log(κ₀/κ) + φ + G − ε
```

Where:
- **κ (per-step cost)**: compute cost per agent action — measured via token counts and wall-clock time
- **φ (within-mode competence)**: how well the agent optimizes within a single strategy class
- **G = I_π(S;D) (information generation)**: mutual information between agent strategies (S) and dataset characteristics (D) — measures exploration diversity
- **ε = E_D[KL(π_D ‖ q_D)] (routing mismatch)**: divergence between the agent's actual mode distribution and the optimal routing — only nonzero when memory/sharing enables conditioning on prior results

The framework predicts that different **architecture configurations** (parallelism × memory) will shift these terms in characteristic ways. We test this with a **2×2 design**:

| Cell | Agents | Memory | Expected BP Signature |
|------|--------|--------|----------------------|
| **d00** | 1 (single) | None | Baseline: φ only, G=0, ε=0 |
| **d10** | 1 (single) | Shared | φ + ε term appears (memory enables conditioning) |
| **d01** | N (parallel) | None | G term appears (independent exploration), ε=0 |
| **d11** | N (parallel) | Shared | All four terms active, potential for highest Δ |

The substrate is a **CIFAR-10 neural network optimization task**: agents modify `train.py` to minimize `val_bpb` (validation bits-per-byte). Each agent session has a fixed time budget and deterministic evaluation (seed=42, MAX_STEPS=585, num_workers=0).

---

## Current State: Phase 02 — Power Calibration

### What Has Been Completed

1. **Phase 00**: Overview and workflow setup
2. **Phase 01 + 01a + 01b**: Deterministic evaluation verified
   - Replaced time-based training loop with step-based (MAX_STEPS=585)
   - Fixed all random seeds (SEED=42, PYTHONHASHSEED, torch deterministic)
   - Verified: 5 runs produce identical val_bpb = 0.925845
3. **Phase 02 (d00)**: All 5 replicates COMPLETE
   - d00_rep1: 12 runs, best=0.855528
   - d00_rep2: 2 runs, best=0.925845
   - d00_rep3: 4 runs, best=0.925845
   - d00_rep4: 14 runs, best=0.921272
   - d00_rep5: 15 runs, best=0.824019
   - **Total: 47 runs across 5 reps**
4. **Phase 02 (d10)**: IN PROGRESS
   - d10_rep1: 3 runs, best=0.925845 ✅ COMPLETE
   - d10_rep2: IN PROGRESS (agent running, ~3 iterations so far)
   - d10_rep3–5: QUEUED

### Key Files to Read (in priority order)

```
# 1. Understand the workflow engine and current state
workflow/state.json                          # Current phase, measurements, decisions
workflow/phases/*.md                         # All 15 phase definitions (the DAG)
workflow/run.py                              # Orchestrator CLI

# 2. Understand the experiment infrastructure
src/agent_parallelization_new/launcher.py    # How experiments are launched
src/agent_parallelization_new/orchestrator.py # Agent lifecycle management
configs/experiment_d00.yaml                  # d00 cell config
configs/experiment_d10.yaml                  # d10 cell config
workflow/scripts/run_calibration.py          # Calibration runner (currently executing)

# 3. Understand the substrate
autoresearch/train.py                        # What agents modify (the optimization target)
autoresearch/prepare.py                      # Evaluation harness (read-only for agents)

# 4. Analysis scripts (some need updates/creation)
workflow/scripts/analyze_calibration.py      # Calibration analysis
workflow/scripts/check_decision_gate.py      # Decision gate evaluation
scripts/label_modes.py                       # Post-hoc mode labeling
scripts/decompose_four_term.py               # BP four-term decomposition estimators

# 5. Results (when available)
runs/experiment_calibration_d00_rep*/        # d00 experiment outputs
runs/experiment_calibration_d10_rep*/        # d10 experiment outputs (partial)

# 6. Theory documents
BP.pdf                                       # The BP framework paper
autoresearch_bp.pdf                          # Application to AutoResearch
autoresearch_bp_revised.pdf                  # Revised version
```

### What Is Currently Running

A background process (`run_calibration.py`) is sequentially executing d10_rep2 through d10_rep5. Each rep launches a single-memory agent (claude-haiku-4-5-20251001) that gets 45 minutes to iteratively modify train.py. The agent has access to shared memory from prior iterations within the same rep.

**You should NOT interfere with the running experiments.** Focus on preparation and analysis.

---

## Immediate Next Actions (after d10 completes)

### Action 1: Complete Phase 02 → Mark as done
When all 10 reps are finished, update `workflow/state.json`.

### Action 2: Phase 02a — Analyze Calibration
Read `workflow/phases/02a_analyze_calibration.md` for full details. Key tasks:
1. Label modes for all runs using `scripts/label_modes.py`
2. Compute per-cell statistics (mean, std, min, max val_bpb)
3. Compute **Cohen's d** = (mean_d10 - mean_d00) / pooled_std
4. Compute **mode diversity** per cell (distinct strategy categories with ≥2 accepted edits)
5. Compute **Jensen gap** and cost variance on token and wall-clock axes
6. Write `workflow/artifacts/calibration_analysis.json` and `workflow/artifacts/calibration_summary.md`

### Action 3: Phase 02b — Decision Gate
Read `workflow/phases/02b_decision_gate.md`. The decision matrix:

| Effect (d) | Modes | Runs | Decision |
|------------|-------|------|----------|
| d > 0.3 | ≥ 3 | ≥ 50 | **proceed** to full 2×2 |
| d > 0.3 | < 3 | any | **extend_budget** |
| d > 0.3 | any | < 30 | **extend_budget** |
| d ≤ 0.3 | any | ≥ 50 | **escalate_cifar100** |
| d ≤ 0.1 | ≤ 1 | any | **structured_search** |

### Action 4: Branch to Next Phase
Based on the gate decision, proceed to one of:
- `03_full_2x2_run` (proceed) — launch d01 and d11 cells
- `04_escalation_cifar100` (escalate) — switch to harder dataset
- `05_structured_search` (search) — restructure agent interface

---

## Parallel Workstreams You Can Start Now

While waiting for d10 to finish, prepare these independently:

### Workstream A: Analysis Infrastructure Validation
- Verify `workflow/scripts/analyze_calibration.py` works with the d00 data that's already complete
- Run it on d00 only to catch any bugs before d10 data arrives
- Check that glob patterns match actual directory names (`runs/experiment_calibration_*`)

### Workstream B: Mode Labeling Pipeline
- Run `scripts/label_modes.py` on completed d00 reps
- Verify the two-level classification (subsystem × change_type) produces meaningful categories
- Check if the output format is compatible with `analyze_calibration.py`

### Workstream C: Four-Term Decomposition Estimator Readiness
- Read `scripts/decompose_four_term.py` and verify it handles the calibration data format
- The estimators need: per-run val_bpb, mode labels, step costs (tokens + wall-clock)
- Prepare to compute the decomposition as soon as calibration analysis is complete

### Workstream D: d01/d11 Config Preparation
- If the decision gate says "proceed", we'll need configs for parallel agent cells
- Read `configs/experiment_d00.yaml` and `configs/experiment_d10.yaml` as templates
- Draft `configs/experiment_d01.yaml` (parallel, no sharing) and `configs/experiment_d11.yaml` (parallel, shared memory)
- Check `src/agent_parallelization_new/` for mode support — d11 is implemented, d01 may need verification

### Workstream E: Visualization and Reporting
- Design figures for the calibration summary:
  - Box plot of val_bpb distributions (d00 vs d10)
  - Per-rep trajectory plots (val_bpb over training iterations)
  - Mode distribution bar charts
  - Jensen gap comparison

---

## Beyond the Current Plan: Additional Experiments to Consider

### 1. Budget Sensitivity Analysis
The current calibration uses 45-min total / 60s training per run. The number of iterations per rep varies dramatically (2–15 runs). Consider:
- Is the budget sufficient for the memory advantage to manifest in d10?
- Would a longer budget (90 min) amplify or wash out the d10-d00 difference?
- A mini-study: 2 reps each at {30, 45, 60, 90} min budgets for d00 and d10

### 2. Agent Model Sensitivity
Currently using `claude-haiku-4-5-20251001`. The BP decomposition assumes agents have meaningful optimization capability. Consider:
- Would a more capable model (Sonnet) produce different mode distributions?
- Is Haiku's competence ceiling limiting the observable φ term?
- Cost-benefit: 1-2 reps with Sonnet as a robustness check

### 3. Substrate Complexity Gradient
Instead of the binary CIFAR-10 → CIFAR-100 escalation, consider a gradient:
- CIFAR-10 with reduced model capacity (fewer channels/layers)
- CIFAR-10 with full capacity (current)
- CIFAR-100 with reduced capacity
- This gives a cleaner signal about where the architecture contrast emerges

### 4. Memory Ablation in d10
The d10 cell provides "shared memory" — but what exactly does the agent see? Consider:
- Ablate memory content: full history vs. best-only vs. last-N-runs
- This directly tests whether ε (routing mismatch) responds to information quality
- Cheap to implement: modify the memory injection in the orchestrator

### 5. Theorem Refinement: Jensen Remainder Decomposition
The BP paper's wall-clock axis involves the Jensen remainder R_α = log E[κ_α] − E[log κ_α]. Consider:
- Is the empirical R_α dominated by a few outlier runs?
- Does R_α differ systematically between d00 and d10?
- Can we decompose R_α further by mode to understand which strategies drive cost variance?

### 6. Counterfactual: Random Search Baseline
A key question: does the agent's strategy (intelligent modification of train.py) outperform random hyperparameter search? Consider:
- Implement a random baseline that samples from the same hyperparameter space the agents explore
- This establishes whether G (information generation) is truly nonzero
- If random search matches agent performance, the entire framework needs recalibration

---

## Constraints and Gotchas

1. **Determinism is critical**: All evaluation uses SEED=42, MAX_STEPS=585, num_workers=0. Never change these.
2. **prepare.py is read-only**: Agents can only modify train.py. The evaluation harness is frozen.
3. **Directory naming**: Actual experiment dirs use `experiment_calibration_d00_rep*` prefix (not just `calibration_d00_rep*`). Check patterns.
4. **Agent model**: Sub-agents are `claude-haiku-4-5-20251001`. The launcher handles API authentication via `claude --print` CLI.
5. **No SLURM**: This runs on a local Mac (darwin, ARM). No GPU — all training is CPU-based with `num_workers=0`.
6. **Git worktrees**: Each agent runs in an isolated git worktree. Don't touch worktree directories under `runs/`.
7. **The background calibration runner is PID 36706** — do not kill it.

---

## Success Criteria for Your Session

By the end of your session, you should have:
- [ ] Read and understood the full workflow DAG (all 15 phases)
- [ ] Validated analysis scripts against existing d00 data
- [ ] Prepared mode labeling pipeline for immediate execution
- [ ] Drafted d01/d11 configs (contingent on "proceed" decision)
- [ ] Identified any bugs or gaps in the analysis/decision pipeline
- [ ] Proposed at least 2 concrete additional experiments with rationale
- [ ] Left a summary of findings and recommendations in `workflow/artifacts/codex_session_notes.md`
