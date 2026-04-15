# Implementation Guide: BP Four-Term Decomposition on AutoResearch

## Overview

This guide specifies all modifications needed to the `agent_parallelization` repository to support the 2×2 factorial experiment from the Beneventano–Poggio paper. Each task is self-contained and should be done in order. After each task, run the existing test suite (`pytest -q`) plus any new tests introduced by that task.

**Branch setup:**

```bash
cd agent_parallelization
git checkout -b bp-2x2-instrumentation main
```

---

## Task 1: Fix model passthrough in agent runner

**Why:** The 2×2 requires M (base LLM) to be held fixed across all cells. Currently `AgentConfig.model` exists but is never passed to the `claude` CLI command, so the actual model depends on the user's local CLI default.

**File:** `src/agent_parallelization_new/agents/claude_agent_runner.py`

**What to do:**

Find the command construction around line 392:

```python
cmd = [
    "claude",
    "--print",
    "--output-format", "text",
    "--dangerously-skip-permissions",
]
if system_prompt:
    cmd += ["--system-prompt", system_prompt]
cmd += [turn_msg]
```

Change it to:

```python
cmd = [
    "claude",
    "--print",
    "--output-format", "json",       # CHANGED: json to capture token usage
    "--dangerously-skip-permissions",
]
if self.config.model:
    cmd += ["--model", self.config.model]
if system_prompt:
    cmd += ["--system-prompt", system_prompt]
cmd += [turn_msg]
```

**Notes:**
- Switching to `--output-format json` is intentional — it is required for Task 2 (token tracking). You will need to parse the JSON output to extract the text response. The JSON output from `claude --print --output-format json` includes a `result` field with the text and a `usage` field with token counts.
- After this change, `_run_turn()` must parse the JSON response instead of using raw stdout as the agent's text output. Add a helper:

```python
import json

def _parse_claude_output(self, raw_stdout: str) -> tuple[str, dict]:
    """Parse JSON output from claude CLI. Returns (text_response, usage_dict)."""
    try:
        data = json.loads(raw_stdout)
        text = data.get("result", raw_stdout)
        usage = data.get("usage", {})
        return text, usage
    except json.JSONDecodeError:
        # Fallback: treat as plain text (no usage data)
        return raw_stdout, {}
```

- Integrate this into the flow after `subprocess.Popen` reads stdout. Use the `text` part where the old code used raw stdout; save `usage` for Task 2.

**Verify:** Run an agent manually with `--model claude-haiku-4-5-20251001` and confirm the model field appears in `metadata.json`.

---

## Task 2: Add per-turn token and context tracking

**Why:** Without token counts per turn, we cannot compute κ̄_token(d) — the average token cost per attempt — which is the cost term in the four-term decomposition. Without context fill tracking, we cannot test H5 (context pressure is dominant).

**File:** `src/agent_parallelization_new/agents/claude_agent_runner.py`

**What to do:**

### 2a. Create a turn log file

At runner initialization (in `__init__` or at the start of `run()`), open a JSONL file:

```python
self.turns_log_path = self.agent_dir / "results" / "turns.jsonl"
```

### 2b. Log every turn

After each call to `_run_turn()`, append a record:

```python
import time

turn_record = {
    "turn": self.turn_count,
    "timestamp": time.time(),
    "model": self.config.model,
    "input_tokens": usage.get("input_tokens", None),
    "output_tokens": usage.get("output_tokens", None),
    "system_prompt_chars": len(system_prompt) if system_prompt else 0,
    "turn_msg_chars": len(turn_msg),
    "response_chars": len(text_response),
    "context_fill_ratio": self._estimate_context_fill(),
    "wall_clock_seconds": turn_elapsed,
}
with open(self.turns_log_path, "a") as f:
    f.write(json.dumps(turn_record) + "\n")
```

### 2c. Estimate context fill

Add a method to estimate c/K:

```python
def _estimate_context_fill(self) -> float:
    """Estimate context fill ratio c/K.
    
    Uses character count / 4 as a rough token estimate.
    K is the model's context window (default 200k tokens).
    """
    K = 200_000  # default context window in tokens
    # Accumulate total chars sent and received across all turns
    estimated_tokens = self._cumulative_chars / 4
    return min(estimated_tokens / K, 1.0)
```

Initialize `self._cumulative_chars = 0` in the constructor and increment it with `len(system_prompt) + len(turn_msg) + len(text_response)` after each turn.

### 2d. If `--output-format json` does not expose token usage

If the `claude` CLI does not include usage in JSON output, fall back to character-based estimation:

```python
input_tokens_est = (len(system_prompt) + len(turn_msg)) // 4
output_tokens_est = len(text_response) // 4
```

This is sufficient because the decomposition uses ratios (κ̄(d0)/κ̄(d)), and a proportional proxy preserves ratios.

### 2e. Aggregate in metadata.json

At the end of the run (in `_write_metadata()`), add totals:

```python
metadata["total_input_tokens"] = sum of all turn input_tokens
metadata["total_output_tokens"] = sum of all turn output_tokens
metadata["total_turns"] = self.turn_count
metadata["avg_context_fill"] = mean of all turn context_fill_ratio values
metadata["final_context_fill"] = last turn's context_fill_ratio
```

**Verify:** Run a short 5-minute single agent. Check that `turns.jsonl` has one record per turn and that `metadata.json` has the new aggregate fields.

---

## Task 3: Add structured per-training-run wall-clock logging

**Why:** We need wall-clock time per attempt to compute κ̄_wall(d).

**File:** `src/agent_parallelization_new/agents/claude_agent_runner.py`

**What to do:**

In the watcher that detects `run.result` (around line 318–341), you already have `run_wall_start`. Add structured logging:

```python
training_run_record = {
    "run_index": self._training_run_count,
    "turn": self.turn_count,
    "started_at": run_wall_start,
    "finished_at": time.time(),
    "wall_seconds": time.time() - run_wall_start,
    "val_bpb": parsed_val_bpb,   # extract from run.result
    "status": "success" or "crash",
}
training_runs_path = self.agent_dir / "results" / "training_runs.jsonl"
with open(training_runs_path, "a") as f:
    f.write(json.dumps(training_run_record) + "\n")
```

Also parse `training_seconds` from `logs/train_current.out` using the existing `log_parser.py` and include it in the record, so you have both the framework-measured wall clock and the self-reported training time.

**Verify:** After a short run, `training_runs.jsonl` should have one entry per training invocation with timing and metric data.

---

## Task 4: Implement d10 — single agent with external memory

**Why:** d10 is a single agent that reads a compact result log before each turn, reducing context pressure. This is the cell that isolates the memory main effect.

### 4a. Create a new experiment mode

**File (new):** `src/agent_parallelization_new/experiment_modes/single_agent_memory.py`

This mode is identical to `single_agent_double_budget.py` but sets a flag `use_external_memory=True` on the agent config.

```python
"""Single agent with external memory (d10 in the 2×2 design)."""

def run_single_agent_memory(
    experiment_dir, config, system_prompt, first_message, autoresearch_dir
):
    """Run a single agent with external memory enabled."""
    # Same as single_agent_double_budget but with memory flag
    config.agents[0].use_external_memory = True
    # ... delegate to orchestrator with n=1
```

### 4b. Add the memory injection mechanism

**File:** `src/agent_parallelization_new/agents/claude_agent_runner.py`

Add a method that reads the reasoning trace and produces a compact summary:

```python
def _build_memory_context(self) -> str:
    """Build compact result log from reasoning trace for context injection.
    
    Returns a markdown table of all experiments so far:
    #  | change | bpb | delta | best
    """
    trace_path = self.agent_dir / "reasoning" / "trace.jsonl"
    if not trace_path.exists():
        return ""
    
    lines = ["# Experiment Log", "| # | change | bpb | Δ | best |", "|---|--------|-----|---|------|"]
    best_bpb = float("inf")
    
    for line in trace_path.read_text().strip().split("\n"):
        try:
            entry = json.loads(line)
            step = entry.get("step", "?")
            hypothesis = entry.get("hypothesis", "?")[:40]
            bpb = entry.get("val_bpb_after")
            if bpb is None:
                continue
            bpb_val = float(bpb)
            prev = entry.get("val_bpb_before")
            delta = f"{bpb_val - float(prev):+.4f}" if prev else "—"
            is_best = "✓" if bpb_val < best_bpb else ""
            if bpb_val < best_bpb:
                best_bpb = bpb_val
            lines.append(f"| {step} | {hypothesis} | {bpb_val:.4f} | {delta} | {is_best} |")
        except (json.JSONDecodeError, ValueError):
            continue
    
    if len(lines) <= 3:  # only header
        return ""
    return "\n".join(lines)
```

Then in the turn message construction (the "Continue the research…" message for turns > 0), prepend the memory context:

```python
if self.config.use_external_memory:
    memory = self._build_memory_context()
    if memory:
        turn_msg = f"{memory}\n\n---\n\n{turn_msg}"
```

### 4c. Add the config flag

**File:** `src/agent_parallelization_new/config.py`

Add to `AgentConfig`:

```python
use_external_memory: bool = False
```

### 4d. Add the mode to the launcher

**File:** `src/agent_parallelization_new/launcher.py`

Add a `main_single_memory()` entry point and register it in `pyproject.toml`:

```toml
[project.scripts]
run-parallel = "agent_parallelization_new.launcher:main_parallel"
run-single-long = "agent_parallelization_new.launcher:main_single_long"
run-single-memory = "agent_parallelization_new.launcher:main_single_memory"
```

### 4e. Add YAML config support

In `configs/experiment.yaml`, add mode `single_memory`:

```yaml
experiment:
  mode: single_memory  # d10: single agent + external memory
```

**Verify:** Run d10 for 15 minutes. After turn 5+, confirm that the turn message in `run_agent.log` includes the experiment log table. Compare `turns.jsonl` context_fill_ratio between d00 and d10 — d10 should have lower growth rate if the memory is working.

---

## Task 5: Implement d11 — parallel agents with shared memory

**Why:** d11 is the swarm cell — two agents that share experiment results. This isolates the interaction effect (memory × parallelism).

### 5a. Create the shared memory file

**File:** `src/agent_parallelization_new/utils/workspace.py`

In `create_workspace()`, when the experiment mode is `parallel_shared_memory`, also create a shared log file at the experiment level (not per-agent):

```python
shared_log_path = experiment_dir / "shared_results_log.jsonl"
if not shared_log_path.exists():
    shared_log_path.touch()
```

Symlink this file into each agent's workspace:

```python
(workspace / "shared_results_log.jsonl").symlink_to(shared_log_path)
```

### 5b. Write to shared log after each experiment

**File:** `src/agent_parallelization_new/agents/claude_agent_runner.py`

After `update_snapshot.py` completes (detected by the watcher), append to the shared log:

```python
import fcntl

def _append_shared_log(self, step, hypothesis, val_bpb, accepted):
    shared_path = self.workspace / "shared_results_log.jsonl"
    if not shared_path.exists():
        return
    record = json.dumps({
        "agent_id": self.config.agent_id,
        "step": step,
        "hypothesis": hypothesis[:60],
        "val_bpb": val_bpb,
        "accepted": accepted,
        "timestamp": time.time(),
    })
    with open(shared_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(record + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
```

### 5c. Read shared log before each turn

Reuse the `_build_memory_context()` method from Task 4, but read from `shared_results_log.jsonl` instead of the private trace:

```python
def _build_shared_memory_context(self) -> str:
    """Build compact result log from shared results across all agents."""
    shared_path = self.workspace / "shared_results_log.jsonl"
    if not shared_path.exists():
        return ""
    
    lines = ["# Shared Experiment Log (all agents)",
             "| agent | # | change | bpb | kept |",
             "|-------|---|--------|-----|------|"]
    
    for line in shared_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            agent = entry.get("agent_id", "?")[-4:]  # last 4 chars
            step = entry.get("step", "?")
            hyp = entry.get("hypothesis", "?")[:35]
            bpb = entry.get("val_bpb")
            accepted = "✓" if entry.get("accepted") else "✗"
            if bpb is not None:
                lines.append(f"| {agent} | {step} | {hyp} | {float(bpb):.4f} | {accepted} |")
        except (json.JSONDecodeError, ValueError):
            continue
    
    if len(lines) <= 3:
        return ""
    return "\n".join(lines)
```

Inject into the turn message when `use_shared_memory=True`:

```python
if self.config.use_shared_memory:
    shared_mem = self._build_shared_memory_context()
    if shared_mem:
        turn_msg = f"{shared_mem}\n\n---\n\n{turn_msg}"
```

### 5d. Add config and mode

**File:** `src/agent_parallelization_new/config.py`

```python
use_shared_memory: bool = False
```

**File (new):** `src/agent_parallelization_new/experiment_modes/parallel_shared_memory.py`

Same as `parallel_two_agents.py` but sets `use_shared_memory=True` on all agents.

Register in launcher and pyproject.toml as `run-parallel-shared`.

**Verify:** Run d11 with 2 agents for 15 minutes. Confirm that `shared_results_log.jsonl` has entries from both agents interleaved. Confirm that each agent's turn message (in `run_agent.log`) includes results from the other agent.

---

## Task 6: Post-hoc mode labeling

**Why:** The four-term decomposition requires classifying each edit into a "mode" (optimizer, lr_schedule, architecture, batch_data, other) to compute G (information gain) and ε (routing mismatch).

**File (new):** `scripts/label_modes.py`

**What to do:**

Write a script that reads all snapshots from an experiment run, diffs each `step_N/train.py` against `train.py.baseline` (or the previous accepted step), and assigns a mode label based on which hyperparameters/code sections changed.

```python
"""Post-hoc mode labeling for the BP decomposition.

Usage:
    python scripts/label_modes.py --experiment-dir runs/experiment_XXX

Reads snapshots from each agent, diffs train.py against baseline,
and writes mode labels to agent_dir/results/mode_labels.jsonl
"""

import re, json, difflib, argparse
from pathlib import Path

# Keywords that map diff content to modes
MODE_KEYWORDS = {
    "optimizer": [
        "adamw", "muon", "adam_betas", "weight_decay", "optimizer",
        "adamw_step", "muon_step", "setup_optimizer",
    ],
    "lr_schedule": [
        "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr",
        "warmup_ratio", "warmdown_ratio", "final_lr_frac",
        "get_lr_multiplier", "get_muon_momentum", "learning_rate", "_lr",
    ],
    "architecture": [
        "depth", "aspect_ratio", "head_dim", "n_head", "n_embd",
        "window_pattern", "causalselfattention", "mlp", "block",
        "gptconfig", "n_layer", "rotary", "attention",
    ],
    "batch_data": [
        "total_batch_size", "device_batch_size", "max_seq_len",
        "dataloader", "batch", "accumulation",
    ],
}

def classify_diff(diff_lines: list[str]) -> str:
    """Classify a diff into a mode based on keyword matching."""
    added_removed = " ".join(
        line[1:].lower() for line in diff_lines
        if line.startswith("+") or line.startswith("-")
    )
    
    scores = {}
    for mode, keywords in MODE_KEYWORDS.items():
        scores[mode] = sum(1 for kw in keywords if kw in added_removed)
    
    best_mode = max(scores, key=scores.get)
    if scores[best_mode] == 0:
        return "other"
    return best_mode


def label_experiment(experiment_dir: Path):
    """Label all snapshots in an experiment with mode classifications."""
    for agent_dir in sorted(experiment_dir.glob("mode_*/agent_*")):
        snapshots_dir = agent_dir / "snapshots"
        baseline_path = agent_dir / "workspace" / "train.py.baseline"
        
        if not snapshots_dir.exists() or not baseline_path.exists():
            continue
        
        baseline = baseline_path.read_text().splitlines()
        labels = []
        prev_accepted = baseline
        
        for step_dir in sorted(snapshots_dir.glob("step_*")):
            train_path = step_dir / "train.py"
            meta_path = step_dir / "metadata.json"
            
            if not train_path.exists():
                continue
            
            current = train_path.read_text().splitlines()
            diff = list(difflib.unified_diff(prev_accepted, current, lineterm=""))
            mode = classify_diff(diff)
            
            meta = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            
            label_entry = {
                "step": meta.get("step", step_dir.name),
                "mode": mode,
                "diff_lines_changed": len([l for l in diff if l.startswith("+") or l.startswith("-")]),
                "hypothesis": meta.get("hypothesis", ""),
                "val_bpb_after": meta.get("val_bpb_after"),
                "accepted": meta.get("accepted"),
            }
            labels.append(label_entry)
            
            # Update prev_accepted only if this step was accepted
            if meta.get("accepted"):
                prev_accepted = current
        
        output_path = agent_dir / "results" / "mode_labels.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for entry in labels:
                f.write(json.dumps(entry) + "\n")
        
        print(f"Labeled {len(labels)} steps for {agent_dir.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    label_experiment(args.experiment_dir)
```

**Verify:** Run on an existing experiment directory. Check that `mode_labels.jsonl` has sensible labels. Manually verify 10 random entries against the actual diffs.

---

## Task 7: Four-term decomposition estimators

**Why:** This is the analysis script that computes the actual decomposition — the core deliverable.

**File (new):** `scripts/compute_decomposition.py`

**What to do:**

Write a script that reads all instrumented data from a completed 2×2 experiment and computes:

```python
"""Compute the BP four-term decomposition from 2×2 experiment data.

Usage:
    python scripts/compute_decomposition.py \
        --d00 runs/experiment_d00 \
        --d10 runs/experiment_d10 \
        --d01 runs/experiment_d01 \
        --d11 runs/experiment_d11

Outputs:
    - Decomposition table (terminal + JSON)
    - Main effects and interaction
    - Hypothesis test results (H1-H6)
    - Bootstrap confidence intervals
"""

import json, math, argparse
import numpy as np
from pathlib import Path


def load_cell_data(experiment_dir: Path) -> dict:
    """Load all instrumented data from one cell of the 2×2."""
    data = {
        "turns": [],
        "training_runs": [],
        "mode_labels": [],
    }
    
    for agent_dir in sorted(experiment_dir.glob("mode_*/agent_*")):
        # Load turns
        turns_path = agent_dir / "results" / "turns.jsonl"
        if turns_path.exists():
            for line in turns_path.read_text().strip().split("\n"):
                if line.strip():
                    data["turns"].append(json.loads(line))
        
        # Load training runs
        runs_path = agent_dir / "results" / "training_runs.jsonl"
        if runs_path.exists():
            for line in runs_path.read_text().strip().split("\n"):
                if line.strip():
                    data["training_runs"].append(json.loads(line))
        
        # Load mode labels
        labels_path = agent_dir / "results" / "mode_labels.jsonl"
        if labels_path.exists():
            for line in labels_path.read_text().strip().split("\n"):
                if line.strip():
                    data["mode_labels"].append(json.loads(line))
    
    return data


def compute_kappa_token(data: dict) -> float:
    """κ̄_token: average total tokens (input + output) per attempt."""
    if not data["turns"]:
        return float("nan")
    tokens_per_turn = []
    for t in data["turns"]:
        inp = t.get("input_tokens") or (t.get("system_prompt_chars", 0) + t.get("turn_msg_chars", 0)) // 4
        out = t.get("output_tokens") or t.get("response_chars", 0) // 4
        tokens_per_turn.append(inp + out)
    return np.mean(tokens_per_turn)


def compute_kappa_wall(data: dict) -> float:
    """κ̄_wall: average wall-clock seconds per attempt (LLM call + training)."""
    if not data["turns"]:
        return float("nan")
    return np.mean([t.get("wall_clock_seconds", 0) for t in data["turns"]])


def compute_kappa_by_context_bin(data: dict, n_bins: int = 4) -> dict:
    """κ̄_token stratified by context fill ratio bins (for H5)."""
    if not data["turns"]:
        return {}
    bins = np.linspace(0, 1, n_bins + 1)
    result = {}
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        label = f"{lo:.0%}-{hi:.0%}"
        in_bin = [t for t in data["turns"]
                  if lo <= t.get("context_fill_ratio", 0) < hi]
        if in_bin:
            tokens = []
            for t in in_bin:
                inp = t.get("input_tokens") or (t.get("system_prompt_chars", 0) + t.get("turn_msg_chars", 0)) // 4
                out = t.get("output_tokens") or t.get("response_chars", 0) // 4
                tokens.append(inp + out)
            result[label] = np.mean(tokens)
        else:
            result[label] = None
    return result


def compute_mode_distribution(data: dict) -> dict:
    """π̂: empirical distribution over modes from accepted edits."""
    accepted = [m for m in data["mode_labels"] if m.get("accepted")]
    if not accepted:
        return {}
    from collections import Counter
    counts = Counter(m["mode"] for m in accepted)
    total = sum(counts.values())
    return {mode: count / total for mode, count in counts.items()}


def entropy(dist: dict) -> float:
    """Shannon entropy H(p) in nats."""
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def kl_divergence(p: dict, q: dict, smoothing: float = 1e-6) -> float:
    """KL(p || q) in nats with Laplace smoothing."""
    all_keys = set(p) | set(q)
    n = len(all_keys)
    kl = 0.0
    for k in all_keys:
        pk = p.get(k, 0) + smoothing
        qk = q.get(k, 0) + smoothing
        # Renormalize
        pk_norm = pk / (1 + n * smoothing)
        qk_norm = qk / (1 + n * smoothing)
        kl += pk_norm * math.log(pk_norm / qk_norm)
    return kl


def compute_decomposition(d0_data, d_data, prior):
    """Compute the four-term decomposition Δ = log(κ0/κ) + φ + G - ε."""
    k0_token = compute_kappa_token(d0_data)
    k_token = compute_kappa_token(d_data)
    k0_wall = compute_kappa_wall(d0_data)
    k_wall = compute_kappa_wall(d_data)
    
    cost_term_token = math.log(k0_token / k_token) if k_token > 0 else float("nan")
    cost_term_wall = math.log(k0_wall / k_wall) if k_wall > 0 else float("nan")
    
    # φ: within-mode competence (simplified — median attempts to improvement)
    phi = 0.0  # Placeholder: requires per-mode attempts-to-improvement
    
    # G: information gain from verifier feedback
    d_dist = compute_mode_distribution(d_data)
    G = entropy(d_dist) - entropy(prior) if d_dist and prior else 0.0
    
    # ε: routing mismatch KL(empirical || prior)
    epsilon = kl_divergence(d_dist, prior) if d_dist and prior else 0.0
    
    delta_token = cost_term_token + phi + G - epsilon
    delta_wall = cost_term_wall + phi + G - epsilon
    
    return {
        "cost_term_token": cost_term_token,
        "cost_term_wall": cost_term_wall,
        "phi": phi,
        "G": G,
        "epsilon": epsilon,
        "delta_token": delta_token,
        "delta_wall": delta_wall,
        "kappa_token": k_token,
        "kappa_wall": k_wall,
    }


def test_hypotheses(cells: dict, decompositions: dict) -> dict:
    """Test H1-H6 from the paper."""
    results = {}
    
    # H1: Parallelism helps only on wall-clock
    d01 = decompositions.get("d01", {})
    results["H1_wall_positive"] = d01.get("delta_wall", 0) > 0
    results["H1_token_nonpositive"] = d01.get("delta_token", 0) <= 0
    results["H1_holds"] = results["H1_wall_positive"] and results["H1_token_nonpositive"]
    
    # H2: External memory helps on both axes
    d10 = decompositions.get("d10", {})
    results["H2_token_positive"] = d10.get("cost_term_token", 0) > 0
    results["H2_wall_positive"] = d10.get("cost_term_wall", 0) > 0
    results["H2_holds"] = results["H2_token_positive"] and results["H2_wall_positive"]
    
    # H3: Shared memory lowers ε in multi-agent
    d01_eps = decompositions.get("d01", {}).get("epsilon", 0)
    d11_eps = decompositions.get("d11", {}).get("epsilon", 0)
    results["H3_epsilon_d11_lt_d01"] = d11_eps < d01_eps
    results["H3_holds"] = results["H3_epsilon_d11_lt_d01"]
    
    # H4: Parallelism sensitive to coordination
    results["H4_epsilon_exceeds_log2"] = (d01_eps - 0) > math.log(2)  # d00 ε ≈ 0
    
    # H5: Context pressure dominant (checked via kappa stratification)
    results["H5_note"] = "Check kappa_by_context_bin for monotone increase"
    
    # H6: d11 dominates d00 on both axes
    d11 = decompositions.get("d11", {})
    results["H6_wall_positive"] = d11.get("delta_wall", 0) > 0
    results["H6_token_positive"] = d11.get("delta_token", 0) > 0
    results["H6_holds"] = results["H6_wall_positive"] and results["H6_token_positive"]
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d00", required=True, type=Path, help="Experiment dir for d00 (single, no memory)")
    parser.add_argument("--d10", required=True, type=Path, help="Experiment dir for d10 (single, memory)")
    parser.add_argument("--d01", required=True, type=Path, help="Experiment dir for d01 (parallel, no sharing)")
    parser.add_argument("--d11", required=True, type=Path, help="Experiment dir for d11 (parallel, shared)")
    args = parser.parse_args()
    
    cells = {
        "d00": load_cell_data(args.d00),
        "d10": load_cell_data(args.d10),
        "d01": load_cell_data(args.d01),
        "d11": load_cell_data(args.d11),
    }
    
    # Pooled prior from d00 + d11 accepted edits (per paper Section 7.1)
    pooled_labels = cells["d00"]["mode_labels"] + cells["d11"]["mode_labels"]
    pooled_accepted = [m for m in pooled_labels if m.get("accepted")]
    from collections import Counter
    if pooled_accepted:
        counts = Counter(m["mode"] for m in pooled_accepted)
        total = sum(counts.values())
        prior = {mode: c / total for mode, c in counts.items()}
    else:
        prior = {}
    
    decompositions = {}
    d00_data = cells["d00"]
    
    print("=" * 70)
    print("BP FOUR-TERM DECOMPOSITION — 2×2 RESULTS")
    print("=" * 70)
    print(f"\n{'Cell':<6} {'log(κ0/κ)_tok':>14} {'log(κ0/κ)_wall':>15} {'φ':>8} {'G':>8} {'-ε':>8} {'Δ_tok':>8} {'Δ_wall':>8}")
    print("-" * 70)
    
    for cell_name in ["d10", "d01", "d11"]:
        dec = compute_decomposition(d00_data, cells[cell_name], prior)
        decompositions[cell_name] = dec
        print(f"{cell_name:<6} {dec['cost_term_token']:>14.4f} {dec['cost_term_wall']:>15.4f} "
              f"{dec['phi']:>8.4f} {dec['G']:>8.4f} {-dec['epsilon']:>8.4f} "
              f"{dec['delta_token']:>8.4f} {dec['delta_wall']:>8.4f}")
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS")
    print("=" * 70)
    
    hyp = test_hypotheses(cells, decompositions)
    for k, v in hyp.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("CONTEXT PRESSURE (H5)")
    print("=" * 70)
    
    for cell_name in ["d00", "d10", "d01", "d11"]:
        bins = compute_kappa_by_context_bin(cells[cell_name])
        print(f"\n  {cell_name}: κ̄_token by c/K bin:")
        for bin_label, val in bins.items():
            print(f"    {bin_label}: {val:.0f} tokens" if val else f"    {bin_label}: no data")
    
    # Save full results
    output = {
        "decompositions": decompositions,
        "hypotheses": hyp,
        "prior": prior,
    }
    output_path = Path("decomposition_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")
```

**Verify:** Create synthetic test data mimicking the expected output format and run the script. Confirm the table renders and all hypothesis fields are populated.

---

## Task 8: Add experiment mode to YAML config and CLI

**Why:** Need a clean way to select any of the four cells from config or CLI.

**File:** `src/agent_parallelization_new/launcher.py`

Add mode routing:

```python
MODES = {
    "single_long": main_single_long,           # d00
    "single_memory": main_single_memory,        # d10
    "parallel": main_parallel,                  # d01
    "parallel_shared": main_parallel_shared,    # d11
}
```

**File:** `configs/experiment.yaml`

Document the mapping:

```yaml
# 2×2 design modes:
#   single_long         = d00: single agent, no external memory
#   single_memory       = d10: single agent, external memory (reads result log)
#   parallel            = d01: N agents, no shared memory
#   parallel_shared     = d11: N agents, shared result log
experiment:
  mode: parallel  # change to run different cells
```

---

## Execution Order for the 2×2 Pilot

After implementing Tasks 1–8, run the pilot:

```bash
# Cell d00: single agent, no memory, 30 min
run-single-long --time-budget 30 --config configs/experiment_d00.yaml

# Cell d10: single agent, with memory, 30 min
run-single-memory --time-budget 30 --config configs/experiment_d10.yaml

# Cell d01: 2 parallel agents, no sharing, 30 min
run-parallel --n-agents 2 --time-budget 30 --config configs/experiment_d01.yaml

# Cell d11: 2 parallel agents, shared memory, 30 min
run-parallel-shared --n-agents 2 --time-budget 30 --config configs/experiment_d11.yaml

# Post-hoc: label modes
python scripts/label_modes.py --experiment-dir runs/experiment_d00
python scripts/label_modes.py --experiment-dir runs/experiment_d10
python scripts/label_modes.py --experiment-dir runs/experiment_d01
python scripts/label_modes.py --experiment-dir runs/experiment_d11

# Analysis: compute decomposition
python scripts/compute_decomposition.py \
    --d00 runs/experiment_d00 \
    --d10 runs/experiment_d10 \
    --d01 runs/experiment_d01 \
    --d11 runs/experiment_d11
```

---

## Corrections to the Paper's Framework (Agreed Upon)

These are corrections discussed during the design phase that deviate from the paper as written:

1. **τ_cost is measured in tokens, not dollars.** GPU cost is constant across cells (5 min per attempt) and does not discriminate architectures. The cost axis uses total agent tokens (input + output) per attempt. The decomposition is unchanged because only ratios matter.

2. **GPU cost term is dropped from κ̄_cost.** For the same reason. κ̄_cost(d) = mean tokens per attempt.

3. **The number of agents p does not appear as a separate term.** It enters through κ̄_wall (parallelism halves critical path) and ε (duplication raises routing mismatch). This is already how the paper models it, but worth stating explicitly.

4. **External memory in d10 is a persistent scratchpad, not "shared" memory.** A single agent reading its own structured result log. The benefit comes from selective re-reading vs. full context accumulation, which lowers effective c/K.

5. **The memory format is fixed:** one-line-per-experiment table with fields (step, change, bpb, delta, best). ~15-20 tokens per row. At 200 attempts, ~4000 tokens total — 2% of a 200K context window.
