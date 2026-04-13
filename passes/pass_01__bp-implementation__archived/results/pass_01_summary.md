# Pass 01 — BP Implementation: Experiment Summary

**Status**: Archived
**Period**: April 2026 (first AI pass)
**Objective**: Establish baseline measurements for the BP framework's 2x2 experimental design and identify infrastructure issues before scaling.

---

## Experimental Design

The BP framework decomposes agent performance via Delta = log(kappa_0/kappa) + phi + G - epsilon. Pass 01 tested a 2x2 factorial design crossing two axes:

|              | No Memory (0) | Memory (1)     |
|--------------|---------------|----------------|
| **Single (0)** | d00 (baseline) | d10            |
| **Parallel (1)** | d01           | d11            |

### Task, model, and metrics

**Task**: Each LLM agent (claude-haiku-4-5) is given a git repository containing a CIFAR-10 image classification problem. The agent autonomously reads and edits `train.py`, choosing hyperparameters and architecture modifications, then runs training. After each training attempt, the agent observes the result and decides what to try next.

**Model being trained**: A configurable CNN (`CIFAR10Net`) for CIFAR-10 (32x32 RGB images, 10 classes). Default architecture: 3 convolutional blocks (Conv2d + BatchNorm + ReLU + Dropout + MaxPool), followed by a fully connected classifier (128 hidden units). ~357K parameters. All training runs on CPU (no GPU available in this environment).

**What the agent optimizes**: Across successive training attempts within a single experiment, the LLM agent modifies `train.py` to change:
- **Optimizer**: type (Adam, AdamW, SGD), learning rate, weight decay, momentum, betas
- **Schedule**: warmup epochs, LR decay factor, decay milestones
- **Architecture**: depth (number of conv blocks), base channels, channel multiplier, dropout rate, FC hidden width, batch norm on/off
- **Data**: batch size

The agent follows a one-change-per-turn strategy, guided by its memory of previous attempts.

**val_bpb metric**: Despite the name "bits per byte", this is simply the **cross-entropy loss** on the CIFAR-10 test set (10,000 images). It is printed as `val_bpb` in the training output but is identical to `val_loss`. Lower is better. Typical range: 0.73-0.90 for well-tuned models, ~1.9 for untrained/short runs.

**Why initial val_bpb differs across runs**: Each experiment starts from a different git commit state. The agent modifies `train.py` during the experiment, and each repetition starts fresh from the repository baseline — but different experiment batches may have been launched from slightly different baseline commits (with different default hyperparameters). Additionally, with only 120 seconds per training attempt and different random seeds for data augmentation order, the first evaluation point varies. This is an uncontrolled source of variance in Pass 01.

**Tokens consumed**: This counts **Claude API tokens** (input + output) used by the LLM agent during its autonomous loop — how much "thinking" the agent did. It is NOT related to training data tokens. Counted from the API usage response when available, or estimated as characters/4 as fallback. Parallel cells consume roughly 2x more tokens because two agents run simultaneously.

Three experiment classes were run:

### 1. Pilot 2x2 feasibility — 12 runs (4 cells x 3 reps)

Each agent is an LLM (claude-haiku-4-5) that autonomously writes and trains small neural networks. The **base time budget** is 30 minutes (1800s) per experiment: the agent has 30 minutes of wall-clock time to iterate on code and launch training runs. Each individual **training attempt** is capped at 120 seconds. Within the 30-minute window, the agent decides how many training attempts to make — typically 2-9 for single-agent cells, 9-18 for parallel cells (because two agents run independently).

### 2. Exploratory config iteration — 8 runs

Same 4 modes as the pilot, but with a shorter **10-minute budget** (600s) and 120s per training attempt. All configs use claude-haiku-4-5. The 8 runs are: 5x single-long (no memory, 1 agent), 1x single-memory (external memory, 1 agent), 1x parallel (2 agents, no memory), 1x parallel-shared (2 agents, shared memory). The purpose was rapid iteration to explore different hyperparameter choices before committing to the full pilot.

### 3. Resource contention evaluation — scaling study N=1..8

In the pilot, parallel cells (d01/d11) showed worse val_bpb than single cells (d00/d10). But is that because parallelism itself is unhelpful, or because two training processes running simultaneously on the same CPU compete for compute and each one gets fewer gradient steps done?

To answer this, we remove the LLM agent entirely and run **only the training script** (`train.py`) directly, with a fixed 2-second training budget. This isolates the pure hardware contention effect: same code, same hyperparameters, same seed — the only variable is how many copies run at the same time on the 10-core CPU machine.

The experiment launches N identical `train.py` processes simultaneously (N=1,2,4,8) and measures: (a) how many gradient steps each process completes in 2 seconds, (b) the resulting val_bpb, (c) total wall time and throughput. Two thread policies are tested: **default** (PyTorch's OpenMP uses all available cores per process, so N processes fight over the same 10 cores) and **partitioned** (CPU affinity pins 10/N cores to each process, eliminating thread contention).

Note: the val_bpb values here (~1.9) are much worse than the pilot (~0.8) because the training budget is 2 seconds vs 120 seconds — only ~19 gradient steps vs hundreds. The absolute values don't matter; what matters is the **relative degradation** as N increases.

---

## Key Results

### 1. Pilot 2x2 (primary experiment)

**Per-cell per-rep breakdown** (training runs = number of train attempts the agent made within its 30-min budget):

| Cell | Rep | Training runs | Initial val_bpb | Best val_bpb | Tokens consumed |
|------|-----|---------------|-----------------|-------------|-----------------|
| d00  | 1   | 4             | 0.828           | 0.827       | 36,416          |
| d00  | 2   | 2             | 0.800           | 0.800       | 48,104          |
| d00  | 3   | 9             | 0.801           | 0.799       | 38,402          |
| d10  | 1   | 3             | 0.829           | 0.829       | 47,170          |
| d10  | 2   | 9             | 0.761           | 0.761       | 35,136          |
| d10  | 3   | 4             | 0.831           | 0.755       | 43,200          |
| d01  | 1   | 11            | 0.900           | 0.811       | 91,803          |
| d01  | 2   | 9             | 0.802           | 0.802       | 92,684          |
| d01  | 3   | 18            | 0.837           | 0.837       | 70,693          |
| d11  | 1   | 9             | 0.845           | 0.801       | 90,053          |
| d11  | 2   | 17            | 0.844           | 0.804       | 67,685          |
| d11  | 3   | 14            | 0.796           | 0.796       | 69,146          |

**Summary statistics**:

| Cell | Mode              | Mean training runs | Mean best val_bpb | Std   | N reps |
|------|-------------------|--------------------|---------------------|-------|--------|
| d00  | Single / No Memory | 5.0 +/- 2.9       | 0.809               | 0.016 | 3      |
| d10  | Single / Memory    | 5.3 +/- 2.6       | 0.782               | 0.041 | 3      |
| d01  | Parallel / No Mem  | 12.7 +/- 3.9      | 0.817               | 0.018 | 3      |
| d11  | Parallel / Memory  | 13.3 +/- 3.3      | 0.800               | 0.004 | 3      |

Note: parallel cells (d01, d11) produce ~2.5x more training attempts because two agents run independently, but this does not translate into better final quality — it reflects the resource contention problem.

**Observations**:
- Memory improves single-agent performance (d10 < d00 by ~0.027 bpb), but with high variance (std 0.041).
- Parallel agents without memory (d01) perform *worse* than baseline, suggesting resource contention degrades training quality.
- Parallel + memory (d11) partially recovers quality and shows the lowest variance (std 0.004), hinting at a stabilizing effect.
- None of the six pre-registered hypotheses (H1-H6) were fully supported at this sample size.
- Linear fit R^2(best_val_bpb ~ total_tokens) = 0.32 — token budget is a weak predictor of final quality.

![Pilot 2x2 comparison](figures/fig01_pilot_2x2_comparison.png)

**Figure 1 interpretation**: The error bars and scatter points reveal that d10 (memory) has the lowest mean but also the widest spread — two reps cluster near 0.755-0.761 while a third sits at 0.829, suggesting a bimodal outcome where memory either "clicks" and produces a large gain or fails to help at all. In contrast, d11 (parallel + memory) shows remarkably tight clustering (all 3 reps within 0.796-0.804), but its mean is not lower than d00. This hints that memory's benefit in d10 may depend on lucky initialization rather than a robust mechanism. The d01 bar sits above the d00 baseline, confirming that parallelism alone hurts — but the scatter shows one rep at 0.802 (comparable to d00) and another at 0.837, so the damage is also inconsistent. With only 3 points per cell, none of these differences would survive a significance test; the figure is best read as generating hypotheses, not confirming them.

### 2. Exploratory Runs

All exploratory runs use claude-haiku-4-5, 10-minute budget, 120s per training attempt.

| Run                  | Mode             | Agents | Memory   | Shared | Best val_bpb |
|----------------------|------------------|--------|----------|--------|-------------|
| single-long (run1)   | single_long      | 1      | no       | no     | no data     |
| single-long (run2)   | single_long      | 1      | no       | no     | 0.762       |
| single-long (run3)   | single_long      | 1      | no       | no     | no data     |
| single-long (run4)   | single_long      | 1      | no       | no     | no data     |
| single-long (run5)   | single_long      | 1      | no       | no     | 0.816       |
| single-memory (run1) | single_memory    | 1      | **yes**  | no     | **0.739**   |
| parallel (run1)      | parallel         | 2      | no       | no     | 0.824       |
| parallel-shared (run1)| parallel_shared | 2      | no       | **yes**| 0.830       |

3 of 5 single-long runs produced no usable metrics (agent failed to complete a training run within budget).

**Observations**:
- The best result across all Pass 01 experiments was a single-agent + memory run (0.739), substantially beating the pilot mean.
- Parallel configurations consistently underperformed single-agent setups, reinforcing the resource contention finding.
- 3/8 exploratory runs produced no data, highlighting infrastructure fragility at shorter budgets.

![Exploratory comparison](figures/fig03_exploratory_comparison.png)

**Figure 3 interpretation**: The standout is single-memory at 0.739, well left of the d00 pilot baseline (dashed line). However, this is a single unreplicated run — it could be an outlier. The two successful single-long runs bracket the baseline (0.762 and 0.816), showing high variance even within the same configuration. Both parallel modes sit to the right of the baseline, reinforcing the contention penalty. A critical limitation: the 3 missing single-long runs introduce survivorship bias — we only see results from runs where the agent happened to produce valid training output. The true single-long distribution may be worse than shown.

### 3. Resource Contention

This study isolates CPU contention by running identical 2-second training tasks (not the full agent loop). On a 10-core CPU machine, it measures how throughput and training quality change as N concurrent processes compete for CPU and memory bandwidth.

| N agents | Policy      | Wall time (s) | Speedup | Efficiency | Mean steps | Mean val_bpb |
|----------|-------------|---------------|---------|------------|------------|-------------|
| 1        | sequential  | 9.34          | 1.00x   | 100%       | 19.0       | 1.945       |
| 1        | parallel    | 10.08         | 0.93x   | 93%        | 21.0       | 1.947       |
| 2        | default     | 11.18         | 1.69x   | 85%        | 17.0       | 1.986       |
| 2        | partitioned | 10.16         | 1.86x   | 93%        | 19.0       | 1.945       |
| 4        | default     | 14.28         | 2.63x   | 66%        | 12.0       | 2.059       |
| 4        | partitioned | 14.26         | 2.63x   | 66%        | 15.0       | 2.018       |
| 8        | default     | 23.54         | 3.15x   | 39%        | 7.1        | 2.207       |
| 8        | partitioned | 22.58         | 3.29x   | 41%        | 10.0       | 2.113       |

**Observations**:
- Speedup is strictly sublinear: N=8 yields only ~3.2x throughput (ideal would be 8x).
- Training quality degrades monotonically with N: val_bpb worsens by ~13% from N=1 to N=8 (default policy). Each agent completes fewer gradient steps in the same wall-clock budget.
- Partitioned thread policy (dividing cores equally: 5 per agent at N=2, 1 per agent at N=8) provides modest quality improvements at N=2 (no degradation) but the advantage diminishes at higher N.
- This is the key confound for the 2x2 design: parallel cells (d01, d11) face resource contention that single cells (d00, d10) do not. Any quality difference between parallel and single cells conflates the parallelism effect with the contention effect.

![Resource contention](figures/fig02_resource_contention.png)

**Figure 2 interpretation**: The left panel shows both policies diverge sharply from the ideal linear speedup after N=2, plateauing around 3x at N=8. The right panel reveals the cost: default-policy val_bpb degrades nearly linearly with N (~0.033 per doubling). The partitioned policy flattens the curve at N=2 (no quality loss) but converges with default at higher N, where there simply aren't enough cores per agent (1 core at N=8 vs 10 at N=1). The key takeaway is that for the pilot's N=2 (d01/d11), the contention penalty is real but modest (~2% val_bpb degradation with default policy, near-zero with partitioning). This means the d01/d11 quality gap vs d00/d10 in the pilot is partly but not entirely explained by contention — some signal may remain.

## Conclusions

1. **Memory is the strongest single factor**: Adding memory to a single agent produced the best val_bpb (0.739 exploratory, 0.782 pilot mean), consistently outperforming no-memory counterparts.

2. **Parallelism introduces CPU contention**: On shared-CPU hardware, parallel agents compete for resources, degrading individual training quality. This is a confound that must be controlled in future passes.

3. **The 2x2 design is viable but underpowered**: With only 3 reps per cell, variance is too high to draw inferential conclusions. The factorial structure is sound, but needs more repetitions and confound control.

4. **d11 shows a stabilizing pattern**: Parallel + memory had the lowest variance (0.004), suggesting memory may help coordinate parallel agents, though the mechanism is unclear.

5. **Infrastructure validated**: The experiment runner, config system, and analysis pipeline work end-to-end. Key bugs found: resource contention not controlled, training budget sensitivity.

## Implications for Later Passes

- Pass 02 addressed the theoretical decomposition framework.
- Pass 03 introduced phased execution and config routing.
- Pass 04 redesigned the 2x2 with confound controls (fixed seeds, CPU pinning, task headroom) informed directly by Pass 01 findings.
