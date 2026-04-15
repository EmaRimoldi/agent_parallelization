# Probe Analysis Report (Waves 1, 2, 3)

**Generated**: 2026-04-13 09:43

## Probe Results Summary

| Probe | Name | Runs | Non-BL | Categories | Entropy | Jaccard | Switch | Improvements | Best bpb |
|-------|------|------|--------|------------|---------|---------|--------|-------------|----------|
| P01 | parallel_homo | 7 | 5 | 3 | 1.522 | 0.333 | 0.600 | 1 | 0.922538 |
| P02 | parallel_diverse | 14 | 12 | 3 | 1.459 | 1.000 | 0.500 | 0 | 0.979568 |
| P03 | single_baseline | 8 | 7 | 1 | -0.000 | N/A | 0.143 | 0 | 0.935578 |
| P04 | single_short_train | 10 | 9 | 3 | 1.352 | N/A | 0.556 | 0 | 1.103296 |
| P05 | memory_baseline | 7 | 6 | 3 | 1.459 | N/A | 0.667 | 1 | 0.919108 |
| P06 | shared_diverse | 10 | 8 | 3 | 1.061 | 0.333 | 0.625 | 0 | 0.948354 |
| P07 | shared_extended | 18 | 16 | 3 | 1.477 | 0.667 | 0.875 | 1 | 0.906257 |
| P08 | memory_extended | 14 | 13 | 4 | 1.826 | N/A | 0.769 | 0 | 0.981923 |
| P09 | diverse_extended | 29 | 27 | 5 | 1.961 | 0.800 | 0.778 | 0 | 0.971015 |
| P10 | homo_fixed | 14 | 12 | 3 | 1.555 | 1.000 | 0.750 | 0 | 0.959791 |
| P11 | single_hightemp | 21 | 20 | 5 | 1.292 | N/A | 0.550 | 0 | 0.933486 |
| P12 | shared_fixed | 41 | 39 | 5 | 1.662 | 0.400 | 0.282 | 2 | 0.914051 |
| P13 | dual_hightemp | 44 | 42 | 3 | 0.913 | 0.667 | 0.333 | 0 | 0.961317 |
| P14 | hightemp_memory | - | - | - | - | - | - | - | - |
| P15 | seeded_guidance | - | - | - | - | - | - | - | - |
| P16 | optimal_baseline | - | - | - | - | - | - | - | - |
| P17 | full_stack | - | - | - | - | - | - | - | - |
| P18 | seeded_parallel | - | - | - | - | - | - | - | - |

## Per-Agent Details

### P01: parallel_homo

**agent_0**: 4 runs, categories: optimization, other
  Sequence: other -> optimization -> optimization
**agent_1**: 3 runs, categories: regularization, other
  Sequence: regularization -> other

### P02: parallel_diverse

**agent_0**: 7 runs, categories: optimization, regularization, other
  Sequence: optimization -> optimization -> other -> optimization -> regularization -> other
**agent_1**: 7 runs, categories: optimization, regularization, other
  Sequence: other -> other -> regularization -> optimization -> optimization -> optimization

### P03: single_baseline

**agent_0**: 8 runs, categories: regularization
  Sequence: regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization

### P04: single_short_train

**agent_0**: 10 runs, categories: optimization, regularization, other
  Sequence: other -> other -> regularization -> other -> optimization -> other -> other -> optimization -> optimization

### P05: memory_baseline

**agent_0**: 7 runs, categories: optimization, regularization, other
  Sequence: optimization -> optimization -> optimization -> other -> regularization -> other

### P06: shared_diverse

**agent_0**: 7 runs, categories: optimization, regularization, data_pipeline
  Sequence: optimization -> optimization -> regularization -> optimization -> optimization -> data_pipeline
**agent_1**: 3 runs, categories: optimization
  Sequence: optimization -> optimization

### P07: shared_extended

**agent_0**: 3 runs, categories: regularization, other
  Sequence: regularization -> other
**agent_1**: 15 runs, categories: optimization, regularization, other
  Sequence: optimization -> optimization -> regularization -> optimization -> regularization -> optimization -> other -> optimization -> regularization -> optimization -> regularization -> other -> optimization -> optimization

### P08: memory_extended

**agent_0**: 14 runs, categories: optimization, regularization, other, architecture
  Sequence: other -> other -> optimization -> regularization -> other -> regularization -> optimization -> other -> optimization -> architecture -> other -> regularization -> regularization

### P09: diverse_extended

**agent_0**: 14 runs, categories: optimization, regularization, other, architecture
  Sequence: optimization -> other -> architecture -> optimization -> regularization -> other -> other -> other -> optimization -> optimization -> architecture -> optimization -> regularization
**agent_1**: 15 runs, categories: data_pipeline, optimization, regularization, other, architecture
  Sequence: data_pipeline -> optimization -> regularization -> other -> optimization -> optimization -> optimization -> other -> optimization -> architecture -> optimization -> optimization -> other -> regularization

### P10: homo_fixed

**agent_0**: 7 runs, categories: optimization, regularization, other
  Sequence: optimization -> optimization -> regularization -> other -> optimization -> other
**agent_1**: 7 runs, categories: optimization, regularization, other
  Sequence: other -> optimization -> other -> optimization -> regularization -> regularization

### P11: single_hightemp

**agent_0**: 21 runs, categories: data_pipeline, optimization, regularization, other, architecture
  Sequence: optimization -> optimization -> data_pipeline -> optimization -> other -> optimization -> optimization -> optimization -> architecture -> optimization -> optimization -> optimization -> architecture -> optimization -> optimization -> regularization -> optimization -> optimization -> optimization -> optimization

### P12: shared_fixed

**agent_0**: 18 runs, categories: regularization, other
  Sequence: other -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization
**agent_1**: 23 runs, categories: data_pipeline, optimization, regularization, other, architecture
  Sequence: other -> regularization -> regularization -> optimization -> optimization -> optimization -> data_pipeline -> data_pipeline -> architecture -> architecture -> regularization -> optimization -> optimization -> regularization -> regularization -> other -> optimization -> optimization -> optimization -> optimization -> regularization -> regularization

### P13: dual_hightemp

**agent_0**: 9 runs, categories: optimization, regularization
  Sequence: optimization -> optimization -> optimization -> regularization -> optimization -> optimization -> optimization -> regularization
**agent_1**: 35 runs, categories: optimization, regularization, architecture
  Sequence: optimization -> optimization -> architecture -> optimization -> optimization -> regularization -> regularization -> architecture -> optimization -> optimization -> optimization -> optimization -> optimization -> optimization -> optimization -> optimization -> optimization -> optimization -> optimization -> optimization -> regularization -> optimization -> optimization -> optimization -> optimization -> optimization -> optimization -> optimization -> optimization -> regularization -> regularization -> optimization -> optimization -> optimization

## Comparative Findings

```
DIVERSITY (P01 vs P02):
  Jaccard: P01=0.333 vs P02=1.000 (similar/P01 more diverse)
  Entropy: P01=1.522 vs P02=1.459
  Runs: P01=7 vs P02=14
  Improvements: P01=1 vs P02=0
  NO SIGNAL: Diversity injection did not reduce Jaccard

TRAINING HEADROOM (P03 vs P04):
  Total runs: P03=8 (60s) vs P04=10 (30s), ratio=1.25x
  Mean train time: P03=60s vs P04=30s
  Entropy: P03=-0.000 vs P04=1.352
  Improvements: P03=0 vs P04=0
  NO SIGNAL: Run count ratio only 1.2x (threshold: 1.3x)

MEMORY + DIVERSITY (P05 vs P06):
  Runs: P05=7 vs P06=10
  Switch rate: P05=0.667 vs P06=0.625
  Improvements: P05=1 vs P06=0
  Best bpb: P05=0.919108 vs P06=0.948354
  NO SIGNAL: Shared memory + diversity not better

EXTENDED BUDGET: shared+diverse (P06 vs P07):
  Runs: P06=10 vs P07=18
  Best bpb: P06=0.948354 vs P07=0.906257
  Improvements: P06=0 vs P07=1
  SIGNAL: Extended budget improved best bpb by 0.042097

EXTENDED BUDGET: single+memory (P05 vs P08):
  Runs: P05=7 vs P08=14
  Best bpb: P05=0.919108 vs P08=0.981923
  Improvements: P05=1 vs P08=0
  NO SIGNAL: Extended budget did not improve best bpb

EXTENDED BUDGET: parallel diverse (P02 vs P09):
  Runs: P02=14 vs P09=29
  Best bpb: P02=0.979568 vs P09=0.971015
  Improvements: P02=0 vs P09=0
  SIGNAL: Extended budget improved best bpb by 0.008553

EXTENDED BUDGET: parallel homo (P01 vs P10):
  Runs: P01=7 vs P10=14
  Best bpb: P01=0.922538 vs P10=0.959791
  Improvements: P01=1 vs P10=0
  NO SIGNAL: Extended budget did not improve best bpb

TEMPERATURE EFFECT (P03 default vs P11 temp=1.2):
  Runs: P03=8 vs P11=21
  Best bpb: P03=0.935578 vs P11=0.933486

REAL SHARED MEMORY (P07 broken vs P12 fixed):
  Runs: P07=18 vs P12=41
  Best bpb: P07=0.906257 vs P12=0.914051

DEGRADATION CHECK (P08):
  agent_0: 13 non-baseline runs
  Best bpb: 0.981923
  Worst bpb: 1.548404
  Spread: 0.566481 (larger = more degradation)
  DEGRADATION: Agent quality spread > 0.5 bpb

DEGRADATION CHECK (P11):
  agent_0: 20 non-baseline runs
  Best bpb: 0.933486
  Worst bpb: 2.305029
  Spread: 1.371543 (larger = more degradation)
  DEGRADATION: Agent quality spread > 0.5 bpb

DUAL vs SINGLE HIGH-TEMP (P11 vs P13):
  Runs: P11=21 vs P13=44
  Best bpb: P11=0.933486 vs P13=0.961317
  Worst bpb: P11=2.305029 vs P13=7.876028

CROSS-PROBE SUMMARY:
  P01 (W1): runs=7, entropy=1.522, improvements=1, best_bpb=0.922538, score=24.6
  P02 (W1): runs=14, entropy=1.459, improvements=0, best_bpb=0.979568, score=21.3
  P03 (W1): runs=8, entropy=-0.000, improvements=0, best_bpb=0.935578, score=8.0
  P04 (W1): runs=10, entropy=1.352, improvements=0, best_bpb=1.103296, score=16.8
  P05 (W1): runs=7, entropy=1.459, improvements=1, best_bpb=0.919108, score=24.3
  P06 (W1): runs=10, entropy=1.061, improvements=0, best_bpb=0.948354, score=15.3
  P07 (W2): runs=18, entropy=1.477, improvements=1, best_bpb=0.906257, score=35.4
  P08 (W2): runs=14, entropy=1.826, improvements=0, best_bpb=0.981923, score=23.1
  P09 (W2): runs=29, entropy=1.961, improvements=0, best_bpb=0.971015, score=38.8
  P10 (W2): runs=14, entropy=1.555, improvements=0, best_bpb=0.959791, score=21.8
  P11 (W3): runs=21, entropy=1.292, improvements=0, best_bpb=0.933486, score=27.5
  P12 (W3): runs=41, entropy=1.662, improvements=2, best_bpb=0.914051, score=69.3
  P13 (W3): runs=44, entropy=0.913, improvements=0, best_bpb=0.961317, score=48.6
  BEST PROBE: P12 (score=69.3)
```

## Strategy Category Distribution

| Category | Count | % |
|----------|-------|---|
| optimization | 105 | 48.6% |
| regularization | 60 | 27.8% |
| other | 36 | 16.7% |
| architecture | 10 | 4.6% |
| data_pipeline | 5 | 2.3% |
