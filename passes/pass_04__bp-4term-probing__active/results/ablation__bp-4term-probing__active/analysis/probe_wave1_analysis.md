# Wave 1 Probe Analysis Report

**Generated**: 2026-04-13 05:38

## Probe Results Summary

| Probe | Name | Runs | Non-BL | Categories | Entropy | Jaccard | Switch | Improvements | Best bpb |
|-------|------|------|--------|------------|---------|---------|--------|-------------|----------|
| P01 | parallel_homo | 7 | 5 | 3 | 1.522 | 0.333 | 0.600 | 1 | 0.922538 |
| P02 | parallel_diverse | 14 | 12 | 3 | 1.459 | 1.000 | 0.500 | 0 | 0.979568 |
| P03 | single_baseline | 8 | 7 | 1 | -0.000 | N/A | 0.143 | 0 | 0.935578 |
| P04 | single_short_train | 10 | 9 | 3 | 1.352 | N/A | 0.556 | 0 | 1.103296 |
| P05 | memory_baseline | 7 | 6 | 3 | 1.459 | N/A | 0.667 | 1 | 0.919108 |
| P06 | shared_diverse | 10 | 8 | 3 | 1.061 | 0.333 | 0.625 | 0 | 0.948354 |

## Per-Agent Details

### P01: parallel_homo

**agent_0**: 4 runs, categories: other, optimization
  Sequence: other -> optimization -> optimization
**agent_1**: 3 runs, categories: regularization, other
  Sequence: regularization -> other

### P02: parallel_diverse

**agent_0**: 7 runs, categories: other, optimization, regularization
  Sequence: optimization -> optimization -> other -> optimization -> regularization -> other
**agent_1**: 7 runs, categories: other, optimization, regularization
  Sequence: other -> other -> regularization -> optimization -> optimization -> optimization

### P03: single_baseline

**agent_0**: 8 runs, categories: regularization
  Sequence: regularization -> regularization -> regularization -> regularization -> regularization -> regularization -> regularization

### P04: single_short_train

**agent_0**: 10 runs, categories: other, optimization, regularization
  Sequence: other -> other -> regularization -> other -> optimization -> other -> other -> optimization -> optimization

### P05: memory_baseline

**agent_0**: 7 runs, categories: other, optimization, regularization
  Sequence: optimization -> optimization -> optimization -> other -> regularization -> other

### P06: shared_diverse

**agent_0**: 7 runs, categories: regularization, optimization, data_pipeline
  Sequence: optimization -> optimization -> regularization -> optimization -> optimization -> data_pipeline
**agent_1**: 3 runs, categories: optimization
  Sequence: optimization -> optimization

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

CROSS-PROBE SUMMARY:
  P01: runs=7, entropy=1.522, improvements=1, score=24.6
  P02: runs=14, entropy=1.459, improvements=0, score=21.3
  P03: runs=8, entropy=-0.000, improvements=0, score=8.0
  P04: runs=10, entropy=1.352, improvements=0, score=16.8
  P05: runs=7, entropy=1.459, improvements=1, score=24.3
  P06: runs=10, entropy=1.061, improvements=0, score=15.3
  BEST PROBE: P01 (score=24.6)
```

## Strategy Category Distribution

| Category | Count | % |
|----------|-------|---|
| optimization | 20 | 42.6% |
| regularization | 13 | 27.7% |
| other | 13 | 27.7% |
| data_pipeline | 1 | 2.1% |
