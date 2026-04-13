# Phase 01b: Debug Non-Determinism Sources

## Goal

Identify and fix any remaining sources of non-determinism that were not caught by Phase 01.

## Background

Phase 01a showed that the train.py modifications did not achieve full determinism. This phase systematically identifies the remaining sources.

## Tasks

### 1. Identify non-deterministic operations

Run train.py with torch determinism warnings enabled:

```python
import torch
torch.use_deterministic_algorithms(True)  # strict mode, will raise on non-deterministic ops
```

If this raises a `RuntimeError`, it will name the specific operation that lacks a deterministic implementation.

### 2. Common sources to check

- **Scatter/gather operations**: Some indexing operations are non-deterministic on GPU but deterministic on CPU.
- **BatchNorm**: Should be deterministic on CPU. Verify by replacing with `GroupNorm` temporarily.
- **Dropout**: Deterministic if torch seed is set. Verify by checking if `torch.manual_seed` is called before the model is created.
- **Data augmentation**: RandomCrop, RandomHorizontalFlip use torch.rand internally. They should be deterministic if `torch.manual_seed` is set. Check if any augmentation reseeds internally.
- **Python hash randomization**: Set `PYTHONHASHSEED` as an environment variable *before* Python starts, not just in-process. This may require a wrapper script.
- **Thread-level non-determinism**: `torch.set_num_threads(1)` forces single-threaded execution.

### 3. Progressively isolate

1. Run with the simplest possible model (1 epoch, tiny batch) → deterministic?
2. Add back augmentation → still deterministic?
3. Add back full epochs → still deterministic?
4. Add back full batch size → still deterministic?

The step where determinism breaks reveals the source.

### 4. Apply fixes and return to verification

After identifying and fixing all sources, return to Phase 01a for re-verification.

## Required Inputs

- Modified train.py and prepare.py from Phase 01
- The verification failure output from Phase 01a (in logs/)

## Expected Outputs

- Identified non-determinism source(s)
- Fixed train.py and/or prepare.py
- Ready for re-verification

## Success Criteria

- All identified non-determinism sources have been addressed
- Ready to re-run Phase 01a

## Failure Modes

- If determinism is impossible on this platform (e.g., due to a torch bug), document the minimum achievable variance and proceed with that as the noise floor.
- If the noise floor is < 0.001 (vs ~0.04 before), that is acceptable for the research goals even if not mathematically exact.

## Next Phase

On completion: return to `01a_verify_determinism` for re-verification
