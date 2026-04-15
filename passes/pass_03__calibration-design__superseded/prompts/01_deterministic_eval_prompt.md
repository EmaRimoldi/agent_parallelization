<!-- AUTO-GENERATED PROMPT — DO NOT EDIT -->
<!-- Phase: 01_deterministic_eval | Branch: main | Generated: 2026-04-12T15:12:31 -->

## Injected Context

- **Current phase**: `01_deterministic_eval`
- **Branch**: `main`
- **Completed phases**: 00_overview
- **Repo root**: `/Users/emanuelerimoldi/Documents/GitHub/agent_parallelization`
- **Workflow dir**: `/Users/emanuelerimoldi/Documents/GitHub/agent_parallelization/workflow`

---
# Phase 01: Make Evaluation Deterministic

## Goal

Eliminate all sources of non-determinism in the training and evaluation pipeline so that running the same `train.py` twice produces identical `val_bpb` to floating-point precision.

## Background

The current `autoresearch/train.py` uses `seed = int(time.time() * 1000) % (2**32)` — a time-based seed that changes on every execution. Combined with unseeded numpy/random, unseeded DataLoader shuffling, and multi-worker data loading (`num_workers=2`), this produces ~0.04–0.05 standard deviation in val_bpb across runs of identical code. This noise floor is the same magnitude as the architecture contrasts we are trying to measure.

Making evaluation deterministic is the single highest-ROI change in the entire research program.

## Tasks

### 1. Modify `autoresearch/train.py`

Replace the time-based seed with a fixed seed and add full determinism controls:

```python
import os
import random
import numpy as np
import torch

SEED = 42

def set_deterministic_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Call `set_deterministic_seed()` at the very top of the script, before any other torch/numpy operations.

Find and replace the existing seed line:
```python
# REMOVE THIS:
seed = int(time.time() * 1000) % (2**32)
# REPLACE WITH:
set_deterministic_seed(SEED)
```

### 2. Fix DataLoader determinism in `autoresearch/train.py` and/or `autoresearch/prepare.py`

Wherever DataLoader is created with `shuffle=True`, add a seeded generator:

```python
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    ...,
    shuffle=True,
    num_workers=0,       # CHANGED from 2 to 0
    generator=g,         # ADDED
)
```

Set `num_workers=0` for all DataLoaders to eliminate multiprocessing non-determinism.

### 3. Modify `autoresearch/prepare.py`

Add the same determinism preamble if prepare.py does any random operations (data splitting, augmentation seeding):

```python
import os, random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = "42"
```

### 4. Preserve the original files

Before modifying, create backups:
```bash
cp autoresearch/train.py autoresearch/train.py.pre_deterministic
cp autoresearch/prepare.py autoresearch/prepare.py.pre_deterministic
```

### 5. Verify the changes compile

```bash
cd autoresearch && python -c "import train; print('train.py imports OK')"
cd autoresearch && python -c "import prepare; print('prepare.py imports OK')"
```

## Required Inputs

- `autoresearch/train.py` (current version with time-based seed)
- `autoresearch/prepare.py` (current version)

## Expected Outputs

- Modified `autoresearch/train.py` with fixed seed and full determinism
- Modified `autoresearch/prepare.py` with fixed seed
- Backup copies of both originals
- Both files import without errors

## Success Criteria

- The time-based seed line is removed
- `set_deterministic_seed(42)` is called before any torch/numpy operations
- All DataLoaders use `num_workers=0` and a seeded generator
- PYTHONHASHSEED is set
- Both files import cleanly

## Failure Modes

- `torch.use_deterministic_algorithms(True)` may raise errors for some operations. Use `warn_only=True` to identify which operations are non-deterministic, then address them.
- If the model uses operations without deterministic implementations on CPU, you may need to set `warn_only=True` instead of strict mode.

## Next Phase

On success: proceed to `01a_verify_determinism`
