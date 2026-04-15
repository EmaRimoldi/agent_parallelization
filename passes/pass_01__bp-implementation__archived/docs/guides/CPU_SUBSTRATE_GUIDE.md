# CPU Training Substrate: CIFAR-10

## Purpose

This document specifies a CPU-only training substrate that replaces the GPU-based nanochat in the `autoresearch/` directory. The goal is to preserve the exact same experimental loop (agent modifies `train.py`, runs training, reads `val_bpb`-equivalent metric) while removing the GPU requirement so that the 2×2 pilot can run locally.

## Structural requirements

The substrate must satisfy these invariants to be compatible with the `agent_parallelization` framework:

1. A `prepare.py` that downloads data, exposes constants, and provides an `evaluate` function.
2. A `train.py` that the agent modifies — containing model, optimizer, hyperparameters, and training loop.
3. A scalar metric printed to stdout in the format `val_loss: X.XXXXXX` (lower is better).
4. A fixed time budget per run (2 minutes wall-clock on CPU).
5. Training randomness from seed/shuffling that produces noisy but unbiased evaluation.
6. Enough optimization surface across 5 modes (optimizer, lr_schedule, architecture, batch_data, other).

## Setup instructions

### Step 1: Create the substrate directory

```bash
mkdir -p autoresearch
cd autoresearch
```

### Step 2: Create `prepare.py`

```python
"""Fixed constants, data download, and evaluation harness.
DO NOT MODIFY — this is read-only for the agent.
"""

import os
import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ─── Constants ───────────────────────────────────────────────────────
TIME_BUDGET = 120          # 2 minutes wall-clock for training
EVAL_BATCH_SIZE = 256
NUM_CLASSES = 10
INPUT_CHANNELS = 3
IMAGE_SIZE = 32
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─── Data ────────────────────────────────────────────────────────────
_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

_transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def get_train_loader(batch_size: int, num_workers: int = 2) -> DataLoader:
    """Returns training dataloader. Downloads data on first call."""
    dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=_transform_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=False, drop_last=True)


def get_val_loader(batch_size: int = EVAL_BATCH_SIZE, num_workers: int = 2) -> DataLoader:
    """Returns validation dataloader. Downloads data on first call."""
    dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=_transform_val)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)


@torch.no_grad()
def evaluate_loss(model: nn.Module, device: str = "cpu") -> float:
    """Evaluate model on validation set. Returns average cross-entropy loss."""
    model.eval()
    val_loader = get_val_loader()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / total_samples


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, device: str = "cpu") -> float:
    """Evaluate model on validation set. Returns accuracy in [0, 1]."""
    model.eval()
    val_loader = get_val_loader()
    correct = 0
    total = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


def download_data():
    """Download CIFAR-10 if not already present."""
    datasets.CIFAR10(DATA_DIR, train=True, download=True)
    datasets.CIFAR10(DATA_DIR, train=False, download=True)
    print("Data ready.")


if __name__ == "__main__":
    download_data()
```

### Step 3: Create `train.py`

This is the file the agent modifies. It contains a baseline CNN with enough knobs for all 5 modes.

```python
"""CIFAR-10 training script — the agent modifies this file.

Metric: val_loss (cross-entropy on validation set, lower is better).
Time budget: 2 minutes wall-clock for training.
Device: CPU only.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from prepare import (
    TIME_BUDGET, NUM_CLASSES, INPUT_CHANNELS, IMAGE_SIZE,
    get_train_loader, evaluate_loss, evaluate_accuracy,
)

# ─── Architecture hyperparameters ────────────────────────────────────
DEPTH = 3                    # number of conv blocks
BASE_CHANNELS = 32           # channels in first conv layer
CHANNEL_MULT = 2             # channel multiplier per block
USE_BATCHNORM = True         # batch normalization
DROPOUT_RATE = 0.0           # dropout after each block
FC_HIDDEN = 128              # hidden units in classifier head

# ─── Optimizer hyperparameters ───────────────────────────────────────
OPTIMIZER = "adam"            # "adam", "sgd", "adamw"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9               # only for SGD
ADAM_BETAS = (0.9, 0.999)

# ─── LR schedule hyperparameters ────────────────────────────────────
USE_LR_SCHEDULE = True
WARMUP_EPOCHS = 2
LR_DECAY_FACTOR = 0.1
LR_DECAY_EPOCHS = [60, 80]   # decay at these epochs (if reached)

# ─── Batch / data hyperparameters ────────────────────────────────────
BATCH_SIZE = 128
NUM_WORKERS = 2

# ─── Model ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv -> optional BN -> ReLU -> optional Dropout."""
    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CIFAR10Net(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    def __init__(self):
        super().__init__()
        layers = []
        in_ch = INPUT_CHANNELS
        out_ch = BASE_CHANNELS
        for i in range(DEPTH):
            layers.append(ConvBlock(in_ch, out_ch, USE_BATCHNORM, DROPOUT_RATE))
            in_ch = out_ch
            out_ch = min(out_ch * CHANNEL_MULT, 512)  # cap at 512

        self.features = nn.Sequential(*layers)

        # Compute feature map size after DEPTH pooling layers
        feat_size = IMAGE_SIZE // (2 ** DEPTH)
        feat_dim = in_ch * feat_size * feat_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, FC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(FC_HIDDEN, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ─── Training ────────────────────────────────────────────────────────

def build_optimizer(model):
    if OPTIMIZER == "adam":
        return optim.Adam(model.parameters(), lr=LEARNING_RATE,
                          betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adamw":
        return optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                           betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        return optim.SGD(model.parameters(), lr=LEARNING_RATE,
                         momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")


def build_scheduler(optimizer, steps_per_epoch):
    if not USE_LR_SCHEDULE:
        return None
    # Cosine annealing over the full training time
    # Estimate total steps from time budget (rough: 2 min ≈ ~40 epochs on CPU)
    total_steps = steps_per_epoch * 100  # upper bound
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


def main():
    device = "cpu"
    t_start = time.time()

    # Seed for reproducibility within a run (but different across runs via clock)
    seed = int(time.time() * 1000) % (2**32)
    torch.manual_seed(seed)

    # Data
    train_loader = get_train_loader(BATCH_SIZE, NUM_WORKERS)
    steps_per_epoch = len(train_loader)

    # Model
    model = CIFAR10Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, steps_per_epoch)

    # Training loop with time budget
    total_training_time = 0.0
    step = 0
    epoch = 0

    while True:
        model.train()
        epoch += 1
        for images, labels in train_loader:
            step_start = time.time()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            step += 1
            dt = time.time() - step_start
            total_training_time += dt

            # Check time budget
            if total_training_time >= TIME_BUDGET:
                break

            # Fast-fail on NaN
            if torch.isnan(loss):
                print("ERROR: NaN loss detected, aborting.")
                print("---")
                print(f"val_loss:          nan")
                print(f"val_accuracy:      0.0")
                print(f"training_seconds:  {total_training_time:.1f}")
                print(f"total_seconds:     {time.time() - t_start:.1f}")
                return

        if total_training_time >= TIME_BUDGET:
            break

    # Final evaluation
    val_loss = evaluate_loss(model, device)
    val_acc = evaluate_accuracy(model, device)
    t_end = time.time()

    param_count = sum(p.numel() for p in model.parameters())

    print("---")
    print(f"val_loss:          {val_loss:.6f}")
    print(f"val_accuracy:      {val_acc:.4f}")
    print(f"training_seconds:  {total_training_time:.1f}")
    print(f"total_seconds:     {t_end - t_start:.1f}")
    print(f"total_steps:       {step}")
    print(f"total_epochs:      {epoch}")
    print(f"param_count:       {param_count}")


if __name__ == "__main__":
    main()
```

### Step 4: Create a minimal `program.md`

```markdown
# AutoResearch — CIFAR-10 CPU Substrate

You are an autonomous ML researcher optimizing a CIFAR-10 classifier on CPU.

## Goal
Minimize `val_loss` (validation cross-entropy loss) — **lower is better**.

## Files
- `train.py` — the ONLY file you modify. Contains model, optimizer, hyperparameters.
- `prepare.py` — read-only. Data loading, evaluation, constants.

## Metric
After each 2-minute training run, `train.py` prints `val_loss: X.XXXXXX`.
This is the number you are optimizing.

## What you can change in train.py
- Architecture: DEPTH, BASE_CHANNELS, CHANNEL_MULT, USE_BATCHNORM, DROPOUT_RATE, FC_HIDDEN
- Optimizer: OPTIMIZER type, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM, ADAM_BETAS
- LR schedule: USE_LR_SCHEDULE, WARMUP_EPOCHS, LR_DECAY_FACTOR, LR_DECAY_EPOCHS
- Batch/data: BATCH_SIZE, NUM_WORKERS
- Model architecture: ConvBlock, CIFAR10Net (add residual connections, change pooling, etc.)
- Anything else in train.py

## What you cannot change
- prepare.py (read-only)
- The evaluation function
- The time budget (2 minutes, enforced by TIME_BUDGET in prepare.py)
```

### Step 5: Download data once

```bash
cd autoresearch
python prepare.py
```

### Step 6: Verify baseline

```bash
python train.py
```

Expected output (approximate, CPU-dependent):
```
---
val_loss:          1.450000
val_accuracy:      0.4800
training_seconds:  120.0
total_seconds:     125.3
total_steps:       ~1200
total_epochs:      ~3
param_count:       ~120000
```

The baseline should give ~1.3–1.6 val_loss. There is substantial room for improvement (state-of-art simple CNNs reach ~0.5 val_loss on CIFAR-10), which ensures agents have enough optimization surface.

---

## Compatibility with the framework

### Metric name change

The framework's `log_parser.py` and `worker_loop.sh` grep for `val_bpb:`. Two options:

**Option A (recommended):** Change `train.py` to also print `val_bpb: X.XXXXXX` as an alias for `val_loss`. Add this line after the `val_loss` print:

```python
print(f"val_bpb:           {val_loss:.6f}")
```

This makes the substrate a drop-in replacement with zero framework changes.

**Option B:** Change all `val_bpb` references in the framework to `val_loss`. This is cleaner but requires modifying `log_parser.py`, `worker_loop.sh` templates in `training_harness.py`, the agent prompts, and `schema.py`.

**Recommendation:** Use Option A for the pilot. It is a one-line addition to `train.py` and avoids touching framework code.

### Worker scripts

The framework generates `start_gpu_worker.sh`, `run_on_worker.sh`, `stop_gpu_worker.sh` via `training_harness.py`. These use SLURM and `CUDA_VISIBLE_DEVICES`. For CPU-only local runs:

**Modify `training_harness.py`** to detect a `local_cpu` mode (or add a config flag `slurm.enabled: false`) and generate simpler scripts:

**`start_gpu_worker.sh` (CPU version):**
```bash
#!/bin/bash
# CPU mode — no GPU allocation needed
echo "cpu_worker_$$"
touch gpu_allocated_at
```

**`run_on_worker.sh` (CPU version):**
```bash
#!/bin/bash
# CPU mode — run training directly
rm -f run.result
RUN_COUNT=$((RUN_COUNT + 1))
RUN_LOG=$(printf 'logs/train_run_%03d.out' "$RUN_COUNT")
mkdir -p logs
python train.py > "$RUN_LOG" 2>&1
cp "$RUN_LOG" logs/train_current.out
VAL=$(grep 'val_bpb:' logs/train_current.out | head -1 | awk '{print $2}') || VAL=""
VRAM="0.0"
if [ -n "$VAL" ]; then
    echo "TRAINING DONE" > run.result
    echo "val_bpb: $VAL" >> run.result
    echo "peak_vram_mb: 0.0" >> run.result
    echo "TRAINING DONE"
    echo "val_bpb: $VAL"
    echo "peak_vram_mb: 0.0"
else
    ERRMSG=$(tail -5 logs/train_current.out | tr '\n' ' ')
    echo "TRAINING FAILED: $ERRMSG" > run.result
    echo "TRAINING FAILED: $ERRMSG"
fi
```

**`stop_gpu_worker.sh` (CPU version):**
```bash
#!/bin/bash
# CPU mode — nothing to release
echo "CPU worker stopped"
```

### Agent prompt adjustments

In `templates/agent_system_prompt.md`, the only change needed is:

1. Replace references to GPU/VRAM with CPU.
2. Replace `val_bpb` with `val_loss` in explanations (but keep `val_bpb` as the grepped metric name per Option A).
3. Change "5 minutes" to "2 minutes" for training time.

These can be parameterized via the existing template placeholder system (`{{TRAIN_TIME_BUDGET_MIN}}`).

### Config for CIFAR-10 local runs

Create `configs/experiment_cifar10.yaml`:

```yaml
experiment:
  id: null
  mode: parallel            # change per cell: single_long / single_memory / parallel / parallel_shared
  runs_dir: runs

agents:
  n: 2
  model: claude-sonnet-4-6  # or claude-haiku-4-5-20251001 for cheaper pilot
  time_budget_minutes: 30
  train_time_budget_seconds: 120   # 2 minutes for CIFAR-10 on CPU
  temperature: null
  cuda_devices: null               # not used in CPU mode

slurm:
  enabled: false                   # CPU-only, no SLURM

templates:
  system_prompt: templates/agent_system_prompt.md
  first_message: templates/agent_first_message.md
```

---

## Properties of this substrate

| Property | nanochat (GPU) | CIFAR-10 (CPU) | Match? |
|----------|---------------|----------------|--------|
| Single file agent modifies | train.py | train.py | ✓ |
| Fixed time budget per run | 5 min | 2 min | ✓ |
| Scalar noisy metric (lower=better) | val_bpb | val_loss | ✓ |
| Noise source | training randomness | training randomness | ✓ |
| Mode: optimizer | Muon/AdamW params | Adam/SGD/AdamW params | ✓ |
| Mode: lr_schedule | warmup/warmdown ratios | cosine/step schedule | ✓ |
| Mode: architecture | depth, attention, embeddings | depth, channels, BN, residuals | ✓ |
| Mode: batch_data | batch size, seq len | batch size, num workers | ✓ |
| Mode: other | regularization, precision | dropout, init, augmentation | ✓ |
| Optimization surface | ~11% improvement (Karpathy) | ~60%+ improvement possible | ✓ (more room) |
| Hardware | H100 GPU required | any CPU | ✓ (key advantage) |
| Cost per attempt | ~$0.22 (GPU + tokens) | tokens only | ✓ (cheaper) |

---

## What this substrate does NOT test

- GPU-specific effects (VRAM pressure, kernel compilation time, flash attention)
- The exact nanochat model and dataset
- Results comparable to the AutoResearch leaderboard

The purpose is to validate the BP decomposition framework and test hypotheses H1–H6 on the cheapest possible substrate. If the decomposition works here, it generalizes; if it doesn't, the failure is informative regardless of substrate.
