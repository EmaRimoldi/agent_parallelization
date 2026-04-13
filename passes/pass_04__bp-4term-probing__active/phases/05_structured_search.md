# Phase 05: Fallback — Structured Hyperparameter Search

## Goal

Replace free-form code editing with a structured config-based search where each agent proposes hyperparameter configurations instead of arbitrary code changes. This gives exact mode labeling and clean theoretical mapping.

## Background

Free-form code editing produces strategies that are hard to classify into modes. If mode identification remains degenerate even with deterministic evaluation and extended budgets, structured search provides a clean alternative where:
- Each configuration dimension maps directly to a BP mode
- Mode distributions are exact (based on which dimensions were changed)
- Evaluation is deterministic by construction
- The mapping to BP theory is transparent

## Tasks

### 1. Define the configuration space

Create a config template `autoresearch/config_template.yaml`:

```yaml
# Model architecture
architecture:
  depth: 4              # 2, 3, 4, 5, 6
  base_channels: 32     # 16, 32, 64, 128
  use_batchnorm: true   # true, false
  dropout_rate: 0.0     # 0.0, 0.1, 0.2, 0.3, 0.5

# Optimizer
optimizer:
  type: "sgd"           # sgd, adam, adamw
  learning_rate: 0.01   # range: 1e-4 to 1.0
  weight_decay: 0.0     # range: 0 to 0.1
  momentum: 0.9         # range: 0 to 0.99 (SGD only)

# Scheduler
scheduler:
  type: "none"          # none, cosine, step
  warmup_epochs: 0      # 0, 1, 2, 5
  min_lr: 1e-6          # range: 1e-8 to 1e-4

# Data
data:
  batch_size: 64        # 32, 64, 128, 256
  augmentation: "basic" # none, basic, strong

# Training
training:
  epochs: 50            # auto-determined by time budget
  gradient_clip: 0.0    # 0.0 (off), 1.0, 5.0
```

### 2. Create a config-driven train.py

Create `autoresearch/train_from_config.py` that:
- Reads a YAML/JSON config file
- Constructs the model, optimizer, scheduler, and data loaders from config values
- Trains and evaluates, reporting val_bpb
- Is fully deterministic (fixed seeds)

### 3. Create a config-based agent interface

Modify the agent prompt to instruct agents to:
- Output a JSON config instead of editing train.py
- The orchestrator writes the config and runs `train_from_config.py`
- Each config change is automatically labeled by which dimensions changed

### 4. Define mode mapping

Each config change is labeled by the section it modifies:
- `architecture`: any change to architecture.* → mode "architecture"
- `optimizer`: any change to optimizer.* → mode "optimizer"
- `scheduler`: any change to scheduler.* → mode "scheduler"
- `data`: any change to data.* → mode "data"
- `training`: any change to training.* → mode "training"

Multi-dimensional changes get the mode of the most-changed section.

### 5. Test the structured interface

Run a small test with one agent:
```bash
python scripts/run_parallel_experiment.py \
  --config configs/experiment_structured_d00.yaml \
  --output-dir runs/test_structured
```

### 6. Re-enter the main pipeline

After implementing structured search, return to Phase 03 (full 2x2 run) with the new interface.

## Required Inputs

- Current autoresearch pipeline
- Understanding of the model's hyperparameter space

## Expected Outputs

- `autoresearch/config_template.yaml`
- `autoresearch/train_from_config.py`
- Updated agent prompts for config-based search
- Mode labeling is now exact (config-dimension-based)

## Success Criteria

- An agent can propose configs and get deterministic val_bpb evaluations
- Mode labeling is automatic and exact
- The structured interface produces diverse configs across agents

## Failure Modes

- Agents don't follow the config format: add validation and error messages
- Config space is too small: agents quickly exhaust the grid. Allow continuous ranges.
- Config space is too large: agents explore randomly without learning. Add a warm-start config.

## Next Phase

On completion: return to `03_full_2x2_run` with the structured search interface
