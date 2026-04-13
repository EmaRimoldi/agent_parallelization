# Phase 04: Escalation — Switch to CIFAR-100

## Goal

Replace CIFAR-10 with CIFAR-100 to increase task difficulty, creating more room for architecture effects and strategy diversity.

## Background

The calibration phase showed that the CIFAR-10 task is too simple for architecture differences to matter (Cohen's d ≤ 0.3). CIFAR-100 has 100 classes instead of 10, making it substantially harder for a simple CNN. This should:
- Increase the gap between good and bad approaches (larger Δ)
- Force agents to explore more diverse strategies (richer mode distributions)
- Make parallelism and memory more valuable

## Tasks

### 1. Modify `autoresearch/prepare.py`

Change the dataset from CIFAR-10 to CIFAR-100:

```python
# CHANGE THIS:
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, ...)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, ...)

# TO THIS:
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, ...)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, ...)
```

Also update the number of output classes if hardcoded:
```python
# If the model has num_classes=10 anywhere, change to num_classes=100
```

### 2. Update the baseline `autoresearch/train.py`

Ensure the model architecture supports 100 output classes:
- Check the final linear layer dimension
- If hardcoded to 10, change to 100

### 3. Verify the change works

```bash
cd autoresearch && python train.py
```

Expected: training completes, val_bpb is reported (will be higher than CIFAR-10).

### 4. Re-verify determinism

The deterministic modifications from Phase 01 should still work. Run verification:

```bash
python workflow/scripts/verify_determinism.py
```

### 5. Update agent prompts if needed

Check `templates/agent_system_prompt.md` and `templates/agent_first_message.md`:
- If they mention CIFAR-10 specifically, update to CIFAR-100
- Adjust any accuracy expectations (CIFAR-100 is harder)

### 6. Record the dataset change

```bash
python workflow/run.py measure '{"dataset": "cifar100", "num_classes": 100}'
python workflow/run.py log "Escalated to CIFAR-100"
```

## Required Inputs

- Current `autoresearch/prepare.py` and `autoresearch/train.py`
- Internet access to download CIFAR-100 dataset

## Expected Outputs

- Modified prepare.py loading CIFAR-100
- Modified train.py with 100-class output
- Verified deterministic evaluation on CIFAR-100
- Updated agent prompts

## Success Criteria

- Training completes on CIFAR-100 without errors
- Determinism verification passes (5 identical runs)
- val_bpb is in a reasonable range (higher than CIFAR-10)

## Failure Modes

- Out of memory: CIFAR-100 has the same image size as CIFAR-10, so memory shouldn't be an issue.
- Training too slow: the same model architecture should train at similar speed. If it's too slow, reduce epochs.
- Model too simple: if a simple CNN can't learn CIFAR-100 at all (val_bpb doesn't improve from random), increase model capacity.

## Next Phase

After dataset change and determinism verification: re-enter the main pipeline at `01_deterministic_eval` (the verify step will fast-path since we just verified)

In practice, the workflow will loop back to `02_power_calibration` with the new dataset.
