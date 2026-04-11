# AutoResearch CIFAR-10 Substrate

This directory is the CPU training substrate used by the agent orchestration
framework in this repository.

Files:
- `prepare.py`: read-only data download and evaluation harness
- `train.py`: the only file the research agent should modify
- `program.md`: short substrate-specific instructions

Metric:
- `val_bpb` is emitted as an alias for validation loss so the existing
  framework can parse results without renaming downstream fields.
