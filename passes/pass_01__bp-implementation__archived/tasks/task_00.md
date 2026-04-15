# Task 0: Branch setup and CPU substrate

You are implementing a series of modifications to the `agent_parallelization` repository. Read `docs/guides/IMPLEMENTATION_GUIDE.md` and `docs/guides/CPU_SUBSTRATE_GUIDE.md` for full context.

## This task

1. Create and switch to branch `bp-2x2-instrumentation` from main:
```bash
git checkout -b bp-2x2-instrumentation main
```

2. Create the CIFAR-10 CPU training substrate as specified in `docs/guides/CPU_SUBSTRATE_GUIDE.md`. Specifically:
   - Create `autoresearch/prepare.py` with the exact content from the guide
   - Create `autoresearch/train.py` with the exact content from the guide
   - Create `autoresearch/program.md` with the content from the guide

3. Add `val_bpb` alias to `autoresearch/train.py` — after the `val_loss` print line, add:
```python
print(f"val_bpb:           {val_loss:.6f}")
```
This ensures compatibility with the framework's log parser which greps for `val_bpb:`.

4. Run `cd autoresearch && python prepare.py` to download CIFAR-10 data.

5. Run `cd autoresearch && python train.py` to verify the baseline works. Training should complete in ~2 minutes and print `val_loss`, `val_bpb`, `val_accuracy`, etc.

6. Commit everything:
```bash
git add -A
git commit -m "Add CIFAR-10 CPU substrate and implementation guides"
```

## Success criteria
- Branch `bp-2x2-instrumentation` exists
- `autoresearch/prepare.py` and `autoresearch/train.py` exist and are runnable
- `python autoresearch/train.py` completes in ~2 minutes on CPU and prints `val_bpb: X.XXXXXX`
- All files committed

Do NOT proceed to other tasks. Stop after committing.
