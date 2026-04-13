# Figure Archive

All tracked repository figures live under this directory.

## Subfolders

- `pass_01_pilot/`
  Pass 01 pilot figures generated from the original 2x2 BP experiment.
- `pass_02_workflow_calibration/`
  Pass 02 calibration and decision-gate figures produced by the `workflow/` system.
  `workflow/` is the phase-based orchestration layer for Pass 02, not a separate pass.
- `pass_02_archive_overview/`
  Pass 02 repository-level overview figures used by the top-level README to summarize the archive and experiment history.

## Canonical usage

- Use this directory for README-linked figures and tracked archival visuals.
- Keep raw logs, JSON summaries, and per-phase notes in their native experiment folders.
- If a script renders figures for the repository docs, it should write them here rather than into `docs/` or `workflow/artifacts/`.
