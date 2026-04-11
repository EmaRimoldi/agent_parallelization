"""Post-hoc mode labeling for the BP decomposition.

Usage:
    python scripts/label_modes.py --experiment-dir runs/experiment_XXX

Reads snapshots from each agent, diffs train.py against baseline,
and writes mode labels to agent_dir/results/mode_labels.jsonl.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path


MODE_KEYWORDS = {
    "optimizer": [
        "adamw",
        "muon",
        "adam_betas",
        "weight_decay",
        "optimizer",
        "adamw_step",
        "muon_step",
        "setup_optimizer",
    ],
    "lr_schedule": [
        "embedding_lr",
        "unembedding_lr",
        "matrix_lr",
        "scalar_lr",
        "warmup_ratio",
        "warmdown_ratio",
        "final_lr_frac",
        "get_lr_multiplier",
        "get_muon_momentum",
        "learning_rate",
        "_lr",
    ],
    "architecture": [
        "depth",
        "aspect_ratio",
        "head_dim",
        "n_head",
        "n_embd",
        "window_pattern",
        "causalselfattention",
        "mlp",
        "block",
        "gptconfig",
        "n_layer",
        "rotary",
        "attention",
    ],
    "batch_data": [
        "total_batch_size",
        "device_batch_size",
        "max_seq_len",
        "dataloader",
        "batch",
        "accumulation",
    ],
}


def classify_diff(diff_lines: list[str]) -> str:
    """Classify a diff into a mode based on keyword matching."""
    added_removed = " ".join(
        line[1:].lower()
        for line in diff_lines
        if line.startswith("+") or line.startswith("-")
    )

    scores = {}
    for mode, keywords in MODE_KEYWORDS.items():
        scores[mode] = sum(1 for kw in keywords if kw in added_removed)

    best_mode = max(scores, key=scores.get)
    if scores[best_mode] == 0:
        return "other"
    return best_mode


def label_experiment(experiment_dir: Path):
    """Label all snapshots in an experiment with mode classifications."""
    for agent_dir in sorted(experiment_dir.glob("mode_*/agent_*")):
        snapshots_dir = agent_dir / "snapshots"
        baseline_path = agent_dir / "workspace" / "train.py.baseline"

        if not snapshots_dir.exists() or not baseline_path.exists():
            continue

        baseline = baseline_path.read_text().splitlines()
        labels = []
        prev_accepted = baseline

        for step_dir in sorted(snapshots_dir.glob("step_*")):
            train_path = step_dir / "train.py"
            meta_path = step_dir / "metadata.json"

            if not train_path.exists():
                continue

            current = train_path.read_text().splitlines()
            diff = list(difflib.unified_diff(prev_accepted, current, lineterm=""))
            mode = classify_diff(diff)

            meta = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())

            label_entry = {
                "step": meta.get("step", meta.get("step_index", step_dir.name)),
                "mode": mode,
                "diff_lines_changed": len(
                    [line for line in diff if line.startswith("+") or line.startswith("-")]
                ),
                "hypothesis": meta.get("hypothesis", ""),
                "val_bpb_after": meta.get("val_bpb_after"),
                "accepted": meta.get("accepted"),
            }
            labels.append(label_entry)

            if meta.get("accepted"):
                prev_accepted = current

        output_path = agent_dir / "results" / "mode_labels.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            for entry in labels:
                fh.write(json.dumps(entry) + "\n")

        print(f"Labeled {len(labels)} steps for {agent_dir.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    label_experiment(args.experiment_dir)
