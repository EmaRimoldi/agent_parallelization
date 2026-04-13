"""Post-hoc observable mode labeling for the BP decomposition.

Usage:
    python scripts/label_modes.py --experiment-dir runs/experiment_XXX

Reads snapshot metadata plus train.py diffs and writes
agent_dir/results/mode_labels.jsonl.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path


MODE_KEYWORDS = {
    "optimization": [
        "optimizer",
        "adam",
        "adamw",
        "muon",
        "learning rate",
        "lr",
        "scheduler",
        "warmup",
        "beta",
        "momentum",
    ],
    "regularization": [
        "dropout",
        "weight_decay",
        "weight decay",
        "regularization",
        "label smoothing",
        "clip",
    ],
    "architecture": [
        "depth",
        "width",
        "hidden",
        "channel",
        "channels",
        "base channels",
        "conv",
        "attention",
        "head",
        "layer",
        "block",
        "embedding",
        "mlp",
        "batchnorm",
    ],
    "data_pipeline": [
        "batch",
        "dataloader",
        "dataset",
        "augment",
        "mixup",
        "cutmix",
        "crop",
        "flip",
        "worker",
    ],
    "training_loop": [
        "max steps",
        "step budget",
        "steps",
        "epoch",
        "epochs",
        "accumulation",
        "clip grad",
        "gradient accumulation",
    ],
    "memory_or_coordination": [
        "memory",
        "cache",
        "shared",
        "workspace",
        "retrieve",
        "coordination",
    ],
}

ACCEPTED_DECISIONS = {
    "promoted_after_reevaluation",
}


def classify_text_category(text: str) -> str:
    lowered = text.lower().replace("_", " ")
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    scores = {}
    for mode, keywords in MODE_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            normalized = keyword.lower().replace("_", " ")
            if " " in normalized:
                score += int(normalized in lowered)
            else:
                score += int(normalized in tokens)
        scores[mode] = score
    best_mode = max(scores, key=scores.get)
    return best_mode if scores[best_mode] > 0 else "other"


def diff_changed_lines(before: list[str], after: list[str]) -> list[str]:
    diff = difflib.unified_diff(before, after, lineterm="")
    return [
        line[1:]
        for line in diff
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]


def classify_diff_category(changed_lines: list[str]) -> tuple[str, dict[str, int]]:
    lowered = " ".join(line.lower() for line in changed_lines)
    scores = {
        mode: sum(1 for keyword in keywords if keyword in lowered)
        for mode, keywords in MODE_KEYWORDS.items()
    }
    best_mode = max(scores, key=scores.get)
    return (best_mode if scores[best_mode] > 0 else "other"), scores


def load_trace_by_step(agent_dir: Path) -> dict[int, dict]:
    trace_path = agent_dir / "reasoning" / "trace.jsonl"
    entries: dict[int, dict] = {}
    if not trace_path.exists():
        return entries
    for raw_line in trace_path.read_text().splitlines():
        if not raw_line.strip():
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        step_index = entry.get("step_index")
        if isinstance(step_index, int):
            entries[step_index] = entry
    return entries


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for raw_line in path.read_text().splitlines():
        if not raw_line.strip():
            continue
        try:
            rows.append(json.loads(raw_line))
        except json.JSONDecodeError:
            continue
    return rows


def choose_mode(
    hypothesis_category: str,
    diff_category: str,
    baseline_diff_category: str,
) -> tuple[str, str]:
    if hypothesis_category != "other":
        return hypothesis_category, "hypothesis"
    if diff_category != "other":
        return diff_category, "accepted_diff"
    if baseline_diff_category != "other":
        return baseline_diff_category, "baseline_diff"
    return "other", "fallback"


def is_reevaluation_placeholder(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith("reevaluate") or lowered.startswith("re-evaluate")


def build_snapshot_index(agent_dir: Path) -> dict[str, dict]:
    snapshots_dir = agent_dir / "snapshots"
    baseline_path = agent_dir / "workspace" / "train.py.baseline"
    workspace_train_path = agent_dir / "workspace" / "train.py"

    if not snapshots_dir.exists():
        return {}

    if baseline_path.exists():
        baseline_lines = baseline_path.read_text().splitlines()
        baseline_source = "train.py.baseline"
    elif workspace_train_path.exists():
        baseline_lines = workspace_train_path.read_text().splitlines()
        baseline_source = "workspace_train.py_fallback"
    else:
        first_snapshot = next(iter(sorted(snapshots_dir.glob("step_*/train.py"))), None)
        if first_snapshot is None:
            return {}
        baseline_lines = first_snapshot.read_text().splitlines()
        baseline_source = "first_snapshot_fallback"

    accepted_reference_lines = baseline_lines
    trace_by_step = load_trace_by_step(agent_dir)
    snapshot_index: dict[str, dict] = {}

    for step_dir in sorted(snapshots_dir.glob("step_*")):
        train_path = step_dir / "train.py"
        meta_path = step_dir / "metadata.json"
        if not train_path.exists() or not meta_path.exists():
            continue

        try:
            metadata = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue

        current_lines = train_path.read_text().splitlines()
        step_index = metadata.get("step_index")
        if not isinstance(step_index, int):
            try:
                step_index = int(step_dir.name.split("_")[1])
            except (IndexError, ValueError):
                step_index = -1

        trace_entry = trace_by_step.get(step_index, {})
        hypothesis = str(metadata.get("hypothesis") or trace_entry.get("hypothesis") or "")
        expected_effect = str(
            metadata.get("expected_effect") or trace_entry.get("expected_effect") or ""
        )
        hypothesis_category = str(
            metadata.get("strategy_category")
            or trace_entry.get("strategy_category")
            or classify_text_category(f"{hypothesis} {expected_effect}")
        )

        accepted_diff_lines = diff_changed_lines(accepted_reference_lines, current_lines)
        baseline_diff_lines = diff_changed_lines(baseline_lines, current_lines)
        diff_category, diff_scores = classify_diff_category(accepted_diff_lines)
        baseline_diff_category, baseline_scores = classify_diff_category(baseline_diff_lines)

        commit = str(metadata.get("git_commit") or "")
        if commit:
            snapshot_index[commit] = {
                "step": step_index,
                "hypothesis": hypothesis,
                "expected_effect": expected_effect,
                "hypothesis_category": hypothesis_category,
                "diff_category": diff_category,
                "baseline_diff_category": baseline_diff_category,
                "diff_scores": diff_scores,
                "baseline_diff_scores": baseline_scores,
                "baseline_source": baseline_source,
                "diff_lines_changed": len(accepted_diff_lines),
                "baseline_diff_lines_changed": len(baseline_diff_lines),
                "shared_memory_entries_visible": metadata.get("shared_memory_entries_visible"),
                "prior_trace_entries_visible": metadata.get("prior_trace_entries_visible"),
            }

        if metadata.get("accepted") is True:
            accepted_reference_lines = current_lines

    return snapshot_index


def build_label_from_training_run(
    row: dict,
    snapshot_entry: dict | None,
    agent_name: str,
    prior_entry: dict | None = None,
) -> dict:
    commit = str(row.get("candidate_commit") or row.get("commit") or "")
    evaluation_kind = str(row.get("evaluation_kind") or "primary")

    raw_hypothesis = str(row.get("hypothesis") or "")
    raw_expected_effect = str(row.get("expected_effect") or "")
    prior_hypothesis = str((prior_entry or {}).get("hypothesis") or "")
    prior_expected = str((prior_entry or {}).get("expected_effect") or "")

    if evaluation_kind == "reevaluation" and prior_entry and (
        not raw_hypothesis or is_reevaluation_placeholder(raw_hypothesis)
    ):
        hypothesis = prior_hypothesis
        expected_effect = prior_expected
    else:
        hypothesis = str(raw_hypothesis or (snapshot_entry or {}).get("hypothesis") or prior_hypothesis)
        expected_effect = str(
            raw_expected_effect
            or (snapshot_entry or {}).get("expected_effect")
            or prior_expected
        )

    strategy_category = str(row.get("strategy_category") or "").strip()
    if evaluation_kind == "reevaluation" and prior_entry and (
        not strategy_category or strategy_category == "other"
    ):
        strategy_category = str(prior_entry.get("hypothesis_category") or "")

    if strategy_category and strategy_category != "other":
        hypothesis_category = strategy_category
    else:
        hypothesis_category = str(
            (snapshot_entry or {}).get("hypothesis_category")
            or (prior_entry or {}).get("hypothesis_category")
            or classify_text_category(f"{hypothesis} {expected_effect}")
        )

    diff_category = str((snapshot_entry or {}).get("diff_category") or "other")
    baseline_diff_category = str((snapshot_entry or {}).get("baseline_diff_category") or "other")
    mode, mode_source = choose_mode(hypothesis_category, diff_category, baseline_diff_category)
    if (
        evaluation_kind == "reevaluation"
        and prior_entry
        and mode == "other"
        and str(prior_entry.get("mode") or "other") != "other"
    ):
        mode = str(prior_entry["mode"])
        mode_source = "candidate_history"
    promotion_decision = row.get("promotion_decision")

    return {
        "step": row.get("turn", row.get("run_index")),
        "run_index": row.get("run_index"),
        "mode": mode,
        "mode_source": "training_run_hypothesis" if mode_source == "hypothesis" else mode_source,
        "hypothesis_category": hypothesis_category,
        "diff_category": diff_category,
        "baseline_diff_category": baseline_diff_category,
        "diff_scores": (snapshot_entry or {}).get("diff_scores", {}),
        "baseline_diff_scores": (snapshot_entry or {}).get("baseline_diff_scores", {}),
        "baseline_source": (snapshot_entry or {}).get("baseline_source"),
        "diff_lines_changed": (snapshot_entry or {}).get("diff_lines_changed", 0),
        "baseline_diff_lines_changed": (snapshot_entry or {}).get("baseline_diff_lines_changed", 0),
        "candidate_commit": commit or None,
        "candidate_id": row.get("candidate_id") or (
            f"{agent_name}:{commit[:12]}" if commit else f"{agent_name}:run{row.get('run_index')}"
        ),
        "hypothesis": hypothesis,
        "expected_effect": expected_effect,
        "val_bpb_after": row.get("val_bpb"),
        "accepted": promotion_decision in ACCEPTED_DECISIONS,
        "promotion_decision": promotion_decision,
        "evaluation_kind": evaluation_kind,
        "baseline_candidate": row.get("baseline_candidate"),
        "shared_memory_entries_visible": row.get("shared_memory_context_entries"),
        "prior_trace_entries_visible": (snapshot_entry or {}).get("prior_trace_entries_visible"),
    }


def build_labels_from_snapshots(agent_dir: Path, snapshot_index: dict[str, dict]) -> list[dict]:
    labels = []
    for commit, snapshot_entry in sorted(
        snapshot_index.items(),
        key=lambda item: int(item[1].get("step", -1)),
    ):
        hypothesis_category = str(snapshot_entry.get("hypothesis_category") or "other")
        diff_category = str(snapshot_entry.get("diff_category") or "other")
        baseline_diff_category = str(snapshot_entry.get("baseline_diff_category") or "other")
        mode, mode_source = choose_mode(hypothesis_category, diff_category, baseline_diff_category)
        labels.append(
            {
                "step": snapshot_entry.get("step"),
                "mode": mode,
                "mode_source": mode_source,
                "hypothesis_category": hypothesis_category,
                "diff_category": diff_category,
                "baseline_diff_category": baseline_diff_category,
                "diff_scores": snapshot_entry.get("diff_scores", {}),
                "baseline_diff_scores": snapshot_entry.get("baseline_diff_scores", {}),
                "baseline_source": snapshot_entry.get("baseline_source"),
                "diff_lines_changed": snapshot_entry.get("diff_lines_changed", 0),
                "baseline_diff_lines_changed": snapshot_entry.get("baseline_diff_lines_changed", 0),
                "candidate_commit": commit or None,
                "candidate_id": (
                    f"{agent_dir.name}:{commit[:12]}" if commit else f"{agent_dir.name}:step{snapshot_entry.get('step')}"
                ),
                "hypothesis": snapshot_entry.get("hypothesis", ""),
                "expected_effect": snapshot_entry.get("expected_effect", ""),
                "val_bpb_after": None,
                "accepted": None,
                "shared_memory_entries_visible": snapshot_entry.get("shared_memory_entries_visible"),
                "prior_trace_entries_visible": snapshot_entry.get("prior_trace_entries_visible"),
            }
        )
    return labels


def label_experiment(experiment_dir: Path) -> None:
    """Label all snapshots in an experiment with observable mode classifications."""
    for agent_dir in sorted(experiment_dir.glob("mode_*/agent_*")):
        snapshot_index = build_snapshot_index(agent_dir)
        training_runs = load_jsonl(agent_dir / "results" / "training_runs.jsonl")
        labels = []

        if training_runs:
            prior_labels_by_candidate: dict[str, dict] = {}
            for row in training_runs:
                commit = str(row.get("candidate_commit") or row.get("commit") or "")
                candidate_id = str(
                    row.get("candidate_id")
                    or (f"{agent_dir.name}:{commit[:12]}" if commit else f"{agent_dir.name}:unknown")
                )
                snapshot_entry = snapshot_index.get(commit)
                label_entry = build_label_from_training_run(
                    row,
                    snapshot_entry,
                    agent_dir.name,
                    prior_entry=prior_labels_by_candidate.get(candidate_id),
                )
                labels.append(label_entry)
                prior_labels_by_candidate[candidate_id] = label_entry
        else:
            labels = build_labels_from_snapshots(agent_dir, snapshot_index)

        output_path = agent_dir / "results" / "mode_labels.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            for entry in labels:
                fh.write(json.dumps(entry) + "\n")

        mode_counts = Counter(entry["mode"] for entry in labels)
        print(
            f"Labeled {len(labels)} steps for {agent_dir.name}: "
            + ", ".join(f"{mode}={count}" for mode, count in sorted(mode_counts.items()))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    label_experiment(args.experiment_dir)
