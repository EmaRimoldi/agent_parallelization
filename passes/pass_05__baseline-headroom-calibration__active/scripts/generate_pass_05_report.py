"""Generate Pass 05 README, summary tables, and figures.

This script reads the baseline-headroom calibration outputs under `runs/` and
materializes the pass artifact under `passes/pass_05__baseline-headroom-calibration__active/`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
PASS_DIR = ROOT / "passes" / "pass_05__baseline-headroom-calibration__active"
RESULTS_DIR = PASS_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

RUN_SPECS = [
    (
        "default_fixed1170",
        "runs/baseline_headroom_calibration_fixed1170",
        1170,
        "Default healthy-mistuned baseline screen",
    ),
    (
        "extended_fixed1170",
        "runs/baseline_headroom_calibration_extended_targeted_fixed1170",
        1170,
        "Broader model / optimizer / regularization screen",
    ),
    (
        "refinement_fixed585",
        "runs/baseline_refinement_custom_fixed585",
        585,
        "Shorter-step refinement screen",
    ),
    (
        "refinement_fixed1170",
        "runs/baseline_refinement_custom_fixed1170",
        1170,
        "Intermediate-width / head / mild-dropout refinement",
    ),
]

RECOMMENDED_RUN = "refinement_fixed1170"
RECOMMENDED_BASELINE = "width30_lr_low"
RECOMMENDED_QSTAR = 0.824
MIN_DELTA = 0.005


@dataclass
class BaselineRecord:
    run_id: str
    run_path: str
    fixed_steps: int
    purpose: str
    baseline_id: str
    baseline_val_bpb: float
    edit_count: int
    raw_wins: int
    raw_win_rate: float
    category_count: int
    winning_categories: list[str]
    q3: float | None
    hits_at_q3: int | None
    best_edit_val_bpb: float | None
    best_edit_trial: str | None
    category_bests: dict[str, tuple[float, float, str]]


def load_rows() -> tuple[list[BaselineRecord], list[dict]]:
    records: list[BaselineRecord] = []
    trials: list[dict] = []
    for run_id, run_rel, fixed_steps, purpose in RUN_SPECS:
        run_path = ROOT / run_rel
        payload_path = run_path / "baseline_headroom_results.json"
        payload = json.loads(payload_path.read_text())
        raw_results = payload["results"]
        trials.extend(dict(row, run_id=run_id, fixed_steps=fixed_steps) for row in raw_results)
        baseline_ids = sorted({row["baseline_id"] for row in raw_results if row["is_baseline"]})
        for baseline_id in baseline_ids:
            baseline = next(
                row for row in raw_results
                if row["baseline_id"] == baseline_id and row["is_baseline"]
            )
            baseline_val = float(baseline["val_bpb"])
            edits = [
                row for row in raw_results
                if row["baseline_id"] == baseline_id
                and not row["is_baseline"]
                and row.get("val_bpb") is not None
            ]
            category_bests: dict[str, tuple[float, float, str]] = {}
            raw_wins = 0
            for row in edits:
                val = float(row["val_bpb"])
                improvement = baseline_val - val
                if improvement >= MIN_DELTA:
                    raw_wins += 1
                category = row["category"]
                prior = category_bests.get(category)
                if prior is None or val < prior[0]:
                    category_bests[category] = (val, improvement, row["id"])

            winning = sorted(
                category for category, (_, improvement, _) in category_bests.items()
                if improvement >= MIN_DELTA
            )
            winning_vals = sorted(
                val for val, improvement, _ in category_bests.values()
                if improvement >= MIN_DELTA
            )
            q3 = winning_vals[2] if len(winning_vals) >= 3 else None
            hits_at_q3 = (
                sum(1 for row in edits if float(row["val_bpb"]) <= q3)
                if q3 is not None else None
            )
            best_edit = min(edits, key=lambda row: float(row["val_bpb"]), default=None)
            records.append(
                BaselineRecord(
                    run_id=run_id,
                    run_path=run_rel,
                    fixed_steps=fixed_steps,
                    purpose=purpose,
                    baseline_id=baseline_id,
                    baseline_val_bpb=baseline_val,
                    edit_count=len(edits),
                    raw_wins=raw_wins,
                    raw_win_rate=(raw_wins / len(edits) if edits else 0.0),
                    category_count=len(winning),
                    winning_categories=winning,
                    q3=q3,
                    hits_at_q3=hits_at_q3,
                    best_edit_val_bpb=(
                        float(best_edit["val_bpb"]) if best_edit is not None else None
                    ),
                    best_edit_trial=best_edit["id"] if best_edit is not None else None,
                    category_bests=category_bests,
                )
            )
    return records, trials


def write_tables(records: list[BaselineRecord], trials: list[dict]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with (TABLES_DIR / "baseline_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run_id",
                "fixed_steps",
                "baseline_id",
                "baseline_val_bpb",
                "raw_wins",
                "edit_count",
                "raw_win_rate",
                "category_count",
                "winning_categories",
                "q3",
                "hits_at_q3",
                "best_edit_trial",
                "best_edit_val_bpb",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "run_id": record.run_id,
                    "fixed_steps": record.fixed_steps,
                    "baseline_id": record.baseline_id,
                    "baseline_val_bpb": f"{record.baseline_val_bpb:.6f}",
                    "raw_wins": record.raw_wins,
                    "edit_count": record.edit_count,
                    "raw_win_rate": f"{record.raw_win_rate:.6f}",
                    "category_count": record.category_count,
                    "winning_categories": ",".join(record.winning_categories),
                    "q3": "" if record.q3 is None else f"{record.q3:.6f}",
                    "hits_at_q3": "" if record.hits_at_q3 is None else record.hits_at_q3,
                    "best_edit_trial": record.best_edit_trial or "",
                    "best_edit_val_bpb": (
                        "" if record.best_edit_val_bpb is None
                        else f"{record.best_edit_val_bpb:.6f}"
                    ),
                }
            )

    with (TABLES_DIR / "trial_results.csv").open("w", newline="") as fh:
        fieldnames = [
            "run_id",
            "fixed_steps",
            "id",
            "baseline_id",
            "category",
            "is_baseline",
            "val_bpb",
            "total_seconds",
            "status",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in trials:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    payload = {
        "recommended": {
            "run_id": RECOMMENDED_RUN,
            "baseline_id": RECOMMENDED_BASELINE,
            "q_star": RECOMMENDED_QSTAR,
        },
        "baselines": [
            {
                "run_id": record.run_id,
                "fixed_steps": record.fixed_steps,
                "baseline_id": record.baseline_id,
                "baseline_val_bpb": record.baseline_val_bpb,
                "raw_wins": record.raw_wins,
                "edit_count": record.edit_count,
                "raw_win_rate": record.raw_win_rate,
                "category_count": record.category_count,
                "winning_categories": record.winning_categories,
                "q3": record.q3,
                "hits_at_q3": record.hits_at_q3,
                "best_edit_trial": record.best_edit_trial,
                "best_edit_val_bpb": record.best_edit_val_bpb,
            }
            for record in records
        ],
    }
    (TABLES_DIR / "pass_05_summary.json").write_text(json.dumps(payload, indent=2))


def _candidate_label(record: BaselineRecord) -> str:
    return f"{record.baseline_id}\n{record.run_id.replace('_', ' ')}"


def plot_screen_overview(records: list[BaselineRecord]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda rec: (rec.fixed_steps, rec.baseline_val_bpb))
    labels = [_candidate_label(rec) for rec in ordered]
    values = [rec.baseline_val_bpb for rec in ordered]
    colors = [
        "#2f7d32" if rec.baseline_id == RECOMMENDED_BASELINE and rec.run_id == RECOMMENDED_RUN
        else ("#5271a5" if rec.fixed_steps == 1170 else "#b8792e")
        for rec in ordered
    ]

    plt.figure(figsize=(12, 8))
    y = list(range(len(ordered)))
    plt.barh(y, values, color=colors)
    for idx, rec in enumerate(ordered):
        if rec.q3 is not None:
            plt.scatter(rec.q3, idx, color="black", marker="D", s=28, zorder=3)
            plt.text(rec.q3 + 0.005, idx, f"q3={rec.q3:.3f}", va="center", fontsize=8)
    plt.axvline(RECOMMENDED_QSTAR, color="#9b1b30", linestyle="--", linewidth=1.5, label="recommended q*")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("val_bpb (lower is better)")
    plt.title("Pass 05 baseline candidates and category-derived q*")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure-01-baseline-screen-overview.png", dpi=180)
    plt.close()


def plot_gate_scatter(records: list[BaselineRecord]) -> None:
    plt.figure(figsize=(9, 6))
    for rec in records:
        color = "#5271a5" if rec.fixed_steps == 1170 else "#b8792e"
        marker = "*" if rec.baseline_id == RECOMMENDED_BASELINE and rec.run_id == RECOMMENDED_RUN else "o"
        size = 240 if marker == "*" else 80
        plt.scatter(rec.raw_win_rate, rec.category_count, s=size, color=color, marker=marker, alpha=0.85)
        if rec.category_count >= 3:
            plt.text(rec.raw_win_rate + 0.01, rec.category_count + 0.03, rec.baseline_id, fontsize=8)
    plt.axhline(3, color="black", linestyle="--", linewidth=1, label=">=3 category gate")
    plt.axvspan(0.10, 0.30, color="#d8ead3", alpha=0.55, label="original raw-win gate")
    plt.xlabel("raw edit win rate")
    plt.ylabel("winning category count")
    plt.title("Gate diagnostics: category richness vs task permissiveness")
    plt.ylim(-0.2, 5.4)
    plt.xlim(-0.02, 1.05)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure-02-gate-diagnostics.png", dpi=180)
    plt.close()


def plot_category_heatmap(records: list[BaselineRecord]) -> None:
    selected_ids = {
        ("default_fixed1170", "narrow_lr_low"),
        ("default_fixed1170", "overregularized_lr_low"),
        ("extended_fixed1170", "sgd_baseline"),
        ("refinement_fixed1170", "width30_lr_low"),
        ("refinement_fixed1170", "fc96_lr_low"),
        ("refinement_fixed1170", "dropout005_lr_low"),
    }
    selected = [
        rec for rec in records
        if (rec.run_id, rec.baseline_id) in selected_ids
    ]
    categories = sorted({cat for rec in selected for cat in rec.category_bests})
    matrix: list[list[float]] = []
    for rec in selected:
        row = []
        for cat in categories:
            row.append(rec.category_bests.get(cat, (rec.baseline_val_bpb, 0.0, ""))[1])
        matrix.append(row)

    plt.figure(figsize=(11, 4.8))
    im = plt.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.08, vmax=0.12)
    plt.colorbar(im, label="best improvement over baseline")
    plt.xticks(range(len(categories)), categories, rotation=35, ha="right")
    plt.yticks(range(len(selected)), [_candidate_label(rec) for rec in selected], fontsize=8)
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            plt.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8)
    plt.title("Best improvement by strategy category")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure-03-category-improvement-heatmap.png", dpi=180)
    plt.close()


def plot_recommended_detail(records: list[BaselineRecord], trials: list[dict]) -> None:
    rec = next(
        rec for rec in records
        if rec.run_id == RECOMMENDED_RUN and rec.baseline_id == RECOMMENDED_BASELINE
    )
    rows = [
        row for row in trials
        if row["run_id"] == RECOMMENDED_RUN
        and row["baseline_id"] == RECOMMENDED_BASELINE
    ]
    rows = sorted(rows, key=lambda row: (row["is_baseline"], row["category"], row["id"]))
    labels = ["baseline" if row["is_baseline"] else row["id"].split("__", 1)[1] for row in rows]
    values = [float(row["val_bpb"]) for row in rows]
    colors = ["#555555" if row["is_baseline"] else "#5271a5" for row in rows]

    plt.figure(figsize=(12, 5.5))
    x = list(range(len(rows)))
    plt.bar(x, values, color=colors)
    plt.axhline(rec.baseline_val_bpb, color="black", linestyle="--", linewidth=1, label="baseline")
    plt.axhline(RECOMMENDED_QSTAR, color="#9b1b30", linestyle="--", linewidth=1.5, label="q*=0.824")
    plt.xticks(x, labels, rotation=35, ha="right", fontsize=8)
    plt.ylabel("val_bpb")
    plt.title("Recommended baseline: width30_lr_low at 1170 fixed steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure-04-recommended-baseline-detail.png", dpi=180)
    plt.close()


def write_markdown(records: list[BaselineRecord]) -> None:
    recommended = next(
        rec for rec in records
        if rec.run_id == RECOMMENDED_RUN and rec.baseline_id == RECOMMENDED_BASELINE
    )
    total_trials = sum(rec.edit_count + 1 for rec in records)
    top_rows = sorted(
        [rec for rec in records if rec.category_count >= 3],
        key=lambda rec: (abs(rec.raw_win_rate - 0.45), rec.baseline_val_bpb),
    )[:8]

    table_lines = []
    for rec in top_rows:
        table_lines.append(
            "| {baseline} | {run} | {steps} | {base:.6f} | {wins}/{edits} | {cats} | {q3} |".format(
                baseline=rec.baseline_id,
                run=rec.run_id,
                steps=rec.fixed_steps,
                base=rec.baseline_val_bpb,
                wins=rec.raw_wins,
                edits=rec.edit_count,
                cats=", ".join(rec.winning_categories),
                q3="NA" if rec.q3 is None else f"{rec.q3:.6f}",
            )
        )

    md = f"""# Pass 05 - Baseline Headroom Calibration

**Status**: Active
**Period**: April 14, 2026
**Objective**: Find a healthy but non-trivial AutoResearch baseline before running the reviewer-grade BP 2x2.

## Research Question

Pass 04 showed that the previous task was too close to a narrow local optimum:
only a small learning-rate region reliably improved validation loss. Pass 05 asks:

**Can we choose a baseline and fixed-step evaluator where several distinct strategy categories can improve the model, without making the task trivially easy?**

This pass is deliberately non-agentic. It calibrates the task geometry before spending LLM budget on architecture comparisons.

## What Changed

Pass 05 added a controlled baseline-headroom calibration tool:

- fixed-step evaluator support with `AUTOSEARCH_MAX_STEPS`;
- isolated workspaces for every baseline/edit trial;
- baseline and edit panels covering optimizer/LR, scheduler, capacity, regularization, and batch/data;
- JSON/CSV/Markdown outputs for calibration screens;
- reviewer-grade cost and hitting-time instrumentation from the preceding protocol work.

The working `autoresearch/train.py` baseline was updated to the selected candidate:

```text
DEPTH = 3
BASE_CHANNELS = 30
FC_HIDDEN = 128
OPTIMIZER = adam
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.0
USE_LR_SCHEDULE = False
BATCH_SIZE = 128
AUTOSEARCH_MAX_STEPS = 1170
```

## Experiments

| screen | fixed steps | trials | purpose |
| --- | ---: | ---: | --- |
| `baseline_headroom_calibration_fixed1170` | 1170 | 43 | default healthy-mistuned baseline screen |
| `baseline_headroom_calibration_extended_targeted_fixed1170` | 1170 | 38 | broader model / optimizer / regularization screen |
| `baseline_refinement_custom_fixed585` | 585 | 40 | shorter-step refinement screen |
| `baseline_refinement_custom_fixed1170` | 1170 | 40 | intermediate-width / head / mild-dropout refinement |

Total controlled evaluations summarized here: **{total_trials}**.

## Key Figures

![Baseline screen overview](results/figures/figure-01-baseline-screen-overview.png)

**Figure 1**: baseline quality and category-derived `q3` thresholds. The selected baseline is not the weakest; it sits near the better end while still preserving multiple useful intervention categories.

![Gate diagnostics](results/figures/figure-02-gate-diagnostics.png)

**Figure 2**: raw edit win rate versus number of winning strategy categories. The original 10-30% raw-win gate was too strict for this panel because several duplicate edits within the same category can win. The more useful diagnostic is category richness plus negative controls.

![Category improvement heatmap](results/figures/figure-03-category-improvement-heatmap.png)

**Figure 3**: best improvement by strategy category. Good candidates expose multiple positive categories while retaining negative or weak categories.

![Recommended baseline detail](results/figures/figure-04-recommended-baseline-detail.png)

**Figure 4**: detailed trial outcomes for `width30_lr_low`.

## Decision

Recommended baseline:

```text
baseline_id = {recommended.baseline_id}
run = {recommended.run_id}
baseline val_bpb = {recommended.baseline_val_bpb:.6f}
q* = {RECOMMENDED_QSTAR:.3f}
```

Winning categories:

| category | best trial | best val_bpb | improvement |
| --- | --- | ---: | ---: |
"""
    for category, (val, improvement, trial_id) in sorted(recommended.category_bests.items()):
        if improvement >= MIN_DELTA:
            md += f"| {category} | `{trial_id}` | {val:.6f} | {improvement:.6f} |\n"

    md += f"""
Negative / near-negative controls:

| trial | category | val_bpb | delta vs baseline |
| --- | --- | ---: | ---: |
"""
    for category, (val, improvement, trial_id) in sorted(recommended.category_bests.items()):
        if improvement < MIN_DELTA:
            md += f"| `{trial_id}` | {category} | {val:.6f} | {improvement:.6f} |\n"

    md += f"""
## Candidate Comparison

| baseline | screen | steps | baseline val_bpb | raw wins | winning categories | q3 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
{chr(10).join(table_lines)}

## Why Not 585 Steps

The 585-step screens created broad headroom, but almost every reasonable edit won.
That is useful for debugging, but weak for confirmatory architecture claims. At 1170 steps, the task remains learnable while retaining more negative controls.

## Next Step

Run a small agentic pilot on `width30_lr_low` before the full 2x2:

```text
fixed-step evaluator
AUTOSEARCH_MAX_STEPS = 1170
serialized evaluator
q* = 0.824
separate agent_deliberation_wall_time and evaluator_wall_time
true independent replicates
```

## Artifacts

- Summary table: `results/tables/baseline_summary.csv`
- Trial table: `results/tables/trial_results.csv`
- Machine-readable summary: `results/tables/pass_05_summary.json`
- Source calibration reports remain under `runs/baseline_*`.
"""
    (PASS_DIR / "README.md").write_text(md)
    (RESULTS_DIR / "pass_05_summary.md").write_text(md.replace("results/figures/", "figures/").replace("results/tables/", "tables/"))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    records, trials = load_rows()
    write_tables(records, trials)
    plot_screen_overview(records)
    plot_gate_scatter(records)
    plot_category_heatmap(records)
    plot_recommended_detail(records, trials)
    write_markdown(records)
    print(f"Wrote {PASS_DIR / 'README.md'}")
    print(f"Wrote {RESULTS_DIR / 'pass_05_summary.md'}")
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
