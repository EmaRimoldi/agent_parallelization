#!/usr/bin/env python3
"""Render lightweight SVG figures for the repository README.

The figures are derived from the stable Phase 02 calibration artifacts:
- workflow/artifacts/calibration_analysis.json
- workflow/artifacts/calibration_analysis_current.json
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "pass_02_workflow_calibration"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def wrap_svg(title: str, subtitle: str, body: str, width: int = 1000, height: int = 420) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title>{title}</title>
  <desc>{subtitle}</desc>
  <rect width="{width}" height="{height}" fill="#f8fafc"/>
  <rect x="16" y="16" width="{width-32}" height="{height-32}" rx="20" fill="#ffffff" stroke="#dbe4ee"/>
  <text x="40" y="56" font-family="Arial, Helvetica, sans-serif" font-size="26" font-weight="700" fill="#0f172a">{title}</text>
  <text x="40" y="84" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#475569">{subtitle}</text>
  {body}
</svg>
"""


def bar(value: float, max_value: float, width: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, (value / max_value) * width)


def render_quality_figure(best_d00: float, best_d10: float, mean_d00: float, mean_d10: float) -> str:
    max_value = max(best_d00, best_d10, mean_d00, mean_d10)
    chart_width = 320
    origin_x = 70
    body = []
    colors = {"d00": "#0f766e", "d10": "#b45309"}

    def panel(x0: int, title: str, left_label: str, left_value: float, right_label: str, right_value: float) -> None:
        body.append(f'<text x="{x0}" y="130" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#0f172a">{title}</text>')
        rows = [
            (left_label, left_value, colors["d00"], 170),
            (right_label, right_value, colors["d10"], 245),
        ]
        for label, value, color, y in rows:
            bw = bar(value, max_value, chart_width)
            body.append(f'<text x="{x0}" y="{y-18}" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#334155">{label}</text>')
            body.append(f'<rect x="{x0}" y="{y}" width="{chart_width}" height="28" rx="8" fill="#e2e8f0"/>')
            body.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="28" rx="8" fill="{color}"/>')
            body.append(f'<text x="{x0 + chart_width + 16}" y="{y+19}" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#0f172a">{fmt(value, 3)}</text>')

    panel(origin_x, "Best-of-Rep Mean (lower is better)", "d00", best_d00, "d10", best_d10)
    panel(origin_x + 430, "All-Runs Mean (lower is better)", "d00", mean_d00, "d10", mean_d10)
    body.append('<text x="70" y="360" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#475569">Interpretation: the descriptive best-of-rep view and the stricter all-runs view both fail to show a d10 quality advantage.</text>')
    return wrap_svg(
        "Phase 02 Quality Snapshot",
        "Deterministic calibration summary for d00 (single/no memory) vs d10 (single/external memory).",
        "".join(body),
    )


def render_behavior_figure(runs_d00: int, runs_d10: int, per_rep_d00: float, per_rep_d10: float, cv_d00: float, cv_d10: float) -> str:
    max_runs = max(runs_d00, runs_d10)
    max_per_rep = max(per_rep_d00, per_rep_d10)
    max_cv = max(cv_d00, cv_d10)
    body = []
    colors = {"d00": "#1d4ed8", "d10": "#7c3aed"}

    def metric_block(x0: int, title: str, d00_value: float, d10_value: float, max_value: float, digits: int = 1) -> None:
        body.append(f'<text x="{x0}" y="130" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#0f172a">{title}</text>')
        for idx, (label, value, color) in enumerate((("d00", d00_value, colors["d00"]), ("d10", d10_value, colors["d10"]))):
            y = 175 + idx * 74
            bw = bar(value, max_value, 220)
            body.append(f'<text x="{x0}" y="{y-18}" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#334155">{label}</text>')
            body.append(f'<rect x="{x0}" y="{y}" width="220" height="28" rx="8" fill="#e2e8f0"/>')
            body.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="28" rx="8" fill="{color}"/>')
            body.append(f'<text x="{x0 + 238}" y="{y+19}" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#0f172a">{fmt(value, digits)}</text>')

    metric_block(70, "Total Training Runs", float(runs_d00), float(runs_d10), float(max_runs), digits=0)
    metric_block(390, "Mean Runs / Rep", per_rep_d00, per_rep_d10, max_per_rep, digits=1)
    metric_block(690, "Wall-Clock CV", cv_d00, cv_d10, max_cv, digits=3)
    body.append('<text x="70" y="360" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#475569">Interpretation: d10 iterates faster and more consistently, but this throughput gain did not translate into a quality gain in calibration.</text>')
    return wrap_svg(
        "Phase 02 Search Behavior",
        "Throughput and cost-variance snapshot from the completed d00/d10 deterministic calibration.",
        "".join(body),
    )


def render_gate_figure(best_d: float, all_runs_d: float, accepted_repeat_d00: int, accepted_repeat_d10: int) -> str:
    body = []
    zero_x = 500
    scale = 220
    body.append('<text x="70" y="128" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#0f172a">Decision Split</text>')
    body.append('<text x="70" y="154" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#475569">Positive values favor d00 here because d10 was worse on quality.</text>')
    body.append(f'<line x1="{zero_x}" y1="180" x2="{zero_x}" y2="310" stroke="#94a3b8" stroke-width="2"/>')
    for y, label, value, color in (
        (215, "Best-of-rep Cohen's d", best_d, "#b91c1c"),
        (270, "All-runs Cohen's d", all_runs_d, "#0f766e"),
    ):
        x = zero_x if value >= 0 else zero_x + value * scale
        width = abs(value) * scale
        body.append(f'<text x="70" y="{y+6}" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#334155">{label}</text>')
        body.append(f'<rect x="{x:.1f}" y="{y-14}" width="{width:.1f}" height="28" rx="8" fill="{color}"/>')
        body.append(f'<text x="{zero_x + width + 14 if value >= 0 else x - 64:.1f}" y="{y+6}" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#0f172a">{fmt(value, 3)}</text>')
    body.append('<text x="70" y="350" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#0f172a">Accepted Modes With ≥2 Accepted Edits</text>')
    for idx, (label, value, color) in enumerate((("d00", accepted_repeat_d00, "#1d4ed8"), ("d10", accepted_repeat_d10, "#7c3aed"))):
        x0 = 70 + idx * 180
        body.append(f'<rect x="{x0}" y="370" width="120" height="28" rx="8" fill="#e2e8f0"/>')
        body.append(f'<rect x="{x0}" y="370" width="{20 + value * 40}" height="28" rx="8" fill="{color}"/>')
        body.append(f'<text x="{x0+8}" y="389" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#ffffff">{label}</text>')
        body.append(f'<text x="{x0+138}" y="389" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#0f172a">{value}</text>')
    return wrap_svg(
        "Phase 02 Gate Readout",
        "The descriptive best-of-rep view suggested a medium difference, but the stricter all-runs + accepted-mode gate remained effectively negative.",
        "".join(body),
    )


def main() -> None:
    phase02 = load_json(REPO_ROOT / "workflow" / "artifacts" / "calibration_analysis.json")
    rigorous = load_json(REPO_ROOT / "workflow" / "artifacts" / "calibration_analysis_current.json")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    quality_svg = render_quality_figure(
        best_d00=phase02["cells"]["d00"]["best_per_rep_mean"],
        best_d10=phase02["cells"]["d10"]["best_per_rep_mean"],
        mean_d00=rigorous["d00"]["mean_val_bpb"],
        mean_d10=rigorous["d10"]["mean_val_bpb"],
    )
    behavior_svg = render_behavior_figure(
        runs_d00=phase02["cells"]["d00"]["n_runs"],
        runs_d10=phase02["cells"]["d10"]["n_runs"],
        per_rep_d00=phase02["comparisons"]["iteration_throughput"]["d00_mean_runs"],
        per_rep_d10=phase02["comparisons"]["iteration_throughput"]["d10_mean_runs"],
        cv_d00=phase02["cells"]["d00"]["wall_clock_cv"],
        cv_d10=phase02["cells"]["d10"]["wall_clock_cv"],
    )
    gate_svg = render_gate_figure(
        best_d=phase02["comparisons"]["best_of_rep"]["cohens_d"],
        all_runs_d=rigorous["cohens_d"],
        accepted_repeat_d00=rigorous["d00"]["modes_with_2plus_accepted"],
        accepted_repeat_d10=rigorous["d10"]["modes_with_2plus_accepted"],
    )

    (FIGURES_DIR / "phase02_quality.svg").write_text(quality_svg)
    (FIGURES_DIR / "phase02_behavior.svg").write_text(behavior_svg)
    (FIGURES_DIR / "phase02_gate.svg").write_text(gate_svg)
    print("Wrote Phase 02 README figures to", FIGURES_DIR)


if __name__ == "__main__":
    main()
