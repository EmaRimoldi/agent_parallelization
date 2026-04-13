#!/usr/bin/env python3
"""Render archive overview figures for the repository README.

This script produces stable PNG assets that summarize the two main
experimental passes plus the pre-pilot smoke runs. It only reads committed
artifacts and does not touch active experiment directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "figures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[str] = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


FONT_H1 = load_font(44, bold=True)
FONT_H2 = load_font(28, bold=True)
FONT_H3 = load_font(22, bold=True)
FONT_BODY = load_font(19)
FONT_SMALL = load_font(16)


COLORS = {
    "bg": "#f7f3eb",
    "ink": "#102022",
    "muted": "#55656a",
    "line": "#9aa8ab",
    "pilot": "#c86f31",
    "followup": "#2d8f86",
    "workflow": "#3e6bb2",
    "smoke": "#d4b04c",
    "bundle": "#4f7a43",
    "panel": "#fffdf8",
}


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    return box[2] - box[0], box[3] - box[1]


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = COLORS["ink"],
    spacing: int = 4,
) -> None:
    width, height = text_size(draw, text, font)
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0 - width) / 2
    y = y0 + (y1 - y0 - height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center")


def box(
    draw: ImageDraw.ImageDraw,
    coords: tuple[int, int, int, int],
    title: str,
    body: str,
    color: str,
) -> None:
    x0, y0, x1, y1 = coords
    draw.rounded_rectangle(coords, radius=22, fill=COLORS["panel"], outline=color, width=4)
    draw.rounded_rectangle((x0, y0, x1, y0 + 56), radius=22, fill=color, outline=color, width=4)
    draw.rectangle((x0, y0 + 28, x1, y0 + 56), fill=color)
    centered_text(draw, (x0 + 18, y0 + 8, x1 - 18, y0 + 50), title, FONT_H3, fill="white")
    centered_text(draw, (x0 + 20, y0 + 66, x1 - 20, y1 - 16), body, FONT_BODY, fill=COLORS["ink"])


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = COLORS["line"]) -> None:
    draw.line((start, end), fill=fill, width=6)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) > abs(ey - sy):
        sign = 1 if ex > sx else -1
        draw.polygon([(ex, ey), (ex - 18 * sign, ey - 10), (ex - 18 * sign, ey + 10)], fill=fill)
    else:
        sign = 1 if ey > sy else -1
        draw.polygon([(ex, ey), (ex - 10, ey - 18 * sign), (ex + 10, ey - 18 * sign)], fill=fill)


def draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    labels: Iterable[str],
    values: Iterable[float],
    color: str,
    title: str,
    y_label: str | None = None,
    max_value: float | None = None,
    lower_is_better: bool = False,
) -> None:
    x0, y0, x1, y1 = area
    draw.rounded_rectangle(area, radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.text((x0 + 18, y0 + 14), title, font=FONT_H3, fill=COLORS["ink"])
    chart_left = x0 + 70
    chart_top = y0 + 62
    chart_bottom = y1 - 68
    chart_right = x1 - 24
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=COLORS["line"], width=2)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=COLORS["line"], width=2)
    labels = list(labels)
    values = list(values)
    if not values:
        return
    vmax = max_value if max_value is not None else max(values) * 1.15
    vmax = max(vmax, 1e-6)
    bar_area = chart_right - chart_left
    step = bar_area / max(len(values), 1)
    bar_width = min(82, step * 0.6)
    for i, (label, value) in enumerate(zip(labels, values)):
        cx = chart_left + step * (i + 0.5)
        left = int(cx - bar_width / 2)
        right = int(cx + bar_width / 2)
        height = 0 if value <= 0 else (value / vmax) * (chart_bottom - chart_top - 10)
        top = int(chart_bottom - height)
        draw.rounded_rectangle((left, top, right, chart_bottom), radius=8, fill=color, outline=color)
        value_text = f"{value:.2f}" if value < 100 else f"{int(value)}"
        vw, vh = text_size(draw, value_text, FONT_SMALL)
        draw.text((cx - vw / 2, top - vh - 6), value_text, font=FONT_SMALL, fill=COLORS["ink"])
        lw, _ = text_size(draw, label, FONT_SMALL)
        draw.text((cx - lw / 2, chart_bottom + 12), label, font=FONT_SMALL, fill=COLORS["ink"])
    if y_label:
        draw.text((x0 + 16, y0 + 48), y_label, font=FONT_SMALL, fill=COLORS["muted"])
    if lower_is_better:
        note = "Lower is better"
        nw, nh = text_size(draw, note, FONT_SMALL)
        draw.text((chart_right - nw, y0 + 16), note, font=FONT_SMALL, fill=COLORS["muted"])


def draw_error_bars(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    labels: Iterable[str],
    means: Iterable[float],
    lows: Iterable[float],
    highs: Iterable[float],
    color: str,
    title: str,
) -> None:
    x0, y0, x1, y1 = area
    draw.rounded_rectangle(area, radius=18, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.text((x0 + 18, y0 + 14), title, font=FONT_H3, fill=COLORS["ink"])
    labels = list(labels)
    means = list(means)
    lows = list(lows)
    highs = list(highs)
    ymin = min(lows)
    ymax = max(highs)
    pad = max((ymax - ymin) * 0.15, 0.02)
    ymin -= pad
    ymax += pad
    chart_left = x0 + 72
    chart_top = y0 + 60
    chart_bottom = y1 - 70
    chart_right = x1 - 24
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=COLORS["line"], width=2)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=COLORS["line"], width=2)
    step = (chart_right - chart_left) / max(len(labels), 1)
    bar_width = min(82, step * 0.6)

    def scale(v: float) -> float:
        if ymax <= ymin:
            return chart_bottom
        return chart_bottom - (v - ymin) / (ymax - ymin) * (chart_bottom - chart_top)

    for i, label in enumerate(labels):
        cx = chart_left + step * (i + 0.5)
        mean_y = scale(means[i])
        low_y = scale(lows[i])
        high_y = scale(highs[i])
        left = int(cx - bar_width / 2)
        right = int(cx + bar_width / 2)
        draw.rounded_rectangle((left, mean_y, right, chart_bottom), radius=8, fill=color, outline=color)
        draw.line((cx, high_y, cx, low_y), fill=COLORS["ink"], width=3)
        draw.line((cx - 10, high_y, cx + 10, high_y), fill=COLORS["ink"], width=3)
        draw.line((cx - 10, low_y, cx + 10, low_y), fill=COLORS["ink"], width=3)
        value_text = f"{means[i]:.3f}"
        vw, vh = text_size(draw, value_text, FONT_SMALL)
        draw.text((cx - vw / 2, mean_y - vh - 8), value_text, font=FONT_SMALL, fill=COLORS["ink"])
        lw, _ = text_size(draw, label, FONT_SMALL)
        draw.text((cx - lw / 2, chart_bottom + 12), label, font=FONT_SMALL, fill=COLORS["ink"])
    note = "Repeated incumbent mean ± 95% CI"
    draw.text((chart_left, y0 + 16), note, font=FONT_SMALL, fill=COLORS["muted"])


def render_flow(smoke_count: int) -> None:
    image = Image.new("RGB", (1800, 1120), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((70, 44), "Experiment Archive Flow", font=FONT_H1, fill=COLORS["ink"])
    draw.text(
        (70, 104),
        "Repository-level summary of the recorded BP 2x2 experiment passes and follow-up analyses.",
        font=FONT_BODY,
        fill=COLORS["muted"],
    )

    smoke_box = (90, 190, 550, 400)
    pilot_box = (675, 190, 1135, 400)
    follow_box = (1250, 120, 1710, 365)
    workflow_box = (1250, 430, 1710, 695)
    decision_box = (675, 500, 1135, 760)
    bundle_box = (675, 860, 1135, 1040)

    box(
        draw,
        smoke_box,
        "Pre-pilot smoke",
        f"{smoke_count} short bring-up runs on Apr 11\n\n"
        "5 x baseline single-agent smoke\n"
        "1 x d10 memory smoke\n"
        "1 x d01 parallel smoke\n"
        "1 x d11 shared-memory smoke\n\n"
        "Path: runs/experiment_exp_20260411_*",
        COLORS["smoke"],
    )
    box(
        draw,
        pilot_box,
        "Pass 1: Original 2x2 pilot",
        "12 runs = 4 cells x 3 reps\n"
        "30 min per rep on CIFAR-10 CPU substrate\n\n"
        "Raw runs: runs/experiment_pilot_*\n"
        "Aggregates: results/ and archives/pass_02_theory_validation_bundle_20260412/artifacts/",
        COLORS["pilot"],
    )
    box(
        draw,
        follow_box,
        "Pass 2A: Theory follow-ups",
        "Noise assay\n"
        "Repeated incumbent reevaluation\n"
        "Cost variance / Jensen analysis\n"
        "Context-sweep feasibility\n"
        "Protocol smoke and compliance audit\n\n"
        "Path: archives/pass_02_theory_validation_bundle_20260412/experiments/",
        COLORS["followup"],
    )
    box(
        draw,
        workflow_box,
        "Pass 2B: Workflow calibration",
        "Deterministic evaluation fix\n"
        "d00 vs d10 calibration: 5 reps each\n"
        "Phase 02 analysis + decision gate\n"
        "Exploratory d01/d11 override launch\n\n"
        "Path: workflow/ + runs/experiment_calibration_*",
        COLORS["workflow"],
    )
    box(
        draw,
        decision_box,
        "Current reading",
        "Pilot pass suggested interesting cost-side effects,\n"
        "but the follow-up and calibration passes weakened the\n"
        "strong quality claim.\n\n"
        "Best current status:\n"
        "promising but not yet rigorous\n\n"
        "Strict gate currently favors structured_search.",
        COLORS["muted"],
    )
    box(
        draw,
        bundle_box,
        "Reviewer bundle",
        "Top-level README archive\n"
        "archives/pass_02_theory_validation_bundle_20260412/\n"
        "workflow/artifacts/\n"
        "archives/pass_02_theory_validation_bundle_20260412/theory/\n"
        "autoresearch_bp_revised.pdf\n\n"
        "This is the stable handoff package.",
        COLORS["bundle"],
    )

    arrow(draw, (550, 295), (675, 295))
    arrow(draw, (1135, 250), (1250, 230))
    arrow(draw, (1135, 340), (1250, 510))
    arrow(draw, (905, 400), (905, 500))
    arrow(draw, (1480, 695), (1040, 695))
    arrow(draw, (905, 760), (905, 860))

    image.save(OUT_DIR / "experiment_archive_flow.png")


def render_summary(
    smoke_count: int,
    pilot: dict,
    noise: dict,
    repmeans: dict,
    calibration: dict,
    calibration_runs: list[dict],
) -> None:
    image = Image.new("RGB", (1900, 1200), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "Experiment Archive Summary", font=FONT_H1, fill=COLORS["ink"])
    draw.text(
        (70, 98),
        "Counts and headline metrics from the tracked smoke, pilot, follow-up, and calibration passes.",
        font=FONT_BODY,
        fill=COLORS["muted"],
    )

    pilot_reps = sum(len(pilot["raw_metrics"][cell]["experiments"]) for cell in pilot["raw_metrics"])
    pilot_training_runs = sum(
        exp["total_training_runs"]
        for cell in pilot["raw_metrics"].values()
        for exp in cell["experiments"]
    )
    calibration_training_runs = sum(item.get("n_training_runs", 0) for item in calibration_runs)
    reeval_total = sum(cell["n"] for cell in repmeans["cells"].values())
    noise_total = len(noise["baseline"]["runs"]) + len(noise["best_d10"]["runs"])

    draw_bar_chart(
        draw,
        (70, 170, 660, 575),
        ["Smoke", "Pilot", "Calib", "Noise", "Reeval"],
        [smoke_count, pilot_reps, len(calibration_runs), noise_total, reeval_total],
        COLORS["workflow"],
        "Recorded experiment counts",
        y_label="Distinct runs / evaluations",
    )

    cells = ["d00", "d10", "d01", "d11"]
    draw_error_bars(
        draw,
        (720, 170, 1830, 575),
        cells,
        [repmeans["cells"][cell]["mean_val_bpb"] for cell in cells],
        [repmeans["cells"][cell]["ci95_low"] for cell in cells],
        [repmeans["cells"][cell]["ci95_high"] for cell in cells],
        COLORS["pilot"],
        "Repeated incumbent reevaluation across all four cells",
    )

    draw_bar_chart(
        draw,
        (70, 635, 900, 1110),
        ["Pilot train", "Calib train"],
        [pilot_training_runs, calibration_training_runs],
        COLORS["followup"],
        "Training runs captured in major passes",
        y_label="Logged training runs",
    )

    combo_labels = ["Noise base", "Noise d10", "Cal d00", "Cal d10"]
    combo_values = [
        noise["baseline"]["mean_val_bpb"],
        noise["best_d10"]["mean_val_bpb"],
        calibration["d00"]["mean_val_bpb"],
        calibration["d10"]["mean_val_bpb"],
    ]
    draw_bar_chart(
        draw,
        (960, 635, 1470, 1110),
        combo_labels,
        combo_values,
        COLORS["bundle"],
        "Quality checks that changed the interpretation",
        y_label="Mean val_bpb",
        max_value=max(combo_values) * 1.2,
        lower_is_better=True,
    )

    draw_bar_chart(
        draw,
        (1520, 635, 1830, 1110),
        ["d00", "d10"],
        [
            calibration["d00"]["jensen_gap_wall"],
            calibration["d10"]["jensen_gap_wall"],
        ],
        COLORS["smoke"],
        "Wall Jensen gap",
        y_label="Empirical R_alpha",
    )

    notes = [
        "Pilot best repeat means after reevaluation:",
        f"d10 {repmeans['cells']['d10']['mean_val_bpb']:.3f}, d00 {repmeans['cells']['d00']['mean_val_bpb']:.3f},",
        f"d11 {repmeans['cells']['d11']['mean_val_bpb']:.3f}, d01 {repmeans['cells']['d01']['mean_val_bpb']:.3f}.",
        "Calibration all-runs view remains close to null:",
        f"d00 {calibration['d00']['mean_val_bpb']:.3f} vs d10 {calibration['d10']['mean_val_bpb']:.3f},",
        f"Cohen's d {calibration['cohens_d']:.3f} ({calibration['effect_interpretation']}).",
    ]
    y = 1140
    for line in notes:
        draw.text((70, y), line, font=FONT_SMALL, fill=COLORS["muted"])
        y += 20

    image.save(OUT_DIR / "experiment_archive_summary.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke_count = len(list((ROOT / "runs").glob("experiment_exp_20260411_*")))
    pilot = load_json(ROOT / "results" / "pilot_raw_data.json")
    noise = load_json(ROOT / "archives/pass_02_theory_validation_bundle_20260412" / "experiments" / "noise_assay" / "noise_summary.json")
    repmeans = load_json(ROOT / "archives/pass_02_theory_validation_bundle_20260412" / "experiments" / "followup_01" / "replicated_means_summary.json")
    calibration = load_json(ROOT / "workflow" / "artifacts" / "calibration_analysis_current.json")
    calibration_runs = load_json(ROOT / "workflow" / "artifacts" / "calibration_runs.json")
    render_flow(smoke_count)
    render_summary(smoke_count, pilot, noise, repmeans, calibration, calibration_runs)
    print(f"Wrote {OUT_DIR / 'experiment_archive_flow.png'}")
    print(f"Wrote {OUT_DIR / 'experiment_archive_summary.png'}")


if __name__ == "__main__":
    main()
