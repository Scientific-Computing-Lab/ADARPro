#!/usr/bin/env python3
"""Build the PNAS main and SI figures from authoritative retained artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "paper/figure_assets/selected"
CONTEXTS = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
CONTEXT_LABELS = {
    "Artery": "Artery",
    "Brain": "Brain",
    "Liver": "Liver",
    "MuscleSkeletal": "Skeletal muscle",
    "Combined": "Combined",
}
COLORS = {
    "navy": "#143642",
    "blue": "#2166AC",
    "sky": "#67A9CF",
    "teal": "#138D90",
    "green": "#2A9D67",
    "amber": "#D99000",
    "red": "#C64242",
    "purple": "#7562A8",
    "gray": "#68737D",
    "light": "#E8EEF1",
    "pale_blue": "#E7F1F7",
    "pale_green": "#E9F5EE",
    "pale_amber": "#FFF4D9",
    "pale_red": "#FAE9E8",
    "ink": "#172126",
}


def configure() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "ADARPro-2026",
            "savefig.dpi": 450,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_asset(
    axis: plt.Axes,
    name: str,
    bounds: tuple[float, float, float, float],
    alpha: float = 1.0,
    zorder: int = 1,
    crop: tuple[float, float, float, float] | None = None,
) -> plt.Axes:
    """Place a curated transparent illustration inside an otherwise vector panel."""
    path = ASSET_DIR / f"{name}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing curated figure asset {path}; run generate_figure_assets.py"
        )
    image = mpimg.imread(path)
    if crop is not None:
        x0, y0, x1, y1 = crop
        height, width = image.shape[:2]
        image = image[
            round(y0 * height) : round(y1 * height),
            round(x0 * width) : round(x1 * width),
        ]
    inset = axis.inset_axes(bounds, transform=axis.transAxes, zorder=zorder)
    inset.imshow(image, interpolation="lanczos", alpha=alpha)
    inset.set_axis_off()
    return inset


def section_rule(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    title: str,
    color: str,
    detail: str | None = None,
) -> None:
    """Draw a compact editorial label with a colored hierarchy rule."""
    axis.plot([x, x + width], [y, y], color=color, lw=2.1, solid_capstyle="round")
    axis.text(
        x,
        y - 0.025,
        title,
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=COLORS["ink"],
    )
    if detail:
        axis.text(
            x,
            y - 0.075,
            detail,
            ha="left",
            va="top",
            fontsize=6.2,
            color=COLORS["gray"],
            linespacing=1.25,
        )


def panel_label(axis: plt.Axes, value: str) -> None:
    axis.text(
        -0.07,
        1.04,
        value,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def clean_axis(axis: plt.Axes, grid_axis: str | None = "x") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        axis.grid(axis=grid_axis, color="#D8DEE2", linewidth=0.55, zorder=0)
    axis.set_axisbelow(True)


def draw_rna_duplex(
    axis: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    height: float,
    color: str,
    *,
    target_color: str | None = None,
    partner_color: str | None = None,
    n_pairs: int = 11,
    linewidth: float = 1.0,
) -> tuple[float, float]:
    """Draw a compact, explicitly paired RNA duplex in axis coordinates."""
    x = np.linspace(x0, x1, n_pairs)
    phase = np.linspace(0, 2 * np.pi, n_pairs)
    top = y + height / 2 + 0.06 * height * np.sin(phase)
    bottom = y - height / 2 + 0.06 * height * np.sin(phase)
    axis.plot(x, top, color=color, lw=linewidth, solid_capstyle="round", zorder=3)
    axis.plot(x, bottom, color=color, lw=linewidth, solid_capstyle="round", zorder=3)
    target_index = n_pairs // 2
    for index, (xpos, y_top, y_bottom) in enumerate(zip(x, top, bottom)):
        axis.plot(
            [xpos, xpos],
            [y_bottom, y_top],
            color="#9AAAB2",
            lw=max(0.45, linewidth * 0.55),
            zorder=2,
        )
        radius = 0.007 if index == target_index else 0.0046
        axis.add_patch(
            Circle(
                (xpos, y_top),
                radius,
                facecolor=(
                    target_color
                    if index == target_index and target_color is not None
                    else "white"
                ),
                edgecolor=color,
                lw=0.65,
                zorder=4,
            )
        )
        axis.add_patch(
            Circle(
                (xpos, y_bottom),
                radius,
                facecolor=(
                    partner_color
                    if index == target_index and partner_color is not None
                    else "white"
                ),
                edgecolor=color,
                lw=0.65,
                zorder=4,
            )
        )
    return x[target_index], top[target_index]


def save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        if suffix == "pdf":
            fixed_time = dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc)
            metadata: dict[str, object] = {
                "Creator": "ADARPro figure pipeline",
                "Producer": "Matplotlib",
                "CreationDate": fixed_time,
                "ModDate": fixed_time,
            }
        elif suffix == "svg":
            metadata = {"Creator": "ADARPro figure pipeline", "Date": "2026-09-10"}
        else:
            metadata = {"Software": "ADARPro figure pipeline"}
        figure.savefig(
            output / f"{stem}.{suffix}", bbox_inches="tight", metadata=metadata
        )
    plt.close(figure)


def copy_source(frame: pd.DataFrame, source: Path, name: str) -> Path:
    source.mkdir(parents=True, exist_ok=True)
    path = source / name
    frame.to_csv(path, index=False)
    return path


def read_ci(path: str, metric: str, **selectors: str) -> pd.Series:
    frame = pd.read_csv(ROOT / path)
    subset = frame[frame["metric"] == metric]
    for column, value in selectors.items():
        subset = subset[subset[column] == value]
    if len(subset) != 1:
        raise RuntimeError(
            f"Expected one row in {path}: {metric}, {selectors}; got {len(subset)}"
        )
    return subset.iloc[0]


def rounded_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 8,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["ink"],
    )


def benchmark_split_table(source: Path) -> pd.DataFrame:
    records = []
    for split, filename in (
        ("Train", "train"),
        ("Validation", "valid"),
        ("Test", "test"),
    ):
        path = ROOT / f"data/human/Combined/{filename}.metadata.csv"
        frame = pd.read_csv(path, usecols=["pair_id"])
        records.append(
            {
                "split": split,
                "pair_count": int(frame["pair_id"].nunique()),
                "source": path.relative_to(ROOT).as_posix(),
            }
        )
    result = pd.DataFrame(records)
    copy_source(result, source, "figure1_benchmark_split.csv")
    return result


def figure1(output: Path, source: Path) -> None:
    figure = plt.figure(figsize=(7.0, 5.65))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.08, 1],
        height_ratios=[1.35, 0.86],
        hspace=0.24,
        wspace=0.22,
    )
    ax_a = figure.add_subplot(grid[0, :])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 1])
    for axis in (ax_a, ax_b, ax_c):
        axis.set_axis_off()

    panel_label(ax_a, "A")
    ax_a.set_title(
        "A layered map of recoverable information",
        loc="left",
        fontweight="bold",
    )
    ax_a.axvspan(0.00, 0.325, color="#F4F8FA", zorder=0)
    ax_a.axvspan(0.337, 0.662, color="#F2F8F5", zorder=0)
    ax_a.axvspan(0.674, 1.00, color="#FBF6EA", zorder=0)
    add_asset(
        ax_a,
        "rna_substrate_editorial",
        (0.020, 0.355, 0.285, 0.345),
        zorder=2,
    )
    target_asset = add_asset(
        ax_a,
        "target_pair_context",
        (0.365, 0.395, 0.280, 0.255),
        zorder=2,
    )
    target_asset.add_patch(
        plt.Rectangle(
            (0.31, 0.05),
            0.38,
            0.90,
            transform=target_asset.transAxes,
            facecolor=COLORS["pale_green"],
            edgecolor=COLORS["green"],
            lw=0.7,
            alpha=0.20,
            zorder=0,
        )
    )
    add_asset(
        ax_a,
        "matched_cell_state",
        (0.725, 0.320, 0.235, 0.405),
        alpha=0.92,
        zorder=2,
    )

    section_rule(
        ax_a,
        0.025,
        0.86,
        0.26,
        "INTRINSIC SUBSTRATE",
        COLORS["blue"],
        "sequence · base pairing · predicted fold",
    )
    section_rule(
        ax_a,
        0.355,
        0.86,
        0.27,
        "TARGET SELECTIVITY",
        COLORS["green"],
        "focal A · predicted partner · aligned window",
    )
    section_rule(
        ax_a,
        0.695,
        0.86,
        0.27,
        "CELLULAR / TRANSCRIPT CONTEXT",
        COLORS["amber"],
        "transcript state · ADAR access · regulators",
    )
    ax_a.text(
        0.505,
        0.665,
        "focal A",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["amber"],
    )
    ax_a.add_patch(
        FancyArrowPatch(
            (0.505, 0.638),
            (0.505, 0.555),
            arrowstyle="-|>",
            mutation_scale=6.5,
            color=COLORS["amber"],
            lw=0.75,
            zorder=4,
        )
    )
    ax_a.text(
        0.505,
        0.400,
        "predicted partner",
        ha="center",
        va="center",
        fontsize=6.0,
        color=COLORS["teal"],
    )
    ax_a.add_patch(
        FancyArrowPatch(
            (0.505, 0.425),
            (0.505, 0.488),
            arrowstyle="-|>",
            mutation_scale=6.5,
            color=COLORS["teal"],
            lw=0.75,
            zorder=4,
        )
    )
    ax_a.plot(
        [0.405, 0.405, 0.605, 0.605],
        [0.376, 0.363, 0.363, 0.376],
        color=COLORS["green"],
        lw=0.8,
    )
    ax_a.text(
        0.505,
        0.340,
        "aligned focal window",
        ha="center",
        va="center",
        fontsize=6.1,
        color=COLORS["gray"],
    )
    ax_a.text(
        0.165,
        0.235,
        "strong recoverable signal",
        ha="center",
        fontsize=6.2,
        color=COLORS["blue"],
        fontweight="bold",
    )
    ax_a.text(
        0.505,
        0.235,
        "target centering adds information",
        ha="center",
        fontsize=6.2,
        color=COLORS["green"],
        fontweight="bold",
    )
    ax_a.text(
        0.835,
        0.235,
        "tested bulk proxies add little residual signal",
        ha="center",
        fontsize=6.2,
        color=COLORS["amber"],
        fontweight="bold",
    )
    for xpos, endpoint, color in [
        (0.165, 0.400, COLORS["blue"]),
        (0.505, 0.500, COLORS["green"]),
        (0.835, 0.600, COLORS["amber"]),
    ]:
        ax_a.add_patch(
            FancyArrowPatch(
                (xpos, 0.205),
                (endpoint, 0.112),
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0",
                mutation_scale=8.5,
                color=color,
                lw=0.95,
                alpha=0.92,
                zorder=2,
            )
        )
    ax_a.add_patch(
        plt.Rectangle(
            (0.31, 0.038),
            0.38,
            0.066,
            facecolor="#EEF3F5",
            edgecolor="none",
            zorder=3,
        )
    )
    ax_a.plot(
        [0.31, 0.69],
        [0.104, 0.104],
        color=COLORS["navy"],
        lw=1.05,
        solid_capstyle="butt",
        zorder=4,
    )
    ax_a.text(
        0.50,
        0.070,
        "OBSERVED ENDOGENOUS EDITING",
        ha="center",
        va="center",
        color=COLORS["navy"],
        fontsize=7.0,
        fontweight="bold",
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)

    panel_label(ax_b, "B")
    ax_b.set_title(
        "Controlled questions, not model parts", loc="left", fontweight="bold"
    )
    categories = [
        (
            "BIOLOGICAL CONTEXT",
            "Do measured cellular annotations add residual information?",
            COLORS["amber"],
        ),
        (
            "SUPERVISION",
            "Does respecting duplex organization alter what is learned?",
            COLORS["green"],
        ),
        (
            "REPRESENTATION",
            "What is lost between global duplex and focal-site views?",
            COLORS["blue"],
        ),
        (
            "GENERALIZATION",
            "Which determinants transfer, and where do they fail?",
            COLORS["purple"],
        ),
    ]
    for index, (title, detail, color) in enumerate(categories):
        y = 0.80 - 0.22 * index
        ax_b.add_patch(Circle((0.055, y + 0.025), 0.024, color=color, alpha=0.95))
        ax_b.text(
            0.055,
            y + 0.025,
            str(index + 1),
            color="white",
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="bold",
        )
        ax_b.text(
            0.105,
            y + 0.055,
            title,
            fontsize=6.4,
            color=color,
            fontweight="bold",
            va="center",
        )
        ax_b.text(
            0.105, y + 0.002, detail, fontsize=6.4, color=COLORS["ink"], va="center"
        )
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)

    panel_label(ax_c, "C")
    ax_c.set_title(
        "Evidence was separated from final validation", loc="left", fontweight="bold"
    )
    split_counts = benchmark_split_table(source).set_index("split")["pair_count"]
    split = {
        "Train": (int(split_counts["Train"]), COLORS["blue"]),
        "Validation": (int(split_counts["Validation"]), COLORS["green"]),
        "Test": (int(split_counts["Test"]), COLORS["red"]),
    }
    total = sum(value[0] for value in split.values())
    ax_c.text(
        0.02,
        0.89,
        "GLOBAL PAIR-LEVEL PARTITION",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=COLORS["navy"],
        fontweight="bold",
    )
    x = 0.02
    split_centers: dict[str, float] = {}
    for label, (count, color) in split.items():
        width = 0.94 * count / total
        split_centers[label] = x + width / 2
        ax_c.add_patch(
            plt.Rectangle(
                (x, 0.72), width, 0.12, facecolor=color, edgecolor="white", lw=0.8
            )
        )
        ax_c.text(
            x + width / 2,
            0.78,
            f"{label}\n{count}",
            ha="center",
            va="center",
            color="white",
            fontsize=5.45 if label == "Validation" else 5.8,
            fontweight="bold",
        )
        x += width
    stages = [
        (
            0.205,
            "1  DEVELOP",
            "pair-disjoint OOF\nfive folds",
            COLORS["blue"],
            COLORS["pale_blue"],
        ),
        (
            0.545,
            "2  FREEZE",
            "code · weights\nthresholds · hashes",
            COLORS["green"],
            COLORS["pale_green"],
        ),
        (
            0.855,
            "3  EVALUATE",
            "canonical test\nno retuning",
            COLORS["red"],
            COLORS["pale_red"],
        ),
    ]
    card_widths = [0.255, 0.235, 0.225]
    for index, ((xpos, title, detail, color, _fill), card_width) in enumerate(
        zip(stages, card_widths)
    ):
        left = xpos - card_width / 2
        ax_c.plot(
            [left, left + card_width],
            [0.490, 0.490],
            color=color,
            lw=1.5,
            solid_capstyle="butt",
            zorder=3,
        )
        ax_c.add_patch(
            FancyArrowPatch(
                (split_centers[["Train", "Validation", "Test"][index]], 0.705),
                (xpos, 0.510),
                arrowstyle="-|>",
                mutation_scale=7.0,
                color=color,
                lw=0.9,
                alpha=0.88,
                zorder=4,
            )
        )
        ax_c.text(
            xpos,
            0.425,
            title,
            ha="center",
            va="center",
            fontsize=6.05,
            color=color,
            fontweight="bold",
        )
        ax_c.text(
            xpos,
            0.300,
            detail,
            ha="center",
            va="center",
            fontsize=5.55,
            color=COLORS["ink"],
            linespacing=1.18,
        )
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    figure.subplots_adjust(left=0.055, right=0.99, top=0.965, bottom=0.045)
    save(figure, output, "figure1_systematic_dissection")


def scope_tables(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recover retained within-context and transfer results for the scope map."""
    within = pd.read_csv(ROOT / "results/table1_within_context.csv")
    baseline = pd.read_csv(ROOT / "results/matrix_baseline_auroc.csv", index_col=0)
    bioaware = pd.read_csv(ROOT / "results/matrix_bioaware_auroc.csv", index_col=0)
    baseline.index.name = "train_context"
    bioaware.index.name = "train_context"

    human = set(CONTEXTS)
    transfer_rows = []
    for train_context in baseline.index:
        for eval_context in baseline.columns:
            if train_context == eval_context:
                group = "Matched context"
            elif train_context in human and eval_context in human:
                group = "Across human endpoints"
            elif train_context not in human and eval_context not in human:
                group = "Across nonhuman datasets"
            else:
                group = "Across human/nonhuman"
            transfer_rows.append(
                {
                    "train_context": train_context,
                    "eval_context": eval_context,
                    "transfer_group": group,
                    "baseline_auroc": baseline.loc[train_context, eval_context],
                    "bioaware_auroc": bioaware.loc[train_context, eval_context],
                    "source": "results/matrix_baseline_auroc.csv; "
                    "results/matrix_bioaware_auroc.csv",
                }
            )
    transfer = pd.DataFrame(transfer_rows)
    diagonal = within[within.train_context == within.eval_context].copy()
    pivot = baseline.reset_index()
    copy_source(diagonal, source, "figure2_within_context_metrics.csv")
    copy_source(transfer, source, "figure2_transfer_groups.csv")
    copy_source(pivot, source, "figure2_baseline_transfer_auroc.csv")
    return diagonal, baseline, transfer


def figure_scope(output: Path, source: Path) -> None:
    """Show how much sequence--structure models capture and where they transfer."""
    within, baseline, transfer = scope_tables(source)
    contexts = [
        "Artery",
        "Brain",
        "Liver",
        "MuscleSkeletal",
        "Combined",
        "Octopus",
        "Ptychodera",
        "Strongylocentrotus",
    ]
    labels = [
        "Artery",
        "Brain",
        "Liver",
        "Skeletal muscle",
        "Combined",
        "Octopus",
        "Ptychodera",
        "Strongylocentrotus",
    ]
    human_n = len(CONTEXTS)
    figure = plt.figure(figsize=(7.0, 6.25))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[0.92, 1.25],
        height_ratios=[1.06, 0.94],
        hspace=0.48,
        wspace=0.42,
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[:, 1])
    ax_c = figure.add_subplot(grid[1, 0])

    context_key = figure.add_axes([0.105, 0.872, 0.84, 0.108])
    context_key.set_axis_off()
    context_key.axvspan(0.00, 0.575, color="#F5F9FB", zorder=0)
    context_key.axvspan(0.59, 1.00, color="#FCF8EE", zorder=0)
    context_key.text(
        0.015,
        0.92,
        "HUMAN ENDPOINTS",
        fontsize=6.8,
        fontweight="bold",
        color=COLORS["blue"],
        va="top",
    )
    add_asset(
        context_key,
        "human_contexts",
        (0.015, 0.30, 0.435, 0.46),
        zorder=2,
    )
    human_names = ["Artery", "Brain", "Liver", "Skeletal muscle"]
    human_x = [0.060, 0.165, 0.270, 0.385]
    for xpos, name in zip(human_x, human_names):
        context_key.text(
            xpos,
            0.19,
            name,
            ha="center",
            va="center",
            fontsize=5.25,
            color=COLORS["ink"],
        )
    context_key.scatter(
        [0.505],
        [0.535],
        s=210,
        facecolor="white",
        edgecolor=COLORS["blue"],
        linewidth=0.9,
        zorder=2,
    )
    context_key.text(
        0.505,
        0.535,
        r"$\Sigma$",
        ha="center",
        va="center",
        fontsize=7.0,
        color=COLORS["blue"],
        fontweight="bold",
    )
    context_key.text(
        0.505,
        0.19,
        "Combined\n(pooled)",
        ha="center",
        va="center",
        fontsize=5.25,
        color=COLORS["ink"],
        linespacing=1.0,
    )
    context_key.plot([0.582, 0.582], [0.10, 0.90], color="#C8D2D7", lw=0.8)
    context_key.text(
        0.605,
        0.92,
        "NONHUMAN DATASETS",
        fontsize=6.8,
        fontweight="bold",
        color=COLORS["amber"],
        va="top",
    )
    marine_crops = [
        (0.00, 0.08, 0.385, 0.72),
        (0.39, 0.03, 0.61, 0.90),
        (0.63, 0.08, 1.00, 0.72),
    ]
    marine_names = ["Octopus", "Ptychodera", "Strongylocentrotus"]
    marine_x = [0.665, 0.790, 0.925]
    for xpos, crop, name in zip(marine_x, marine_crops, marine_names):
        add_asset(
            context_key,
            "marine_lineages",
            (xpos - 0.040, 0.30, 0.080, 0.46),
            zorder=2,
            crop=crop,
        )
        context_key.text(
            xpos,
            0.19,
            name,
            ha="center",
            va="center",
            fontsize=5.0 if name != "Strongylocentrotus" else 4.55,
            color=COLORS["ink"],
        )
    context_key.set_xlim(0, 1)
    context_key.set_ylim(0, 1)

    panel_label(ax_a, "A")
    baseline_within = (
        within[within.variant == "baseline"].set_index("train_context").loc[contexts]
    )
    ypos = np.arange(len(contexts))
    ax_a.scatter(
        baseline_within.auroc,
        ypos - 0.13,
        color=COLORS["blue"],
        s=23,
        label="AUROC",
        zorder=3,
    )
    ax_a.scatter(
        baseline_within.f1,
        ypos + 0.13,
        color=COLORS["green"],
        s=23,
        marker="s",
        label="F1",
        zorder=3,
    )
    ax_a.axhline(3.5, color=COLORS["blue"], lw=0.6, alpha=0.45)
    ax_a.axhline(human_n - 0.5, color=COLORS["gray"], lw=0.9, ls="--")
    ax_a.set_yticks(ypos, labels)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.75, 0.95)
    ax_a.set_xlabel("Within-context performance")
    ax_a.set_title(
        "Substantial signal within each dataset", loc="left", fontweight="bold"
    )
    ax_a.legend(
        frameon=False,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        fontsize=6.3,
        handletextpad=0.35,
        labelspacing=0.35,
    )
    clean_axis(ax_a)

    panel_label(ax_b, "B")
    heatmap = baseline.loc[contexts, contexts]
    image_handle = ax_b.imshow(
        heatmap.to_numpy(),
        vmin=0.65,
        vmax=0.94,
        cmap="YlGnBu",
        aspect="auto",
    )
    ax_b.set_xticks(np.arange(len(contexts)), labels, rotation=47, ha="right")
    ax_b.set_yticks(np.arange(len(contexts)), labels)
    ax_b.axhline(human_n - 0.5, color="white", lw=1.2)
    ax_b.axvline(human_n - 0.5, color="white", lw=1.2)
    ax_b.axhline(3.5, color="white", lw=0.55, alpha=0.8)
    ax_b.axvline(3.5, color="white", lw=0.55, alpha=0.8)
    for index in range(len(contexts)):
        ax_b.add_patch(
            plt.Rectangle(
                (index - 0.47, index - 0.47),
                0.94,
                0.94,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
                alpha=0.95,
            )
        )
    for row in range(len(contexts)):
        for column in range(len(contexts)):
            value = heatmap.iloc[row, column]
            red, green, blue, _ = image_handle.cmap(image_handle.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax_b.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.9,
                color="white" if luminance < 0.48 else COLORS["ink"],
            )
    ax_b.set_xlabel("Evaluation context")
    ax_b.set_ylabel("")
    ax_b.set_title(
        "Transfer exposes lineage and context boundaries",
        loc="left",
        fontweight="bold",
        pad=18,
    )
    ax_b.text(
        0.0,
        1.008,
        "rows = training · columns = evaluation",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=COLORS["gray"],
    )
    colorbar = figure.colorbar(image_handle, ax=ax_b, fraction=0.04, pad=0.025)
    colorbar.set_label("AUROC", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)

    panel_label(ax_c, "C")
    groups = [
        "Matched context",
        "Across human endpoints",
        "Across nonhuman datasets",
        "Across human/nonhuman",
    ]
    group_labels = [
        "Matched",
        "Human\ntransfer",
        "Nonhuman\ntransfer",
        "Cross-\nlineage",
    ]
    rng = np.random.default_rng(2701)
    for index, group in enumerate(groups):
        values = transfer.loc[
            transfer.transfer_group == group, "baseline_auroc"
        ].to_numpy()
        jitter = rng.uniform(-0.09, 0.09, size=len(values))
        ax_c.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=15,
            color=COLORS["blue"] if index < 3 else COLORS["amber"],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        ax_c.plot(
            [index - 0.20, index + 0.20],
            [np.mean(values), np.mean(values)],
            color=COLORS["ink"],
            lw=1.7,
            zorder=3,
        )
    ax_c.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS["blue"],
                markeredgecolor="white",
                markersize=4.5,
                label="matched / within-group",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS["amber"],
                markeredgecolor="white",
                markersize=4.5,
                label="human–nonhuman",
            ),
            plt.Line2D(
                [0],
                [0],
                color=COLORS["ink"],
                lw=1.7,
                label="group mean",
            ),
        ],
        frameon=False,
        ncol=1,
        fontsize=5.55,
        loc="lower left",
        borderaxespad=0.4,
        handlelength=1.5,
        labelspacing=0.22,
    )
    ax_c.set_xticks(np.arange(len(groups)), group_labels)
    ax_c.set_ylim(0.65, 0.95)
    ax_c.set_ylabel("Transfer AUROC")
    ax_c.set_title(
        "Generalization is structured, not universal", loc="left", fontweight="bold"
    )
    clean_axis(ax_c, "y")

    figure.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.09)
    save(figure, output, "figure2_sequence_scope")


def evidence_table() -> pd.DataFrame:
    rows = [
        (
            "Learning & supervision",
            "Substrate-aware learning",
            "Pair-first expanded supervision",
            "selected",
            "More labels; correct pair weight",
        ),
        (
            "Learning & supervision",
            "Supervision",
            "Continuous + ordinal auxiliary labels",
            "limited",
            "Small ranking gain",
        ),
        (
            "Learning & supervision",
            "Cross-tissue",
            "Shared four-tissue training",
            "limited",
            "Transfer below preset criteria",
        ),
        (
            "Target representation",
            "Target selectivity",
            "Aligned target + helix-partner view",
            "selected",
            "Passes shifted-window control",
        ),
        (
            "Target representation",
            "Target selectivity",
            "Focal node / contact geometry",
            "limited",
            "Localizes target effect",
        ),
        (
            "Target representation",
            "Representation",
            "Frozen RNA-language embeddings",
            "not supported",
            "F1-only; no ranking gain",
        ),
        (
            "Target representation",
            "Representation",
            "Substrate-wide all-A decoder",
            "not supported",
            "Stopped for futility",
        ),
        (
            "Target representation",
            "Capacity",
            "Larger / deeper graph networks",
            "not supported",
            "No consistent benefit",
        ),
        (
            "Target representation",
            "Capacity",
            "Graph Transformer / GATv2 expansion",
            "not pursued",
            "Simpler validated model sufficient",
        ),
        (
            "Substrate & cellular context",
            "Substrate propensity",
            "Pair geometry / editing burden",
            "supported, not yet actionable",
            "Substrate propensity detectable",
        ),
        (
            "Substrate & cellular context",
            "Substrate propensity",
            "Hierarchical engagement gate",
            "limited",
            "Ranking gain; unstable F1",
        ),
        (
            "Substrate & cellular context",
            "Biological context",
            "Bulk H3K36me3 overlap",
            "limited",
            "Liver AUPRC only",
        ),
        (
            "Substrate & cellular context",
            "Biological context",
            "RBP binding × bulk expression",
            "not supported",
            "No residual gain",
        ),
        (
            "Substrate & cellular context",
            "Biological context",
            "Transcript / repeat annotation panel",
            "not supported",
            "No site-level gain",
        ),
        (
            "Substrate & cellular context",
            "Cellular state",
            "Matched donor/cell multimodality",
            "deferred",
            "Matched data unavailable",
        ),
        (
            "Substrate & cellular context",
            "RNA state",
            "Matched in vivo structure / isoforms",
            "deferred",
            "Matched data unavailable",
        ),
        (
            "Integration",
            "Integration",
            "Fixed whole-duplex + local-augmented logits",
            "selected",
            "Complementary predictions",
        ),
        (
            "Integration",
            "Integration",
            "Unified end-to-end architecture",
            "not supported",
            "Below frozen ensemble",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["group", "scale", "hypothesis", "status", "interpretation"],
    )


def substrate_realization_tables(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap_rows = []
    model_rows = []
    for context in CONTEXTS[:4]:
        pair_path = (
            ROOT
            / f"experiments/runs/pair_engagement/{context}/oof_pair_predictions.parquet"
        )
        pairs = pd.read_parquet(pair_path)
        pairs = pairs[pairs["model"] == "training_mean"]
        pairs = pairs[
            ["pair_id", "edited_fraction", "geometry__genomic_gap"]
        ].drop_duplicates("pair_id")
        pairs["gap_log10"] = np.log10(1 + pairs["geometry__genomic_gap"])
        pairs["gap_bin"] = pd.qcut(
            pairs["gap_log10"], 8, labels=False, duplicates="drop"
        )
        summary = (
            pairs.groupby("gap_bin", observed=True)
            .agg(
                gap_log10=("gap_log10", "mean"),
                edited_fraction=("edited_fraction", "mean"),
                n_pairs=("pair_id", "nunique"),
            )
            .reset_index()
        )
        summary["context"] = context
        summary["spearman"] = (
            pairs[["geometry__genomic_gap", "edited_fraction"]]
            .corr(method="spearman")
            .iloc[0, 1]
        )
        summary["source"] = pair_path.relative_to(ROOT).as_posix()
        gap_rows.append(summary)

        metrics_path = (
            ROOT / f"experiments/runs/pair_engagement_ablation/{context}/metrics.csv"
        )
        ci_path = (
            ROOT
            / f"experiments/runs/pair_engagement_ablation/{context}/pair_bootstrap_ci.csv"
        )
        metrics = pd.read_csv(metrics_path).set_index("model")
        ci = pd.read_csv(ci_path)
        delta = ci[
            (ci.model == "combined_16")
            & (ci.reference == "graph_intrinsic")
            & (ci.metric == "r2")
        ].iloc[0]
        model_rows.append(
            {
                "context": context,
                "graph_only_r2": metrics.loc["graph_intrinsic", "r2"],
                "geometry_plus_graph_r2": metrics.loc["combined_16", "r2"],
                "delta_r2_lower": delta["lower"],
                "delta_r2_median": delta["median"],
                "delta_r2_upper": delta["upper"],
                "any_high_auroc": metrics.loc["combined_16", "any_edited_auroc"],
                "source": metrics_path.relative_to(ROOT).as_posix(),
            }
        )
    gap = pd.concat(gap_rows, ignore_index=True)
    models = pd.DataFrame(model_rows)
    copy_source(gap, source, "figure2_gap_editing_burden.csv")
    copy_source(models, source, "figure2_pair_engagement.csv")
    return gap, models


def figure2(output: Path, source: Path) -> None:
    frame = evidence_table()
    copy_source(frame, source, "figure2_evidence_map.csv")
    gap, engagement = substrate_realization_tables(source)
    status_colors = {
        "selected": COLORS["green"],
        "supported, not yet actionable": COLORS["teal"],
        "limited": COLORS["amber"],
        "not supported": COLORS["red"],
        "deferred": COLORS["purple"],
        "not pursued": COLORS["gray"],
    }
    status_markers = {
        "selected": "o",
        "supported, not yet actionable": "D",
        "limited": "^",
        "not supported": "X",
        "deferred": "s",
        "not pursued": "P",
    }
    figure = plt.figure(figsize=(7.0, 7.15))
    grid = figure.add_gridspec(
        2, 2, height_ratios=[1.48, 0.78], hspace=0.36, wspace=0.36
    )
    ax_a = figure.add_subplot(grid[0, :])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 1])

    panel_label(ax_a, "A")
    ax_a.set_axis_off()
    ax_a.set_title(
        "Evidence map: which information changes the recoverable signal?",
        loc="left",
        fontweight="bold",
    )
    ax_a.axvspan(0.00, 0.145, color="#F7F9FA", zorder=0)
    ax_a.text(
        0.072,
        0.965,
        "INFORMATION\nLEVELS",
        ha="center",
        va="top",
        fontsize=6.35,
        fontweight="bold",
        color=COLORS["navy"],
        linespacing=1.0,
    )
    headers = ["Tested formulation", "Outcome", "Interpretation"]
    xpos = [0.170, 0.625, 0.785]
    for index, (x, header) in enumerate(zip(xpos, headers)):
        ax_a.text(
            x,
            0.94,
            header,
            fontsize=7.3,
            fontweight="bold",
            color=COLORS["navy"],
            va="bottom",
            ha="left",
        )
    group_colors = {
        "Learning & supervision": COLORS["green"],
        "Target representation": COLORS["blue"],
        "Substrate & cellular context": COLORS["amber"],
        "Integration": COLORS["purple"],
    }
    y = 0.895
    row_number = 0
    group_spans: dict[str, tuple[float, float, str]] = {}
    for group, group_rows in frame.groupby("group", sort=False):
        color = group_colors[group]
        header_y = y
        ax_a.add_patch(
            plt.Rectangle(
                (0.158, y - 0.014),
                0.827,
                0.029,
                facecolor=color,
                edgecolor="none",
                alpha=0.065,
                zorder=0,
            )
        )
        ax_a.plot(
            [0.158, 0.158],
            [y - 0.014, y + 0.015],
            color=color,
            lw=1.8,
            solid_capstyle="butt",
        )
        ax_a.plot(
            [0.158, 0.985],
            [y + 0.015, y + 0.015],
            color=color,
            lw=0.55,
            alpha=0.7,
            solid_capstyle="butt",
        )
        ax_a.text(
            0.170,
            y,
            group.upper(),
            fontsize=6.25,
            fontweight="bold",
            color=color,
            va="center",
        )
        y -= 0.036
        for _, row in group_rows.iterrows():
            if row_number % 2 == 0:
                ax_a.add_patch(
                    plt.Rectangle(
                        (0.158, y - 0.016), 0.827, 0.033, color="#F6F8F9", lw=0
                    )
                )
            ax_a.text(xpos[0], y, row["hypothesis"], fontsize=6.55, va="center")
            marker_size = {
                "selected": 25,
                "supported, not yet actionable": 25,
                "limited": 29,
                "not supported": 30,
                "deferred": 25,
                "not pursued": 28,
            }[row["status"]]
            ax_a.scatter(
                [0.605],
                [y],
                s=marker_size,
                marker=status_markers[row["status"]],
                color=status_colors[row["status"]],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            display_status = (
                "Supported†"
                if row["status"] == "supported, not yet actionable"
                else row["status"].capitalize()
            )
            ax_a.text(
                xpos[1],
                y,
                display_status,
                fontsize=6.0,
                va="center",
                color=status_colors[row["status"]],
            )
            ax_a.text(xpos[2], y, row["interpretation"], fontsize=6.25, va="center")
            y -= 0.0365
            row_number += 1
        group_spans[group] = (header_y, y + 0.017, color)

    level_labels = {
        "Learning & supervision": "LEARNING",
        "Target representation": "TARGET-LOCAL",
        "Substrate & cellular context": "SUBSTRATE /\nCONTEXT",
        "Integration": "INTEGRATION",
    }
    for group, (header_y, lower_y, color) in group_spans.items():
        center_y = (header_y + lower_y) / 2
        ax_a.add_patch(
            plt.Rectangle(
                (0.010, lower_y + 0.003),
                0.125,
                header_y - lower_y - 0.006,
                facecolor=color,
                edgecolor="none",
                alpha=0.055,
                zorder=0,
            )
        )
        ax_a.plot(
            [0.135, 0.158],
            [header_y, header_y],
            color=color,
            lw=1.0,
            solid_capstyle="butt",
            zorder=2,
        )
        if group == "Learning & supervision":
            # Equal pair weighting: substrates with unequal row counts contribute once.
            for line_y, count in [
                (center_y + 0.025, 4),
                (center_y - 0.018, 7),
            ]:
                sites = np.linspace(0.036, 0.086, count)
                ax_a.plot(
                    [sites[0], sites[-1]],
                    [line_y, line_y],
                    color=COLORS["navy"],
                    lw=0.65,
                    zorder=2,
                )
                ax_a.scatter(
                    sites,
                    np.full_like(sites, line_y),
                    s=7.5,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.55,
                    zorder=3,
                )
                ax_a.add_patch(
                    FancyArrowPatch(
                        (0.090, line_y),
                        (0.116, center_y + 0.003),
                        arrowstyle="-|>",
                        mutation_scale=5.5,
                        color=color,
                        lw=0.65,
                    )
                )
            ax_a.add_patch(
                Circle(
                    (0.119, center_y + 0.003),
                    0.008,
                    facecolor=color,
                    edgecolor="white",
                    lw=0.4,
                    zorder=4,
                )
            )
        elif group == "Target representation":
            add_asset(
                ax_a,
                "target_pair_context",
                (0.022, center_y - 0.070, 0.102, 0.135),
                alpha=0.98,
                zorder=2,
            )
            ax_a.plot(
                [0.060, 0.060, 0.091, 0.091],
                [center_y - 0.040, center_y - 0.050, center_y - 0.050, center_y - 0.040],
                color=color,
                lw=0.65,
                zorder=4,
            )
        elif group == "Substrate & cellular context":
            add_asset(
                ax_a,
                "substrate_organization",
                (0.018, center_y - 0.010, 0.110, 0.105),
                alpha=0.98,
                zorder=2,
            )
            add_asset(
                ax_a,
                "matched_cell_state",
                (0.043, center_y - 0.105, 0.060, 0.080),
                alpha=0.90,
                zorder=2,
            )
            ax_a.plot(
                [0.070, 0.070],
                [center_y - 0.025, center_y - 0.010],
                color=color,
                lw=0.65,
                alpha=0.75,
                zorder=3,
            )
        else:
            integration_y = center_y + 0.018
            for start_y, branch_color in [
                (integration_y + 0.020, COLORS["blue"]),
                (integration_y - 0.020, COLORS["purple"]),
            ]:
                ax_a.add_patch(
                    FancyArrowPatch(
                        (0.032, start_y),
                        (0.086, integration_y),
                        arrowstyle="-|>",
                        mutation_scale=5.5,
                        color=branch_color,
                        lw=0.75,
                    )
                )
            ax_a.add_patch(
                plt.Rectangle(
                    (0.087, integration_y - 0.014),
                    0.034,
                    0.028,
                    facecolor="white",
                    edgecolor=color,
                    lw=0.75,
                    zorder=3,
                )
            )
            ax_a.text(
                0.104,
                integration_y,
                "½Σ",
                ha="center",
                va="center",
                fontsize=4.7,
                fontweight="bold",
                color=color,
                zorder=4,
            )
        label_y = lower_y + (0.008 if group == "Integration" else 0.014)
        ax_a.text(
            0.072,
            label_y,
            level_labels[group],
            ha="center",
            va="bottom",
            fontsize=5.05 if group == "Integration" else 5.15,
            fontweight="bold",
            color=color,
            linespacing=0.95,
        )
    ax_a.text(
        0.99,
        0.018,
        "Comparators, effect sizes, and intervals: Table S2.",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=COLORS["gray"],
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)

    panel_label(ax_b, "B")
    for context, color in zip(
        CONTEXTS[:4],
        [COLORS["blue"], COLORS["purple"], COLORS["teal"], COLORS["amber"]],
    ):
        subset = gap[gap.context == context]
        rho = subset.spearman.iloc[0]
        ax_b.plot(
            subset.gap_log10,
            subset.edited_fraction,
            marker="o",
            ms=3.2,
            lw=1.1,
            color=color,
            label=f"{CONTEXT_LABELS[context]}  ρ={rho:.2f}",
        )
    ax_b.set_xlabel("Mean log$_{10}$(inter-arm gap + 1) within octile")
    ax_b.set_ylabel("Mean edited-site fraction")
    ax_b.set_title(
        "Geometry tracks\nsubstrate-level editing burden",
        loc="left",
        fontweight="bold",
        fontsize=8.3,
    )
    ax_b.legend(frameon=False, fontsize=6.1)
    clean_axis(ax_b, "both")

    panel_label(ax_c, "C")
    ordered = engagement.set_index("context").loc[CONTEXTS[:4]]
    y = np.arange(4)
    ax_c.errorbar(
        ordered.delta_r2_median,
        y,
        xerr=np.vstack(
            [
                ordered.delta_r2_median - ordered.delta_r2_lower,
                ordered.delta_r2_upper - ordered.delta_r2_median,
            ]
        ),
        fmt="o",
        ms=4.5,
        capsize=2,
        color=COLORS["teal"],
    )
    for index, (_, row) in enumerate(ordered.iterrows()):
        ax_c.text(
            row.delta_r2_upper + 0.008,
            index,
            f"R²={row.geometry_plus_graph_r2:.2f}",
            va="center",
            fontsize=6.4,
            color=COLORS["gray"],
        )
    ax_c.axvline(0, color=COLORS["ink"], lw=0.7)
    ax_c.set_yticks(y, [CONTEXT_LABELS[x] for x in CONTEXTS[:4]])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Geometry + graph − graph-only ΔR²\n(95% pair-clustered interval)")
    ax_c.set_title(
        "Geometry improves\npair-level burden models",
        loc="left",
        fontweight="bold",
        fontsize=8.3,
    )
    ax_c.set_xlim(-0.01, 0.34)
    clean_axis(ax_c)
    figure.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.08)
    save(figure, output, "figure2_evidence_map")


def pair_balance_table() -> pd.DataFrame:
    rows = []
    for context in CONTEXTS:
        frame = pd.read_csv(
            ROOT
            / f"experiments/runs/richer_supervision/binary_pair_balanced/{context}/pair_bootstrap_ci.csv"
        )
        frame = frame[frame.metric.isin(["auroc", "auprc", "f1"])].copy()
        frame["context"] = context
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def local_integration_table() -> pd.DataFrame:
    rows = []
    for context in ["MuscleSkeletal", "Liver"]:
        path = (
            ROOT
            / f"experiments/runs/richer_supervision/pair_local_ensemble_comparison/{context}/pair_bootstrap_ci.csv"
        )
        frame = pd.read_csv(path)
        selected = frame[
            (frame.candidate == "pair_balanced")
            & (frame.reference == "pair_local_ensemble")
            & frame.metric.isin(["auroc", "auprc", "f1"])
        ].copy()
        selected[["lower", "median", "upper"]] = -selected[
            ["upper", "median", "lower"]
        ].to_numpy()
        selected["context"] = context
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def architecture_panel(axis: plt.Axes, weights: list[float]) -> None:
    axis.set_axis_off()
    axis.set_title(
        "Selected predictor: complementary views, fixed integration",
        loc="left",
        fontweight="bold",
    )
    axis.add_patch(
        plt.Rectangle(
            (0.015, 0.205),
            0.190,
            0.640,
            facecolor=COLORS["pale_green"],
            edgecolor="#CFDCD5",
            lw=0.55,
            zorder=0,
        )
    )
    axis.add_patch(
        plt.Rectangle(
            (0.285, 0.555),
            0.695,
            0.290,
            facecolor=COLORS["pale_blue"],
            edgecolor="none",
            zorder=0,
        )
    )
    axis.add_patch(
        plt.Rectangle(
            (0.285, 0.205),
            0.695,
            0.290,
            facecolor="#F3EFF8",
            edgecolor="none",
            zorder=0,
        )
    )
    axis.text(
        0.110,
        0.805,
        "PAIR-FIRST INPUT",
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=COLORS["green"],
    )
    add_asset(
        axis,
        "rna_substrate_editorial",
        (0.032, 0.455, 0.155, 0.245),
        alpha=0.98,
        zorder=2,
    )
    axis.text(
        0.110,
        0.335,
        "all valid high/low sites",
        ha="center",
        va="center",
        fontsize=6.0,
        color=COLORS["gray"],
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.192, 0.525),
            (0.234, 0.525),
            arrowstyle="-|>",
            mutation_scale=8.0,
            color=COLORS["green"],
            lw=1.1,
            zorder=4,
        )
    )
    axis.add_patch(
        Circle(
            (0.247, 0.525),
            0.010,
            facecolor=COLORS["green"],
            edgecolor="white",
            lw=0.7,
            zorder=5,
        )
    )
    for end, color in [
        ((0.315, 0.690), COLORS["blue"]),
        ((0.315, 0.345), COLORS["purple"]),
    ]:
        axis.add_patch(
            FancyArrowPatch(
                (0.257, 0.525),
                end,
                arrowstyle="-|>",
                mutation_scale=8.0,
                color=color,
                lw=1.05,
                zorder=4,
            )
        )

    for xpos, label in [
        (0.385, "REPRESENTATION"),
        (0.565, "COMPUTATION"),
        (0.720, "LOGIT"),
        (0.890, "FIXED COMBINATION"),
    ]:
        axis.text(
            xpos,
            0.925,
            label,
            ha="center",
            va="center",
            fontsize=5.45,
            fontweight="bold",
            color=COLORS["gray"],
        )
    axis.text(
        0.305,
        0.800,
        "WHOLE-DUPLEX VIEW",
        fontsize=6.6,
        fontweight="bold",
        color=COLORS["blue"],
        va="center",
    )
    graph_nodes = [
        (0.330, 0.690),
        (0.365, 0.745),
        (0.400, 0.690),
        (0.365, 0.635),
        (0.435, 0.740),
        (0.435, 0.640),
    ]
    for start, end in [
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 5),
    ]:
        axis.plot(
            [graph_nodes[start][0], graph_nodes[end][0]],
            [graph_nodes[start][1], graph_nodes[end][1]],
            color=COLORS["sky"],
            lw=0.8,
            zorder=2,
        )
    for xpos, ypos in graph_nodes:
        axis.add_patch(
            Circle(
                (xpos, ypos),
                0.010,
                facecolor="white",
                edgecolor=COLORS["blue"],
                lw=1.0,
                zorder=3,
            )
        )

    axis.text(
        0.305,
        0.455,
        "TARGET-ALIGNED VIEW",
        fontsize=6.6,
        fontweight="bold",
        color=COLORS["purple"],
        va="center",
    )
    add_asset(
        axis,
        "target_pair_context",
        (0.310, 0.270, 0.145, 0.145),
        alpha=0.98,
        zorder=2,
    )

    for y0, color, label in [
        (0.69, COLORS["blue"], "graph encoder"),
        (0.34, COLORS["purple"], "graph + local encoder"),
    ]:
        axis.add_patch(
            plt.Rectangle(
                (0.500, y0 - 0.052),
                0.135,
                0.104,
                facecolor="white",
                edgecolor=color,
                lw=0.8,
                zorder=2,
            )
        )
        axis.text(
            0.5675,
            y0,
            label,
            ha="center",
            va="center",
            fontsize=5.9,
            color=COLORS["ink"],
            zorder=3,
        )
        axis.add_patch(
            FancyArrowPatch(
                (0.455, y0),
                (0.496, y0),
                arrowstyle="-|>",
                mutation_scale=7.5,
                color=color,
                lw=0.9,
            )
        )
        axis.add_patch(
            FancyArrowPatch(
                (0.639, y0),
                (0.685, y0),
                arrowstyle="-|>",
                mutation_scale=7.5,
                color=color,
                lw=0.9,
            )
        )

    for y0, color, label in [
        (0.69, COLORS["blue"], "$z_1$"),
        (0.34, COLORS["purple"], "$z_2$"),
    ]:
        axis.plot(
            [0.697, 0.735],
            [y0, y0],
            color=color,
            lw=1.35,
            solid_capstyle="round",
            zorder=3,
        )
        axis.text(
            0.716,
            y0 + 0.037,
            label,
            color=color,
            fontsize=6.5,
            fontweight="bold",
            ha="center",
            va="center",
        )
    for start, end, color in [
        ((0.738, 0.69), (0.825, 0.560), COLORS["blue"]),
        ((0.738, 0.34), (0.825, 0.490), COLORS["purple"]),
    ]:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8.0,
                color=color,
                lw=0.9,
            )
        )
    axis.add_patch(
        plt.Rectangle(
            (0.825, 0.450),
            0.155,
            0.150,
            facecolor="white",
            edgecolor=COLORS["amber"],
            lw=1.0,
            zorder=2,
        )
    )
    axis.text(
        0.9025,
        0.525,
        rf"$z = {weights[0]:.1f}z_1 + {weights[1]:.1f}z_2$",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=COLORS["ink"],
    )
    axis.plot([0.015, 0.980], [0.135, 0.135], color="#D3DCE0", lw=0.65)
    axis.text(
        0.497,
        0.070,
        "independent training  ·  fixed weights  ·  no stacker  ·  no post-test fitting",
        ha="center",
        fontsize=6.25,
        fontweight="bold",
        color=COLORS["navy"],
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)


def figure3(output: Path, source: Path) -> None:
    manifest = json.loads(
        (ROOT / "provenance/final_pipeline_manifest.json").read_text(encoding="utf-8")
    )
    weights = manifest["candidate_specification"]["weights"]
    if len(weights) != 2 or not np.isclose(sum(weights), 1.0):
        raise RuntimeError(f"Unexpected final ensemble weights: {weights}")
    control_path = (
        ROOT / "experiments/runs/richer_supervision/local_alignment_control/"
        "MuscleSkeletal/summary.json"
    )
    control = json.loads(control_path.read_text(encoding="utf-8"))
    shift_match = re.search(r"(\d+)-nt-shifted", control["scope"])
    if shift_match is None:
        raise RuntimeError("Could not recover shifted-window control offset")
    shift_nt = int(shift_match.group(1))

    pair = pair_balance_table()
    integration = local_integration_table()
    supervision = pd.read_csv(
        ROOT / "experiments/runs/richer_supervision/pair_balance_comparison/metrics.csv"
    )
    focal_metrics = pd.read_csv(
        ROOT / "experiments/runs/richer_supervision/pooling_comparison/metrics.csv"
    )
    alignment_metrics = pd.read_csv(
        ROOT
        / "experiments/runs/richer_supervision/local_alignment_control/MuscleSkeletal/metrics.csv"
    )
    copy_source(pair, source, "figure3_pair_balance_ci.csv")
    copy_source(integration, source, "figure3_integration_ci.csv")
    copy_source(supervision, source, "figure3_supervision_decomposition.csv")
    copy_source(focal_metrics, source, "figure3_focal_metrics.csv")
    copy_source(alignment_metrics, source, "figure3_alignment_metrics.csv")
    copy_source(
        pd.DataFrame(
            [
                {
                    "quantity": "pair_model_logit_weight",
                    "value": weights[0],
                    "source": "provenance/final_pipeline_manifest.json",
                },
                {
                    "quantity": "local_augmented_graph_logit_weight",
                    "value": weights[1],
                    "source": "provenance/final_pipeline_manifest.json",
                },
                {
                    "quantity": "shifted_control_offset_nt",
                    "value": shift_nt,
                    "source": control_path.relative_to(ROOT).as_posix(),
                },
            ]
        ),
        source,
        "figure3_model_specification.csv",
    )

    figure = plt.figure(figsize=(7.0, 7.6))
    grid = figure.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 0.72], hspace=0.62, wspace=0.48
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])
    ax_e = figure.add_subplot(grid[2, :])

    metric_colors = {
        "auroc": COLORS["blue"],
        "auprc": COLORS["teal"],
        "f1": COLORS["green"],
    }
    panel_label(ax_a, "A")
    model_order = ["development_reference", "expanded_site", "expanded_pair"]
    model_labels = [
        "Original\nreference",
        "All high/low\nrow-wise",
        "All high/low\npair-first",
    ]
    x = np.arange(3)
    for metric in ["auroc", "auprc", "f1"]:
        values = supervision.set_index("model").loc[model_order, metric]
        ax_a.plot(
            x,
            values,
            marker="o",
            lw=1.4,
            ms=4.2,
            color=metric_colors[metric],
            label=metric.upper(),
        )
        ax_a.text(
            2.10,
            values.iloc[-1],
            metric.upper(),
            color=metric_colors[metric],
            fontsize=6.3,
            fontweight="bold",
            va="center",
        )
    ax_a.set_xticks(x, model_labels)
    ax_a.set_xlim(-0.10, 2.48)
    ax_a.set_ylim(0.75, 0.905)
    ax_a.set_ylabel("Muscle development OOF")
    ax_a.set_title(
        "Label expansion and pair weighting both help",
        loc="left",
        fontweight="bold",
    )
    ax_a.text(
        0.5,
        0.755,
        "+ valid labels",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=COLORS["gray"],
    )
    ax_a.text(
        1.5,
        0.755,
        "+ pair-aware weighting",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=COLORS["gray"],
    )
    clean_axis(ax_a, "y")

    panel_label(ax_b, "B")
    positions = np.arange(len(CONTEXTS))
    for offset, metric in zip([-0.18, 0, 0.18], ["auroc", "auprc", "f1"]):
        subset = pair[pair.metric == metric].set_index("context").loc[CONTEXTS]
        ax_b.errorbar(
            100 * subset["median"],
            positions + offset,
            xerr=np.vstack(
                [
                    100 * (subset["median"] - subset["lower"]),
                    100 * (subset["upper"] - subset["median"]),
                ]
            ),
            fmt="o",
            ms=4.2,
            capsize=1.8,
            color=metric_colors[metric],
            label=metric.upper(),
        )
    ax_b.axvline(0, color=COLORS["ink"], lw=0.7)
    ax_b.set_yticks(positions, [CONTEXT_LABELS[x] for x in CONTEXTS])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Pair-first − internal reference (points)")
    ax_b.set_title("Pair-first gains replicate", loc="left", fontweight="bold")
    ax_b.legend(
        frameon=False,
        ncol=3,
        loc="upper right",
        borderaxespad=0.4,
        fontsize=6.2,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    clean_axis(ax_b)

    panel_label(ax_c, "C")
    representation_rows = []
    for label, frame, model in [
        ("Whole-duplex\nonly", alignment_metrics, "pair_balanced"),
        ("Focal-node\ngraph", focal_metrics, "focal"),
        ("Graph +\naligned local", alignment_metrics, "aligned_local"),
        (f"Graph + shifted\n+{shift_nt} nt", alignment_metrics, "shifted_local"),
    ]:
        row = frame[frame.model == model].iloc[0]
        representation_rows.append(
            {
                "representation": label,
                **{metric: row[metric] for metric in ["auroc", "auprc", "f1"]},
            }
        )
    representations = pd.DataFrame(representation_rows)
    copy_source(representations, source, "figure3_representation_metrics.csv")
    x = np.arange(len(representations))
    width = 0.24
    for index, metric in enumerate(["auroc", "auprc", "f1"]):
        ax_c.bar(
            x + (index - 1) * width,
            representations[metric],
            width=width,
            color=metric_colors[metric],
            label=metric.upper(),
            zorder=2,
        )
    ax_c.set_xticks(x, representations.representation, rotation=0, ha="center")
    ax_c.tick_params(axis="x", labelsize=6.4, pad=3)
    ax_c.set_ylim(0.64, 0.94)
    ax_c.set_ylabel("Development OOF performance")
    ax_c.set_title(
        "Target-centered local information\nimproves prediction",
        loc="left",
        fontweight="bold",
    )
    clean_axis(ax_c, "y")
    ax_c.legend(
        frameon=False,
        ncol=3,
        loc="upper left",
        fontsize=6.1,
        columnspacing=0.8,
        handletextpad=0.3,
    )

    panel_label(ax_d, "D")
    metrics = ["AUROC", "AUPRC", "F1"]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["green"]]
    for i, metric in enumerate(["auroc", "auprc", "f1"]):
        for j, context in enumerate(["MuscleSkeletal", "Liver"]):
            row = integration[
                (integration.context == context) & (integration.metric == metric)
            ].iloc[0]
            ypos = i + (-0.1 if j == 0 else 0.1)
            ax_d.errorbar(
                100 * row["median"],
                ypos,
                xerr=np.array(
                    [
                        [100 * (row["median"] - row["lower"])],
                        [100 * (row["upper"] - row["median"])],
                    ]
                ),
                fmt="o",
                ms=4.4,
                capsize=2,
                color=colors[i],
                markerfacecolor=colors[i] if j == 0 else "white",
                markeredgecolor=colors[i],
                markeredgewidth=0.9,
            )
    ax_d.axvline(0, color=COLORS["ink"], lw=0.7)
    ax_d.set_yticks(range(3), metrics)
    ax_d.set_xlabel(
        "Fixed ensemble − whole-duplex-only predictor\n(points; 95% pair-clustered CI)"
    )
    ax_d.set_title(
        "Independently optimized views are complementary",
        loc="left",
        fontweight="bold",
    )
    ax_d.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=COLORS["gray"],
                markerfacecolor=COLORS["gray"],
                lw=0,
                label="Skeletal muscle",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=COLORS["gray"],
                markerfacecolor="white",
                markeredgecolor=COLORS["gray"],
                lw=0,
                label="Liver",
            ),
        ],
        frameon=False,
        ncol=1,
        fontsize=6.2,
        loc="upper right",
        borderaxespad=0.35,
        labelspacing=0.25,
        handletextpad=0.25,
    )
    ax_d.set_xlim(0, 4.25)
    clean_axis(ax_d)

    panel_label(ax_e, "E")
    architecture_panel(ax_e, weights)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.06)
    save(figure, output, "figure3_organizing_principles")


def final_tables(
    source: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(ROOT / "results/adarpro/final_locked_test/metrics.csv")
    rows = []
    for context in CONTEXTS:
        ci = pd.read_csv(
            ROOT / f"results/adarpro/final_locked_test/{context}_pair_bootstrap_ci.csv"
        )
        ci["context"] = context
        rows.append(ci)
    confidence = pd.concat(rows, ignore_index=True)
    transitions = pd.read_csv(
        ROOT / "figures/final/source_tables/final_error_transitions.csv"
    )
    matched = pd.read_csv(ROOT / "figures/final/source_tables/final_matched_recall.csv")
    matched_ci = pd.read_csv(
        ROOT / "figures/final/source_tables/final_matched_recall_pair_bootstrap_ci.csv"
    )
    copy_source(metrics, source, "figure4_locked_metrics.csv")
    copy_source(confidence, source, "figure4_locked_metric_ci.csv")
    copy_source(transitions, source, "figure4_error_transitions.csv")
    merged = matched.merge(
        matched_ci[matched_ci.metric == "false_positives"], on="context", how="left"
    )
    copy_source(merged, source, "figure4_matched_recall.csv")
    return metrics, confidence, transitions, merged


def figure4(output: Path, source: Path) -> None:
    metrics, confidence, transitions, _matched = final_tables(source)
    figure = plt.figure(figsize=(7.0, 7.0))
    grid = figure.add_gridspec(
        3, 2, height_ratios=[1.12, 0.82, 0.88], hspace=0.52, wspace=0.42
    )
    ax_a = figure.add_subplot(grid[0, :])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 1])
    ax_d = figure.add_subplot(grid[2, 0])
    ax_e = figure.add_subplot(grid[2, 1])

    panel_label(ax_a, "A")
    metric_colors = {
        "auroc": COLORS["blue"],
        "auprc": COLORS["teal"],
        "f1": COLORS["green"],
    }
    positions = np.arange(len(CONTEXTS))
    for offset, metric in zip([-0.18, 0, 0.18], ["auroc", "auprc", "f1"]):
        subset = (
            confidence[confidence.metric == metric].set_index("context").loc[CONTEXTS]
        )
        ax_a.errorbar(
            100 * subset["median"],
            positions + offset,
            xerr=np.vstack(
                [
                    100 * (subset["median"] - subset["lower"]),
                    100 * (subset["upper"] - subset["median"]),
                ]
            ),
            fmt="o",
            ms=4.8,
            capsize=2,
            color=metric_colors[metric],
            label=metric.upper(),
        )
    ax_a.axvline(0, color=COLORS["ink"], lw=0.8)
    ax_a.set_yticks(positions, [CONTEXT_LABELS[x] for x in CONTEXTS])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("ADARPro − published AdarEdit (points; 95% pair-clustered CI)")
    ax_a.set_title(
        "Frozen canonical held-out evaluation",
        loc="left",
        fontweight="bold",
    )
    ax_a.legend(
        frameon=False,
        ncol=3,
        loc="lower right",
        borderaxespad=0.35,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    clean_axis(ax_a)

    panel_label(ax_b, "B")
    base = (
        metrics[metrics.model == "published_baseline_reproduced"]
        .set_index("context")
        .loc[CONTEXTS]
    )
    cand = (
        metrics[metrics.model == "adarpro_pair_local_equal_logit_ensemble"]
        .set_index("context")
        .loc[CONTEXTS]
    )
    y = np.arange(len(CONTEXTS))
    for index, context in enumerate(CONTEXTS):
        ax_b.plot(
            [base.loc[context, "f1"], cand.loc[context, "f1"]],
            [index, index],
            color="#CBD3D8",
            lw=2,
            zorder=1,
        )
    ax_b.scatter(base.f1, y, color=COLORS["gray"], s=24, label="AdarEdit", zorder=2)
    ax_b.scatter(cand.f1, y, color=COLORS["green"], s=24, label="ADARPro", zorder=3)
    ax_b.set_yticks(y, [CONTEXT_LABELS[x] for x in CONTEXTS])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.80, 0.905)
    ax_b.set_xlabel("F1")
    ax_b.set_title("Absolute held-out F1", loc="left", fontweight="bold")
    ax_b.legend(
        frameon=False,
        ncol=2,
        fontsize=6.2,
        loc="upper left",
        borderaxespad=0.35,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    clean_axis(ax_b)

    panel_label(ax_c, "C")
    brier = confidence[confidence.metric == "brier"].set_index("context").loc[CONTEXTS]
    reduction = -brier["median"]
    ax_c.errorbar(
        reduction,
        y,
        xerr=np.vstack([reduction + brier["upper"], -brier["lower"] - reduction]),
        fmt="o",
        ms=4.5,
        capsize=2,
        color=COLORS["purple"],
    )
    ax_c.axvline(0, color=COLORS["ink"], lw=0.7)
    ax_c.set_yticks(y, [CONTEXT_LABELS[x] for x in CONTEXTS])
    ax_c.invert_yaxis()
    ax_c.set_xticks([0.00, 0.01, 0.02, 0.03])
    ax_c.set_xlabel("Brier-score reduction (95% pair-clustered CI)")
    ax_c.set_title(
        "Calibration-sensitive error decreases", loc="left", fontweight="bold"
    )
    clean_axis(ax_c)

    panel_label(ax_d, "D")
    transitions = transitions.set_index("context").loc[CONTEXTS]
    ax_d.barh(
        y - 0.15,
        100 * transitions.fp_rescue_rate,
        height=0.27,
        color=COLORS["teal"],
        label="FP rescued",
    )
    ax_d.barh(
        y + 0.15,
        100 * transitions.tp_damage_rate,
        height=0.27,
        color=COLORS["red"],
        label="TP damaged",
    )
    ax_d.set_yticks(y, [CONTEXT_LABELS[x] for x in CONTEXTS])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Sites (%) at frozen threshold")
    ax_d.set_title("FP rescue exceeds TP damage", loc="left", fontweight="bold")
    ax_d.legend(
        frameon=False,
        loc="lower right",
        ncol=1,
        fontsize=5.7,
        borderaxespad=0.4,
    )
    clean_axis(ax_d)
    ax_d.set_xlim(0, 56)

    panel_label(ax_e, "E")
    ax_e.set_axis_off()
    ax_e.set_title("Present recoverability boundary", loc="left", fontweight="bold")
    boundary = [
        (
            0.70,
            "Recoverable",
            "whole-duplex signal\ntarget-centered signal",
            COLORS["green"],
            COLORS["pale_green"],
        ),
        (
            0.41,
            "Partially recoverable",
            "pair-level propensity\ncross-context transfer",
            COLORS["amber"],
            COLORS["pale_amber"],
        ),
        (
            0.12,
            "Unmeasured at matched scale",
            "cell/donor state · in vivo fold\nregulator exposure",
            COLORS["purple"],
            "#F1EDF7",
        ),
    ]
    for y0, title, detail, color, fill in boundary:
        ax_e.add_patch(
            plt.Rectangle(
                (0.02, y0),
                0.96,
                0.235,
                facecolor=fill,
                edgecolor="none",
                zorder=0,
            )
        )
        ax_e.plot(
            [0.02, 0.98],
            [y0 + 0.235, y0 + 0.235],
            color=color,
            lw=0.72,
            alpha=0.55,
            solid_capstyle="butt",
            zorder=2,
        )
        ax_e.add_patch(
            plt.Rectangle(
                (0.033, y0 + 0.026),
                0.010,
                0.183,
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )
        ax_e.plot(
            [0.735, 0.735],
            [y0 + 0.028, y0 + 0.207],
            color="#CFD8DC",
            lw=0.55,
            zorder=2,
        )
        ax_e.text(
            0.070,
            y0 + 0.182,
            title.upper(),
            ha="left",
            va="center",
            fontsize=6.4,
            color=color,
            fontweight="bold",
            zorder=3,
        )
        ax_e.text(
            0.070,
            y0 + 0.095,
            detail,
            ha="left",
            va="center",
            fontsize=5.9,
            color=COLORS["ink"],
            linespacing=1.18,
            zorder=3,
        )

    add_asset(
        ax_e,
        "rna_substrate_editorial",
        (0.755, 0.725, 0.205, 0.170),
        alpha=0.98,
        zorder=2,
    )
    add_asset(
        ax_e,
        "substrate_organization",
        (0.755, 0.435, 0.205, 0.175),
        alpha=0.98,
        zorder=2,
    )
    add_asset(
        ax_e,
        "matched_cell_state",
        (0.775, 0.135, 0.165, 0.190),
        alpha=0.94,
        zorder=2,
    )
    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)
    figure.subplots_adjust(left=0.12, right=0.99, top=0.97, bottom=0.07)
    save(figure, output, "figure4_locked_validation")


def supplementary_figures(output: Path, source: Path) -> None:
    # S1: internal development scores make the evaluation regime explicit.
    frames = []
    for context in ["MuscleSkeletal", "Liver"]:
        frame = pd.read_csv(
            ROOT
            / f"experiments/runs/richer_supervision/pair_local_ensemble_comparison/{context}/metrics.csv"
        )
        frame["context"] = context
        frames.append(frame)
    dev = pd.concat(frames, ignore_index=True)
    copy_source(dev, source, "figureS1_development_metrics.csv")
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), sharey=True)
    models = ["development_reference", "pair_balanced", "local", "pair_local_ensemble"]
    labels = [
        "Internal\nreference",
        "Pair-first",
        "Local-aug.\ngraph",
        "Fixed\nensemble",
    ]
    for axis, metric in zip(axes, ["auroc", "auprc", "f1"]):
        for j, context in enumerate(["MuscleSkeletal", "Liver"]):
            subset = dev[dev.context == context].set_index("model").loc[models]
            axis.plot(
                np.arange(4),
                subset[metric],
                marker="o",
                color=COLORS["navy"] if j == 0 else COLORS["amber"],
                alpha=1 if j == 0 else 0.8,
                label=CONTEXT_LABELS[context],
            )
        axis.set_xticks(range(4), labels, rotation=20, ha="right")
        axis.set_title(metric.upper())
        clean_axis(axis, "y")
    axes[0].set_ylabel("Pair-disjoint development OOF")
    axes[-1].legend(frameon=False)
    figure.suptitle(
        "Development comparisons use an identical-fold internal reference",
        fontweight="bold",
        fontsize=10,
    )
    figure.tight_layout()
    save(figure, output, "figureS1_development_regime")

    # S2: negative biological feature formulations and matched controls.
    selected = []
    specs = [
        (
            "RBP suppressors",
            "experiments/runs/residual_pilot/MuscleSkeletal/pair_bootstrap_ci.csv",
            {"model": "baseline_plus_biological_suppressors"},
        ),
        (
            "All predicted RBP binding",
            "experiments/runs/residual_pilot/MuscleSkeletal/pair_bootstrap_ci.csv",
            {"model": "baseline_plus_all_predicted_binding"},
        ),
        (
            "Matched random features",
            "experiments/runs/residual_pilot/MuscleSkeletal/pair_bootstrap_ci.csv",
            {"model": "baseline_plus_matched_random"},
        ),
        (
            "IRAlu geometry",
            "experiments/runs/genomic_residual_v2/MuscleSkeletal/pair_bootstrap_ci.csv",
            {"model": "baseline_plus_geometry"},
        ),
        (
            "H3K36me3, Liver",
            "experiments/runs/chromatin_residual/Liver/pair_bootstrap_ci.csv",
            {"model": "baseline_plus_h3k36me3_multiscale_occupancy"},
        ),
        (
            "H3K36me3 +1 Mb control",
            "experiments/runs/chromatin_residual/Liver/pair_bootstrap_ci.csv",
            {"model": "baseline_plus_shift1mb_control"},
        ),
    ]
    for label, path, selectors in specs:
        for metric in ["auroc", "auprc", "f1"]:
            row = read_ci(path, metric, **selectors)
            selected.append(
                {
                    "feature": label,
                    "metric": metric,
                    "lower": row["lower"],
                    "median": row["median"],
                    "upper": row["upper"],
                    "source": path,
                }
            )
    negative = pd.DataFrame(selected)
    copy_source(negative, source, "figureS2_biological_features_ci.csv")
    figure, axis = plt.subplots(figsize=(7.0, 3.7))
    feature_order = [spec[0] for spec in specs][::-1]
    y = np.arange(len(feature_order))
    for offset, metric in zip([-0.16, 0, 0.16], ["auroc", "auprc", "f1"]):
        subset = (
            negative[negative.metric == metric].set_index("feature").loc[feature_order]
        )
        axis.errorbar(
            100 * subset["median"],
            y + offset,
            xerr=np.vstack(
                [
                    100 * (subset["median"] - subset["lower"]),
                    100 * (subset["upper"] - subset["median"]),
                ]
            ),
            fmt="o",
            ms=4,
            capsize=2,
            color={
                "auroc": COLORS["blue"],
                "auprc": COLORS["teal"],
                "f1": COLORS["green"],
            }[metric],
            label=metric.upper(),
        )
    axis.axvline(0, color=COLORS["ink"], lw=0.7)
    axis.set_xlabel("Candidate − same-fold reference (percentage points)")
    axis.set_yticks(y, feature_order)
    axis.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.68, -0.18),
    )
    clean_axis(axis)
    figure.suptitle(
        "Tested biological annotations did not add robust residual information",
        fontweight="bold",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(figure, output, "figureS2_biological_feature_controls")

    # S3: natural-prevalence robustness.
    natural_frames = []
    for context in ["MuscleSkeletal", "Liver"]:
        frame = pd.read_csv(
            ROOT
            / f"experiments/runs/richer_supervision/natural_prevalence/pair_local_comparison/{context}/metrics.csv"
        )
        frame["context"] = context
        natural_frames.append(frame)
    natural = pd.concat(natural_frames, ignore_index=True)
    copy_source(natural, source, "figureS3_natural_prevalence_metrics.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    model_order = [
        "development_reference",
        "pair_balanced",
        "local",
        "pair_local_ensemble",
    ]
    model_labels = ["Internal ref.", "Pair-first", "Local-aug. graph", "Ensemble"]
    for axis, context in zip(axes, ["MuscleSkeletal", "Liver"]):
        subset = natural[natural.context == context].set_index("model").loc[model_order]
        x = np.arange(4)
        axis.bar(
            x - 0.17, subset.auprc, width=0.34, color=COLORS["teal"], label="AUPRC"
        )
        axis.bar(x + 0.17, subset.f1, width=0.34, color=COLORS["green"], label="F1")
        axis.set_xticks(x, model_labels, rotation=18, ha="right")
        axis.set_title(CONTEXT_LABELS[context], fontweight="bold")
        axis.set_ylabel("Natural-prevalence performance")
        clean_axis(axis, "y")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    save(figure, output, "figureS3_natural_prevalence")

    # S4: preregistered unified comparison.
    unified = pd.read_csv(
        ROOT
        / "experiments/runs/unified/comparison_primary/MuscleSkeletal/pair_bootstrap_ci.csv"
    )
    unified = unified[
        (unified.candidate == "unified") & (unified.reference == "frozen_ensemble")
    ].copy()
    copy_source(unified, source, "figureS4_unified_vs_ensemble_ci.csv")
    shown = unified[
        unified.metric.isin(
            ["auroc", "auprc", "f1", "precision", "recall", "specificity", "brier"]
        )
    ].copy()
    order = ["auroc", "auprc", "f1", "precision", "recall", "specificity", "brier"]
    shown["order"] = shown.metric.map({x: i for i, x in enumerate(order)})
    shown = shown.sort_values("order", ascending=False)
    figure, axis = plt.subplots(figsize=(5.7, 3.0))
    y = np.arange(len(shown))
    colors = [
        COLORS["red"]
        if (row.metric != "brier" and row["median"] < 0)
        or (row.metric == "brier" and row["median"] > 0)
        else COLORS["green"]
        for _, row in shown.iterrows()
    ]
    for i, (_, row) in enumerate(shown.iterrows()):
        axis.errorbar(
            100 * row["median"],
            i,
            xerr=np.array(
                [
                    [100 * (row["median"] - row["lower"])],
                    [100 * (row["upper"] - row["median"])],
                ]
            ),
            fmt="o",
            color=colors[i],
            capsize=2.2,
        )
    axis.axvline(0, color=COLORS["ink"], lw=0.8)
    axis.set_yticks(
        y, [x.upper() if x != "brier" else "Brier (increase)" for x in shown.metric]
    )
    axis.set_xlabel("Unified − frozen ensemble (percentage points)")
    axis.set_title(
        "Preregistered unified model did not surpass fixed prediction-level integration",
        loc="left",
        fontweight="bold",
    )
    clean_axis(axis)
    figure.tight_layout()
    save(figure, output, "figureS4_unified_negative_result")

    # S5: canonical ROC and PR curves from saved curve source table.
    curves = pd.read_csv(ROOT / "figures/final/source_tables/final_roc_pr_curves.csv")
    copy_source(curves, source, "figureS5_locked_roc_pr_curves.csv")
    figure, axes = plt.subplots(2, 5, figsize=(7.0, 3.1), sharex="row", sharey="row")
    for column, context in enumerate(CONTEXTS):
        for row_index, curve in enumerate(["ROC", "PR"]):
            axis = axes[row_index, column]
            subset = curves[(curves.context == context) & (curves.curve == curve)]
            for model, color, style in [
                ("Published AdarEdit", COLORS["gray"], "--"),
                ("adarpro_pair_local_equal_logit_ensemble", COLORS["green"], "-"),
            ]:
                values = subset[subset.model == model]
                axis.plot(values.x, values.y, color=color, linestyle=style, lw=1.1)
            if curve == "ROC":
                axis.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.5, linestyle=":")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            clean_axis(axis, None)
            if row_index == 0:
                axis.set_title(CONTEXT_LABELS[context], fontsize=7.5, fontweight="bold")
            if column == 0:
                axis.set_ylabel("TPR" if curve == "ROC" else "Precision")
            if row_index == 1:
                axis.set_xlabel("Recall")
            elif row_index == 0:
                axis.set_xlabel("FPR")
    figure.legend(
        [
            plt.Line2D([0], [0], color=COLORS["gray"], linestyle="--"),
            plt.Line2D([0], [0], color=COLORS["green"]),
        ],
        ["AdarEdit", "ADARPro"],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
    )
    figure.suptitle(
        "Canonical held-out ROC and precision–recall curves",
        fontsize=10,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(figure, output, "figureS5_locked_curves")

    # S6: final calibration.
    calibration = pd.read_csv(
        ROOT / "figures/final/source_tables/final_calibration.csv"
    )
    copy_source(calibration, source, "figureS6_locked_calibration.csv")
    figure, axes = plt.subplots(1, 5, figsize=(7.0, 1.75), sharex=True, sharey=True)
    for axis, context in zip(axes, CONTEXTS):
        subset = calibration[calibration.context == context]
        for model, color, style in [
            ("Published AdarEdit", COLORS["gray"], "--"),
            ("adarpro_pair_local_equal_logit_ensemble", COLORS["green"], "-"),
        ]:
            values = subset[subset.model == model]
            axis.plot(
                values.mean_predicted_probability,
                values.observed_fraction,
                color=color,
                linestyle=style,
                marker="o",
                ms=2,
                lw=0.9,
            )
        axis.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.5, linestyle=":")
        axis.set_title(CONTEXT_LABELS[context], fontsize=7.3)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        clean_axis(axis, None)
    axes[0].set_ylabel("Observed")
    for axis in axes:
        axis.set_xlabel("Predicted")
    figure.suptitle("Canonical held-out calibration", fontsize=9.5, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save(figure, output, "figureS6_locked_calibration")

    # S7: tissue-dependent site changes over stable substrate rankings.
    variation_path = (
        ROOT / "experiments/runs/post_analysis/tissue_variation/per_site_variation.parquet"
    )
    switch_path = (
        ROOT / "experiments/runs/post_analysis/tissue_variation/strict_switch_matrix.csv"
    )
    stability_path = (
        ROOT
        / "experiments/runs/post_analysis/pair_stability/cross_tissue_pair_stability.csv"
    )
    variation = pd.read_parquet(variation_path)
    variation = variation[variation.observed_tissues == 4].copy()
    switches = pd.read_csv(switch_path)
    stability = pd.read_csv(stability_path)
    copy_source(
        pd.DataFrame(
            {
                "editing_index_range": np.sort(variation.editing_index_range),
                "fraction_at_or_above": 1
                - np.arange(len(variation), dtype=float) / len(variation),
            }
        ),
        source,
        "figureS7_editing_range_survival.csv",
    )

    figure = plt.figure(figsize=(7.0, 5.0))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.1],
        width_ratios=[1.05, 0.95],
        hspace=0.56,
        wspace=0.42,
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, :])

    ordered_range = np.sort(variation.editing_index_range.to_numpy(dtype=float))
    survival = 100 * (1 - np.arange(len(ordered_range)) / len(ordered_range))
    ax_a.plot(ordered_range, survival, color=COLORS["purple"], lw=1.7)
    for threshold in (1, 5, 10, 15):
        fraction = 100 * variation.editing_index_range.ge(threshold).mean()
        ax_a.scatter(threshold, fraction, s=24, color=COLORS["purple"], zorder=3)
        ax_a.text(
            threshold + 0.8,
            fraction + 1.5,
            f"{fraction:.1f}%",
            fontsize=6.7,
            color=COLORS["purple"],
            va="bottom",
        )
    ax_a.set_xlim(0, 50)
    ax_a.set_ylim(0, 102)
    ax_a.set_xlabel("Within-site EditingIndex range (percentage points)")
    ax_a.set_ylabel("Sites at or above range (%)")
    ax_a.set_title(
        "Matched sites change across tissues",
        loc="left",
        fontweight="bold",
    )
    ax_a.text(
        0.99,
        0.95,
        f"{len(variation):,} sites observed in all four tissues",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color=COLORS["gray"],
    )
    panel_label(ax_a, "A")
    clean_axis(ax_a, "both")

    tissues = ["Artery", "Brain", "Liver", "MuscleSkeletal"]
    tissue_labels = ["Artery", "Brain", "Liver", "Skeletal\nmuscle"]
    matrix = np.full((4, 4), np.nan)
    counts = np.zeros((4, 4), dtype=int)
    for row in switches.itertuples(index=False):
        i = tissues.index(row.edited_tissue)
        j = tissues.index(row.nonedited_tissue)
        matrix[i, j] = row.strict_switch_per_10000_shared
        counts[i, j] = row.strict_switch_sites
    image_plot = ax_b.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=np.nanmax(matrix))
    for i in range(4):
        for j in range(4):
            if i == j:
                ax_b.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#F1F3F4",
                        edgecolor="white",
                        lw=1,
                    )
                )
                continue
            color = "white" if matrix[i, j] > 0.48 * np.nanmax(matrix) else COLORS["ink"]
            ax_b.text(
                j,
                i,
                f"{counts[i, j]:,}\n({matrix[i, j]:.1f})",
                ha="center",
                va="center",
                fontsize=6.2,
                color=color,
                linespacing=1.0,
            )
    ax_b.set_xticks(range(4), tissue_labels)
    ax_b.set_yticks(range(4), tissue_labels)
    ax_b.set_xlabel("Nonedited tissue (<1%)")
    ax_b.set_ylabel("Edited tissue (≥15%)")
    ax_b.set_title("Strict state switches are asymmetric", loc="left", fontweight="bold")
    colorbar = figure.colorbar(image_plot, ax=ax_b, fraction=0.046, pad=0.04)
    colorbar.set_label("Switches per 10,000 shared sites", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6.4)
    panel_label(ax_b, "B")

    pairs = [
        ("Artery", "Brain"),
        ("Artery", "Liver"),
        ("Artery", "MuscleSkeletal"),
        ("Brain", "Liver"),
        ("Brain", "MuscleSkeletal"),
        ("Liver", "MuscleSkeletal"),
    ]
    pair_labels = [
        "Artery–Brain",
        "Artery–Liver",
        "Artery–Muscle",
        "Brain–Liver",
        "Brain–Muscle",
        "Liver–Muscle",
    ]
    y = np.arange(len(pairs))[::-1]
    metric_specs = [
        ("mean_editing_spearman", "Mean EditingIndex", COLORS["blue"], 0.11),
        ("high_fraction_spearman", "High-editing fraction", COLORS["teal"], -0.11),
    ]
    for metric, label, color, offset in metric_specs:
        ordered = []
        for first, second in pairs:
            row = stability[
                stability.first_tissue.eq(first)
                & stability.second_tissue.eq(second)
                & stability.metric.eq(metric)
            ]
            if len(row) != 1:
                raise RuntimeError(f"missing pair-stability row: {first}, {second}, {metric}")
            ordered.append(row.iloc[0])
        values = pd.DataFrame(ordered)
        ax_c.errorbar(
            values.spearman,
            y + offset,
            xerr=np.vstack(
                [values.spearman - values.lower, values.upper - values.spearman]
            ),
            fmt="o",
            ms=4.2,
            capsize=2,
            color=color,
            label=label,
        )
    ax_c.set_yticks(y, pair_labels)
    ax_c.set_xlim(0.5, 1.0)
    ax_c.axvline(0, color=COLORS["ink"], lw=0.6)
    ax_c.set_xlabel("Spearman correlation across Alu pairs (95% pair-bootstrap interval)")
    ax_c.set_title(
        "Substrate rankings remain correlated across tissues on identical site sets",
        loc="left",
        fontweight="bold",
    )
    ax_c.legend(frameon=False, ncol=2, loc="lower right")
    panel_label(ax_c, "C")
    clean_axis(ax_c)
    figure.suptitle(
        "Tissue-dependent editing overlays stable substrate rankings",
        fontsize=10.5,
        fontweight="bold",
        y=0.995,
    )
    figure.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.10)
    save(figure, output, "figureS7_tissue_context_and_pair_stability")

    # S8: post-hoc diagnostics of the selected statistical and local principles.
    within = pd.read_csv(
        ROOT / "experiments/runs/post_analysis/within_duplex/within_duplex_summary.csv"
    )
    pair_size = pd.read_csv(
        ROOT
        / "experiments/runs/richer_supervision/pair_balance_diagnostic/raw_site_count_quintiles.csv"
    )
    strata = pd.read_csv(
        ROOT
        / "experiments/runs/target_local_interpretation/MuscleSkeletal/geometry_strata.csv"
    )
    copy_source(pair_size, source, "figureS8_pair_size_quintiles.csv")
    copy_source(strata, source, "figureS8_target_structure_strata.csv")

    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    models = [
        "development_reference",
        "pair_first",
        "focal",
        "target_local",
        "fixed_ensemble",
    ]
    model_labels = ["Internal\nreference", "Pair-first", "Focal", "Target-\ncentered", "Fixed\nensemble"]
    x = np.arange(len(models))
    for context, color, marker in [
        ("Liver", COLORS["teal"], "o"),
        ("MuscleSkeletal", COLORS["amber"], "s"),
    ]:
        values = within[within.context.eq(context)].set_index("model").loc[models]
        ax_a.errorbar(
            x,
            values.mean_within_pair_auc,
            yerr=np.vstack(
                [
                    values.mean_within_pair_auc - values.lower,
                    values.upper - values.mean_within_pair_auc,
                ]
            ),
            marker=marker,
            ms=4,
            capsize=2,
            lw=1.1,
            color=color,
            label=CONTEXT_LABELS[context],
        )
    ax_a.set_xticks(x, model_labels)
    ax_a.set_ylim(0.87, 0.97)
    ax_a.set_ylabel("Mean within-pair AUROC")
    ax_a.set_title("Within-duplex discrimination", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, loc="lower right")
    panel_label(ax_a, "A")
    clean_axis(ax_a, "y")

    ax_b.plot(
        pair_size.raw_site_count_median,
        pair_size.pair_gain_vs_site_balanced,
        color=COLORS["green"],
        marker="o",
        ms=5,
        lw=1.4,
    )
    ax_b.axhline(0, color=COLORS["ink"], lw=0.7)
    for row in pair_size.itertuples(index=False):
        ax_b.text(
            row.raw_site_count_median,
            row.pair_gain_vs_site_balanced + (0.006 if row.pair_gain_vs_site_balanced >= 0 else -0.009),
            f"Q{int(row.raw_site_count_quintile) + 1}",
            ha="center",
            va="center",
            fontsize=6.5,
            color=COLORS["green"],
        )
    ax_b.set_xlabel("Median eligible sites per pair")
    ax_b.set_ylabel("Log-loss gain from pair-first\n(>0 favors pair-first)")
    ax_b.set_title("Pair-first learning redistributes gain", loc="left", fontweight="bold")
    panel_label(ax_b, "B")
    clean_axis(ax_b, "both")

    model_colors = {"local": COLORS["blue"], "ensemble": COLORS["purple"]}
    target = strata[strata.stratum.eq("target_paired")].copy()
    target["level"] = pd.Categorical(
        target["level"], categories=["unpaired", "paired"], ordered=True
    )
    target = target.sort_values(["level", "candidate"])
    levels = ["Unpaired", "Paired"]
    for index, candidate in enumerate(["local", "ensemble"]):
        values = target[target.candidate.eq(candidate)].sort_values("level")
        ax_c.scatter(
            np.arange(2) + (index - 0.5) * 0.12,
            values.mean_brier_difference,
            s=28,
            color=model_colors[candidate],
            label="Local model" if candidate == "local" else "Fixed ensemble",
        )
    ax_c.axhline(0, color=COLORS["ink"], lw=0.7)
    ax_c.set_xticks(range(2), levels)
    ax_c.set_ylabel("Candidate − pair-first Brier")
    ax_c.set_title("Gain is not a paired/unpaired split", loc="left", fontweight="bold")
    ax_c.legend(frameon=False, loc="lower left")
    panel_label(ax_c, "C")
    clean_axis(ax_c, "y")

    stem = strata[strata.stratum.eq("stem_extent")].copy()
    order = ["0", "1-5", "6-10", "11-20", ">20"]
    stem["level"] = pd.Categorical(stem["level"], categories=order, ordered=True)
    for index, candidate in enumerate(["local", "ensemble"]):
        values = stem[stem.candidate.eq(candidate)].sort_values("level")
        ax_d.plot(
            np.arange(len(order)) + (index - 0.5) * 0.08,
            values.mean_brier_difference,
            marker="o" if candidate == "local" else "s",
            ms=4,
            lw=1.0,
            color=model_colors[candidate],
            label="Local model" if candidate == "local" else "Fixed ensemble",
        )
    ax_d.axhline(0, color=COLORS["ink"], lw=0.7)
    ax_d.set_xticks(range(len(order)), order)
    ax_d.set_xlabel("Contiguous target stem extent (nt)")
    ax_d.set_ylabel("Candidate − pair-first Brier")
    ax_d.set_title("Benefit spans stem extents", loc="left", fontweight="bold")
    panel_label(ax_d, "D")
    clean_axis(ax_d, "y")
    figure.suptitle(
        "Post-hoc diagnostics support the selected learning and representation principles",
        fontsize=10.5,
        fontweight="bold",
        y=0.995,
    )
    figure.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.10, hspace=0.48, wspace=0.38)
    save(figure, output, "figureS8_post_analysis_diagnostics")


def write_manifest(source: Path, output: Path) -> None:
    rows = []
    for path in sorted(source.glob("*.csv")):
        if path.name == "figure_source_manifest.csv":
            continue
        rows.append(
            {
                "artifact": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "role": "quantitative figure source table",
            }
        )
    for path in sorted(output.glob("*.pdf")):
        rows.append(
            {
                "artifact": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "role": "rendered publication figure",
            }
        )
    for path in sorted(ASSET_DIR.glob("*.png")):
        rows.append(
            {
                "artifact": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "role": "curated non-quantitative GPT Image 1.5 sub-asset",
            }
        )
    for name in ("generation_manifest.json", "selection_manifest.json"):
        path = ASSET_DIR.parent / name
        rows.append(
            {
                "artifact": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "role": "generated-asset provenance",
            }
        )
    pd.DataFrame(rows).to_csv(source / "figure_source_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("paper/figures"))
    parser.add_argument("--source", type=Path, default=Path("paper/source_tables"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    source = args.source if args.source.is_absolute() else ROOT / args.source
    configure()
    figure1(output, source)
    figure_scope(output, source)
    figure2(output, source)
    figure3(output, source)
    figure4(output, source)
    supplementary_figures(output, source)
    write_manifest(source, output)
    print(f"PASS: rendered {len(list(output.glob('*.pdf')))} vector figures")


if __name__ == "__main__":
    main()
