#!/usr/bin/env python3
"""Build the PNAS main and SI figures from authoritative retained artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
CONTEXT_LABELS = {
    "Artery": "Artery",
    "Brain": "Brain",
    "Liver": "Liver",
    "MuscleSkeletal": "Muscle skeletal",
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
            "font.family": "DejaVu Sans",
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


def save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        if suffix == "pdf":
            fixed_time = dt.datetime(2026, 8, 25, tzinfo=dt.timezone.utc)
            metadata: dict[str, object] = {
                "Creator": "ADARPro figure pipeline",
                "Producer": "Matplotlib",
                "CreationDate": fixed_time,
                "ModDate": fixed_time,
            }
        elif suffix == "svg":
            metadata = {"Creator": "ADARPro figure pipeline", "Date": "2026-08-25"}
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
    figure = plt.figure(figsize=(7.0, 5.5))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 0.92])
    ax_a = figure.add_subplot(grid[:, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 1])
    for axis in (ax_a, ax_b, ax_c):
        axis.set_axis_off()

    panel_label(ax_a, "A")
    ax_a.set_title(
        "Biological hierarchy of endogenous editing", loc="left", fontweight="bold"
    )
    rounded_box(
        ax_a,
        (0.06, 0.79),
        0.85,
        0.13,
        "Tissue / cellular state\nADAR activity · processing · occupancy",
        COLORS["pale_amber"],
        COLORS["amber"],
        8.4,
        "bold",
    )
    rounded_box(
        ax_a,
        (0.10, 0.48),
        0.77,
        0.20,
        "",
        COLORS["pale_blue"],
        COLORS["blue"],
        8.5,
        "bold",
    )
    ax_a.plot([0.19, 0.78], [0.58, 0.58], color=COLORS["navy"], lw=1.2)
    left = Polygon(
        [
            [0.21, 0.55],
            [0.40, 0.55],
            [0.40, 0.52],
            [0.46, 0.58],
            [0.40, 0.64],
            [0.40, 0.61],
            [0.21, 0.61],
        ],
        closed=True,
        color=COLORS["sky"],
        alpha=0.9,
    )
    right = Polygon(
        [
            [0.76, 0.55],
            [0.57, 0.55],
            [0.57, 0.52],
            [0.51, 0.58],
            [0.57, 0.64],
            [0.57, 0.61],
            [0.76, 0.61],
        ],
        closed=True,
        color=COLORS["teal"],
        alpha=0.9,
    )
    ax_a.add_patch(left)
    ax_a.add_patch(right)
    for x in np.linspace(0.43, 0.54, 7):
        ax_a.plot([x, 0.97 - x], [0.555, 0.605], color=COLORS["gray"], lw=0.65)
    ax_a.text(
        0.49,
        0.655,
        "Inverted-Alu RNA substrate",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    rounded_box(
        ax_a,
        (0.16, 0.16),
        0.65,
        0.17,
        "Target-centered view\nlocal sequence + helix partner",
        COLORS["pale_green"],
        COLORS["green"],
        8.4,
        "bold",
    )
    ax_a.scatter(
        [0.485],
        [0.285],
        s=85,
        color=COLORS["red"],
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )
    ax_a.text(
        0.485,
        0.285,
        "A",
        color="white",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        zorder=6,
    )
    for start, end in [((0.49, 0.79), (0.49, 0.69)), ((0.49, 0.48), (0.49, 0.34))]:
        ax_a.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                color=COLORS["gray"],
                lw=1.0,
            )
        )
    ax_a.text(
        0.49,
        0.735,
        "realization / engagement",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["gray"],
    )
    ax_a.text(
        0.49,
        0.405,
        "local chemical selectivity",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["gray"],
    )
    ax_a.text(
        0.49,
        0.06,
        "Prediction must respect both shared substrate state\nand the fate of a specific nucleotide.",
        ha="center",
        va="center",
        color=COLORS["navy"],
        fontsize=8,
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)

    panel_label(ax_b, "B")
    ax_b.set_title("Systematic hypothesis map", loc="left", fontweight="bold")
    categories = [
        (
            "Biological context",
            "RBP · transcript/repeat · chromatin · structure",
            COLORS["pale_amber"],
            COLORS["amber"],
        ),
        (
            "Supervision",
            "all high/low sites · continuous/ordinal labels",
            COLORS["pale_green"],
            COLORS["green"],
        ),
        (
            "Representation",
            "whole duplex · focal site · aligned partner",
            COLORS["pale_blue"],
            COLORS["blue"],
        ),
        (
            "Integration",
            "transfer · ensemble · unified end-to-end",
            "#F1ECF7",
            COLORS["purple"],
        ),
    ]
    for index, (title, detail, fill, edge) in enumerate(categories):
        y = 0.78 - 0.21 * index
        rounded_box(
            ax_b, (0.02, y), 0.96, 0.15, f"{title}\n{detail}", fill, edge, 7.6, "bold"
        )
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)

    panel_label(ax_c, "C")
    ax_c.set_title("Leakage-safe evidence funnel", loc="left", fontweight="bold")
    split_counts = benchmark_split_table(source).set_index("split")["pair_count"]
    split = {
        "Train": (int(split_counts["Train"]), COLORS["blue"]),
        "Validation": (int(split_counts["Validation"]), COLORS["teal"]),
        "Test": (int(split_counts["Test"]), COLORS["red"]),
    }
    total = sum(value[0] for value in split.values())
    x = 0.02
    for label, (count, color) in split.items():
        width = 0.94 * count / total
        ax_c.add_patch(
            FancyBboxPatch(
                (x, 0.67),
                width,
                0.15,
                boxstyle="round,pad=0.005",
                facecolor=color,
                edgecolor="white",
            )
        )
        split_label = "Valid." if label == "Validation" else label
        ax_c.text(
            x + width / 2,
            0.745,
            f"{split_label}\n{count}",
            ha="center",
            va="center",
            color="white",
            fontsize=5.8,
            fontweight="bold",
        )
        x += width
    rounded_box(
        ax_c,
        (0.02, 0.37),
        0.41,
        0.16,
        "Five pair-disjoint folds\nfeature/model selection",
        COLORS["pale_blue"],
        COLORS["blue"],
        7.6,
        "bold",
    )
    rounded_box(
        ax_c,
        (0.55, 0.37),
        0.41,
        0.16,
        "Freeze code, weights,\nthresholds, hashes",
        COLORS["pale_green"],
        COLORS["green"],
        7.6,
        "bold",
    )
    rounded_box(
        ax_c,
        (0.29, 0.07),
        0.41,
        0.16,
        "One canonical evaluation\nno post-test tuning",
        COLORS["pale_red"],
        COLORS["red"],
        7.6,
        "bold",
    )
    for start, end in [((0.43, 0.45), (0.55, 0.45)), ((0.76, 0.37), (0.61, 0.23))]:
        ax_c.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                color=COLORS["gray"],
                lw=0.9,
            )
        )
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    figure.subplots_adjust(
        left=0.04, right=0.99, top=0.97, bottom=0.04, wspace=0.22, hspace=0.25
    )
    save(figure, output, "figure1_systematic_dissection")


def evidence_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(
        label: str,
        group: str,
        status: str,
        source_path: str,
        metric: str = "auroc",
        invert: bool = False,
        **selectors: str,
    ) -> None:
        row = read_ci(source_path, metric, **selectors)
        lower, median, upper = (
            float(row["lower"]),
            float(row["median"]),
            float(row["upper"]),
        )
        if invert:
            lower, median, upper = -upper, -median, -lower
        rows.append(
            {
                "hypothesis": label,
                "group": group,
                "status": status,
                "metric": metric,
                "lower": lower,
                "median": median,
                "upper": upper,
                "source": source_path,
            }
        )

    add(
        "Pair-first expanded supervision",
        "Supervision",
        "selected",
        "experiments/runs/richer_supervision/pair_balance_comparison/pair_bootstrap_ci.csv",
        candidate="expanded_pair",
        reference="development_reference",
    )
    add(
        "Target-aligned local view",
        "Representation",
        "selected",
        "experiments/runs/richer_supervision/local_alignment_control/MuscleSkeletal/pair_bootstrap_ci.csv",
        candidate="aligned_local",
        reference="pair_balanced",
    )
    add(
        "Fixed global + local ensemble",
        "Integration",
        "selected",
        "experiments/runs/richer_supervision/pair_local_ensemble_comparison/MuscleSkeletal/pair_bootstrap_ci.csv",
        invert=True,
        candidate="pair_balanced",
        reference="pair_local_ensemble",
    )
    add(
        "Shared four-tissue training",
        "Integration",
        "limited",
        "experiments/runs/joint_multilabel/vs_independent/global_pair_bootstrap_ci.csv",
        context="macro",
    )
    add(
        "Continuous/ordinal auxiliary loss",
        "Supervision",
        "limited",
        "experiments/runs/richer_supervision/pair_balanced_multitask_comparison/pair_bootstrap_ci.csv",
        invert=True,
        candidate="expanded_pair",
        reference="multitask_pair",
    )
    add(
        "Substrate-engagement gate",
        "Biology × AI",
        "limited",
        "experiments/runs/hierarchical_gate/binary_natural/MuscleSkeletal/pair_bootstrap_ci.csv",
        model="expanded_plus_combined_engagement",
    )
    add(
        "IRAlu/transcript architecture",
        "Biology",
        "not promoted",
        "experiments/runs/genomic_residual_v2/MuscleSkeletal/pair_bootstrap_ci.csv",
        model="baseline_plus_geometry",
    )
    add(
        "H3K36me3 peak overlap",
        "Biology",
        "not promoted",
        "experiments/runs/chromatin_residual/Liver/pair_bootstrap_ci.csv",
        model="baseline_plus_h3k36me3_multiscale_occupancy",
    )
    add(
        "RBP suppressor panel",
        "Biology",
        "not promoted",
        "experiments/runs/residual_pilot/MuscleSkeletal/pair_bootstrap_ci.csv",
        model="baseline_plus_biological_suppressors",
    )
    add(
        "RiNALMo target embedding",
        "Representation",
        "not promoted",
        "experiments/runs/foundation_residual/rinalmo_mega_target/MuscleSkeletal/pair_bootstrap_ci.csv",
        candidate="foundation_target",
        reference="base_recalibrated",
    )
    add(
        "Substrate-wide all-A decoder",
        "Representation",
        "terminated",
        "experiments/runs/substrate_wide/joint_binary_partial_futility/MuscleSkeletal/pair_bootstrap_ci.csv",
        candidate="control",
        reference="pair_balanced",
    )
    add(
        "Unified end-to-end model",
        "Integration",
        "terminated",
        "experiments/runs/unified/comparison_primary/MuscleSkeletal/pair_bootstrap_ci.csv",
        candidate="unified",
        reference="frozen_ensemble",
    )
    return pd.DataFrame(rows)


def figure2(output: Path, source: Path) -> None:
    frame = evidence_table()
    copy_source(frame, source, "figure2_evidence_map.csv")
    colors = {
        "selected": COLORS["green"],
        "limited": COLORS["amber"],
        "not promoted": COLORS["gray"],
        "terminated": COLORS["red"],
    }
    frame = frame.iloc[::-1].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    y = np.arange(len(frame))
    for index, row in frame.iterrows():
        axis.errorbar(
            100 * row["median"],
            index,
            xerr=np.array(
                [
                    [100 * (row["median"] - row["lower"])],
                    [100 * (row["upper"] - row["median"])],
                ]
            ),
            fmt="o",
            ms=5.2,
            capsize=2.2,
            color=colors[row["status"]],
            ecolor=colors[row["status"]],
            zorder=3,
        )
    axis.axvline(0, color=COLORS["ink"], lw=0.8)
    axis.set_yticks(y, frame["hypothesis"])
    axis.set_xlabel("Pair-disjoint development ΔAUROC (percentage points)")
    axis.set_title(
        "Incremental information is concentrated in supervision and target-local representation",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlim(-8.5, 6.5)
    clean_axis(axis)
    from matplotlib.lines import Line2D

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            label=label.title(),
        )
        for label, color in colors.items()
    ]
    axis.legend(
        handles=legend,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.23),
    )
    axis.text(
        0.995,
        0.01,
        "95% Alu-pair bootstrap intervals; comparisons use their prespecified same-fold reference",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=COLORS["gray"],
    )
    figure.subplots_adjust(left=0.36, right=0.98, top=0.91, bottom=0.20)
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
    rounded_box(
        axis,
        (0.02, 0.60),
        0.26,
        0.20,
        "All high/low sites\npair-first sampler",
        COLORS["pale_green"],
        COLORS["green"],
        7.3,
        "bold",
    )
    rounded_box(
        axis,
        (0.38, 0.72),
        0.27,
        0.15,
        "Whole-duplex GAT\nglobal representation",
        COLORS["pale_blue"],
        COLORS["blue"],
        7.2,
        "bold",
    )
    rounded_box(
        axis,
        (0.38, 0.49),
        0.27,
        0.15,
        "Aligned local CNN\ntarget + helix partner",
        "#F1ECF7",
        COLORS["purple"],
        7.2,
        "bold",
    )
    rounded_box(
        axis,
        (0.76, 0.60),
        0.21,
        0.20,
        f"Equal-logit mean\n{weights[0]:.1f} / {weights[1]:.1f}",
        COLORS["pale_amber"],
        COLORS["amber"],
        7.4,
        "bold",
    )
    for start, end in [
        ((0.28, 0.70), (0.38, 0.79)),
        ((0.28, 0.70), (0.38, 0.56)),
        ((0.65, 0.79), (0.76, 0.70)),
        ((0.65, 0.56), (0.76, 0.70)),
    ]:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                color=COLORS["gray"],
                lw=0.9,
            )
        )
    axis.text(
        0.51,
        0.31,
        "independently trained components",
        ha="center",
        fontsize=7,
        color=COLORS["gray"],
    )
    axis.text(
        0.51,
        0.17,
        "No ensemble-weight search",
        ha="center",
        fontsize=8,
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
    alignment_metrics = pd.read_csv(
        ROOT
        / "experiments/runs/richer_supervision/local_alignment_control/MuscleSkeletal/metrics.csv"
    )
    copy_source(pair, source, "figure3_pair_balance_ci.csv")
    copy_source(integration, source, "figure3_integration_ci.csv")
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
                    "quantity": "target_local_logit_weight",
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

    figure = plt.figure(figsize=(7.0, 7.1))
    grid = figure.add_gridspec(
        3, 2, height_ratios=[0.72, 1.05, 0.72], hspace=0.55, wspace=0.38
    )
    ax_a = figure.add_subplot(grid[0, :])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 1])
    ax_d = figure.add_subplot(grid[2, :])
    architecture_panel(ax_a, weights)
    panel_label(ax_a, "A")

    panel_label(ax_b, "B")
    metric_colors = {
        "auroc": COLORS["blue"],
        "auprc": COLORS["teal"],
        "f1": COLORS["green"],
    }
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
    ax_b.set_xlabel("Pair-balanced − internal reference (points)")
    ax_b.set_title("Pair-first expanded supervision", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, ncol=3, loc="lower right")
    clean_axis(ax_b)

    panel_label(ax_c, "C")
    shown = alignment_metrics[
        alignment_metrics.model.isin(
            ["pair_balanced", "aligned_local", "shifted_local"]
        )
    ].copy()
    labels = {
        "pair_balanced": "Global pair",
        "aligned_local": "Aligned local",
        "shifted_local": f"Shifted +{shift_nt} nt",
    }
    x = np.arange(3)
    width = 0.24
    for index, metric in enumerate(["auroc", "auprc", "f1"]):
        values = [shown.loc[shown.model == model, metric].iloc[0] for model in labels]
        ax_c.bar(
            x + (index - 1) * width,
            values,
            width=width,
            color=metric_colors[metric],
            label=metric.upper(),
            zorder=2,
        )
    ax_c.set_xticks(x, labels.values(), rotation=16, ha="right")
    ax_c.set_ylim(0.64, 0.94)
    ax_c.set_ylabel("Development OOF performance")
    ax_c.set_title(
        "Alignment, not added capacity, carries signal", loc="left", fontweight="bold"
    )
    clean_axis(ax_c, "y")
    ax_c.legend(frameon=False, ncol=3, loc="upper right")

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
                alpha=1 if j == 0 else 0.55,
            )
    ax_d.axvline(0, color=COLORS["ink"], lw=0.7)
    ax_d.set_yticks(range(3), metrics)
    ax_d.set_xlabel(
        "Fixed ensemble − global pair model (percentage points; 95% pair CI)"
    )
    ax_d.set_title(
        "Global and target-local predictors are complementary",
        loc="left",
        fontweight="bold",
    )
    ax_d.text(
        0.99,
        0.04,
        "Muscle Skeletal (solid)    Liver (light)",
        transform=ax_d.transAxes,
        ha="right",
        fontsize=7,
        color=COLORS["gray"],
    )
    ax_d.set_xlim(0, 3.6)
    clean_axis(ax_d)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.08)
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
    metrics, confidence, transitions, matched = final_tables(source)
    figure = plt.figure(figsize=(7.0, 6.0))
    grid = figure.add_gridspec(
        2, 3, height_ratios=[1.12, 0.88], hspace=0.48, wspace=0.52
    )
    ax_a = figure.add_subplot(grid[0, :])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 1])
    ax_d = figure.add_subplot(grid[1, 2])

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
    ax_a.set_xlabel("ADARPro − published AdarEdit (percentage points; 95% pair CI)")
    ax_a.set_title(
        "One-use canonical test: improvement in every endpoint",
        loc="left",
        fontweight="bold",
    )
    ax_a.legend(frameon=False, ncol=3, loc="lower right")
    clean_axis(ax_a)

    panel_label(ax_b, "B")
    x = np.arange(len(CONTEXTS))
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
    ax_b.plot(x, base.f1, "o--", color=COLORS["gray"], lw=1.2, ms=4, label="AdarEdit")
    ax_b.plot(x, cand.f1, "o-", color=COLORS["green"], lw=1.5, ms=4, label="ADARPro")
    ax_b.set_xticks(x, ["Art.", "Brain", "Liver", "Muscle", "Comb."], rotation=0)
    ax_b.set_ylim(0.79, 0.905)
    ax_b.set_ylabel("F1")
    ax_b.set_title("Absolute held-out F1", loc="left", fontweight="bold")
    ax_b.legend(frameon=False)
    clean_axis(ax_b, "y")

    panel_label(ax_c, "C")
    transitions = transitions.set_index("context").loc[CONTEXTS]
    y = np.arange(len(CONTEXTS))
    ax_c.barh(
        y - 0.15,
        100 * transitions.fp_rescue_rate,
        height=0.27,
        color=COLORS["teal"],
        label="Baseline FP rescued",
    )
    ax_c.barh(
        y + 0.15,
        100 * transitions.tp_damage_rate,
        height=0.27,
        color=COLORS["red"],
        label="Baseline TP damaged",
    )
    ax_c.set_yticks(y, ["Art.", "Brain", "Liver", "Muscle", "Comb."])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Sites (%) at frozen threshold")
    ax_c.set_title("FP rescue vs. TP damage", loc="left", fontweight="bold")
    ax_c.legend(frameon=False, loc="upper right", fontsize=6.2)
    clean_axis(ax_c)
    ax_c.set_xlim(0, 56)

    panel_label(ax_d, "D")
    matched = matched.set_index("context").loc[CONTEXTS]
    avoided = -matched["median"]
    ax_d.errorbar(
        avoided,
        y,
        xerr=np.vstack([avoided + matched["upper"], -matched["lower"] - avoided]),
        fmt="o",
        ms=4.5,
        capsize=2,
        color=COLORS["blue"],
    )
    ax_d.axvline(0, color=COLORS["ink"], lw=0.7)
    ax_d.set_yticks(y, ["Art.", "Brain", "Liver", "Muscle", "Comb."])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("False positives avoided")
    ax_d.set_title("Matched-recall FP reduction", loc="left", fontweight="bold")
    clean_axis(ax_d)
    figure.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.10)
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
    labels = ["Internal\nreference", "Pair-first", "Target-local", "Fixed\nensemble"]
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
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)
    for axis, metric in zip(axes, ["auroc", "auprc", "f1"]):
        subset = negative[negative.metric == metric].iloc[::-1]
        y = np.arange(len(subset))
        axis.errorbar(
            100 * subset["median"],
            y,
            xerr=np.vstack(
                [
                    100 * (subset["median"] - subset["lower"]),
                    100 * (subset["upper"] - subset["median"]),
                ]
            ),
            fmt="o",
            ms=4,
            capsize=2,
            color=COLORS["gray"],
        )
        axis.axvline(0, color=COLORS["ink"], lw=0.7)
        axis.set_title(metric.upper())
        axis.set_xlabel("Δ points")
        axis.set_yticks(y, subset.feature if axis is axes[0] else [])
        clean_axis(axis)
    figure.suptitle(
        "Tested bulk biological annotations did not add robust residual information",
        fontweight="bold",
        fontsize=10,
    )
    figure.tight_layout()
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
    model_labels = ["Internal ref.", "Pair-first", "Local", "Ensemble"]
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
    figure2(output, source)
    figure3(output, source)
    figure4(output, source)
    supplementary_figures(output, source)
    write_manifest(source, output)
    print(f"PASS: rendered {len(list(output.glob('*.pdf')))} vector figures")


if __name__ == "__main__":
    main()
