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
        (0.02, 0.79),
        0.40,
        0.14,
        "Intrinsic duplex properties\nsequence + predicted\nsecondary structure",
        COLORS["pale_blue"],
        COLORS["blue"],
        6.9,
        "bold",
    )
    rounded_box(
        ax_a,
        (0.58, 0.79),
        0.40,
        0.14,
        "Cell / transcript state\nprocessing + ADAR\nregulators",
        COLORS["pale_amber"],
        COLORS["amber"],
        6.9,
        "bold",
    )
    rounded_box(
        ax_a,
        (0.12, 0.49),
        0.76,
        0.18,
        "",
        "#F2F5F6",
        COLORS["gray"],
        8.5,
        "bold",
    )
    ax_a.plot([0.22, 0.78], [0.565, 0.565], color=COLORS["navy"], lw=1.2)
    left = Polygon(
        [
            [0.24, 0.545],
            [0.41, 0.545],
            [0.41, 0.525],
            [0.47, 0.565],
            [0.41, 0.605],
            [0.41, 0.585],
            [0.24, 0.585],
        ],
        closed=True,
        color=COLORS["sky"],
        alpha=0.9,
    )
    right = Polygon(
        [
            [0.76, 0.545],
            [0.59, 0.545],
            [0.59, 0.525],
            [0.53, 0.565],
            [0.59, 0.605],
            [0.59, 0.585],
            [0.76, 0.585],
        ],
        closed=True,
        color=COLORS["teal"],
        alpha=0.9,
    )
    ax_a.add_patch(left)
    ax_a.add_patch(right)
    for x in np.linspace(0.44, 0.56, 7):
        ax_a.plot([x, 1.0 - x], [0.545, 0.585], color=COLORS["gray"], lw=0.65)
    ax_a.text(
        0.50,
        0.635,
        "Substrate available to editing",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    rounded_box(
        ax_a,
        (0.17, 0.21),
        0.66,
        0.16,
        "Target-local selectivity\nsequence · pairing · helix partner",
        COLORS["pale_green"],
        COLORS["green"],
        8.4,
        "bold",
    )
    rounded_box(
        ax_a,
        (0.28, 0.025),
        0.44,
        0.09,
        "Observed editing",
        "#F4F6F7",
        COLORS["gray"],
        8.2,
        "bold",
    )
    for start, end in [
        ((0.24, 0.79), (0.37, 0.67)),
        ((0.78, 0.79), (0.63, 0.67)),
        ((0.50, 0.49), (0.50, 0.38)),
        ((0.50, 0.21), (0.50, 0.13)),
    ]:
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
        0.25,
        0.735,
        "well captured by the\nwhole-duplex graph",
        ha="center",
        va="center",
        fontsize=6.6,
        color=COLORS["gray"],
    )
    ax_a.text(
        0.50,
        0.445,
        "which adenosine is selected?",
        ha="center",
        va="center",
        fontsize=6.6,
        color=COLORS["gray"],
    )
    ax_a.text(
        0.75,
        0.735,
        "bulk proxies:\nlimited resolution",
        ha="center",
        va="center",
        fontsize=6.3,
        color=COLORS["amber"],
    )
    ax_a.text(
        0.80,
        0.425,
        "target alignment\nimproves prediction",
        ha="center",
        va="center",
        fontsize=6.3,
        color=COLORS["green"],
    )
    ax_a.text(
        0.50,
        0.505,
        "editing burden + geometry provide an indirect readout",
        ha="center",
        va="center",
        fontsize=6.1,
        color=COLORS["gray"],
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)

    panel_label(ax_b, "B")
    ax_b.set_title("Questions tested", loc="left", fontweight="bold")
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
    ax_c.set_title("Leakage-safe evaluation", loc="left", fontweight="bold")
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
    figure = plt.figure(figsize=(7.0, 7.0))
    grid = figure.add_gridspec(
        2, 2, height_ratios=[1.48, 0.78], hspace=0.36, wspace=0.36
    )
    ax_a = figure.add_subplot(grid[0, :])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 1])

    panel_label(ax_a, "A")
    ax_a.set_axis_off()
    ax_a.set_title(
        "What adds to the sequence–structure graph?",
        loc="left",
        fontweight="bold",
    )
    headers = ["Tested formulation", "Evidence", "Main finding"]
    xpos = [0.025, 0.565, 0.735]
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
    y = 0.895
    row_number = 0
    for group, group_rows in frame.groupby("group", sort=False):
        ax_a.add_patch(
            plt.Rectangle((0, y - 0.012), 1, 0.030, color=COLORS["pale_blue"], lw=0)
        )
        ax_a.text(
            0.012,
            y + 0.002,
            group.upper(),
            fontsize=6.2,
            fontweight="bold",
            color=COLORS["navy"],
            va="center",
        )
        y -= 0.036
        for _, row in group_rows.iterrows():
            if row_number % 2 == 0:
                ax_a.add_patch(
                    plt.Rectangle((0, y - 0.016), 1, 0.033, color="#F7F8F9", lw=0)
                )
            ax_a.text(xpos[0], y, row["hypothesis"], fontsize=6.45, va="center")
            ax_a.scatter(
                [0.545],
                [y],
                s=27,
                marker=status_markers[row["status"]],
                color=status_colors[row["status"]],
                zorder=3,
            )
            display_status = (
                "Supported*"
                if row["status"] == "supported, not yet actionable"
                else row["status"].capitalize()
            )
            ax_a.text(
                xpos[1],
                y,
                display_status,
                fontsize=5.8,
                va="center",
                color=status_colors[row["status"]],
            )
            ax_a.text(
                xpos[2], y, row["interpretation"], fontsize=6.15, va="center"
            )
            y -= 0.0365
            row_number += 1
    ax_a.text(
        0.99,
        0.030,
        "See Table S2 for comparators, effect sizes, and intervals.",
        ha="right",
        va="bottom",
        fontsize=5.9,
        color=COLORS["gray"],
    )
    ax_a.text(
        0.99,
        0.006,
        "*Reproducible observation not yet useful for site prediction; deferred hypotheses were not tested.",
        ha="right",
        va="bottom",
        fontsize=5.6,
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
    rounded_box(
        axis,
        (0.01, 0.59),
        0.19,
        0.20,
        "All high/low sites\npair-first sampler",
        COLORS["pale_green"],
        COLORS["green"],
        7.3,
        "bold",
    )
    rounded_box(
        axis,
        (0.27, 0.68),
        0.33,
        0.19,
        "Component 1\nwhole-duplex GAT → logit",
        COLORS["pale_blue"],
        COLORS["blue"],
        7.2,
        "bold",
    )
    rounded_box(
        axis,
        (0.27, 0.35),
        0.43,
        0.23,
        "Component 2\nwhole-duplex GAT + aligned\ntarget/helix-partner CNN\nrepresentation fusion → logit",
        "#F1ECF7",
        COLORS["purple"],
        6.6,
        "bold",
    )
    rounded_box(
        axis,
        (0.78, 0.53),
        0.20,
        0.20,
        f"Equal-logit mean\n{weights[0]:.1f} / {weights[1]:.1f}",
        COLORS["pale_amber"],
        COLORS["amber"],
        7.4,
        "bold",
    )
    for start, end in [
        ((0.20, 0.69), (0.27, 0.77)),
        ((0.20, 0.69), (0.27, 0.47)),
        ((0.60, 0.77), (0.78, 0.65)),
        ((0.70, 0.47), (0.78, 0.65)),
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
        0.49,
        0.25,
        "independently trained components",
        ha="center",
        fontsize=7,
        color=COLORS["gray"],
    )
    axis.text(
        0.52,
        0.10,
        "No weight search · no stacker · no post-test fitting",
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
    ax_a.set_xticks(x, model_labels)
    ax_a.set_ylim(0.75, 0.905)
    ax_a.set_ylabel("Muscle development OOF")
    ax_a.set_title(
        "Label expansion and pair weighting both help",
        loc="left",
        fontweight="bold",
    )
    ax_a.legend(frameon=False, ncol=3, fontsize=6.2)
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
    ax_b.legend(frameon=False, ncol=3, loc="upper right")
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
    ax_c.tick_params(axis="x", labelsize=6.2)
    ax_c.set_ylim(0.64, 0.94)
    ax_c.set_ylabel("Development OOF performance")
    ax_c.set_title(
        "Target alignment resolves selectivity",
        loc="left",
        fontweight="bold",
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
                lw=1.2,
                label="Skeletal muscle",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=COLORS["gray"],
                lw=1.2,
                alpha=0.45,
                label="Liver",
            ),
        ],
        frameon=False,
        ncol=2,
        fontsize=6.2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
    )
    ax_d.set_xlim(0, 3.8)
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
    metrics, confidence, transitions, matched = final_tables(source)
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
        "Single-use canonical held-out evaluation",
        loc="left",
        fontweight="bold",
    )
    ax_a.legend(frameon=False, ncol=3, loc="lower right")
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
    ax_b.legend(frameon=False, ncol=2, fontsize=6.2)
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
    matched = matched.set_index("context").loc[CONTEXTS]
    avoided = -matched["median"]
    ax_e.errorbar(
        avoided,
        y,
        xerr=np.vstack([avoided + matched["upper"], -matched["lower"] - avoided]),
        fmt="o",
        ms=4.5,
        capsize=2,
        color=COLORS["blue"],
    )
    ax_e.axvline(0, color=COLORS["ink"], lw=0.7)
    ax_e.set_yticks(y, [CONTEXT_LABELS[x] for x in CONTEXTS])
    ax_e.invert_yaxis()
    ax_e.set_xlabel("False positives avoided at matched recall")
    ax_e.set_title(
        "Post-test descriptive matched-recall analysis",
        loc="left",
        fontweight="bold",
    )
    clean_axis(ax_e)
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
