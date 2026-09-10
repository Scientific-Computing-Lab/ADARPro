"""Licensed biological illustrations and precise code-defined schematics.

Coordinates describe conceptual secondary structure and information flow, not
an observed molecular conformation. Every mark has a stated semantic role.
These replacements require author review under the journal's AI-use policy.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Arc, Polygon
from paper_licensed_art import place_artwork

BLUE = "#2166AC"
TEAL = "#138D90"
GREEN = "#2A9D67"
PURPLE = "#7562A8"
GRAY = "#68737D"
INK = "#172126"


def duplex(ax, *, local=False, graph=False, labels=False, sites=False):
    """Credited RNA artwork with vector annotations on identifiable bases."""
    box = (0.07, 0.18, 0.86, 0.65) if local and labels else (0.025, 0.07, 0.95, 0.86)
    x, y, w, h = place_artwork(ax, "local" if local else "rna", box)

    def point(sx, sy):
        vx, vy, vw, vh = (96, 168, 112, 64) if local else (0, 150, 400, 96)
        return x + w * (sx - vx) / vw, y + h * (1 - (sy - vy) / vh)

    if local:
        # A crop is an enlarged local representation, not a new fold. The source
        # base pair is highlighted in the adapted SVG itself.
        left, bottom = point(107, 227)
        right, top = point(200, 173)
        ax.plot(
            [left, left, right, right],
            [bottom + 0.03, bottom, bottom, bottom + 0.03],
            color=GREEN,
            lw=0.75,
        )
        ax.plot(
            [left, left, right, right],
            [top - 0.03, top, top, top - 0.03],
            color=GREEN,
            lw=0.75,
        )
        tx, ty = point(153.7, 186.4)
        px, py = point(153.7, 214.5)
        if labels:
            ax.annotate(
                "focal A",
                xy=(tx, ty),
                xytext=(tx, 0.97),
                fontsize=6.25,
                color=GREEN,
                ha="center",
                va="top",
                arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 0.65},
            )
            ax.annotate(
                "predicted partner",
                xy=(px, py),
                xytext=(px, 0.045),
                fontsize=5.9,
                color=GREEN,
                ha="center",
                va="bottom",
                arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 0.65},
            )
        else:
            # One readable letter identifies the highlighted adenosine even in
            # the smallest architecture/evidence-map placement.
            ax.text(
                tx,
                min(0.98, ty + 0.13),
                "A",
                color=GREEN,
                fontsize=6.1,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    elif sites:
        for sx, color, face in [
            (145.17, GREEN, GREEN),
            (205.02, BLUE, "white"),
            (288.52, GRAY, "white"),
        ]:
            px, py = point(sx, 183.9)
            ax.scatter(
                [px], [py], s=27, facecolor=face, edgecolor=color, lw=0.95, zorder=6
            )
    elif graph:
        # A sparse overlay identifies the nucleotide/backbone graph view.
        sx = np.array(
            [37.9, 69.63, 95.1, 145.17, 179.37, 205.02, 239.2, 264.85, 297.3, 314.8]
        )
        for sy in [183.9, 217.5]:
            px, py = point(sx, sy)
            ax.scatter(
                px,
                np.full_like(px, py),
                s=3.4,
                facecolor="white",
                edgecolor=BLUE,
                lw=0.4,
                zorder=5,
            )
    if labels and not local:
        for sy, label in [(160, "5′"), (239, "3′")]:
            px, py = point(8, sy)
            ax.text(
                px - 0.013, py, label, fontsize=6.1, color=BLUE, ha="right", va="center"
            )


def context(ax, *, labels=False):
    """Licensed compartment artwork containing generic RNA motifs."""
    x, y, w, h = place_artwork(ax, "cell", (0.025, 0.025, 0.95, 0.95))
    if labels:
        ax.text(
            x + w * 0.73,
            y + h * 0.34,
            "RNA",
            ha="center",
            va="center",
            color=BLUE,
            fontsize=6.2,
        )


def geometry(ax, *, labels=False):
    """Oppositely oriented Alu intervals and their genomic separation."""
    ax.plot([0.05, 0.95], [0.59, 0.59], color="#A9B8C2", lw=0.85)
    ax.add_patch(
        Polygon(
            [(0.07, 0.43), (0.30, 0.43), (0.37, 0.59), (0.30, 0.75), (0.07, 0.75)],
            color=BLUE,
            lw=0,
        )
    )
    ax.add_patch(
        Polygon(
            [(0.93, 0.43), (0.70, 0.43), (0.63, 0.59), (0.70, 0.75), (0.93, 0.75)],
            color=TEAL,
            lw=0,
        )
    )
    ax.plot([0.37, 0.37, 0.63, 0.63], [0.33, 0.26, 0.26, 0.33], color=GRAY, lw=0.8)
    if labels:
        ax.text(
            0.19, 0.59, "Alu", ha="center", va="center", color="white", fontsize=6.2
        )
        ax.text(
            0.81, 0.59, "Alu", ha="center", va="center", color="white", fontsize=6.2
        )
        ax.text(
            0.5, 0.10, "genomic gap", ha="center", va="center", fontsize=6.2, color=GRAY
        )


def learning(ax):
    """Equal pair weights within a class despite unequal eligible site counts."""
    for y, n in [(0.73, 5), (0.30, 10)]:
        xs = np.linspace(0.12, 0.65, n)
        ax.plot([0.10, 0.69], [y, y], color="#A9B8C2", lw=0.8)
        ax.scatter(xs, [y] * n, s=6, color=GREEN)
        ax.text(0.83, y, r"$w$", ha="center", va="center", fontsize=8.0, color=GREEN)


def draw_motif(axis, name, bounds, *, zorder=2, **_):
    """Place native vectors or credited source art; names identify concepts."""
    ax = axis.inset_axes(bounds, transform=axis.transAxes, zorder=zorder)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    width = axis.figure.get_figwidth() * axis.get_position().width * bounds[2]
    labels = width > 1.18
    if name in {
        "rna_substrate_editorial",
        "whole_duplex_graph_view",
        "benchmark_pair_unit_editorial",
        "target_local_editorial",
    }:
        duplex(
            ax,
            local=name == "target_local_editorial",
            graph=name == "whole_duplex_graph_view",
            labels=labels,
            sites=name == "benchmark_pair_unit_editorial",
        )
    elif name in {"search_context_scale", "matched_cell_state"}:
        context(ax, labels=labels)
    elif name in {"substrate_context_editorial", "substrate_organization"}:
        geometry(ax, labels=labels)
    elif name == "pair_weighting_editorial":
        learning(ax)
    elif name == "development_fold_pool_editorial":
        for i, color in enumerate([BLUE, "#67A9CF", TEAL, GREEN, PURPLE]):
            x = 0.04 + i * 0.19
            for y in [0.25, 0.51, 0.77]:
                ax.plot([x, x + 0.13], [y, y], color=color, lw=1.1)
    elif name == "frozen_specification_editorial":
        for i, label in enumerate(["code", "weights", "thresholds"]):
            y = 0.76 - i * 0.26
            ax.plot([0.08, 0.91], [y - 0.08, y - 0.08], color="#C4CED3", lw=0.7)
            ax.text(0.1, y, label, color=INK, fontsize=6.2, va="center")
        ax.add_patch(
            Arc((0.84, 0.79), 0.12, 0.20, theta1=0, theta2=180, color=GRAY, lw=0.85)
        )
    else:
        raise ValueError(f"No code-defined motif for {name}")
    return ax
