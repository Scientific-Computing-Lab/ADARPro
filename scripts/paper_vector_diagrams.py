"""Code-defined scientific schematics; no raster, tracing, or image-model assets.

Coordinates describe conceptual secondary structure and information flow, not
an observed molecular conformation. Every mark has a stated semantic role.
These replacements require author review under the journal's AI-use policy.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Arc, Ellipse, PathPatch, Polygon
from matplotlib.path import Path

BLUE = "#2166AC"
TEAL = "#138D90"
GREEN = "#2A9D67"
PURPLE = "#7562A8"
GRAY = "#68737D"
INK = "#172126"


def curve(ax, vertices, color=BLUE, lw=1.2, alpha=1.0):
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(vertices) - 1)
    ax.add_patch(
        PathPatch(
            Path(vertices, codes),
            fill=False,
            color=color,
            lw=lw,
            capstyle="round",
            alpha=alpha,
        )
    )


def duplex(ax, *, local=False, graph=False, labels=False, sites=False):
    """Antiparallel arms of one RNA, joined by an unpaired loop; no DNA helix."""
    xs = np.linspace(0.09, 0.83, 23)
    top = 0.67 + 0.025 * np.cos(xs * 12)
    bottom = 0.30 + 0.020 * np.sin(xs * 10)
    # One schematic bulge makes the imperfect-duplex convention explicit.
    top[6:9] += [0.04, 0.11, 0.04]
    if local:
        ax.axvspan(0.35, 0.64, ymin=0.16, ymax=0.88, color=GREEN, alpha=0.085)
        for x in (0.35, 0.64):
            ax.plot([x, x], [0.16, 0.87], color=GREEN, lw=0.55, linestyle=(0, (2, 2)))
    ax.plot(xs, top, color=BLUE, lw=1.5, solid_capstyle="round")
    ax.plot(xs, bottom, color=TEAL, lw=1.5, solid_capstyle="round")
    curve(
        ax,
        [(xs[-1], top[-1]), (1.02, 0.77), (1.01, 0.15), (xs[-1], bottom[-1])],
        color=GRAY,
        lw=1.15,
    )
    for i, x in enumerate(xs):
        if i in (6, 7, 8, 17):
            continue
        ax.plot(
            [x, x],
            [bottom[i] + 0.035, top[i] - 0.035],
            color="#A3B4BE",
            lw=0.75 if graph else 0.6,
            zorder=1,
        )
    if graph:
        ax.scatter(xs, top, s=5, facecolor="white", edgecolor=BLUE, lw=0.5, zorder=3)
        ax.scatter(xs, bottom, s=5, facecolor="white", edgecolor=TEAL, lw=0.5, zorder=3)
    elif not sites:
        for x, y, color in zip(xs, top, [BLUE] * len(xs)):
            ax.plot([x, x], [y - 0.016, y + 0.016], color=color, lw=1.0)
    if local:
        target = 12
        x = xs[target]
        ax.plot([x, x], [bottom[target], top[target]], color=GREEN, lw=1.1, zorder=4)
        ax.scatter(
            [x, x],
            [top[target], bottom[target]],
            s=28 if labels else 14,
            facecolor=[GREEN, "white"],
            edgecolor=GREEN,
            lw=0.85,
            zorder=5,
        )
        if labels:
            ax.text(
                x, 0.93, "focal A", ha="center", va="center", fontsize=6.3, color=GREEN
            )
            ax.text(
                x,
                0.035,
                "predicted partner",
                ha="center",
                va="center",
                fontsize=6.0,
                color=GRAY,
            )
    if sites:
        for i, color, face in [
            (4, GREEN, GREEN),
            (12, BLUE, "white"),
            (19, GRAY, "white"),
        ]:
            ax.scatter(
                [xs[i]],
                [top[i]],
                s=29,
                facecolor=face,
                edgecolor=color,
                lw=0.95,
                zorder=6,
            )
    if labels and not local:
        ax.text(0.04, top[0], "5′", ha="right", va="center", fontsize=6.0, color=BLUE)
        ax.text(
            0.04, bottom[0], "3′", ha="right", va="center", fontsize=6.0, color=TEAL
        )


def context(ax, *, labels=False):
    """Unmeasured cell/transcript context: RNA in two compartments, no mechanism."""
    ax.add_patch(
        Ellipse(
            (0.5, 0.5), 0.88, 0.78, facecolor="#F5F8FA", edgecolor="#A9B8C2", lw=0.85
        )
    )
    ax.add_patch(
        Ellipse(
            (0.35, 0.51), 0.36, 0.51, facecolor="#E9F1F6", edgecolor="#A9B8C2", lw=0.65
        )
    )
    curve(ax, [(0.22, 0.55), (0.29, 0.76), (0.41, 0.25), (0.49, 0.52)], lw=1.25)
    curve(
        ax,
        [(0.59, 0.61), (0.68, 0.79), (0.75, 0.39), (0.83, 0.61)],
        color=TEAL,
        lw=1.25,
    )
    curve(
        ax,
        [(0.55, 0.34), (0.60, 0.49), (0.70, 0.13), (0.78, 0.35)],
        color=PURPLE,
        lw=1.25,
    )
    if labels:
        ax.text(0.34, 0.22, "RNA", ha="center", fontsize=6.1, color=BLUE)
        ax.text(0.74, 0.77, "RNA", ha="center", fontsize=6.1, color=TEAL)


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
    """Place native vector primitives; legacy asset names only identify concepts."""
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
