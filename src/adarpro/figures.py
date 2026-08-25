"""Generate publication-quality figures from saved ADARpro result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TISSUES = ["Artery", "Brain", "Liver", "MuscleSkeletal"]
PALETTE = {
    "baseline": "#4C78A8",
    "context": "#E45756",
    "control": "#9D9DA1",
    "positive": "#2A9D8F",
}


def configure() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(figure)


def switch_matrix(root: Path, output: Path, source: Path) -> list[str]:
    path = root / "experiments/runs/cross_tissue_switch/switch_comparisons.parquet"
    if not path.exists():
        return []
    comparisons = pd.read_parquet(
        path, columns=["tissue_a", "tissue_b", "label_a_edited"]
    )
    edited = np.where(
        comparisons["label_a_edited"] == 1,
        comparisons["tissue_a"],
        comparisons["tissue_b"],
    )
    nonedited = np.where(
        comparisons["label_a_edited"] == 1,
        comparisons["tissue_b"],
        comparisons["tissue_a"],
    )
    counts = pd.crosstab(
        pd.Categorical(edited, categories=TISSUES),
        pd.Categorical(nonedited, categories=TISSUES),
        dropna=False,
    )
    counts.index.name = "edited_tissue"
    counts.columns.name = "nonedited_tissue"
    counts.to_csv(source / "cross_tissue_switch_matrix.csv")
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    mask = np.eye(len(TISSUES), dtype=bool)
    sns.heatmap(
        counts,
        mask=mask,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar_kws={"label": "strict-switch comparisons"},
        linewidths=0.5,
        ax=axis,
    )
    axis.set_title("Direction of strict editing switches")
    axis.set_xlabel("Tissue with <1% editing")
    axis.set_ylabel("Tissue with ≥15% editing")
    save_figure(figure, output, "cross_tissue_switch_matrix")
    return [str(path)]


def continuous_ablation(root: Path, output: Path, source: Path) -> list[str]:
    path = root / "experiments/runs/continuous_tissue_differences/pair_bootstrap_ci.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path)
    table = table[(table["metric"] == "rmse") & (table["model"] != "tissue_pair")].copy()
    names = {
        "tissue_pair_plus_negative_controls": "Negative-control RBPs",
        "tissue_pair_plus_eclip": "eCLIP × expression",
        "tissue_pair_plus_matched_random": "Matched random",
        "tissue_pair_plus_prismnet": "PrismNet × expression",
        "tissue_pair_plus_suppressors": "Prior suppressors",
        "tissue_pair_plus_rbpnet": "RBPNet × expression",
        "tissue_pair_plus_all_context": "All RBP context",
        "tissue_pair_plus_all_rbp": "All RBP context",
        "binding_without_expression": "Binding without expression",
    }
    table["label"] = table["model"].map(names).fillna(table["model"])
    table = table.sort_values("median")
    table.to_csv(source / "continuous_ablation_rmse_bootstrap_ci.csv", index=False)
    figure, axis = plt.subplots(figsize=(6.0, 4.4))
    positions = np.arange(len(table))
    errors = np.vstack(
        [table["median"] - table["lower"], table["upper"] - table["median"]]
    )
    colors = [PALETTE["control"] if "control" in value.lower() or "random" in value.lower() else PALETTE["context"] for value in table["label"]]
    for index, color in enumerate(colors):
        axis.errorbar(
            table["median"].iloc[index],
            positions[index],
            xerr=errors[:, index : index + 1],
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=2.5,
        )
    axis.scatter(table["median"], positions, c=colors, s=25, zorder=3)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(positions, table["label"])
    axis.set_xlabel(
        "Δ RMSE vs tissue-pair baseline (percentage points; negative = improvement)"
    )
    axis.set_title("RBP context does not improve matched editing differences")
    save_figure(figure, output, "continuous_tissue_ablation")
    return [str(path)]


def eclip_coverage(root: Path, output: Path, source: Path) -> list[str]:
    path = root / "results/adarpro/rbp_feature_audit/eclip_gene_coverage.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path).sort_values("duplex_any_fraction")
    table.to_csv(source / "eclip_gene_coverage.csv", index=False)
    figure, axis = plt.subplots(figsize=(6.2, 4.6))
    axis.barh(table["gene"], 100 * table["duplex_any_fraction"], color=PALETTE["baseline"])
    axis.set_xlabel("Development substrates with any eCLIP peak (%)")
    axis.set_ylabel("")
    axis.set_title("Sparse experimental occupancy in inverted-Alu substrates")
    axis.text(
        0.99,
        0.02,
        "Absence is censored by repeat filtering/multi-mapping",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    save_figure(figure, output, "eclip_duplex_coverage")
    return [str(path)]


def residual_ablation(root: Path, output: Path, source: Path) -> list[str]:
    base = root / "experiments/runs/residual_pilot"
    if not base.exists():
        return []
    paths = sorted(base.glob("*/pair_bootstrap_ci.csv"))
    if not paths:
        return []
    tables = []
    for path in paths:
        table = pd.read_csv(path)
        table["context"] = path.parent.name
        tables.append(table)
    all_ci = pd.concat(tables, ignore_index=True)
    selected = all_ci[
        (all_ci["metric"] == "auroc")
        & all_ci["model"].isin(
            [
                "baseline_logit",
                "baseline_plus_eclip",
                "baseline_plus_predicted_binding",
                "baseline_plus_prismnet",
                "baseline_plus_binding_expression",
                "baseline_plus_biological_suppressors",
                "baseline_plus_broad_rbp",
                "baseline_plus_matched_random",
                "baseline_plus_shifted_eclip_control",
            ]
        )
    ].copy()
    if selected.empty:
        return []
    selected.to_csv(source / "residual_auroc_bootstrap_ci.csv", index=False)
    model_order = selected.groupby("model")["median"].mean().sort_values().index
    contexts = list(dict.fromkeys(selected["context"]))
    figure, axes = plt.subplots(1, len(contexts), figsize=(5.2 * len(contexts), 4.7), squeeze=False)
    for axis, context in zip(axes[0], contexts):
        subset = selected[selected["context"] == context].set_index("model").reindex(model_order).dropna()
        y = np.arange(len(subset))
        errors = np.vstack([subset["median"] - subset["lower"], subset["upper"] - subset["median"]])
        axis.errorbar(subset["median"], y, xerr=errors, fmt="o", color=PALETTE["context"], capsize=2)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y, [value.replace("baseline_plus_", "").replace("_", " ") for value in subset.index])
        axis.set_xlabel("Pair-bootstrap Δ AUROC vs raw OOF baseline")
        axis.set_title(context)
    save_figure(figure, output, "residual_model_ablation")
    return [str(path) for path in paths]


def pair_engagement(root: Path, output: Path, source: Path) -> list[str]:
    base = root / "experiments/runs/pair_engagement"
    paths = [base / context / "oof_pair_predictions.parquet" for context in TISSUES]
    if not all(path.exists() for path in paths):
        return []
    frames = []
    ci_frames = []
    for context, path in zip(TISSUES, paths):
        predictions = pd.read_parquet(path)
        predictions = predictions[predictions["model"] == "ridge_intrinsic"].copy()
        predictions["context"] = context
        frames.append(predictions[["context", "pair_id", "edited_fraction", "prediction"]])
        confidence = pd.read_csv(path.parent / "pair_bootstrap_ci.csv")
        confidence = confidence[
            (confidence["model"] == "ridge_intrinsic")
            & confidence["metric"].isin(["r2", "spearman", "any_edited_auroc"])
        ].copy()
        confidence["context"] = context
        ci_frames.append(confidence)
    values = pd.concat(frames, ignore_index=True)
    confidence = pd.concat(ci_frames, ignore_index=True)
    values.to_csv(source / "pair_engagement_oof_predictions.csv", index=False)
    confidence.to_csv(source / "pair_engagement_bootstrap_ci.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for context, color in zip(TISSUES, sns.color_palette("colorblind", len(TISSUES))):
        subset = values[values["context"] == context]
        axes[0].scatter(
            subset["edited_fraction"],
            subset["prediction"],
            s=8,
            alpha=0.35,
            color=color,
            label=context.replace("MuscleSkeletal", "Muscle"),
            rasterized=True,
        )
    axes[0].plot([0, 0.7], [0, 0.7], color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlim(-0.02, 0.7)
    axes[0].set_ylim(-0.02, 0.7)
    axes[0].set_xlabel("Observed highly edited-site fraction")
    axes[0].set_ylabel("OOF predicted fraction")
    axes[0].set_title("Substrate engagement is predictable")
    axes[0].legend(frameon=False, markerscale=1.8)

    metric = "spearman"
    selected = confidence[confidence["metric"] == metric].set_index("context").loc[TISSUES]
    positions = np.arange(len(selected))
    errors = np.vstack(
        [selected["median"] - selected["lower"], selected["upper"] - selected["median"]]
    )
    axes[1].errorbar(
        positions,
        selected["median"],
        yerr=errors,
        fmt="o",
        color=PALETTE["positive"],
        capsize=3,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions, [value.replace("MuscleSkeletal", "Muscle") for value in TISSUES], rotation=25, ha="right")
    axes[1].set_ylabel("Δ Spearman vs fold training mean")
    axes[1].set_title("Pair-bootstrap 95% intervals")
    save_figure(figure, output, "pair_engagement_prediction")
    return [str(path) for path in paths]


def richer_supervision(root: Path, output: Path, source: Path) -> list[str]:
    base = root / "experiments/runs/richer_supervision/pair_balance_comparison"
    metrics_path = base / "metrics.csv"
    confidence_path = base / "pair_bootstrap_ci.csv"
    if not metrics_path.exists() or not confidence_path.exists():
        return []
    metrics = pd.read_csv(metrics_path)
    confidence = pd.read_csv(confidence_path)
    metrics.to_csv(source / "richer_supervision_metrics.csv", index=False)
    confidence.to_csv(
        source / "richer_supervision_pair_bootstrap_ci.csv", index=False
    )

    labels = {
        "development_reference": "AdarEdit dev reference",
        "expanded_site": "Expanded, site-sampled",
        "expanded_pair": "Expanded, pair-sampled",
    }
    colors = {
        "development_reference": PALETTE["baseline"],
        "expanded_site": "#F2A65A",
        "expanded_pair": PALETTE["positive"],
    }
    displayed_metrics = ["auroc", "auprc", "f1"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))
    positions = np.arange(len(displayed_metrics))
    width = 0.24
    for offset, model in zip((-1, 0, 1), labels):
        row = metrics[metrics["model"] == model].iloc[0]
        axes[0].bar(
            positions + offset * width,
            [row[metric] for metric in displayed_metrics],
            width=width,
            color=colors[model],
            label=labels[model],
        )
    axes[0].set_xticks(positions, [value.upper() for value in displayed_metrics])
    axes[0].set_ylim(0.75, 0.91)
    axes[0].set_ylabel("Grouped OOF metric")
    axes[0].set_title("Pair-first supervision improves every metric")
    axes[0].legend(frameon=False, loc="lower left")

    versus_development = confidence[
        (confidence["candidate"] == "expanded_pair")
        & (confidence["reference"] == "development_reference")
        & confidence["metric"].isin(displayed_metrics)
    ].copy()
    versus_development["comparison"] = "vs AdarEdit dev reference"
    versus_site = confidence[
        (confidence["candidate"] == "expanded_site")
        & (confidence["reference"] == "expanded_pair")
        & confidence["metric"].isin(displayed_metrics)
    ].copy()
    old_lower = versus_site["lower"].copy()
    versus_site["lower"] = -versus_site["upper"]
    versus_site["median"] = -versus_site["median"]
    versus_site["upper"] = -old_lower
    versus_site["comparison"] = "vs site-sampled expanded"
    contrasts = pd.concat([versus_development, versus_site], ignore_index=True)
    metric_order = {name: index for index, name in enumerate(displayed_metrics)}
    contrasts["metric_order"] = contrasts["metric"].map(metric_order)
    contrast_order = {
        "vs AdarEdit dev reference": -0.12,
        "vs site-sampled expanded": 0.12,
    }
    for comparison, color in (
        ("vs AdarEdit dev reference", PALETTE["positive"]),
        ("vs site-sampled expanded", "#F2A65A"),
    ):
        selected = contrasts[contrasts["comparison"] == comparison].sort_values(
            "metric_order"
        )
        y = selected["metric_order"].to_numpy() + contrast_order[comparison]
        errors = np.vstack(
            [
                selected["median"] - selected["lower"],
                selected["upper"] - selected["median"],
            ]
        )
        axes[1].errorbar(
            selected["median"],
            y,
            xerr=errors,
            fmt="o",
            color=color,
            capsize=3,
            label=comparison,
        )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(positions, [value.upper() for value in displayed_metrics])
    axes[1].set_xlabel("Pair-bootstrap metric difference (95% interval)")
    axes[1].set_title("Gain survives pair-clustered uncertainty")
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
    )
    figure.suptitle("Development folds only; canonical test remains sealed", y=1.01)
    save_figure(figure, output, "richer_supervision_frontier")
    return [str(metrics_path), str(confidence_path)]


def target_local_frontier(root: Path, output: Path, source: Path) -> list[str]:
    """Plot the frozen target-local comparison on canonical and natural rows."""

    canonical = (
        root
        / "experiments/runs/richer_supervision/"
        "pair_local_ensemble_comparison/MuscleSkeletal"
    )
    natural = (
        root
        / "experiments/runs/richer_supervision/natural_prevalence/"
        "pair_local_comparison/MuscleSkeletal"
    )
    paths = [
        canonical / "metrics.csv",
        canonical / "pair_bootstrap_ci.csv",
        natural / "metrics.csv",
        natural / "pair_bootstrap_ci.csv",
    ]
    if not all(path.exists() for path in paths):
        return []
    canonical_metrics = pd.read_csv(paths[0])
    canonical_ci = pd.read_csv(paths[1])
    natural_metrics = pd.read_csv(paths[2])
    natural_ci = pd.read_csv(paths[3])
    canonical_metrics.to_csv(source / "target_local_canonical_metrics.csv", index=False)
    canonical_ci.to_csv(source / "target_local_canonical_pair_bootstrap_ci.csv", index=False)
    natural_metrics.to_csv(source / "target_local_natural_metrics.csv", index=False)
    natural_ci.to_csv(source / "target_local_natural_pair_bootstrap_ci.csv", index=False)

    labels = {
        "development_reference": "AdarEdit dev reference",
        "pair_balanced": "Expanded, pair-sampled",
        "local": "Target-local fusion",
        "pair_local_ensemble": "Fixed pair + local",
    }
    colors = {
        "development_reference": PALETTE["baseline"],
        "pair_balanced": "#F2A65A",
        "local": "#5C80BC",
        "pair_local_ensemble": PALETTE["positive"],
    }
    displayed_metrics = ["auroc", "auprc", "f1"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    positions = np.arange(len(displayed_metrics))
    width = 0.19
    offsets = np.arange(-1.5, 2.0, 1.0) * width
    for axis, table, title, limits in (
        (
            axes[0],
            canonical_metrics,
            "Canonical development rows",
            (0.75, 0.93),
        ),
        (
            axes[1],
            natural_metrics,
            "Natural-prevalence development rows",
            (0.20, 0.95),
        ),
    ):
        for offset, model in zip(offsets, labels):
            row = table[table["model"] == model].iloc[0]
            axis.bar(
                positions + offset,
                [row[metric] for metric in displayed_metrics],
                width=width,
                color=colors[model],
                label=labels[model],
            )
        axis.set_xticks(positions, [metric.upper() for metric in displayed_metrics])
        axis.set_ylim(*limits)
        axis.set_ylabel("Grouped OOF metric")
        axis.set_title(title)

    def favorable_rows(table: pd.DataFrame, candidate: str) -> pd.DataFrame:
        selected = table[
            (table["candidate"] == "pair_balanced")
            & (table["reference"] == candidate)
            & table["metric"].isin(displayed_metrics)
        ].copy()
        old_lower = selected["lower"].copy()
        selected["lower"] = -selected["upper"]
        selected["median"] = -selected["median"]
        selected["upper"] = -old_lower
        order = {metric: index for index, metric in enumerate(displayed_metrics)}
        return selected.assign(metric_order=selected["metric"].map(order)).sort_values(
            "metric_order"
        )

    contrasts = [
        (favorable_rows(canonical_ci, "local"), -0.20, "Local, canonical", "#5C80BC"),
        (
            favorable_rows(canonical_ci, "pair_local_ensemble"),
            -0.07,
            "Ensemble, canonical",
            PALETTE["positive"],
        ),
        (favorable_rows(natural_ci, "local"), 0.07, "Local, natural", "#7AA6DC"),
        (
            favorable_rows(natural_ci, "pair_local_ensemble"),
            0.20,
            "Ensemble, natural",
            "#52B69A",
        ),
    ]
    for selected, offset, label, color in contrasts:
        errors = np.vstack(
            [
                selected["median"] - selected["lower"],
                selected["upper"] - selected["median"],
            ]
        )
        axes[2].errorbar(
            selected["median"],
            selected["metric_order"] + offset,
            xerr=errors,
            fmt="o",
            color=color,
            capsize=3,
            label=label,
        )
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_yticks(positions, [metric.upper() for metric in displayed_metrics])
    axes[2].set_xlabel("Gain over pair-sampled model (95% pair CI)")
    axes[2].set_title("Incremental target-local signal")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.04),
    )
    axes[2].legend(frameon=False, loc="lower right")
    figure.suptitle("Development folds only; canonical test remains sealed", y=1.02)
    figure.subplots_adjust(bottom=0.24, wspace=0.34)
    save_figure(figure, output, "target_local_frontier")
    return [str(path) for path in paths]


def run(root: Path, output_dir: Path) -> dict:
    configure()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "source_tables"
    source.mkdir(exist_ok=True)
    generators = [
        switch_matrix,
        continuous_ablation,
        eclip_coverage,
        residual_ablation,
        pair_engagement,
        richer_supervision,
        target_local_frontier,
    ]
    inputs: dict[str, list[str]] = {}
    for generator in generators:
        used = generator(root, output_dir, source)
        if used:
            inputs[generator.__name__] = used
    result = {
        "generated_figures": sorted(path.name for path in output_dir.glob("*.png")),
        "source_tables": sorted(path.name for path in source.glob("*.csv")),
        "input_tables": inputs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    root = args.repository.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    print(json.dumps(run(root, output), indent=2))


if __name__ == "__main__":
    main()
