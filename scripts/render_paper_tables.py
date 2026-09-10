#!/usr/bin/env python3
"""Render manuscript/SI LaTeX tables from authoritative result tables."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "source_tables"
OUTPUT = ROOT / "paper" / "generated"
CONTEXT_LABELS = {"MuscleSkeletal": "Skeletal muscle"}


def tex(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def context_label(value: object) -> str:
    return CONTEXT_LABELS.get(str(value), str(value))


def interval(row: pd.Series, scale: float = 100.0) -> str:
    return f"{scale * row['median']:+.2f} [{scale * row['lower']:+.2f}, {scale * row['upper']:+.2f}]"


def write(name: str, content: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def locked_metrics() -> None:
    df = pd.read_csv(SOURCE / "figure4_locked_metrics.csv")
    model_names = {
        "published_baseline_reproduced": "AdarEdit",
        "adarpro_pair_local_equal_logit_ensemble": "Selected predictor",
    }
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"{tex(context_label(row['context']))} & {model_names[row['model']]} & "
            f"{row['auroc']:.4f} & {row['auprc']:.4f} & {row['f1']:.4f} & "
            f"{row['precision']:.4f} & {row['recall']:.4f} & "
            f"{row['specificity']:.4f} & {row['brier']:.4f} \\\\"
        )
    write(
        "table_s1_locked_metrics.tex",
        r"""\begin{table}[H]
\centering
\caption{Complete one-use canonical held-out metrics. Thresholds were frozen from the published validation partition before test access. Combined is a pooled endpoint.}\label{tab:lockedmetrics}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrr}
\toprule
Context & Model & AUROC & AUPRC & F1 & Precision & Recall & Specificity & Brier \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}}
\end{table}""",
    )


def evidence_map() -> None:
    def select(
        path: str, metric: str, *, invert: bool = False, **selectors: str
    ) -> pd.Series:
        frame = pd.read_csv(ROOT / path)
        selected = frame[frame["metric"] == metric]
        for column, value in selectors.items():
            selected = selected[selected[column] == value]
        if len(selected) != 1:
            raise RuntimeError(
                f"Expected one synthesis row in {path}: {metric}, {selectors}"
            )
        row = selected.iloc[0].copy()
        if invert:
            row[["lower", "median", "upper"]] = -row[
                ["upper", "median", "lower"]
            ].to_numpy()
        return row

    def effect(metric: str, row: pd.Series) -> str:
        label = {"auroc": "AUROC", "auprc": "AUPRC", "f1": "F1", "r2": r"$R^2$"}[metric]
        return f"$\\Delta${label} {interval(row)}"

    rows: list[dict[str, str]] = []

    def add(
        hypothesis: str,
        axis: str,
        comparator: str,
        primary: str,
        secondary: str,
        status: str,
        interpretation: str,
        source: str,
    ) -> None:
        rows.append(
            {
                "hypothesis": hypothesis,
                "axis": axis,
                "comparator": comparator,
                "primary_effect": primary,
                "secondary_behavior": secondary,
                "status": status,
                "interpretation": interpretation,
                "source": source,
            }
        )

    supervision = "experiments/runs/richer_supervision/pair_balance_comparison/pair_bootstrap_ci.csv"
    add(
        "Additional valid high/low labels",
        "Supervision",
        "Original internal reference; same folds",
        effect(
            "auroc",
            select(
                supervision,
                "auroc",
                candidate="expanded_site",
                reference="development_reference",
            ),
        ),
        effect(
            "f1",
            select(
                supervision,
                "f1",
                candidate="expanded_site",
                reference="development_reference",
            ),
        ),
        "Selected",
        "Additional valid labels contribute independently.",
        supervision,
    )
    add(
        "Pair-first weighting",
        "Statistical unit",
        "Expanded row-wise sampling; identical labels",
        effect(
            "auroc",
            select(
                supervision,
                "auroc",
                candidate="expanded_site",
                reference="expanded_pair",
                invert=True,
            ),
        ),
        effect(
            "f1",
            select(
                supervision,
                "f1",
                candidate="expanded_site",
                reference="expanded_pair",
                invert=True,
            ),
        ),
        "Selected",
        "Correcting substrate weight adds a second gain.",
        supervision,
    )
    multitask = "experiments/runs/richer_supervision/pair_balanced_multitask_comparison/pair_bootstrap_ci.csv"
    add(
        "Continuous + ordinal auxiliary labels",
        "Supervision",
        "Pair-first binary model",
        effect(
            "auroc",
            select(
                multitask,
                "auroc",
                candidate="expanded_pair",
                reference="multitask_pair",
                invert=True,
            ),
        ),
        effect(
            "f1",
            select(
                multitask,
                "f1",
                candidate="expanded_pair",
                reference="multitask_pair",
                invert=True,
            ),
        ),
        "Limited",
        "Ranking/Brier gain; negligible F1; adverse natural-prevalence integration.",
        multitask,
    )

    focal = (
        "experiments/runs/richer_supervision/pooling_comparison/pair_bootstrap_ci.csv"
    )
    add(
        "Explicit focal-node retention",
        "Representation",
        "Pair-first whole-duplex pooling",
        effect("auroc", select(focal, "auroc", candidate="focal", reference="pair")),
        effect("f1", select(focal, "f1", candidate="focal", reference="pair")),
        "Limited",
        "Ranking gain; standalone F1 interval crosses 0.",
        focal,
    )
    geometry = (
        "experiments/runs/target_geometry_residual/MuscleSkeletal/pair_bootstrap_ci.csv"
    )
    add(
        "Target/contact geometry",
        "Representation",
        "Pooled-score reference (exploratory)",
        effect(
            "auroc",
            select(geometry, "auroc", model="baseline_plus_multiscale_structure"),
        ),
        effect(
            "f1", select(geometry, "f1", model="baseline_plus_multiscale_structure")
        ),
        "Limited",
        "Ranking and Brier gains localize the target effect.",
        geometry,
    )
    local = "experiments/runs/richer_supervision/local_alignment_control/MuscleSkeletal/pair_bootstrap_ci.csv"
    add(
        "Aligned target/helix-partner representation",
        "Representation",
        "Pair-first whole-duplex-only predictor",
        effect(
            "auroc",
            select(
                local, "auroc", candidate="aligned_local", reference="pair_balanced"
            ),
        ),
        effect(
            "f1",
            select(local, "f1", candidate="aligned_local", reference="pair_balanced"),
        ),
        "Selected",
        "Target-aligned fusion improves all three metrics.",
        local,
    )
    add(
        "96-nt shifted local window",
        "Capacity control",
        "Pair-first whole-duplex-only predictor",
        effect(
            "auroc",
            select(
                local, "auroc", candidate="shifted_local", reference="pair_balanced"
            ),
        ),
        effect(
            "f1",
            select(local, "f1", candidate="shifted_local", reference="pair_balanced"),
        ),
        "Not supported",
        "Matched capacity without target alignment is strongly adverse.",
        local,
    )
    ensemble = "experiments/runs/richer_supervision/pair_local_ensemble_comparison/MuscleSkeletal/pair_bootstrap_ci.csv"
    add(
        "Fixed equal-logit integration",
        "Integration",
        "Whole-duplex-only predictor; muscle",
        effect(
            "auroc",
            select(
                ensemble,
                "auroc",
                candidate="pair_balanced",
                reference="pair_local_ensemble",
                invert=True,
            ),
        ),
        effect(
            "f1",
            select(
                ensemble,
                "f1",
                candidate="pair_balanced",
                reference="pair_local_ensemble",
                invert=True,
            ),
        ),
        "Selected",
        "Complementary predictions; replicated in liver.",
        ensemble,
    )

    engagement_table = pd.read_csv(SOURCE / "figure2_pair_engagement.csv")
    min_effect = 100 * engagement_table["delta_r2_median"].min()
    max_effect = 100 * engagement_table["delta_r2_median"].max()
    add(
        "Pair geometry and editing burden",
        "Substrate propensity",
        "Geometry + graph summaries vs graph summaries",
        rf"$\Delta R^2$ {min_effect:+.2f} to {max_effect:+.2f} points (4 tissues)",
        "All four pair-clustered 95% intervals exclude 0",
        "Supported, not yet actionable",
        "Burden is predictable; consistent with variation in realization/engagement.",
        "paper/source_tables/figure2_pair_engagement.csv",
    )
    gate = "experiments/runs/hierarchical_gate/binary_natural/MuscleSkeletal/pair_bootstrap_ci.csv"
    add(
        "Cross-fitted engagement gate",
        "Biology × AI",
        "Expanded row-wise site predictor",
        effect(
            "auprc", select(gate, "auprc", model="expanded_plus_combined_engagement")
        ),
        effect("f1", select(gate, "f1", model="expanded_plus_combined_engagement")),
        "Limited",
        "AUPRC gain; F1 interval crosses 0; Brier worsens.",
        gate,
    )
    shared = (
        "experiments/runs/joint_multilabel/vs_independent/global_pair_bootstrap_ci.csv"
    )
    add(
        "Shared four-tissue learning",
        "Transfer",
        "Independent tissue models; mean across 4 tissues",
        effect("auroc", select(shared, "auroc", context="macro")),
        effect("f1", select(shared, "f1", context="macro")),
        "Limited",
        "Positive transfer below preset criteria; mean Brier worsens.",
        shared,
    )

    chromatin = "experiments/runs/chromatin_residual/Liver/pair_bootstrap_ci.csv"
    add(
        "Bulk H3K36me3 occupancy",
        "Biological context",
        "Pooled-score reference; liver (exploratory)",
        effect(
            "auprc",
            select(
                chromatin, "auprc", model="baseline_plus_h3k36me3_multiscale_occupancy"
            ),
        ),
        effect(
            "f1",
            select(
                chromatin, "f1", model="baseline_plus_h3k36me3_multiscale_occupancy"
            ),
        ),
        "Limited",
        "Liver: AUPRC only. Muscle threshold effects did not extend to ranking; no stable pair-level gain.",
        chromatin,
    )
    rbp = "experiments/runs/residual_pilot/MuscleSkeletal/pair_bootstrap_ci.csv"
    add(
        "RBP binding × bulk abundance",
        "Biological context",
        "Pooled-score reference (exploratory)",
        effect(
            "auroc", select(rbp, "auroc", model="baseline_plus_biological_suppressors")
        ),
        effect("f1", select(rbp, "f1", model="baseline_plus_biological_suppressors")),
        "Not supported",
        "No consistent screen gain; broader predicted panels were adverse.",
        rbp,
    )
    genomic = (
        "experiments/runs/genomic_residual_v2/MuscleSkeletal/pair_bootstrap_ci.csv"
    )
    add(
        "Transcript/repeat annotation panel",
        "Biological context",
        "Pooled-score reference (exploratory)",
        effect(
            "auroc",
            select(genomic, "auroc", model="baseline_plus_architecture_and_intrinsic"),
        ),
        effect(
            "f1",
            select(genomic, "f1", model="baseline_plus_architecture_and_intrinsic"),
        ),
        "Not supported",
        "No improvement beyond the graph representation.",
        genomic,
    )
    rinalmo = "experiments/runs/foundation_residual/rinalmo_mega_target/MuscleSkeletal/pair_bootstrap_ci.csv"
    add(
        "Frozen RiNALMo target embedding",
        "Representation",
        "Recalibrated cross-fitted baseline",
        effect(
            "auroc",
            select(
                rinalmo,
                "auroc",
                candidate="foundation_target",
                reference="base_recalibrated",
            ),
        ),
        effect(
            "f1",
            select(
                rinalmo,
                "f1",
                candidate="foundation_target",
                reference="base_recalibrated",
            ),
        ),
        "Not supported",
        "F1-only threshold effect; no AUROC/AUPRC gain; failed the preset rule.",
        rinalmo,
    )
    all_a = "experiments/runs/substrate_wide/joint_binary_partial_futility/MuscleSkeletal/pair_bootstrap_ci.csv"
    add(
        "Substrate-wide all-A decoder",
        "Representation",
        "Pair-first predictor on identical rows",
        effect(
            "auroc",
            select(all_a, "auroc", candidate="control", reference="pair_balanced"),
        ),
        effect(
            "f1", select(all_a, "f1", candidate="control", reference="pair_balanced")
        ),
        "Not supported",
        "Prespecified two-fold futility criterion met.",
        all_a,
    )
    unified = "experiments/runs/unified/comparison_primary/MuscleSkeletal/pair_bootstrap_ci.csv"
    add(
        "Unified end-to-end architecture",
        "Integration",
        "Immutable fixed ensemble",
        effect(
            "auroc",
            select(unified, "auroc", candidate="unified", reference="frozen_ensemble"),
        ),
        effect(
            "f1",
            select(unified, "f1", candidate="unified", reference="frozen_ensemble"),
        ),
        "Not supported",
        "AUROC, AUPRC, and Brier probability error were worse.",
        unified,
    )
    add(
        "Supplied Bio-aware variant",
        "Descriptive comparison",
        "Supplied Bio-aware vs compact checkpoints",
        "No common OOF effect estimate",
        "Not uniformly better across endpoints",
        "Mixed; descriptive",
        "Features and architecture change together; capacity is not isolated.",
        "results/adaredit_baseline_metrics.csv",
    )

    df = pd.DataFrame(rows)
    SOURCE.mkdir(parents=True, exist_ok=True)
    df.to_csv(SOURCE / "table_s2_evidence_synthesis.csv", index=False)
    lines = [
        f"{tex(row['axis'])} & {tex(row['hypothesis'])} & {tex(row['comparator'])} & "
        f"{row['primary_effect']} & {tex(row['secondary_behavior'])} & "
        f"{tex(row['status'])} & {tex(row['interpretation'])} \\\\"
        for _, row in df.iterrows()
    ]
    write(
        "table_s2_evidence_map.tex",
        r"""\begin{landscape}
\begingroup
\fontsize{7.0}{7.8}\selectfont
\setlength{\tabcolsep}{1.8pt}
\renewcommand{\arraystretch}{1.10}
\begin{longtable}{>{\raggedright\arraybackslash}p{0.10\linewidth}>{\raggedright\arraybackslash}p{0.15\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.11\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}}
\caption{Development evidence synthesis. Each row states its comparator; estimands differ across rows. Values are percentage-point medians with 95\% pair-clustered intervals unless noted. RBP, transcript/repeat, chromatin, and target/contact residual screens reuse pooled first-stage OOF scores and are exploratory, not fully nested (section A). ``Not supported'' applies to the tested formulation and selection criterion.}\label{tab:evidencemap}\\
\toprule
Axis & Hypothesis & Comparator & Primary effect [95\% CI] & Secondary behavior & Evidence status & Interpretation \\
\midrule
\endfirsthead
\multicolumn{7}{l}{\textit{Table continued}}\\
\toprule
Axis & Hypothesis & Comparator & Primary effect [95\% CI] & Secondary behavior & Evidence status & Interpretation \\
\midrule
\endhead
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{longtable}
\endgroup
\end{landscape}""",
    )


def supervision_decomposition() -> None:
    metrics = pd.read_csv(SOURCE / "figure3_supervision_decomposition.csv").set_index(
        "model"
    )
    ci = pd.read_csv(
        ROOT
        / "experiments/runs/richer_supervision/pair_balance_comparison/pair_bootstrap_ci.csv"
    )
    labels = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "f1": "F1",
    }
    rows = []
    for metric, label in labels.items():
        add_labels = ci[
            (ci.candidate == "expanded_site")
            & (ci.reference == "development_reference")
            & (ci.metric == metric)
        ].iloc[0]
        add_pair = (
            ci[
                (ci.candidate == "expanded_site")
                & (ci.reference == "expanded_pair")
                & (ci.metric == metric)
            ]
            .iloc[0]
            .copy()
        )
        add_pair[["lower", "median", "upper"]] = -add_pair[
            ["upper", "median", "lower"]
        ].to_numpy()
        rows.append(
            f"{label} & {metrics.loc['development_reference', metric]:.4f} & "
            f"{metrics.loc['expanded_site', metric]:.4f} & {interval(add_labels)} & "
            f"{metrics.loc['expanded_pair', metric]:.4f} & {interval(add_pair)} \\\\"
        )
    write(
        "table_s4_supervision_decomposition.tex",
        r"""\begin{table}[H]
\centering
\caption{Same-fold decomposition of expanded supervision and pair-first weighting in muscle development data. Intervals are pair-clustered bootstrap intervals for the incremental step indicated, in percentage points.}\label{tab:supervisiondecomposition}
\begin{tabular}{lrrrrr}
\toprule
Metric & Original reference & All high/low, row-wise & $\Delta$ from labels [95\% CI] & All high/low, pair-first & $\Delta$ from pair weighting [95\% CI] \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


def substrate_realization() -> None:
    gap = pd.read_csv(SOURCE / "figure2_gap_editing_burden.csv")
    models = pd.read_csv(SOURCE / "figure2_pair_engagement.csv").set_index("context")
    rows = []
    for context, group in gap.groupby("context", sort=False):
        row = models.loc[context]
        rows.append(
            f"{tex(context_label(context))} & {group['spearman'].iloc[0]:.3f} & "
            f"{row['graph_only_r2']:.3f} & {row['geometry_plus_graph_r2']:.3f} & "
            f"{100 * row['delta_r2_median']:+.2f} [{100 * row['delta_r2_lower']:+.2f}, {100 * row['delta_r2_upper']:+.2f}] & "
            f"{row['any_high_auroc']:.3f} \\\\"
        )
    write(
        "table_s3_substrate_realization.tex",
        r"""\begin{table}[H]
\centering
\caption{Pair geometry identifies a reproducible component of substrate-level editing burden. Spearman $\rho$ relates genomic inter-arm gap to edited-site fraction. The low-capacity pair model combines genomic geometry with graph-derived summaries; $\Delta R^2$ is relative to graph summaries alone with a 95\% pair-clustered bootstrap interval. ``Any high'' denotes whether a pair contains at least one site with EditingIndex $\geq15\%$. These outcomes do not directly measure molecular realization or engagement.}\label{tab:substraterealization}
\begin{tabular}{lrrrrr}
\toprule
Context & Gap--burden $\rho$ & Graph-only $R^2$ & Combined $R^2$ & $\Delta R^2$ [95\% CI], points & Any-high AUROC \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


def target_controls() -> None:
    alignment = pd.read_csv(SOURCE / "figure3_alignment_metrics.csv")
    focal = pd.read_csv(SOURCE / "figure3_focal_metrics.csv")
    rows = []
    for label, frame, model in [
        ("Whole-duplex pair model", alignment, "pair_balanced"),
        ("Explicit focal node", focal, "focal"),
        ("Graph + aligned target/helix-partner", alignment, "aligned_local"),
        ("Matched-capacity +96-nt shift", alignment, "shifted_local"),
    ]:
        row = frame[frame.model == model].iloc[0]
        rows.append(
            f"{label} & {row['auroc']:.4f} & {row['auprc']:.4f} & {row['f1']:.4f} & {row['brier']:.4f} \\\\"
        )
    write(
        "table_s6_target_controls.tex",
        r"""\begin{table}[H]
\centering
\caption{Target-representation controls in muscle development OOF data. All values are evaluated on the same canonical development rows.}\label{tab:targetcontrols}
\begin{tabular}{lrrrr}
\toprule
Representation & AUROC & AUPRC & F1 & Brier \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


def pair_balance() -> None:
    df = pd.read_csv(SOURCE / "figure3_pair_balance_ci.csv")
    rows = []
    for context, group in df.groupby("context", sort=False):
        values = {row.metric: interval(row) for _, row in group.iterrows()}
        rows.append(
            f"{tex(context_label(context))} & {values['auroc']} & {values['auprc']} & "
            f"{values['f1']} \\\\"
        )
    write(
        "table_s5_pair_balance.tex",
        r"""\begin{table}[H]
\centering
\caption{Pair-first expanded supervision versus the same-fold internal AdarEdit reference. Values are percentage-point median differences with 95\% pair-clustered bootstrap intervals.}\label{tab:pairbalance}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrr}
\toprule
Context & $\Delta$AUROC [95\% CI] & $\Delta$AUPRC [95\% CI] & $\Delta$F1 [95\% CI] \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\end{table}""",
    )


def false_positive_table() -> None:
    errors = pd.read_csv(SOURCE / "figure4_error_transitions.csv")
    matched = pd.read_csv(SOURCE / "figure4_matched_recall.csv").set_index("context")
    rows = []
    for _, row in errors.iterrows():
        mr = matched.loc[row["context"]]
        rows.append(
            f"{tex(context_label(row['context']))} & {int(row['baseline_fp'])} & {int(row['fp_rescued'])} "
            rf"({100 * row['fp_rescue_rate']:.1f}\%) & {int(row['new_fp'])} & "
            rf"{int(row['tp_damaged'])} ({100 * row['tp_damage_rate']:.1f}\%) & "
            f"{int(mr['false_positives_avoided'])} \\\\"
        )
    write(
        "table_s9_false_positives.tex",
        r"""\begin{table}[H]
\centering
\caption{Canonical-test false-positive transitions. Rescue and damage use frozen validation-selected thresholds. The final column is a post-test descriptive analysis that rethresholds the selected predictor to match published-baseline recall exactly.}\label{tab:falsepositives}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Context & Baseline FP & FP corrected (\%) & New FP & TP damaged (\%) & FP avoided at matched recall \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\end{table}""",
    )


def natural_prevalence() -> None:
    df = pd.read_csv(SOURCE / "figureS3_natural_prevalence_metrics.csv")
    names = {
        "development_reference": "Internal reference",
        "pair_balanced": "Pair-first whole-duplex",
        "local": "Local-augmented graph",
        "pair_local_ensemble": "Fixed ensemble",
    }
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{tex(context_label(row['context']))} & {names[row['model']]} & {int(row['n'])} & "
            f"{row['auroc']:.4f} & {row['auprc']:.4f} & {row['f1']:.4f} & "
            f"{row['precision']:.4f} & {row['recall']:.4f} & {int(row['fp'])} \\\\"
        )
    write(
        "table_s7_natural_prevalence.tex",
        r"""\begin{table}[H]
\centering
\caption{Secondary natural-prevalence development evaluation. All eligible high/low observations from unchanged development pairs are included; fold-specific validation thresholds are used.}\label{tab:natural}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrr}
\toprule
Context & Model & $n$ & AUROC & AUPRC & F1 & Precision & Recall & FP \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\end{table}""",
    )


def unified() -> None:
    metrics = pd.read_csv(
        ROOT / "experiments/runs/unified/comparison_primary/MuscleSkeletal/metrics.csv"
    )
    names = {
        "frozen_ensemble": "Frozen ensemble",
        "unified": "Unified end-to-end",
        "local": "Local-augmented graph",
        "development_reference": "Internal reference",
    }
    metric_rows = []
    for _, row in metrics.iterrows():
        metric_rows.append(
            f"{names[row['model']]} & {row['auroc']:.4f} & {row['auprc']:.4f} & "
            f"{row['f1']:.4f} & {row['precision']:.4f} & {row['recall']:.4f} & "
            f"{row['specificity']:.4f} & {row['brier']:.4f} \\\\"
        )
    ci = pd.read_csv(SOURCE / "figureS4_unified_vs_ensemble_ci.csv")
    order = ["auroc", "auprc", "f1", "precision", "recall", "specificity", "brier"]
    lookup = {row.metric: row for _, row in ci.iterrows()}
    ci_rows = [
        f"{metric.upper()} & {interval(lookup[metric])} \\\\" for metric in order
    ]
    write(
        "table_s8_unified.tex",
        r"""\begin{table}[H]
\centering
\caption{Preregistered unified-model comparison on skeletal-muscle development data. The unified architecture was rejected before canonical-test access. Pair-clustered intervals compare the unified model with the immutable ensemble.}\label{tab:unified}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
Model & AUROC & AUPRC & F1 & Precision & Recall & Specificity & Brier \\
\midrule
"""
        + "\n".join(metric_rows)
        + r"""
\bottomrule
\end{tabular}}
\vspace{6pt}

\begin{tabular}{lr}
\toprule
Metric & Unified $-$ ensemble, points [95\% CI] \\
\midrule
"""
        + "\n".join(ci_rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


def experiment_taxonomy() -> None:
    df = pd.read_csv(SOURCE / "search_space_coverage.csv")
    rows = []
    for _, row in df.iterrows():
        status = (
            f"{tex(row['outcome_class'])}"
            + r"\newline\textit{"
            + tex(row["evidence_regime"])
            + "}"
        )
        rows.append(
            f"{tex(row['scale'])} & {tex(row['question'])} & "
            f"{tex(row['candidates_or_measurements'])} & "
            f"{tex(row['control_or_reference'])} & {status} & "
            f"{tex(row['principal_outcome'])} & {tex(row['disposition'])} \\\\"
        )
    write(
        "table_s10_experiment_taxonomy.tex",
        r"""\begin{landscape}
\begingroup
\fontsize{7.5}{8.6}\selectfont
\setlength{\tabcolsep}{1.5pt}
\renewcommand{\arraystretch}{1.00}
\begin{longtable}{>{\raggedright\arraybackslash}p{0.09\linewidth}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.17\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.12\linewidth}>{\raggedright\arraybackslash}p{0.17\linewidth}>{\raggedright\arraybackslash}p{0.11\linewidth}}
\caption{Search-space coverage and final disposition. Rows group the adaptive campaign by question and distinguish controlled, exploratory, descriptive, and untested evidence. ``Deferred'' means suitable data or a validated resource was unavailable, not that the hypothesis failed. Quantitative effects are in Table~\ref{tab:evidencemap}; historical priorities, stopping rules, and run artifacts remain in the machine-readable record.}\label{tab:taxonomy}\\
\toprule
Scale & Question & Candidate information or measurement & Comparator or control & Status and evidence regime & Principal outcome & Disposition \\
\midrule
\endfirsthead
\multicolumn{7}{l}{\textit{Table continued}}\\
\toprule
Scale & Question & Candidate information or measurement & Comparator or control & Status and evidence regime & Principal outcome & Disposition \\
\midrule
\endhead
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{longtable}
\endgroup
\end{landscape}""",
    )


def registry_summary() -> None:
    df = pd.read_csv(ROOT / "experiments/experiments.csv")
    counts = (
        df.groupby("experiment_type", dropna=False).size().sort_values(ascending=False)
    )
    rows = [f"{tex(name)} & {count} \\\\" for name, count in counts.items()]
    write(
        "table_s8_registry.tex",
        r"""\begin{longtable}{p{0.78\textwidth}r}
\caption{Machine-readable experiment-registry coverage. Each of the 128 unique rows records code revision, inputs, fold definition, seed, features, hyperparameters, artifacts, metrics, and canonical-test-access status.}\label{tab:registry}\\
\toprule
Experiment family & Registered runs \\
\midrule
\endfirsthead
\multicolumn{2}{l}{\textit{Table S8 continued}}\\
\toprule
Experiment family & Registered runs \\
\midrule
\endhead
"""
        + "\n".join(rows)
        + r"""
\midrule
Total & 128 \\
\bottomrule
\end{longtable}""",
    )


def manifest() -> None:
    entries = []
    for path in sorted(OUTPUT.glob("table_*.tex")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"file": path.name, "sha256": digest})
    pd.DataFrame(entries).to_csv(OUTPUT / "table_manifest.csv", index=False)


def main() -> None:
    locked_metrics()
    evidence_map()
    supervision_decomposition()
    pair_balance()
    substrate_realization()
    target_controls()
    false_positive_table()
    natural_prevalence()
    unified()
    experiment_taxonomy()
    manifest()
    print("PASS: rendered 10 SI tables from authoritative artifacts")


if __name__ == "__main__":
    main()
