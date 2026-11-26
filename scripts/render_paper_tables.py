#!/usr/bin/env python3
"""Render manuscript/SI LaTeX tables from authoritative result tables."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import paper_sources as sources

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


def minus(text: str) -> str:
    """Set negative numbers with a true minus sign in .tex output; CSV copies keep the ASCII hyphen.

    Only a hyphen that starts a number is replaced, so hyphens inside words or labels
    such as '96-nt' or 'Pair-first' are left alone.
    """
    return re.sub(r"(?<![\w\\{}$-])-(?=\.?\d)", "$-$", text)


def write(name: str, content: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def locked_metrics() -> None:
    df = pd.read_csv(SOURCE / "figure4_locked_metrics.csv")  # written by render_paper_figures from evaluation 3
    model_names = {
        sources.TEST_BASELINE: "AdarEdit",
        sources.TEST_CANDIDATE: "Selected predictor",
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
\caption{\textbf{Ranking, classification, and probability accuracy improve across all five held-out endpoints.} Relative to reproduced AdarEdit, the selected predictor has higher AUROC, AUPRC, and F1 and lower Brier score. Precision and recall show the classification trade-off at thresholds frozen from the published validation partition before the final test. Combined is a pooled endpoint.}\label{tab:lockedmetrics}
\begin{tabular}{l@{\hspace{2\tabcolsep}\hspace{0.6em}}lrrrrrrr}
\toprule
Endpoint & Model & AUROC & AUPRC & F1 & Precision & Recall & Specificity & Brier \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
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

    supervision = sources.relpath("experiments/runs/richer_supervision/pair_balance_comparison/pair_bootstrap_ci.csv")
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
    multitask = sources.relpath("experiments/runs/richer_supervision/pair_balanced_multitask_comparison/pair_bootstrap_ci.csv")
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
        "No clear gain over binary training; added to the graph and focal-node combination, small ranking gain, no F1 change, worse Brier.",
        multitask,
    )

    focal = sources.relpath(
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
    local = sources.relpath("experiments/runs/richer_supervision/local_alignment_control/MuscleSkeletal/pair_bootstrap_ci.csv")
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
        "The target-aligned local predictor improves all three metrics.",
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
    ensemble = sources.relpath("experiments/runs/richer_supervision/pair_local_ensemble_comparison/MuscleSkeletal/pair_bootstrap_ci.csv")
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
        "Complementary predictions; replicated in liver; the selected predictor combines the views in a second stage.",
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
        "Supported (pair level)",
        "Burden is predictable; consistent with variation in realization/engagement.",
        "paper/source_tables/figure2_pair_engagement.csv",
    )
    gap_dir = ROOT / "experiments/runs/step11b/gap_controls"
    gap_med = {"no_gap_stage": [], "shuffled_gap_stage": []}
    excludes = {m: 0 for m in ("auroc", "auprc", "f1", "brier")}
    for context in ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]:
        frame = pd.read_csv(gap_dir / context / "pair_bootstrap_ci.csv")
        for reference in gap_med:
            for metric in excludes:
                row = frame[(frame.candidate == "gap_stage") & (frame.reference == reference) & (frame.metric == metric)].iloc[0]
                if metric == "auroc":
                    gap_med[reference].append(100 * row["median"])
                if reference == "shuffled_gap_stage" and (row["lower"] > 0 or row["upper"] < 0):
                    excludes[metric] += 1
    add(
        "Genomic spacing in the second stage",
        "Substrate propensity",
        "Same second stage without spacing; shuffled spacing",
        rf"AUROC {min(gap_med['no_gap_stage']):+.2f} to {max(gap_med['no_gap_stage']):+.2f} / "
        rf"{min(gap_med['shuffled_gap_stage']):+.2f} to {max(gap_med['shuffled_gap_stage']):+.2f} points (5 endpoints)",
        f"vs shuffled: AUROC, AUPRC, Brier intervals exclude 0 in {excludes['auroc']}, {excludes['auprc']}, "
        f"{excludes['brier']} of 5; F1 in {excludes['f1']} of 5; muscle not in every training run",
        "Selected",
        "The measured spacing improves site ranking.",
        "experiments/runs/step11b/gap_controls",
    )
    gate = "experiments/runs/hierarchical_gate/binary_natural/MuscleSkeletal/pair_bootstrap_ci.csv"
    add(
        "Cross-fitted burden gate",
        "Substrate propensity",
        "Expanded row-wise site predictor",
        effect(
            "auprc", select(gate, "auprc", model="expanded_plus_combined_engagement")
        ),
        effect("f1", select(gate, "f1", model="expanded_plus_combined_engagement")),
        "Limited",
        "AUPRC gain; F1 and Brier intervals cross 0.",
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
        "Positive transfer below preset criteria; mean Brier interval crosses 0.",
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
        "RBP binding, ILF2/ILF3/STRBP panel",
        "Biological context",
        "Pooled-score reference; muscle (exploratory)",
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
        "No stable increment in this screen.",
        genomic,
    )
    rinalmo = "experiments/runs/foundation_residual/rinalmo_mega_target/MuscleSkeletal/pair_bootstrap_ci.csv"
    add(
        "Frozen RiNALMo target embedding",
        "Representation",
        "Recalibrated outer-isolated graph reference",
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
        "F1 threshold effect; no AUROC/AUPRC gain; failed the preset rule.",
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
        "Two-fold futility criterion met.",
        all_a,
    )
    unified = sources.relpath("experiments/runs/unified/comparison_primary/MuscleSkeletal/pair_bootstrap_ci.csv")
    add(
        "Unified end-to-end architecture",
        "Integration",
        "Equal-weight two-model combination",
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
        "results/table1_within_context.csv",
    )

    df = pd.DataFrame(rows)
    SOURCE.mkdir(parents=True, exist_ok=True)
    df.to_csv(SOURCE / "table_s2_evidence_synthesis.csv", index=False)
    lines = [
        minus(
            f"{tex(row['axis'])} & {tex(row['hypothesis'])} & {tex(row['comparator'])} & "
            f"{row['primary_effect']} & {tex(row['secondary_behavior'])} & "
            f"{tex(row['status'])} & {tex(row['interpretation'])} \\\\"
        )
        for _, row in df.iterrows()
    ]
    write(
        "table_s2_evidence_map.tex",
        r"""\begin{landscape}
\begingroup
\fontsize{7.0}{7.8}\selectfont\sffamily
\setlength{\tabcolsep}{1.8pt}
\renewcommand{\arraystretch}{1.10}
\begin{longtable}{>{\raggedright\arraybackslash}p{0.10\linewidth}>{\raggedright\arraybackslash}p{0.15\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.11\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}}
\caption{\textbf{Targeted weighting and local representation yield clearer gains than most tested model extensions.} Genomic geometry adds pair-level information and, as measured spacing in the second stage, site-level ranking, whereas the tested contextual proxies lack consistent site-level gains. Each row names its comparator; different outcomes and references prevent a common ranking of effect sizes. Rows that name no tissue or endpoint count are skeletal-muscle development comparisons, except the descriptive last row. Values are percentage-point medians with 95\% pair-clustered intervals unless noted. RBP, transcript/repeat, chromatin, and target/contact screens reuse pooled first-stage OOF scores and remain exploratory, not fully nested (section~\ref{si:benchmark}). ``Not supported'' concerns the tested formulation and selection criterion.}\label{tab:evidencemap}\\
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
    ci = sources.read_csv(
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
            f"{metrics.loc['expanded_site', metric]:.4f} & {minus(interval(add_labels))} & "
            f"{metrics.loc['expanded_pair', metric]:.4f} & {minus(interval(add_pair))} \\\\"
        )
    write(
        "table_s4_supervision_decomposition.tex",
        r"""\begin{table}[H]
\centering
\caption{\textbf{More valid labels and more balanced substrate weights make separate contributions.} In muscle development data, recovering eligible high/low labels improves prediction under row-wise sampling. Pair-first weighting adds a further gain with the label pool and class balance fixed. The same-fold comparisons isolate the label and weighting effects; the $\Delta$ columns give bootstrap medians with pair-clustered intervals for each indicated step, in percentage points.}\label{tab:supervisiondecomposition}
\begin{tabular}{lrrr@{\hspace{2\tabcolsep}\hspace{0.6em}}rr}
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
            minus(
                f"{tex(context_label(context))} & {group['spearman'].iloc[0]:.3f} & "
                f"{row['graph_only_r2']:.3f} & {row['geometry_plus_graph_r2']:.3f} & "
                f"{100 * row['delta_r2_median']:+.2f} [{100 * row['delta_r2_lower']:+.2f}, {100 * row['delta_r2_upper']:+.2f}] & "
                f"{row['any_high_auroc']:.3f} \\\\"
            )
        )
    write(
        "table_s3_substrate_realization.tex",
        r"""\begin{table}[H]
\centering
\caption{\textbf{Genomic geometry improves prediction of the edited-site fraction in all four tissues.} Larger inter-arm gaps correlate with lower burden (Spearman $\rho$). Burden is the high-editing fraction among eligible high/low sites within a pair. Adding geometry to graph-derived summaries improves pair-level $R^2$; $\Delta R^2$ is relative to graph summaries alone, as a bootstrap median with a 95\% pair-clustered interval. ``Any high'' asks whether a pair contains at least one site with EditingIndex $\geq15\%$. These outcomes measure editing burden, not molecular realization or ADAR engagement.}\label{tab:substraterealization}
\begin{tabular}{lrrrrr}
\toprule
Tissue & Gap--burden $\rho$ & Graph-only $R^2$ & Combined $R^2$ & $\Delta R^2$ [95\% CI], points & Any-high AUROC \\
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
        ("Pair-first whole-duplex model", alignment, "pair_balanced"),
        ("Explicit focal node", focal, "focal"),
        ("Aligned target/helix-partner local predictor", alignment, "aligned_local"),
        ("Matched-capacity 96-nt shift", alignment, "shifted_local"),
        ("Whole-duplex model averaged with aligned local predictor", alignment, "aligned_ensemble"),
        ("Whole-duplex model averaged with shifted local predictor", alignment, "shifted_ensemble"),
    ]:
        row = frame[frame.model == model].iloc[0]
        rows.append(
            f"{label} & {row['auroc']:.4f} & {row['auprc']:.4f} & {row['f1']:.4f} & {row['brier']:.4f} \\\\"
        )
    write(
        "table_s6_target_controls.tex",
        r"""\begin{table}[H]
\centering
\caption{\textbf{The aligned local window adds signal; shifting it loses the gain despite matched model capacity.} Explicit target retention improves ranking over whole-duplex pooling, and the aligned local predictor improves further. Moving the local window while preserving the architecture reduces ranking and F1 and increases Brier error. This control tests target-centered neighborhood information, not partner identity alone. The last two rows average the pair-first whole-duplex model's logits in equal weight with the aligned or the shifted local predictor. All values use the same canonical muscle development rows and out-of-fold predictions.}\label{tab:targetcontrols}
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
        values = {row.metric: minus(interval(row)) for _, row in group.iterrows()}
        rows.append(
            f"{tex(context_label(context))} & {values['auroc']} & {values['auprc']} & "
            f"{values['f1']} \\\\"
        )
    write(
        "table_s5_pair_balance.tex",
        r"""\begin{table}[H]
\centering
\caption{\textbf{Expanded pair-first learning improves ranking and classification across all five endpoints.} The combined change in label coverage and within-class pair weighting outperforms the same-fold internal AdarEdit reference; every interval is positive. Values are percentage-point median differences with 95\% pair-clustered bootstrap intervals. Combined is pooled, and these development comparisons do not isolate weighting from label recovery outside muscle.}\label{tab:pairbalance}
\begin{tabular}{lrrr}
\toprule
Endpoint & $\Delta$AUROC [95\% CI] & $\Delta$AUPRC [95\% CI] & $\Delta$F1 [95\% CI] \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
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
\caption{\textbf{False-positive reduction persists at matched recall.} At frozen validation-selected thresholds, rescue and damage percentages use AdarEdit's false positives and true positives, respectively; ``New FP'' counts new false positives. The final column matches AdarEdit's recall exactly. This rethresholding is post-test descriptive, not model selection.}\label{tab:falsepositives}
\begin{tabular}{lrrrrr}
\toprule
Endpoint & AdarEdit FP & FP corrected (\%) & New FP & TP damaged (\%) & FP avoided at matched recall \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


def natural_prevalence() -> None:
    df = pd.read_csv(SOURCE / "figureS3_natural_prevalence_metrics.csv")
    names = {
        "development_reference": "Internal reference",
        "pair_balanced": "Pair-first whole-duplex",
        "local": "Local predictor",
        "pair_local_ensemble": "Equal-weight average",
        "gap_stage": "Selected predictor",
    }
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{tex(context_label(row['context']))} & {names[row['model']]} & {int(row['n']):,} & "
            f"{row['auroc']:.4f} & {row['auprc']:.4f} & {row['f1']:.4f} & "
            f"{row['precision']:.4f} & {row['recall']:.4f} & {int(row['fp']):,} \\\\"
        )
    write(
        "table_s7_natural_prevalence.tex",
        r"""\begin{table}[H]
\centering
\caption{\textbf{The predictive gains persist when all eligible high/low sites are retained.} In muscle and liver, the selected predictor improves ranking and F1 and reduces false positives relative to the pair-first whole-duplex model. All eligible high/low observations from unchanged development pairs are included; $n$ counts those observations, and thresholds come from fold-specific validation. This secondary evaluation retains the undownsampled class proportions but excludes intermediate editing levels; it is not population-wide prevalence.}\label{tab:natural}
\begin{tabular}{llrrrrrrr}
\toprule
Tissue & Model & $n$ & AUROC & AUPRC & F1 & Precision & Recall & FP \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


METRIC_LABEL = {
    "auroc": "AUROC", "auprc": "AUPRC", "f1": "F1", "precision": "Precision",
    "recall": "Recall", "specificity": "Specificity", "brier": "Brier",
}


def unified() -> None:
    metrics = sources.read_csv(
        ROOT / "experiments/runs/unified/comparison_primary/MuscleSkeletal/metrics.csv"
    )
    names = {
        "frozen_ensemble": "Two-model combination",
        "unified": "Unified end-to-end",
        "local": "Local predictor",
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
        f"{METRIC_LABEL[metric]} & {minus(interval(lookup[metric]))} \\\\" for metric in order
    ]
    write(
        "table_s8_unified.tex",
        r"""\begin{table}[H]
\centering
\caption{\textbf{The joint model loses ranking, classification, and probability accuracy.} In the skeletal-muscle development comparison, the unified architecture has lower AUROC, AUPRC, and F1 and higher Brier error than the two-model combination, and the branch stopped before the final test. Unified-minus-combination differences are bootstrap medians with pair-clustered intervals; positive Brier differences are adverse.}\label{tab:unified}
\begin{tabular}{lrrrrrrr}
\toprule
Model & AUROC & AUPRC & F1 & Precision & Recall & Specificity & Brier \\
\midrule
"""
        + "\n".join(metric_rows)
        + r"""
\bottomrule
\end{tabular}
\vspace{6pt}

\begin{tabular}{lr}
\toprule
Metric & Unified $-$ combination, points [95\% CI] \\
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
\fontsize{7.5}{8.6}\selectfont\sffamily
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{>{\raggedright\arraybackslash}p{0.09\linewidth}>{\raggedright\arraybackslash}p{0.13\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.15\linewidth}>{\raggedright\arraybackslash}p{0.12\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.11\linewidth}}
\caption{\textbf{The search separates supported improvements from unsuccessful tests and questions that lack suitable data.} Rows connect each biological or modeling question to its candidate information, control, evidence regime, and disposition. Limited effects, unsupported formulations, deferred measurements, and superseded directions remain distinct. ``Deferred'' means suitable data or a validated resource was unavailable, not that the hypothesis failed. Table~\ref{tab:evidencemap} gives quantitative effects; historical priorities, stopping rules, and run artifacts remain in the machine-readable record.}\label{tab:taxonomy}\\
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


def gap_controls() -> None:
    labels = {
        "no_gap_stage": "Same stage without spacing",
        "shuffled_gap_stage": "Same stage, shuffled spacing",
        "equal_average": "Equal-weight average, no second stage",
    }
    rows = []
    for context in ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]:
        frame = pd.read_csv(ROOT / f"experiments/runs/step11b/gap_controls/{context}/pair_bootstrap_ci.csv")
        for reference, label in labels.items():
            cells = []
            for metric in ("auroc", "auprc", "f1", "brier"):
                row = frame[(frame.candidate == "gap_stage") & (frame.reference == reference) & (frame.metric == metric)]
                if len(row) != 1:
                    raise RuntimeError(f"{context}: gap_stage/{reference}/{metric}")
                cells.append(minus(interval(row.iloc[0])))
            rows.append(f"{context_label(context)} & {label} & " + " & ".join(cells) + r" \\")
    write(
        "table_s11_gap_controls.tex",
        r"""\begin{table}[H]
\centering
\footnotesize
\caption{\textbf{The measured genomic spacing improves site prediction beyond matched second-stage controls.} The selected predictor's second stage combines the two networks' scores with the logarithm of the spacing between the Alu arms. Each row gives the selected predictor minus a comparator: the same second stage without spacing, the same stage given shuffled spacing, or the equal-weight average of the two networks without a second stage. Values are percentage-point medians with 95\% pair-clustered intervals (2,000 replicates) on development out-of-fold predictions from one training run; for Brier score, negative values are improvements. Across the five tracked training runs, the skeletal-muscle AUROC gain averaged \GapRunsMuscleVsNoGapAUROCMean{} points over the stage without spacing and \GapRunsMuscleVsShuffledAUROCMean{} over shuffled spacing, with intervals excluding zero in \GapRunsMuscleVsNoGapAUROCExcl{} and \GapRunsMuscleVsShuffledAUROCExcl{} of the five runs; in the other four endpoints every AUROC, AUPRC, F1, and Brier interval against both controls excluded zero in all five runs (\texttt{experiments/runs/step11b/gap\_controls\_runs/}). In development, for each outer fold the stage was fitted on the training folds, using the scores of the networks trained for that fold; its penalty and threshold were chosen on the internal-validation fold, and it was applied to the held-out fold. Shuffled spacing permuted the spacing values once across the endpoint's development pairs, with a fixed seed.}\label{tab:gapcontrols}
\begin{tabular}{llrrrr}
\toprule
Endpoint & Comparator & AUROC & AUPRC & F1 & Brier \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


def seed_averaging() -> None:
    rows = []
    for context in ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]:
        frame = pd.read_csv(ROOT / f"experiments/runs/step11/seed_curve/{context}/d_avg.csv")
        cells = []
        for metric in ("auroc", "auprc", "f1"):
            row = frame[(frame.model == "predictor") & (frame.metric == metric)]
            if len(row) != 1:
                raise RuntimeError(f"{context}: predictor/{metric}")
            cells.append(minus(interval(row.iloc[0])))
        rows.append(f"{context_label(context)} & " + " & ".join(cells) + r" \\")
    write(
        "table_s12_seed_averaging.tex",
        r"""\begin{table}[H]
\centering
\footnotesize
\caption{\textbf{Averaging five independently trained copies of each component improves the selected predictor in every endpoint.} Each row gives the five-seed selected predictor minus the mean single-seed predictor on development out-of-fold predictions, as percentage-point medians with 95\% pair-clustered intervals (\SeedAvgBootstrapReplicates{} replicates).}\label{tab:seedavg}
\begin{tabular}{lrrr}
\toprule
Endpoint & AUROC & AUPRC & F1 \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )


def test_contrasts() -> None:
    """Selected predictor minus each comparator on the canonical test (evaluation 3): point difference with 95%
    pair-clustered interval, in percentage points. Aggregate contrasts file only."""
    labels = {
        "adaredit_reference": "Reproduced AdarEdit",
        "pnas_selected_predictor": "First-round predictor",
        "equal_average": "Equal-weight average, no second stage",
    }
    frame = pd.read_csv(ROOT / sources.TEST_CONTRASTS)
    if set(frame.replicates) != {2000}:
        raise RuntimeError(f"{sources.TEST_CONTRASTS}: replicates {sorted(set(frame.replicates))}, caption says 2,000")
    rows = []
    for context in ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]:
        for reference, label in labels.items():
            cells = []
            for metric in ("auroc", "auprc", "f1", "brier"):
                row = frame[(frame.context == context) & (frame.candidate == "gap_stage")
                            & (frame.reference == reference) & (frame.metric == metric)]
                if len(row) != 1:
                    raise RuntimeError(f"{context}: gap_stage/{reference}/{metric}: {len(row)} rows")
                r = row.iloc[0]
                cells.append(minus(f"{100 * r['estimate']:+.2f} [{100 * r['lower']:+.2f}, {100 * r['upper']:+.2f}]"))
            rows.append(f"{context_label(context)} & {label} & " + " & ".join(cells) + r" \\")
    write(
        "table_s13_test_contrasts.tex",
        r"""\begin{table}[H]
\centering
\footnotesize
\caption{\textbf{On the canonical test, the selected predictor improves on reproduced AdarEdit and on the first-round predictor in every endpoint.} Each row gives the selected predictor minus a comparator: reproduced supplied-checkpoint AdarEdit, the first-round predictor, or the equal-weight average of the same two components without a second stage. Values are percentage-point differences with 95\% pair-clustered intervals (2,000 replicates); for Brier score, negative values are improvements. The five endpoints share test pairs and are one test scored five ways.}\label{tab:testcontrasts}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrr}
\toprule
Endpoint & Comparator & AUROC & AUPRC & F1 & Brier \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\end{table}""",
    )


def nonhuman_transfer() -> None:
    """The frozen selected predictor of each human endpoint on AdarEdit's three nonhuman test sets, without
    retraining (results/adarpro/nonhuman_transfer). AUROC; differences in AUROC points with 95% region-bootstrap
    intervals; AUPRC is descriptive."""
    run = ROOT / "results/adarpro/nonhuman_transfer"
    contrasts, metrics = pd.read_csv(run / "contrasts.csv"), pd.read_csv(run / "metrics.csv")
    replicates = json.loads((run / "run_record.json").read_text())["bootstrap"]["replicates"]
    if replicates != 2000:
        raise RuntimeError(f"nonhuman transfer: {replicates} replicates, the caption says 2,000")
    verdict = {"BEATS": "better", "WORSE": "worse"}

    def cell(contrast: str, endpoint: str, species: str) -> pd.Series:
        rows = contrasts[(contrasts.contrast == contrast) & (contrasts.endpoint == endpoint)
                         & (contrasts.species == species)]
        if len(rows) != 1:
            raise RuntimeError(f"nonhuman transfer: {contrast} {endpoint} {species}: {len(rows)} rows")
        return rows.iloc[0]

    def auc(endpoint: str, species: str, model: str, column: str = "auroc") -> float:
        rows = metrics[(metrics.endpoint == endpoint) & (metrics.species == species) & (metrics.model == model)]
        if len(rows) != 1:
            raise RuntimeError(f"nonhuman transfer: {endpoint} {species} {model}: {len(rows)} rows")
        return float(rows.iloc[0][column])

    def difference(row: pd.Series) -> str:
        return minus(f"{100 * row.delta:.2f} [{100 * row.L95:.2f}, {100 * row.U95:.2f}]")

    # Two stacked halves of one table, so that it prints at the size of the other SI tables (16 columns in one
    # tabular had to be shrunk to about 5 pt): the upper half against AdarEdit trained on the same human endpoint,
    # the lower half against AdarEdit trained on the animal, without the spacing stage, and AUPRC.
    upper, lower = [], []
    for endpoint in ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]:
        for species in ["Octopus", "Ptychodera", "Strongylocentrotus"]:
            human = cell("selected_minus_adaredit_human_stronger", endpoint, species)
            own = cell("selected_minus_adaredit_animal_stronger", endpoint, species)
            spacing = cell("selected_minus_no_gap", endpoint, species)
            key = [context_label(endpoint), species if species == "Octopus" else rf"\textit{{{species}}}"]
            upper.append(" & ".join(key + [
                f"{int(human.n_sites):,}", str(int(human.n_regions)),
                f"{human.auroc_a:.3f}", f"{auc(endpoint, species, 'adaredit_human_baseline'):.3f}",
                f"{auc(endpoint, species, 'adaredit_human_bioaware'):.3f}", difference(human),
                verdict.get(human.word, "n.d."),
            ]) + r" \\")
            lower.append(" & ".join(key + [
                f"{own.auroc_b:.3f}", difference(own), verdict.get(own.word, "n.d."),
                f"{spacing.auroc_b:.3f}", difference(spacing), f"{auc(endpoint, species, 'selected', 'auprc'):.3f}",
                f"{auc(endpoint, species, 'adaredit_human_stronger', 'auprc'):.3f}",
            ]) + r" \\")
    write(
        "table_nonhuman_transfer.tex",
        r"""\begin{table}[H]
\centering
\footnotesize
\caption{\textbf{Transfer of the frozen selected predictor to nonhuman datasets.} Each row is the selected predictor of one human endpoint, trained only on human data, scored without retraining on one nonhuman test set (\textit{Ptychodera}, an acorn worm; \textit{Strongylocentrotus}, a sea urchin). AdarEdit's supplied compact and Bio-aware checkpoints are shown for the same human endpoint; $\Delta$ is the selected predictor minus the better of the two. AdarEdit trained on that animal is the better of its two checkpoints trained on that animal's own data. The spacing effect is the selected predictor minus the same networks without the genomic-spacing stage. Differences are in AUROC points with 95\% intervals from 2,000 paired bootstrap replicates resampling whole genomic regions; better and worse mean the interval excludes zero; n.d., the two cannot be told apart. AUPRC is descriptive.}\label{tab:nonhuman}
\begin{tabular}{llrrrrrrl}
\toprule
 & & & & \multicolumn{5}{c}{AUROC, same human endpoint} \\
\cmidrule(lr){5-9}
Endpoint & Animal & Sites & Regions & Selected & Compact & Bio-aware & $\Delta$ [95\%] & \\
\midrule
"""
        + "\n".join(upper)
        + r"""
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{llrrlrrrr}
\toprule
 & & \multicolumn{3}{c}{AdarEdit trained on that animal} & \multicolumn{2}{c}{Without spacing stage} & \multicolumn{2}{c}{AUPRC} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Endpoint & Animal & AUROC & $\Delta$ [95\%] & & AUROC & Spacing effect [95\%] & Selected & Better AdarEdit \\
\midrule
"""
        + "\n".join(lower)
        + r"""
\bottomrule
\end{tabular}
\end{table}""",
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
    gap_controls()
    seed_averaging()
    test_contrasts()
    nonhuman_transfer()
    manifest()
    print("PASS: rendered 14 SI tables from authoritative artifacts")


if __name__ == "__main__":
    main()
