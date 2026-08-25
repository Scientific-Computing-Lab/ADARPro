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
        "adarpro_pair_local_equal_logit_ensemble": "ADARPro",
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
    df = pd.read_csv(SOURCE / "figure2_evidence_map.csv")
    lines = [
        f"{tex(row['scale'])} & {tex(row['hypothesis'])} & {tex(row['status'].title())} & "
        f"{tex(row['interpretation'])} \\\\"
        for _, row in df.iterrows()
    ]
    write(
        "table_s2_evidence_map.tex",
        r"""\begin{table}[H]
\centering
\caption{Development evidence map. ``Not supported'' refers only to the tested measurement and formulation; deferred hypotheses lacked appropriately matched data and were not tested. Quantitative comparisons and their prespecified references are reported in the corresponding sections and source tables.}\label{tab:evidencemap}
\begin{tabularx}{\textwidth}{p{0.16\textwidth}p{0.25\textwidth}p{0.12\textwidth}X}
\toprule
Scale & Tested formulation & Evidence status & Interpretation \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabularx}
\end{table}""",
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
\caption{Same-fold decomposition of expanded supervision and pair-first weighting in muscle development data. Intervals are whole-pair bootstrap intervals for the incremental step indicated, in percentage points.}\label{tab:supervisiondecomposition}
\begin{tabular}{lrrrrr}
\toprule
Metric & Original reference & All high/low, site-balanced & $\Delta$ from labels [95\% CI] & All high/low, pair-first & $\Delta$ from pair weighting [95\% CI] \\
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
\caption{Substrate realization is measurable in development pairs. Spearman $\rho$ relates genomic inter-arm gap to edited-site fraction. The low-capacity pair model combines genomic geometry with graph-derived summaries; $\Delta R^2$ is relative to graph summaries alone with a 95\% whole-pair bootstrap interval. ``Any high'' denotes whether a pair contains at least one site with EditingIndex $\geq15\%$.}\label{tab:substraterealization}
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
        ("Aligned target + partner", alignment, "aligned_local"),
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
\caption{Pair-first expanded supervision versus the same-fold internal AdarEdit reference. Values are percentage-point median differences with 95\% whole-pair bootstrap intervals.}\label{tab:pairbalance}
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
\caption{Canonical-test false-positive transitions. Rescue and damage use frozen validation thresholds. The final column re-thresholds ADARPro to match published-baseline recall exactly.}\label{tab:falsepositives}
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
        "pair_balanced": "Pair-balanced",
        "local": "Target-local",
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
        "local": "Target-local",
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
\caption{Preregistered unified-model comparison on muscle development data. The unified architecture was rejected before canonical-test access.}\label{tab:unified}
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
    df = pd.read_csv(ROOT / "experiments/performance_frontier_opportunities.csv")
    status_names = {
        "terminate_sitewise_broad_panel_advance_geometry_as_engagement": "Broad panel terminated; geometry retained mechanistically",
        "validated_across_all_five_endpoints_selected_component": "Selected component",
        "terminated_from_primary_retain_secondary_ranking_result": "Limited ranking signal; not selected",
        "terminated_from_primary_secondary_shared_signal": "Limited shared signal; not selected",
        "terminated_after_stronger_model_control": "Terminated after stronger-model control",
        "terminated_current_formulations": "Terminated in tested formulation",
        "advance_to_local_fusion_not_standalone": "Advanced to aligned local encoder",
        "validated_complementary_ranking_signal_retain_for_fixed_ensemble": "Validated fallback/complementary view",
        "validated_selected_successor_component": "Selected component",
        "terminated_current_bulk_peak_overlap_formulation": "Bulk peak-overlap formulation terminated",
        "stopped_direct_fusion_after_three_fold_futility": "Stopped for futility",
        "defer_public_icshape_for_coverage_and_context_mismatch": "Deferred: coverage/context mismatch",
        "deferred_missing_authorized_site_by_donor_outcomes": "Deferred: no matched donor outcomes",
        "deferred_future_corpus_not_benchmark_ready": "Deferred: corpus not benchmark-ready",
        "terminated_after_fixed_150m_confirmation": "Terminated after 150M-model confirmation",
        "deferred_after_target_local_controlled_gain": "Deferred after lower-cost local gain",
        "deferred_missing_matched_state_and_outcomes": "Deferred: no matched state/outcomes",
        "deprioritized": "Deprioritized after RBP branch",
        "pair_local_equal_logit_selected": "Selected final integration",
        "terminated_after_preregistered_primary": "Terminated after preregistered primary",
        "terminated": "Terminated",
    }
    rows = []
    for _, row in df.sort_values("rank").iterrows():
        axis = tex(row["frontier"]).replace(" x ", r" $\times$ ")
        direction = tex(row["direction"]).replace(" x ", r" $\times$ ")
        rows.append(
            f"{row['rank']:g} & {axis} & "
            f"{direction} & {tex(status_names[row['initial_status']])} \\\\"
        )
    write(
        "table_s10_experiment_taxonomy.tex",
        r"""\begin{longtable}{p{0.06\textwidth}p{0.14\textwidth}p{0.33\textwidth}p{0.37\textwidth}}
\caption{Complete ranked performance-frontier taxonomy and final campaign decision. Full hypotheses, data sources, leakage risks, promotion rules, kill rules, and result summaries are retained in \texttt{experiments/performance\_frontier\_opportunities.csv}.}\label{tab:taxonomy}\\
\toprule
Rank & Axis & Direction & Final disposition \\
\midrule
\endfirsthead
\multicolumn{4}{l}{\textit{Table S7 continued}}\\
\toprule
Rank & Axis & Direction & Final disposition \\
\midrule
\endhead
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{longtable}""",
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
