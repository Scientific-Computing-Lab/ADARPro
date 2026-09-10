#!/usr/bin/env python3
"""Build the manuscript's numerical source-of-truth ledger and LaTeX macros."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
CONTEXTS = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]
INDIVIDUAL_TISSUES = CONTEXTS[:4]
CONTEXT_MACRO = {
    "Artery": "Artery",
    "Brain": "Brain",
    "Liver": "Liver",
    "MuscleSkeletal": "Muscle",
    "Combined": "Combined",
}
MODEL_MACRO = {
    "published_baseline_reproduced": "AdarEdit",
    "adarpro_pair_local_equal_logit_ensemble": "ADARPro",
}
MODEL_DISPLAY = {
    "published_baseline_reproduced": "AdarEdit",
    "adarpro_pair_local_equal_logit_ensemble": "selected predictor",
}
METRIC_MACRO = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "f1": "FOne",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "brier": "Brier",
}


@dataclass
class Claim:
    claim_id: str
    quantity: str
    exact_value: str
    display_value: str
    unit: str
    epistemic_status: str
    authoritative_artifact: str
    selector: str
    manuscript_use: str
    rounding_rule: str


claims: list[Claim] = []
macros: dict[str, str] = {}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def add(
    claim_id: str,
    quantity: str,
    value: float | str,
    display: str,
    unit: str,
    status: str,
    artifact: Path,
    selector: str,
    use: str,
    rounding: str,
    macro: str | None = None,
) -> None:
    exact = repr(value) if isinstance(value, (float, int)) else value
    claims.append(
        Claim(
            claim_id,
            quantity,
            exact,
            display,
            unit,
            status,
            relative(artifact),
            selector,
            use,
            rounding,
        )
    )
    if macro:
        macros[macro] = display


def add_split_claims() -> None:
    counts: dict[str, int] = {}
    for split, stem in (("Train", "train"), ("Validation", "valid"), ("Test", "test")):
        path = ROOT / f"data/human/Combined/{stem}.metadata.csv"
        value = int(pd.read_csv(path, usecols=["pair_id"])["pair_id"].nunique())
        counts[split] = value
        add(
            f"split_{stem}",
            f"Canonical {split.lower()} pair count",
            value,
            str(value),
            "Alu pairs",
            "benchmark definition",
            path,
            "nunique(pair_id)",
            "Abstract; Results; Methods; Figure 1",
            "integer",
            f"N{split}Pairs",
        )
    total = sum(counts.values())
    path = ROOT / "data/human/Combined/train.metadata.csv"
    add(
        "split_total",
        "Canonical benchmark pair count",
        total,
        str(total),
        "Alu pairs",
        "benchmark definition",
        path,
        "sum of unique pair counts in fixed train/validation/test metadata",
        "Abstract; Results; Methods; Figure 1",
        "integer",
        "NSubstrates",
    )


def add_protocol_claims() -> None:
    """Register numerical design constants used in the manuscript and SI."""
    label_source = ROOT / "src/adarpro/models/final_training.py"
    for claim_id, quantity, value, display, selector, macro in (
        (
            "label_positive_threshold",
            "Positive EditingIndex threshold",
            15,
            "15",
            "binary label: editing_index >= 15",
            "PositiveThresholdPct",
        ),
        (
            "label_negative_threshold",
            "Negative EditingIndex threshold",
            1,
            "1",
            "binary label: editing_index < 1",
            "NegativeThresholdPct",
        ),
    ):
        add(
            claim_id,
            quantity,
            value,
            display,
            "percent EditingIndex",
            "benchmark definition",
            label_source,
            selector,
            "Results; Methods; SI benchmark section",
            "integer percent",
            macro,
        )

    folds_path = ROOT / "data/derived/folds/pair_folds.csv"
    folds = pd.read_csv(folds_path)
    n_folds = int(folds["n_splits"].nunique() == 1 and folds["n_splits"].iloc[0])
    fold_seed = int(folds["fold_seed"].nunique() == 1 and folds["fold_seed"].iloc[0])
    for claim_id, quantity, value, macro in (
        (
            "development_fold_count",
            "Development outer-fold count",
            n_folds,
            "NOuterFolds",
        ),
        (
            "development_fold_seed",
            "Development fold seed",
            fold_seed,
            "DevelopmentFoldSeed",
        ),
    ):
        add(
            claim_id,
            quantity,
            value,
            str(value),
            "count" if "count" in claim_id else "seed",
            "development protocol",
            folds_path,
            "unique n_splits" if "count" in claim_id else "unique fold_seed",
            "Results; Methods; SI leakage-control section",
            "integer",
            macro,
        )

    local_path = ROOT / "src/adarpro/models/multitask_oof.py"
    local_text = local_path.read_text(encoding="utf-8")
    radius = int(
        re.search(r"^LOCAL_RADIUS = (\d+)$", local_text, re.MULTILINE).group(1)
    )
    shift = int(
        re.search(r"^LOCAL_CONTROL_SHIFT = (\d+)$", local_text, re.MULTILINE).group(1)
    )
    for claim_id, quantity, value, macro in (
        (
            "local_window_width",
            "Aligned local-window width",
            2 * radius + 1,
            "LocalWindowWidth",
        ),
        ("local_control_shift", "Shifted-control offset", shift, "LocalControlShift"),
    ):
        add(
            claim_id,
            quantity,
            value,
            str(value),
            "nucleotides",
            "model specification",
            local_path,
            "LOCAL_RADIUS" if "width" in claim_id else "LOCAL_CONTROL_SHIFT",
            "Abstract; Results; Methods; SI target-local section",
            "integer",
            macro,
        )

    manifest_path = ROOT / "provenance/final_pipeline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = {tuple(record["weights"]) for record in manifest["contexts"].values()}
    if weights != {(0.5, 0.5)}:
        raise AssertionError(f"unexpected final ensemble weights: {weights}")
    component_fits = sum(
        len(record["components"]) for record in manifest["contexts"].values()
    )
    add(
        "ensemble_component_weight",
        "Fixed weight assigned to each ensemble component",
        0.5,
        "0.5",
        "weight",
        "frozen model specification",
        manifest_path,
        "all contexts; weights=[0.5, 0.5]",
        "Results; Methods; Figure 3; SI final-training section",
        "one decimal",
        "EnsembleComponentWeight",
    )
    add(
        "final_component_fit_count",
        "Number of final component fits",
        component_fits,
        str(component_fits),
        "trained components",
        "frozen model specification",
        manifest_path,
        "sum(len(context.components))",
        "Results; SI final-training section",
        "integer",
        "NFinalComponentFits",
    )

    unified_path = (
        ROOT
        / "experiments/runs/unified/unified_target_local_interaction_aux_v1/MuscleSkeletal/summary.json"
    )
    unified = json.loads(unified_path.read_text(encoding="utf-8"))
    unified_values = (
        (
            "unified_rows",
            "Unified development row count",
            unified["canonical_rows"],
            "rows",
            "UnifiedRows",
        ),
        (
            "unified_pairs",
            "Unified development pair count",
            unified["pairs"],
            "Alu pairs",
            "UnifiedPairs",
        ),
        (
            "unified_parameters",
            "Unified trainable parameter count",
            unified["trainable_parameters"],
            "parameters",
            "UnifiedParameters",
        ),
        (
            "unified_epochs",
            "Unified maximum training epochs",
            unified["epochs"],
            "epochs",
            "UnifiedEpochs",
        ),
        (
            "unified_aux_weight",
            "Unified substrate auxiliary-loss weight",
            unified["auxiliary_global_loss_weight"],
            "weight",
            "UnifiedAuxWeight",
        ),
    )
    for claim_id, quantity, value, unit, macro in unified_values:
        display = f"{int(value):,}" if unit != "weight" else f"{value:.2f}"
        add(
            claim_id,
            quantity,
            value,
            display,
            unit,
            "preregistered model specification",
            unified_path,
            claim_id.removeprefix("unified_"),
            "Results; SI unified-model section",
            "integer with thousands separator" if unit != "weight" else "two decimals",
            macro,
        )

    unified_config_path = unified_path.parent / "outer_0/config.json"
    unified_config = json.loads(unified_config_path.read_text(encoding="utf-8"))
    decision_path = ROOT / "provenance/unified_development_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    for claim_id, quantity, value, unit, macro, selector in (
        (
            "unified_batch_size",
            "Unified training batch size",
            unified_config["batch_size"],
            "observations",
            "UnifiedBatchSize",
            "batch_size",
        ),
        (
            "unified_base_seed",
            "Unified preregistered base seed",
            decision["comparison"]["seed"],
            "seed",
            "UnifiedBaseSeed",
            "comparison.seed",
        ),
    ):
        add(
            claim_id,
            quantity,
            value,
            f"{int(value):,}",
            unit,
            "preregistered model specification",
            unified_config_path if claim_id == "unified_batch_size" else decision_path,
            selector,
            "SI unified-model section",
            "integer with thousands separator",
            macro,
        )

    report_path = ROOT / "reports/20_unified_end_to_end_result.md"
    report_text = report_path.read_text(encoding="utf-8")
    ensemble_parameters = int(
        re.search(r"([0-9,]+) parameters\nacross the ensemble", report_text)
        .group(1)
        .replace(",", "")
    )
    add(
        "ensemble_parameter_count",
        "Total parameter count across frozen ensemble components",
        ensemble_parameters,
        f"{ensemble_parameters:,}",
        "parameters",
        "preregistered comparator specification",
        report_path,
        "architecture paragraph: parameters across the ensemble",
        "Results; SI unified-model section",
        "integer with thousands separator",
        "EnsembleParameters",
    )

    opportunities_path = ROOT / "experiments/performance_frontier_opportunities.csv"
    opportunities = pd.read_csv(opportunities_path)
    pretraining = opportunities[
        opportunities["direction"] == "Self-supervised IRAlu pretraining"
    ].iloc[0]
    pretraining_pairs = int(
        re.search(r"(\d+) benchmark-excluded", pretraining["effective_power"]).group(1)
    )
    add(
        "pretraining_candidate_pairs",
        "Audited candidate pairs before transcript/foldability filtering",
        pretraining_pairs,
        f"{pretraining_pairs:,}",
        "candidate pairs",
        "deferred feasibility audit",
        opportunities_path,
        "direction=Self-supervised IRAlu pretraining; effective_power",
        "SI foundation-model section",
        "integer with thousands separator",
        "PretrainingCandidatePairs",
    )

    comparison_path = (
        ROOT
        / "experiments/runs/richer_supervision/pair_local_ensemble_comparison/MuscleSkeletal/summary.json"
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    add(
        "bootstrap_replicates",
        "Pair-clustered bootstrap replicate count",
        comparison["bootstrap_replicates"],
        f"{comparison['bootstrap_replicates']:,}",
        "replicates",
        "statistical protocol",
        comparison_path,
        "bootstrap_replicates",
        "Methods; SI leakage-control section",
        "integer with thousands separator",
        "NBootstrapReplicates",
    )

    quarantine_path = ROOT / "provenance/test_quarantine.json"
    quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
    evaluations = int(quarantine["final_locked_test"]["evaluations_consumed"])
    add(
        "canonical_evaluation_count",
        "Canonical test evaluation count",
        evaluations,
        str(evaluations),
        "evaluation",
        "canonical held-out protocol",
        quarantine_path,
        "final_locked_test.evaluations_consumed",
        "Abstract; Results; Methods; SI final-training section",
        "integer",
        "NCanonicalEvaluations",
    )


def add_locked_claims() -> None:
    path = ROOT / "results/adarpro/final_locked_test/metrics.csv"
    frame = pd.read_csv(path)
    metrics = ["auroc", "auprc", "f1", "precision", "recall", "specificity", "brier"]
    for _, row in frame.iterrows():
        context = row["context"]
        model = row["model"]
        for metric in metrics:
            value = float(row[metric])
            display = f"{value:.4f}"
            add(
                f"locked_{context}_{model}_{metric}",
                f"Canonical {context} {MODEL_DISPLAY[model]} {metric.upper()}",
                value,
                display,
                "metric",
                "canonical held-out",
                path,
                f"context={context}; model={model}; column={metric}",
                "Results; Figure 4; Table S1; README",
                "direct four-decimal formatting of frozen value",
                f"{CONTEXT_MACRO[context]}{MODEL_MACRO[model]}{METRIC_MACRO[metric]}",
            )
    pivot = frame.pivot(index="context", columns="model", values=metrics)
    for metric in metrics:
        candidate = pivot[metric]["adarpro_pair_local_equal_logit_ensemble"]
        baseline = pivot[metric]["published_baseline_reproduced"]
        delta = candidate - baseline
        for context in CONTEXTS:
            value = float(delta.loc[context])
            add(
                f"locked_delta_{context}_{metric}",
                f"Canonical {context} selected predictor minus AdarEdit {metric.upper()}",
                value,
                f"{value:.4f}",
                "metric difference",
                "canonical held-out",
                path,
                f"context={context}; candidate minus baseline; column={metric}",
                "Results; Figure 4",
                "direct four-decimal formatting of frozen difference",
                f"{CONTEXT_MACRO[context]}Delta{METRIC_MACRO[metric]}",
            )
        macro = float(delta.mean())
        add(
            f"locked_macro_delta_{metric}",
            f"Macro canonical selected predictor minus AdarEdit {metric.upper()}",
            macro,
            f"{macro:.4f}",
            "metric difference",
            "canonical held-out",
            path,
            f"unweighted mean of five context-specific {metric} differences",
            "Abstract; Results; Discussion; README",
            "direct four-decimal formatting of unrounded context differences",
            f"MacroDelta{METRIC_MACRO[metric]}",
        )

    transitions_path = ROOT / "figures/final/source_tables/final_error_transitions.csv"
    transitions = pd.read_csv(transitions_path).set_index("context")
    matched_path = ROOT / "figures/final/source_tables/final_matched_recall.csv"
    matched = pd.read_csv(matched_path).set_index("context")
    for context in CONTEXTS:
        for column, macro_suffix, unit, digits in (
            ("fp_rescue_rate", "FPRescueRate", "proportion", 1),
            ("tp_damage_rate", "TPDamageRate", "proportion", 1),
        ):
            value = float(transitions.loc[context, column])
            display = f"{100 * value:.{digits}f}"
            add(
                f"locked_{context}_{column}",
                f"Canonical {context} {column}",
                value,
                display,
                unit,
                "canonical held-out descriptive error transition",
                transitions_path,
                f"context={context}; column={column}",
                "Results; Figure 4; Table S9",
                "one decimal after conversion to percent",
                f"{CONTEXT_MACRO[context]}{macro_suffix}",
            )
        avoided = int(matched.loc[context, "false_positives_avoided"])
        add(
            f"locked_{context}_matched_recall_fp_avoided",
            f"Canonical {context} false positives avoided at matched recall",
            avoided,
            str(avoided),
            "sites",
            "canonical held-out descriptive operating-point analysis",
            matched_path,
            f"context={context}; column=false_positives_avoided",
            "Results; Figure 4; Table S9",
            "integer",
            f"{CONTEXT_MACRO[context]}MatchedRecallFPAvoided",
        )


def add_supervision_claims() -> None:
    metrics_path = (
        ROOT / "experiments/runs/richer_supervision/pair_balance_comparison/metrics.csv"
    )
    ci_path = (
        ROOT
        / "experiments/runs/richer_supervision/pair_balance_comparison/pair_bootstrap_ci.csv"
    )
    metrics = pd.read_csv(metrics_path).set_index("model")
    cis = pd.read_csv(ci_path)
    model_names = {
        "development_reference": "InternalReference",
        "expanded_site": "ExpandedSite",
        "expanded_pair": "PairFirst",
    }
    for model, macro_name in model_names.items():
        for metric in ("auroc", "auprc", "f1", "brier"):
            value = float(metrics.loc[model, metric])
            add(
                f"supervision_{model}_{metric}",
                f"Muscle development {model} {metric}",
                value,
                f"{value:.4f}",
                "metric",
                "development OOF",
                metrics_path,
                f"model={model}; column={metric}",
                "Results; Figure 3; SI supervision section",
                "direct four-decimal formatting",
                f"Muscle{macro_name}{METRIC_MACRO[metric]}",
            )
    comparisons = {
        ("expanded_site", "development_reference"): "LabelExpansion",
        ("expanded_pair", "expanded_site"): "PairFirstIncrement",
        ("expanded_pair", "development_reference"): "PairFirstTotal",
    }
    for (candidate, reference), label in comparisons.items():
        source_candidate, source_reference = candidate, reference
        invert = False
        subset = cis[(cis.candidate == candidate) & (cis.reference == reference)]
        if (
            subset.empty
            and candidate == "expanded_pair"
            and reference == "expanded_site"
        ):
            subset = cis[
                (cis.candidate == "expanded_site") & (cis.reference == "expanded_pair")
            ]
            source_candidate, source_reference = "expanded_site", "expanded_pair"
            invert = True
        for metric in ("auroc", "auprc", "f1", "brier"):
            row = subset[subset.metric == metric].iloc[0]
            low, med, high = map(float, (row.lower, row["median"], row.upper))
            if invert:
                low, med, high = -high, -med, -low
            for part, value in (("Median", med), ("Lower", low), ("Upper", high)):
                add(
                    f"supervision_{label}_{metric}_{part.lower()}",
                    f"Muscle development {label} {metric} {part.lower()}",
                    value,
                    f"{100 * value:+.2f}",
                    "percentage points",
                    "development OOF pair-clustered bootstrap",
                    ci_path,
                    f"candidate={source_candidate}; reference={source_reference}; metric={metric}; transformed={invert}",
                    "Results; Figure 3; Table S4",
                    "two decimals after multiplication by 100",
                    f"Muscle{label}{METRIC_MACRO[metric]}{part}",
                )

    for context in CONTEXTS:
        path = (
            ROOT
            / f"experiments/runs/richer_supervision/binary_pair_balanced/{context}/pair_bootstrap_ci.csv"
        )
        frame = pd.read_csv(path)
        for metric in ("auroc", "auprc", "f1"):
            row = frame[frame.metric == metric].iloc[0]
            for part in ("lower", "median", "upper"):
                value = float(row[part])
                add(
                    f"pair_first_{context}_{metric}_{part}",
                    f"{context} pair-first versus internal reference {metric} {part}",
                    value,
                    f"{100 * value:+.2f}",
                    "percentage points",
                    "development OOF pair-clustered bootstrap",
                    path,
                    f"metric={metric}; column={part}",
                    "Results; Figure 3; Table S5",
                    "two decimals after multiplication by 100",
                    f"{CONTEXT_MACRO[context]}PairFirst{METRIC_MACRO[metric]}{part.title()}",
                )


def add_substrate_claims() -> None:
    for context in INDIVIDUAL_TISSUES:
        pred_path = (
            ROOT
            / f"experiments/runs/pair_engagement/{context}/oof_pair_predictions.parquet"
        )
        frame = pd.read_parquet(pred_path)
        frame = frame[frame.model == "training_mean"].drop_duplicates("pair_id")
        rho = float(
            spearmanr(
                frame["geometry__genomic_gap"], frame["edited_fraction"]
            ).statistic
        )
        add(
            f"gap_rho_{context}",
            f"{context} inter-arm genomic gap versus edited-site fraction Spearman rho",
            rho,
            f"{rho:.3f}",
            "Spearman rho",
            "development descriptive observation",
            pred_path,
            "model=training_mean; one row per pair; spearmanr(genomic_gap, edited_fraction)",
            "Results; Figure 2; SI substrate-realization section",
            "three decimals",
            f"{CONTEXT_MACRO[context]}GapRho",
        )
        metrics_path = (
            ROOT / f"experiments/runs/pair_engagement_ablation/{context}/metrics.csv"
        )
        metrics = pd.read_csv(metrics_path).set_index("model")
        for model, macro_name in (
            ("graph_intrinsic", "GraphOnly"),
            ("combined_16", "CombinedGeometry"),
        ):
            for metric in ("r2", "spearman", "any_edited_auroc"):
                value = float(metrics.loc[model, metric])
                metric_macro = {
                    "r2": "RSquared",
                    "spearman": "Spearman",
                    "any_edited_auroc": "AnyEditedAUROC",
                }[metric]
                add(
                    f"engagement_{context}_{model}_{metric}",
                    f"{context} pair-level {model} {metric}",
                    value,
                    f"{value:.3f}",
                    "metric",
                    "development pair-level OOF",
                    metrics_path,
                    f"model={model}; column={metric}",
                    "Results; Figure 2; Table S3",
                    "three decimals",
                    f"{CONTEXT_MACRO[context]}Engagement{macro_name}{metric_macro}",
                )


def add_local_and_unified_claims() -> None:
    align_metrics_path = (
        ROOT
        / "experiments/runs/richer_supervision/local_alignment_control/MuscleSkeletal/metrics.csv"
    )
    align_metrics = pd.read_csv(align_metrics_path).set_index("model")
    for model, macro_name in (
        ("pair_balanced", "Pair"),
        ("aligned_local", "Local"),
        ("shifted_local", "Shifted"),
    ):
        for metric in ("auroc", "auprc", "f1"):
            value = float(align_metrics.loc[model, metric])
            add(
                f"alignment_{model}_{metric}",
                f"Muscle development {model} {metric}",
                value,
                f"{value:.4f}",
                "metric",
                "development OOF",
                align_metrics_path,
                f"model={model}; column={metric}",
                "Results; Figure 3; SI target-local section",
                "direct four-decimal formatting",
                f"Muscle{macro_name}{METRIC_MACRO[metric]}",
            )
    alignment_ci_path = (
        ROOT
        / "experiments/runs/richer_supervision/local_alignment_control/MuscleSkeletal/pair_bootstrap_ci.csv"
    )
    alignment_ci = pd.read_csv(alignment_ci_path)
    subset = alignment_ci[
        (alignment_ci.candidate == "aligned_local")
        & (alignment_ci.reference == "shifted_local")
    ]
    for metric in ("auroc", "auprc", "f1"):
        row = subset[subset.metric == metric].iloc[0]
        for part in ("lower", "median", "upper"):
            value = float(row[part])
            add(
                f"alignment_delta_{metric}_{part}",
                f"Aligned minus shifted {metric} {part}",
                value,
                f"{value:.4f}",
                "metric difference",
                "development OOF pair-clustered bootstrap",
                alignment_ci_path,
                f"candidate=aligned_local; reference=shifted_local; metric={metric}; column={part}",
                "Results; Figure 3",
                "direct four-decimal formatting",
                f"AlignedMinusShifted{METRIC_MACRO[metric]}{part.title()}",
            )

    for context in ("MuscleSkeletal", "Liver"):
        path = (
            ROOT
            / f"experiments/runs/richer_supervision/pair_local_ensemble_comparison/{context}/pair_bootstrap_ci.csv"
        )
        frame = pd.read_csv(path)
        subset = frame[
            (frame.candidate == "pair_balanced")
            & (frame.reference == "pair_local_ensemble")
        ]
        for metric in ("auroc", "auprc", "f1"):
            row = subset[subset.metric == metric].iloc[0]
            low, med, high = -float(row.upper), -float(row["median"]), -float(row.lower)
            for part, value in (("Lower", low), ("Median", med), ("Upper", high)):
                add(
                    f"ensemble_{context}_{metric}_{part.lower()}",
                    f"{context} fixed ensemble minus pair model {metric} {part.lower()}",
                    value,
                    f"{value:.4f}",
                    "metric difference",
                    "development OOF pair-clustered bootstrap",
                    path,
                    f"candidate=pair_balanced; reference=pair_local_ensemble; metric={metric}; sign inverted",
                    "Results; Figure 3",
                    "direct four-decimal formatting",
                    f"{CONTEXT_MACRO[context]}EnsembleDelta{METRIC_MACRO[metric]}{part}",
                )

    unified_metrics_path = (
        ROOT / "experiments/runs/unified/comparison_primary/MuscleSkeletal/metrics.csv"
    )
    unified_metrics = pd.read_csv(unified_metrics_path).set_index("model")
    for model, macro_name in (
        ("unified", "Unified"),
        ("frozen_ensemble", "FrozenEnsemble"),
    ):
        for metric in ("auroc", "auprc", "f1", "brier"):
            value = float(unified_metrics.loc[model, metric])
            add(
                f"unified_{model}_{metric}",
                f"Muscle preregistered {model} {metric}",
                value,
                f"{value:.4f}",
                "metric",
                "preregistered development OOF",
                unified_metrics_path,
                f"model={model}; column={metric}",
                "Results; Figure S; Table S",
                "direct four-decimal formatting",
                f"{macro_name}{METRIC_MACRO[metric]}",
            )

    unified_ci_path = (
        ROOT
        / "experiments/runs/unified/comparison_primary/MuscleSkeletal/pair_bootstrap_ci.csv"
    )
    unified_ci = pd.read_csv(unified_ci_path)
    unified_ci = unified_ci[
        (unified_ci.candidate == "unified")
        & (unified_ci.reference == "frozen_ensemble")
    ]
    for metric in ("auroc", "auprc", "f1", "brier"):
        row = unified_ci[unified_ci.metric == metric].iloc[0]
        for part in ("lower", "median", "upper"):
            value = float(row[part])
            add(
                f"unified_delta_{metric}_{part}",
                f"Unified minus frozen ensemble {metric} {part}",
                value,
                f"{value:.4f}",
                "metric difference",
                "preregistered development OOF pair-clustered bootstrap",
                unified_ci_path,
                f"candidate=unified; reference=frozen_ensemble; metric={metric}; column={part}",
                "Results; Figure S4; Table S8",
                "direct four-decimal formatting",
                f"UnifiedDelta{METRIC_MACRO[metric]}{part.title()}",
            )


def add_limited_branch_claims() -> None:
    multitask_path = (
        ROOT
        / "experiments/runs/richer_supervision/pair_balanced_multitask_comparison/pair_bootstrap_ci.csv"
    )
    multitask = pd.read_csv(multitask_path)
    multitask = multitask[
        (multitask.candidate == "expanded_pair")
        & (multitask.reference == "multitask_pair")
    ]
    for metric in ("auroc", "auprc", "f1", "brier"):
        row = multitask[multitask.metric == metric].iloc[0]
        low, med, high = -float(row.upper), -float(row["median"]), -float(row.lower)
        for part, value in (("Lower", low), ("Median", med), ("Upper", high)):
            add(
                f"multitask_delta_{metric}_{part.lower()}",
                f"Multitask minus pair-first binary {metric} {part.lower()}",
                value,
                f"{value:.4f}",
                "metric difference",
                "development OOF pair-clustered bootstrap",
                multitask_path,
                f"stored candidate=expanded_pair; reference=multitask_pair; metric={metric}; sign inverted",
                "Results; SI richer-supervision section",
                "direct four-decimal formatting",
                f"MultitaskDelta{METRIC_MACRO[metric]}{part}",
            )

    shared_path = (
        ROOT
        / "experiments/runs/joint_multilabel/vs_independent/global_pair_bootstrap_ci.csv"
    )
    shared = pd.read_csv(shared_path)
    for metric in ("auroc", "auprc", "f1", "brier"):
        row = shared[(shared.context == "macro") & (shared.metric == metric)].iloc[0]
        for part in ("lower", "median", "upper"):
            value = float(row[part])
            add(
                f"shared_macro_{metric}_{part}",
                f"Shared four-tissue minus independent macro {metric} {part}",
                value,
                f"{value:.4f}",
                "metric difference",
                "development OOF pair-clustered bootstrap",
                shared_path,
                f"context=macro; metric={metric}; column={part}",
                "Results; SI cross-tissue section",
                "direct four-decimal formatting",
                f"SharedMacroDelta{METRIC_MACRO[metric]}{part.title()}",
            )

    all_a_path = (
        ROOT
        / "experiments/runs/substrate_wide/joint_binary_partial_futility/MuscleSkeletal/pair_bootstrap_ci.csv"
    )
    all_a = pd.read_csv(all_a_path)
    all_a = all_a[(all_a.candidate == "control") & (all_a.reference == "pair_balanced")]
    for metric in ("auroc", "auprc", "f1", "brier"):
        row = all_a[all_a.metric == metric].iloc[0]
        for part in ("lower", "median", "upper"):
            value = float(row[part])
            add(
                f"all_a_delta_{metric}_{part}",
                f"All-A decoder minus pair-first {metric} {part}",
                value,
                f"{value:.4f}",
                "metric difference",
                "development OOF pair-clustered bootstrap",
                all_a_path,
                f"candidate=control; reference=pair_balanced; metric={metric}; column={part}",
                "Results; SI substrate-wide section",
                "direct four-decimal formatting",
                f"AllADelta{METRIC_MACRO[metric]}{part.title()}",
            )

    gate_path = (
        ROOT
        / "experiments/runs/hierarchical_gate/binary_natural/MuscleSkeletal/pair_bootstrap_ci.csv"
    )
    gate = pd.read_csv(gate_path)
    gate = gate[gate.model == "expanded_plus_combined_engagement"]
    for metric in ("auroc", "auprc", "f1", "brier"):
        row = gate[gate.metric == metric].iloc[0]
        for part in ("lower", "median", "upper"):
            value = float(row[part])
            add(
                f"engagement_gate_delta_{metric}_{part}",
                f"Engagement gate minus expanded reference {metric} {part}",
                value,
                f"{value:.4f}",
                "metric difference",
                "development OOF pair-clustered bootstrap",
                gate_path,
                f"model=expanded_plus_combined_engagement; metric={metric}; column={part}",
                "Results; SI substrate-realization section",
                "direct four-decimal formatting",
                f"EngagementGateDelta{METRIC_MACRO[metric]}{part.title()}",
            )


def add_synthesis_table_claims() -> None:
    """Add quantitative branches surfaced in the SI evidence synthesis."""

    specifications = [
        (
            "target_geometry",
            ROOT
            / "experiments/runs/target_geometry_residual/MuscleSkeletal/pair_bootstrap_ci.csv",
            {"model": "baseline_plus_multiscale_structure"},
            "development OOF pair-clustered bootstrap",
        ),
        (
            "focal_node",
            ROOT
            / "experiments/runs/richer_supervision/pooling_comparison/pair_bootstrap_ci.csv",
            {"candidate": "focal", "reference": "pair"},
            "development OOF pair-clustered bootstrap",
        ),
        (
            "h3k36me3_liver",
            ROOT / "experiments/runs/chromatin_residual/Liver/pair_bootstrap_ci.csv",
            {"model": "baseline_plus_h3k36me3_multiscale_occupancy"},
            "development OOF pair-clustered bootstrap",
        ),
        (
            "rbp_suppressors",
            ROOT
            / "experiments/runs/residual_pilot/MuscleSkeletal/pair_bootstrap_ci.csv",
            {"model": "baseline_plus_biological_suppressors"},
            "development OOF pair-clustered bootstrap",
        ),
        (
            "transcript_repeat_panel",
            ROOT
            / "experiments/runs/genomic_residual_v2/MuscleSkeletal/pair_bootstrap_ci.csv",
            {"model": "baseline_plus_architecture_and_intrinsic"},
            "development OOF pair-clustered bootstrap",
        ),
        (
            "rinalmo_target",
            ROOT
            / "experiments/runs/foundation_residual/rinalmo_mega_target/MuscleSkeletal/pair_bootstrap_ci.csv",
            {"candidate": "foundation_target", "reference": "base_recalibrated"},
            "development OOF pair-clustered bootstrap",
        ),
    ]
    for prefix, path, selectors, status in specifications:
        frame = pd.read_csv(path)
        for column, value in selectors.items():
            frame = frame[frame[column] == value]
        for metric in ("auroc", "auprc", "f1"):
            row = frame[frame.metric == metric].iloc[0]
            selector = "; ".join(f"{key}={value}" for key, value in selectors.items())
            for part in ("lower", "median", "upper"):
                value = float(row[part])
                add(
                    f"synthesis_{prefix}_{metric}_{part}",
                    f"SI evidence synthesis {prefix} {metric} {part}",
                    value,
                    f"{100 * value:+.2f}",
                    "percentage points",
                    status,
                    path,
                    f"{selector}; metric={metric}; column={part}",
                    "Table S2 evidence synthesis",
                    "two decimals after multiplication by 100",
                )

    for context in INDIVIDUAL_TISSUES:
        path = (
            ROOT
            / f"experiments/runs/pair_engagement_ablation/{context}/pair_bootstrap_ci.csv"
        )
        frame = pd.read_csv(path)
        row = frame[
            (frame.model == "combined_16")
            & (frame.reference == "graph_intrinsic")
            & (frame.metric == "r2")
        ].iloc[0]
        for part in ("lower", "median", "upper"):
            value = float(row[part])
            add(
                f"synthesis_pair_burden_{context}_r2_{part}",
                f"{context} geometry plus graph minus graph-only R2 {part}",
                value,
                f"{100 * value:+.2f}",
                "R2 percentage points",
                "development pair-level OOF pair-clustered bootstrap",
                path,
                f"model=combined_16; reference=graph_intrinsic; metric=r2; column={part}",
                "Figure 2; Table S2; Table S3",
                "two decimals after multiplication by 100",
            )


def add_post_analysis_claims() -> None:
    """Register retained post-hoc development diagnostics."""

    variation_path = (
        ROOT
        / "experiments/runs/post_analysis/tissue_variation/variation_summary.csv"
    )
    variation = pd.read_csv(variation_path)
    all_four = variation[variation.group == "all_four_tissues"].set_index("metric")
    summary_path = ROOT / "experiments/runs/post_analysis/tissue_variation/summary.json"
    summary = json.loads(summary_path.read_text())
    for claim_id, quantity, value, macro in (
        (
            "matched_sites_two_tissues",
            "Sites observed in at least two individual tissues",
            int(summary["sites_observed_in_at_least_two_tissues"]),
            "NMatchedSitesTwoTissues",
        ),
        (
            "matched_sites_four_tissues",
            "Sites observed in all four individual tissues",
            int(summary["sites_observed_in_all_four_tissues"]),
            "NMatchedSitesFourTissues",
        ),
        (
            "strict_switch_sites",
            "Sites with at least one strict edited/nonedited tissue switch",
            int(summary["unique_strict_switch_sites"]),
            "NStrictSwitchSites",
        ),
    ):
        add(
            claim_id,
            quantity,
            value,
            f"{value:,}",
            "sites",
            "post-hoc descriptive development analysis",
            summary_path,
            quantity,
            "Results; SI post-hoc scale diagnostics; Figure S7",
            "integer with thousands separator",
            macro,
        )
    for threshold, macro in (
        (1, "AllFourRangeOnePct"),
        (5, "AllFourRangeFivePct"),
        (15, "AllFourRangeFifteenPct"),
    ):
        row = all_four.loc[f"fraction_range_ge_{threshold}"]
        add(
            f"all_four_range_ge_{threshold}",
            f"Fraction of four-tissue matched sites with EditingIndex range >= {threshold}",
            float(row["value"]),
            f"{100 * row['value']:.1f}",
            "percent",
            "post-hoc descriptive development analysis with pair-clustered interval",
            variation_path,
            f"group=all_four_tissues; metric=fraction_range_ge_{threshold}; column=value",
            "Results; SI post-hoc scale diagnostics; Figure S7",
            "one decimal after multiplication by 100",
            macro,
        )

    stability_path = (
        ROOT
        / "experiments/runs/post_analysis/pair_stability/cross_tissue_pair_stability.csv"
    )
    stability = pd.read_csv(stability_path)
    for metric, stem in (
        ("mean_editing_spearman", "PairMeanEditingRho"),
        ("high_fraction_spearman", "PairHighFractionRho"),
    ):
        subset = stability[stability.metric == metric]
        for bound, reducer in (("Min", min), ("Max", max)):
            value = float(reducer(subset.spearman))
            add(
                f"{metric}_{bound.lower()}",
                f"{metric} across-tissue {bound.lower()}",
                value,
                f"{value:.3f}",
                "Spearman rho",
                "post-hoc descriptive matched-site development analysis",
                stability_path,
                f"metric={metric}; {bound.lower()}(spearman)",
                "Results; SI post-hoc scale diagnostics; Figure S7",
                "three decimals",
                f"{stem}{bound}",
            )

    within_path = (
        ROOT / "experiments/runs/post_analysis/within_duplex/within_duplex_contrasts.csv"
    )
    within = pd.read_csv(within_path)
    for context in ("Liver", "MuscleSkeletal"):
        for candidate, stem in (
            ("target_local", "WithinLocal"),
            ("fixed_ensemble", "WithinEnsemble"),
        ):
            row = within[
                within.context.eq(context) & within.candidate.eq(candidate)
            ].iloc[0]
            context_stem = CONTEXT_MACRO[context]
            for column, suffix in (
                ("lower", "Lower"),
                ("mean_difference", "Mean"),
                ("upper", "Upper"),
            ):
                value = float(row[column])
                add(
                    f"within_pair_{context}_{candidate}_{column}",
                    f"{context} {candidate} minus pair-first mean within-pair AUROC {column}",
                    value,
                    f"{100 * value:+.2f}",
                    "AUROC percentage points",
                    "post-hoc descriptive development OOF pair-bootstrap",
                    within_path,
                    f"context={context}; candidate={candidate}; column={column}",
                    "Results; SI post-hoc scale diagnostics; Figure S8",
                    "two decimals after multiplication by 100",
                    f"{context_stem}{stem}{suffix}",
                )


def write_outputs() -> None:
    ledger_path = PAPER / "numerical_claim_ledger.csv"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(claims[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(claim) for claim in claims)
    macro_path = PAPER / "generated/quantitative_claims.tex"
    lines = [
        "% Generated by scripts/build_paper_claim_ledger.py; do not edit by hand.",
        r"\providecommand{\PaperRoundingPolicy}{Four decimals for absolute metrics; two percentage points for development effects; three decimals for correlations and pair-level metrics.}",
    ]
    lines.extend(
        rf"\providecommand{{\{name}}}{{{value}}}"
        for name, value in sorted(macros.items())
    )
    macro_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: wrote {len(claims)} numerical claims and {len(macros)} LaTeX macros")


def main() -> None:
    add_split_claims()
    add_protocol_claims()
    add_locked_claims()
    add_supervision_claims()
    add_substrate_claims()
    add_local_and_unified_claims()
    add_limited_branch_claims()
    add_synthesis_table_claims()
    add_post_analysis_claims()
    write_outputs()


if __name__ == "__main__":
    main()
