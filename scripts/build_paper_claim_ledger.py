#!/usr/bin/env python3
"""Build the manuscript's numerical source-of-truth ledger and LaTeX macros."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import paper_sources as sources
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
# Test set (R9): the selected predictor is the gap stage (evaluation 3); AdarEdit is unchanged.
MODEL_MACRO = {
    sources.TEST_BASELINE: "AdarEdit",
    sources.TEST_CANDIDATE: "ADARPro",
}
MODEL_DISPLAY = {
    sources.TEST_BASELINE: "AdarEdit",
    sources.TEST_CANDIDATE: "selected predictor",
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

    unified_path = sources.path(
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

    # PNAS read this from its report's prose; step 11c counts graph + CNN alone from the step-4 checkpoints
    report_path = ROOT / "reports/20_unified_end_to_end_result.md"
    counts_path = sources.path(report_path)
    counts = sources.read_json(report_path)
    ensemble_parameters = int(counts["total_parameters"])
    if ensemble_parameters != sum(int(part["parameters"]) for part in counts["components"].values()):
        raise AssertionError(f"{counts_path}: total_parameters is not the sum of its components")
    add(
        "ensemble_parameter_count",
        "Total parameter count across frozen ensemble components",
        ensemble_parameters,
        f"{ensemble_parameters:,}",
        "parameters",
        "preregistered comparator specification",
        counts_path,
        "total_parameters (= components.graph + components.cnn_alone)",
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

    # The text uses this count for the final (test) comparison (main.tex Methods, SI): read it from the test
    # session's summary (evaluation 3), not from a development run, and check every test interval used it.
    comparison_path = sources.path(ROOT / "results/adarpro/final_locked_test/summary.json")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    contrast_replicates = set(pd.read_csv(ROOT / sources.TEST_CONTRASTS)["replicates"].astype(int))
    if contrast_replicates != {int(comparison["bootstrap_replicates"])}:
        raise RuntimeError(f"test contrasts use {contrast_replicates} replicates, summary says "
                           f"{comparison['bootstrap_replicates']}")
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
    """Test-set claims (R9: evaluation 3 via paper_sources; the selected predictor is the gap stage)."""
    pnas_metrics = ROOT / "results/adarpro/final_locked_test/metrics.csv"
    path = sources.path(pnas_metrics)
    frame = sources.read_csv(pnas_metrics)  # AdarEdit and the gap stage only
    if sorted(frame.model.unique()) != sorted(MODEL_MACRO) or len(frame) != 2 * len(CONTEXTS):
        raise RuntimeError(f"{path}: expected AdarEdit and the selected predictor in every endpoint")
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
        candidate = pivot[metric][sources.TEST_CANDIDATE]
        baseline = pivot[metric][sources.TEST_BASELINE]
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

    transitions_pnas = ROOT / "figures/final/source_tables/final_error_transitions.csv"
    transitions_path = sources.path(transitions_pnas)
    transitions = sources.read_csv(transitions_pnas).set_index("context")
    matched_pnas = ROOT / "figures/final/source_tables/final_matched_recall.csv"
    matched_path = sources.path(matched_pnas)
    matched = sources.read_csv(matched_pnas).set_index("context")
    # the transitions must be the selected predictor's against AdarEdit (counts reconcile with metrics.csv)
    by_model = frame.set_index(["context", "model"])
    for context in CONTEXTS:
        base, cand = by_model.loc[(context, sources.TEST_BASELINE)], by_model.loc[(context, sources.TEST_CANDIDATE)]
        row = transitions.loc[context]
        if not (row.baseline_fp == base.fp and row.baseline_tp == base.tp
                and row.baseline_fp - row.fp_rescued + row.new_fp == cand.fp
                and row.baseline_tp - row.tp_damaged + row.fn_recovered == cand.tp
                and matched.loc[context, "baseline_fp"] == base.fp):
            raise RuntimeError(f"{transitions_path}: {context} does not reconcile with {path}")
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
    # The text prints rescue and damage as ranges; with evaluation 3 the extremes are not Combined and Liver, so the
    # ranges get endpoint-free min/max macros (the per-endpoint macros above stay).
    for column, macro_suffix in (("fp_rescue_rate", "FPRescueRate"), ("tp_damage_rate", "TPDamageRate")):
        values = transitions.loc[CONTEXTS, column].astype(float)
        for which, context in (("Min", values.idxmin()), ("Max", values.idxmax())):
            value = float(values.loc[context])
            add(
                f"locked_{column}_{which.lower()}",
                f"{which} over endpoints of canonical {column}",
                value,
                f"{100 * value:.1f}",
                "proportion",
                "canonical held-out descriptive error transition",
                transitions_path,
                f"{which.lower()} over the five endpoints of {column} (attained in {context})",
                "Results; Figure 4",
                "one decimal after conversion to percent",
                f"{macro_suffix}{which}",
            )
    # U3 (WD51): the rescue rate is gross; the text also prints the net false-positive reduction,
    # (rescued - new false positives) / baseline false positives, as an endpoint-free range.
    net = ((transitions.loc[CONTEXTS, "fp_rescued"] - transitions.loc[CONTEXTS, "new_fp"])
           / transitions.loc[CONTEXTS, "baseline_fp"]).astype(float)
    if not (net > 0).all():
        raise RuntimeError("net false-positive reduction is not positive in every endpoint; revise the text")
    for which, context in (("Min", net.idxmin()), ("Max", net.idxmax())):
        value = float(net.loc[context])
        add(
            f"locked_net_fp_reduction_{which.lower()}",
            f"{which} over endpoints of canonical net false-positive reduction",
            value,
            f"{100 * value:.1f}",
            "proportion",
            "canonical held-out descriptive error transition",
            transitions_path,
            f"{which.lower()} over five endpoints of (fp_rescued - new_fp) / baseline_fp (attained in {context})",
            "Results",
            "one decimal after conversion to percent",
            f"NetFPReduction{which}",
        )
    exceeds = int((transitions.loc[CONTEXTS, "fp_rescue_rate"] > transitions.loc[CONTEXTS, "tp_damage_rate"]).sum())
    if exceeds != len(CONTEXTS):  # Fig. 6D title and caption, SI: rescue exceeds damage
        raise RuntimeError(f"FP rescue exceeds TP damage in {exceeds} of {len(CONTEXTS)} endpoints; revise the text")
    add(
        "locked_rescue_exceeds_damage_endpoints",
        "Endpoints whose FP rescue rate exceeds their TP damage rate",
        exceeds,
        str(exceeds),
        "endpoints",
        "canonical held-out descriptive error transition",
        transitions_path,
        "count over five endpoints of fp_rescue_rate > tp_damage_rate",
        "Results; Figure 4; SI final-evaluation section",
        "integer",
    )


def add_test_contrast_claims() -> None:
    """Evaluation 3: the selected predictor (gap stage) minus PNAS's selected predictor and minus the equal average,
    per endpoint, in percentage points (R9). Claims only, plus summary macros, for wording the text may add."""
    path = ROOT / sources.TEST_CONTRASTS
    frame = pd.read_csv(path)
    metrics_path = sources.path(ROOT / "results/adarpro/final_locked_test/metrics.csv")
    all_models = pd.read_csv(metrics_path).set_index(["context", "model"])  # all four models, for the anchors
    pnas = pd.read_csv(ROOT / "results/adarpro/final_locked_test/metrics.csv").set_index(["context", "model"])
    shown = ("auroc", "auprc", "f1", "brier")
    model_of = {
        "adaredit_reference": sources.TEST_BASELINE,
        "pnas_selected_predictor": "pnas_selected_predictor",
        "equal_average": "equal_average",
    }
    for context in CONTEXTS:  # anchors: v3's PNAS predictor is PNAS's published one; estimates are metric differences
        for metric in shown:
            published = pnas.loc[(context, sources.PNAS_TEST_CANDIDATE), metric]
            if abs(all_models.loc[(context, "pnas_selected_predictor"), metric] - published) > 1e-12:
                raise RuntimeError(f"{metrics_path}: {context} pnas_selected_predictor {metric} is not PNAS's value")
            for reference, model in model_of.items():
                row = frame[(frame.context == context) & (frame.candidate == "gap_stage")
                            & (frame.reference == reference) & (frame.metric == metric)]
                if len(row) != 1:
                    raise RuntimeError(f"{path}: {context} gap_stage vs {reference} {metric}: {len(row)} rows")
                difference = (all_models.loc[(context, sources.TEST_CANDIDATE), metric]
                              - all_models.loc[(context, model), metric])
                if abs(float(row.iloc[0]["estimate"]) - difference) > 1e-12:
                    raise RuntimeError(f"{path}: {context} vs {reference} {metric} estimate is not the metric difference")
    labels = {"pnas_selected_predictor": ("PNAS", "PNAS's selected predictor"),
              "equal_average": ("Equal", "the equal average")}
    estimates: dict[tuple[str, str], list[float]] = {}
    for reference, (tag, label) in labels.items():
        for context in CONTEXTS:
            for metric in shown:
                row = frame[(frame.context == context) & (frame.candidate == "gap_stage")
                            & (frame.reference == reference) & (frame.metric == metric)].iloc[0]
                for stat in ("estimate", "lower", "upper"):
                    value = 100 * float(row[stat])
                    add(
                        f"test_gap_stage_vs_{reference}_{context}_{metric}_{stat}",
                        f"Canonical {context} selected predictor minus {label} {metric.upper()} ({stat})",
                        value,
                        f"{value:.2f}",
                        "percentage points",
                        "canonical held-out pair-clustered bootstrap (evaluation 3)",
                        path,
                        f"context={context}; candidate=gap_stage; reference={reference}; metric={metric}; column={stat}",
                        "available for Results (test); no macro",
                        "two decimals (percentage points)",
                    )
                estimates.setdefault((tag, metric), []).append(100 * float(row["estimate"]))
    for tag, metric, which in (
        ("PNAS", "auroc", "Mean"), ("PNAS", "auroc", "Min"), ("PNAS", "auroc", "Max"),
        ("PNAS", "auprc", "Mean"), ("PNAS", "f1", "Mean"), ("Equal", "auroc", "Min"), ("Equal", "auroc", "Max"),
    ):
        values = estimates[(tag, metric)]
        value = {"Mean": sum(values) / len(values), "Min": min(values), "Max": max(values)}[which]
        reference = next(r for r, (t, _) in labels.items() if t == tag)
        add(
            f"test_gap_stage_vs_{reference}_{metric}_{which.lower()}",
            f"{which} over endpoints of the selected predictor minus {labels[reference][1]} {metric.upper()}",
            value,
            f"{value:.2f}",
            "percentage points",
            "canonical held-out (evaluation 3)",
            path,
            f"{which.lower()} over the five endpoints of estimate (point difference); "
            f"candidate=gap_stage; reference={reference}; metric={metric}",
            "available for Results (test), Abstract",
            "two decimals (percentage points) of the unrounded values",
            f"TestVs{tag}{METRIC_MACRO[metric]}{which}",
        )


def add_supervision_claims() -> None:
    metrics_path = sources.path(
        ROOT / "experiments/runs/richer_supervision/pair_balance_comparison/metrics.csv"
    )
    ci_path = sources.path(
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
        pnas_path = (
            ROOT
            / f"experiments/runs/richer_supervision/binary_pair_balanced/{context}/pair_bootstrap_ci.csv"
        )
        path = sources.path(pnas_path)
        frame = sources.read_csv(pnas_path)  # candidate=pair_balanced; reference=development_reference
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
                    f"candidate=pair_balanced; reference=development_reference; metric={metric}; column={part}",
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
    align_metrics_path = sources.path(
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
    alignment_ci_path = sources.path(
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
        path = sources.path(
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

    unified_metrics_path = sources.path(
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

    unified_ci_path = sources.path(
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
    multitask_path = sources.path(
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
            sources.path(
                ROOT
                / "experiments/runs/richer_supervision/pooling_comparison/pair_bootstrap_ci.csv"
            ),
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

    within_path = sources.path(
        ROOT / "experiments/runs/post_analysis/within_duplex/within_duplex_contrasts.csv"
    )
    within = pd.read_csv(within_path)
    for context in ("Liver", "MuscleSkeletal"):
        # The *WithinEnsemble* macros are the selected predictor's (D48): the gap_stage rows. Their claim ids keep
        # PNAS's "fixed_ensemble" (PNAS's selected predictor) so the ledger's ids stay those of
        # step11b/claim_sources.csv; the selector names the row actually read.
        for candidate, claim_tag, label, stem in (
            ("target_local", "target_local", "target_local", "WithinLocal"),
            ("gap_stage", "fixed_ensemble", "selected predictor (gap_stage)", "WithinEnsemble"),
        ):
            row = within[within.context.eq(context) & within.candidate.eq(candidate)]
            if len(row) != 1:
                raise RuntimeError(f"{within_path}: expected one {context}/{candidate} row, got {len(row)}")
            row = row.iloc[0]
            context_stem = CONTEXT_MACRO[context]
            for column, suffix in (
                ("lower", "Lower"),
                ("mean_difference", "Mean"),
                ("upper", "Upper"),
            ):
                value = float(row[column])
                add(
                    f"within_pair_{context}_{claim_tag}_{column}",
                    f"{context} {label} minus pair-first mean within-pair AUROC {column}",
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


def add_gap_stage_claims() -> None:
    """Gap stage vs its controls (step 11b, D51): development OOF, one run (seed 20260821), 2,000 pair replicates.
    Also the sign of the frozen final second stage's spacing weight in every endpoint (models/final_v2)."""
    references = (("no_gap_stage", "NoGap"), ("shuffled_gap_stage", "Shuffled"), ("equal_average", "EqualAverage"))
    medians: dict[tuple[str, str], list[float]] = {}
    for context in CONTEXTS:
        path = ROOT / f"experiments/runs/step11b/gap_controls/{context}/pair_bootstrap_ci.csv"
        frame = pd.read_csv(path)
        for reference, tag in references:
            for metric in ("auroc", "auprc", "f1", "brier"):
                row = frame[(frame.candidate == "gap_stage") & (frame.reference == reference) & (frame.metric == metric)]
                if len(row) != 1:
                    raise RuntimeError(f"{path}: expected one gap_stage/{reference}/{metric} row, got {len(row)}")
                row = row.iloc[0]
                for stat in ("median", "lower", "upper"):
                    value = 100 * float(row[stat])
                    add(
                        f"gap_stage_vs_{reference}_{context}_{metric}_{stat}",
                        f"{context} gap stage minus {reference}, {metric} ({stat})",
                        value,
                        f"{value:.2f}",
                        "percentage points",
                        "development OOF pair-clustered bootstrap (one run)",
                        path,
                        f"candidate=gap_stage; reference={reference}; metric={metric}; {stat}",
                        "Results (genomic organization); SI gap-control table",
                        "two decimals (percentage points)",
                    )
                medians.setdefault((tag, metric), []).append(100 * float(row["median"]))
    for tag in ("NoGap", "Shuffled"):
        values = medians[(tag, "auroc")]
        for which, value in (("Min", min(values)), ("Max", max(values))):
            add(
                f"gap_stage_vs_{tag.lower()}_auroc_{which.lower()}",
                f"{which} over endpoints of the gap stage minus {tag} AUROC median",
                value,
                f"{value:.2f}",
                "percentage points",
                "development OOF pair-clustered bootstrap (one run)",
                ROOT / "experiments/runs/step11b/gap_controls",
                f"{which.lower()} over the five endpoints of median(gap_stage - {tag}) AUROC",
                "Results (genomic organization)",
                "two decimals (percentage points)",
                f"GapStageVs{tag}AUROC{which}",
            )
    negative = 0
    for context in CONTEXTS:
        spec_path = ROOT / f"models/final_v2/gap_stage/{context}/frozen_gap_stage.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        negative += spec["weights"][spec["columns"].index("gap")] < 0
    if negative != len(CONTEXTS):  # the text says "negative in every endpoint"
        raise RuntimeError(f"spacing weight negative in {negative} of {len(CONTEXTS)} endpoints; revise the text")
    add(
        "gap_stage_negative_weight_endpoints",
        "Endpoints whose frozen second-stage spacing weight is negative",
        negative,
        str(negative),
        "endpoints",
        "frozen final model",
        ROOT / "models/final_v2/gap_stage",
        "weights[columns.index('gap')] < 0 in frozen_gap_stage.json",
        "Results (genomic organization)",
        "integer",
    )


def add_seed_averaging_claims() -> None:
    """Five-seed vs single-seed selected predictor (step 11 seed curve, development OOF; D5)."""
    medians = []
    for context in CONTEXTS:
        path = ROOT / f"experiments/runs/step11/seed_curve/{context}/d_avg.csv"
        frame = pd.read_csv(path)
        for metric in ("auroc", "auprc", "f1"):
            row = frame[(frame.model == "predictor") & (frame.metric == metric)]
            if len(row) != 1:
                raise RuntimeError(f"{path}: predictor/{metric}")
            row = row.iloc[0]
            if not row["lower"] > 0:  # the text says every interval is above zero
                raise RuntimeError(f"{context} {metric}: seed-averaging interval reaches zero; revise the text")
            for stat in ("median", "lower", "upper"):
                value = 100 * float(row[stat])
                add(f"seed_avg_{context}_{metric}_{stat}", f"{context} five-seed minus single-seed predictor, {metric} ({stat})",
                    value, f"{value:.2f}", "percentage points", "development OOF pair-clustered bootstrap", path,
                    f"model=predictor; metric={metric}; {stat}", "Results (seed averaging); SI seed-averaging table",
                    "two decimals (percentage points)")
            if metric == "auroc":
                medians.append(100 * float(row["median"]))
    for which, value in (("Min", min(medians)), ("Max", max(medians))):
        add(f"seed_avg_auroc_{which.lower()}", f"{which} over endpoints of the five-seed AUROC gain", value, f"{value:.2f}",
            "percentage points", "development OOF pair-clustered bootstrap", ROOT / "experiments/runs/step11/seed_curve",
            f"{which.lower()} over five endpoints of median AUROC gain", "Results (seed averaging)",
            "two decimals (percentage points)", f"SeedAvgAUROC{which}")


def add_reformer_claims() -> None:
    """Reformer RBP error screen on the selected predictor's development OOF predictions (R8, COMPUTE D57-D64).
    Numbers printed in SI P.3; guards keep the paragraph's wording true."""
    import json
    run = ROOT / "experiments/runs/r8_reformer/gap_seed20260821"
    pcs = json.loads((run / "pc_summary.json").read_text())
    reading = json.loads((run / "reading.json").read_text())
    fp = pd.read_csv(run / "fp_effects.csv")
    if reading["fn"]["significant"] != 0:
        raise RuntimeError("Reformer: a miss feature is significant; SI P.3 says none is")
    for pc in ("PC2", "PC3"):
        if not pcs[pc]["weight_share_dsRNA_binder_models"] < 1.5 * pcs[pc]["even_share_dsRNA_binder_models"]:
            raise RuntimeError(f"Reformer: {pc} is weighted toward dsRNA binders; SI P.3 says neither is")
    if not reading["sweep_max_abs_d"] < 0.2:
        raise RuntimeError("Reformer: aggregate binding separates false positives at some confidence cut")
    status = "development OOF, four tissues pooled (R8)"
    use = "SI P.3 (RBPs, Reformer screen)"
    add("reformer_models", "Reformer RNA-binding models scored", pcs["models"], str(pcs["models"]), "models", status,
        run / "pc_summary.json", "models", use, "integer", "ReformerModels")
    add("reformer_models_hepg2_k562", "Reformer models trained in HepG2 or K562", pcs["models_hepg2_or_k562"],
        str(pcs["models_hepg2_or_k562"]), "models", status, run / "pc_summary.json", "models_hepg2_or_k562", use,
        "integer", "ReformerModelsCellLines")
    value = float(reading["fn"]["largest_abs_d"])
    add("reformer_fn_largest_abs_d", "Largest |d| over Reformer features, confident misses vs correct edited calls",
        value, f"{value:.2f}", "Cohen's d (composition-adjusted)", status, run / "reading.json", "fn.largest_abs_d", use,
        "two decimals", "ReformerMissLargestD")
    for pc, word in (("PC2", "Two"), ("PC3", "Three")):
        row = fp[fp.feature == pc]
        if len(row) != 1:
            raise RuntimeError(f"Reformer fp_effects: {pc}: {len(row)} rows")
        d = float(row.iloc[0]["d_compCtrl"])
        if not d < 0:
            raise RuntimeError(f"Reformer: false positives not lower on {pc}")
        add(f"reformer_fp_{pc.lower()}_d", f"Confident false positives vs correct rejections on Reformer {pc}", d,
            f"$-${abs(d):.2f}", "Cohen's d (composition-adjusted)", status, run / "fp_effects.csv",
            f"feature == {pc}; d_compCtrl", use, "two decimals", f"ReformerFalsePos{word}D")
        share = 100 * float(pcs["variance_explained"][pc])
        add(f"reformer_{pc.lower()}_variance", f"Share of variance of the 225 z-scored Reformer models on {pc}", share,
            f"{share:.0f}", "percent", status, run / "pc_summary.json", f"variance_explained.{pc}", use, "integer percent",
            f"Reformer{word}Variance")


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
    add_test_contrast_claims()
    add_supervision_claims()
    add_substrate_claims()
    add_local_and_unified_claims()
    add_limited_branch_claims()
    add_synthesis_table_claims()
    add_post_analysis_claims()
    add_gap_stage_claims()
    add_seed_averaging_claims()
    add_reformer_claims()
    write_outputs()


if __name__ == "__main__":
    main()
