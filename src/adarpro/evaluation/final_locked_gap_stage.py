"""Second canonical-test session: the frozen gap stage on five-seed graph + CNN-alone finals, scored once.

PNAS integration, steps 13-14. The headline model and every comparison are named in advance in
``provenance/model_selection_policy_v2.json``:

* headline (``gap_stage``): per endpoint, the frozen logistic stage (``models/final_v2/gap_stage/<endpoint>/``)
  on the log-odds averages of five graph and five CNN-alone final models plus the pair's log1p genomic gap;
* comparators, on the same rows in the same session: the frozen equal log-odds average of the same ten models
  (``equal_average``); PNAS's selected predictor, read from its consumed evaluation
  (``results/adarpro/final_locked_test/``, ``pnas_selected_predictor``); and the reproduced supplied-checkpoint
  AdarEdit baseline (``adaredit_reference``), exactly as PNAS scored it.

``final_locked`` is reused for the canonical loader, the AdarEdit baseline and its published-aggregate check, the
component scoring, the frozen-input check and the pair-clustered bootstrap (PNAS's seeds). ``final_locked`` itself is
unchanged, so PNAS's consumed evaluation stays as it ran. The second authorization comes from a successor quarantine
policy (``quarantine.initialize_successor``) that continues PNAS's audit chain.

Subcommands:
  build-manifest  bind every input by hash while the successor policy is development_sealed
  dry-run         the session's scoring path on the published VALIDATION rows; it must reproduce step 11's frozen
                  validation predictions; no test file is opened and nothing is registered
  run             preflight, consume the one-use authorization, score the canonical test once
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adarpro.evaluation.final_locked import (
    BOOTSTRAP_METRICS,
    assert_published_baseline,
    baseline_predictions,
    candidate_predictions,
    load_canonical_table,
    pair_bootstrap_delta,
    verify_artifact_inventory,
    verify_frozen_inputs,
)
from adarpro.finalize import FROZEN_INPUT_PATHS
from adarpro.mapping.bed_pair_gaps import lookup_gaps
from adarpro.models.baseline_oof import torch_load, upstream_modules
from adarpro.models.final_ensemble import validate_checkpoint
from adarpro.models.final_gap_stage import apply_frozen_stage
from adarpro.models.final_training import FINAL_MODES
from adarpro.models.gap_stage import (
    CNN_MODE,
    GRAPH_MODE,
    average_probabilities,
    pair_gap_table,
)
from adarpro.models.residual import metrics
from adarpro.quarantine import (
    CONTEXTS,
    QuarantineViolation,
    consume_final_authorization,
    development_file,
    load_policy,
    sha256_file,
)
from adarpro.registry import register

MANIFEST_SCHEMA = 2
CANDIDATE_NAME = "adarpro_gap_stage_graph_cnn_alone_5seed"
FAMILY = "gap_stage_on_seed_averaged_graph_and_cnn_alone"
NETWORKS = {"graph": GRAPH_MODE, "cnn_alone": CNN_MODE}
NETWORK_PREFIX = {"graph": "graph", "cnn_alone": "cnn"}
SEEDS = 5
HEADLINE, EQUAL_AVERAGE, PNAS, REFERENCE = "gap_stage", "equal_average", "pnas_selected_predictor", "adaredit_reference"
CONTRASTS = ((HEADLINE, REFERENCE), (HEADLINE, PNAS), (HEADLINE, EQUAL_AVERAGE))
BASELINE_MODEL_NAME = "published_baseline_reproduced"  # PNAS's row name in metrics.csv
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260821  # + the endpoint's index in CONTEXTS, as PNAS
SELECTION_POLICY = "provenance/model_selection_policy_v2.json"
PAIR_GAPS = "data/derived/genomic_mapping/bed_pair_gaps.parquet"
PREDECESSOR = {
    "policy": "provenance/test_quarantine.json",
    "manifest": "provenance/final_pipeline_manifest.json",
    "results": "results/adarpro/final_locked_test",
}
FINAL_ROOT = "models/final_v2"
REPRODUCTION_TOLERANCE = 1e-5  # PNAS's reload rule; a flip at the threshold counts only within this distance
PNAS_METRIC_TOLERANCE = 1e-9
FROZEN_INPUT_PATHS_V2 = {
    **FROZEN_INPUT_PATHS,
    "model_selection_policy": SELECTION_POLICY,
    "predecessor_model_selection_policy": FROZEN_INPUT_PATHS["model_selection_policy"],
    "predecessor_manifest": PREDECESSOR["manifest"],
    "predecessor_metrics": f"{PREDECESSOR['results']}/metrics.csv",
    "predecessor_summary": f"{PREDECESSOR['results']}/summary.json",
    "gap_stage_implementation": "src/adarpro/models/final_gap_stage.py",
    "gap_stage_features": "src/adarpro/models/gap_stage.py",
    "pair_gap_implementation": "src/adarpro/mapping/bed_pair_gaps.py",
    "pair_reconstruction": "src/adarpro/mapping/reconstruct.py",
    "pair_gaps": PAIR_GAPS,
    "pair_gaps_provenance": PAIR_GAPS.replace(".parquet", ".provenance.json"),
    "pair_mapping": "data/derived/genomic_mapping/pair_mapping.csv",
    "alu_pair_bed": "data/raw/alu_pairs/Pair_Alu_withStrand.bed",
    "second_session_implementation": "src/adarpro/evaluation/final_locked_gap_stage.py",
}


def file_record(repository: Path, relative: str) -> dict:
    path = repository / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def context_specification(repository: Path, context: str) -> dict:
    """The frozen gap stage and equal average of one endpoint, with every checkpoint bound by hash."""

    stage_dir = Path(FINAL_ROOT) / "gap_stage" / context
    average_dir = Path(FINAL_ROOT) / "equal_average" / context
    stage_path = repository / stage_dir / "frozen_gap_stage.json"
    stage = json.loads(stage_path.read_text())
    stage_summary = json.loads((repository / stage_dir / "summary.json").read_text())
    average = json.loads((repository / average_dir / "summary.json").read_text())
    for name, payload in (("stage", stage), ("stage summary", stage_summary), ("equal average", average)):
        if payload.get("context") != context or payload.get("canonical_test_access") is not False:
            raise RuntimeError(f"{context} {name} is not a development-only artifact of this endpoint")
    if stage_summary["frozen_stage_sha256"] != sha256_file(stage_path):
        raise RuntimeError(f"{context}: frozen stage differs from the one its summary records")
    components = []
    networks = {network: [] for network in NETWORKS}
    for directory, digest in stage["final_models"].items():
        path = Path(directory)
        mode, seed_dir = path.name, path.parent.parent.name
        network = next((n for n, m in NETWORKS.items() if m == mode), None)
        if network is None or path.parent.name != context or not seed_dir.startswith("seed"):
            raise RuntimeError(f"{context}: unexpected final model folder {directory}")
        name = f"{NETWORK_PREFIX[network]}{seed_dir.removeprefix('seed')}"
        checkpoint = (path / "best.pth").relative_to(repository) if path.is_absolute() else path / "best.pth"
        if average["components"].get(name) != mode or average["checkpoint_sha256"].get(name) != digest:
            raise RuntimeError(f"{context}: {name} differs between the gap stage and the equal average")
        if sha256_file(repository / checkpoint) != digest:
            raise RuntimeError(f"{context}: checkpoint hash mismatch for {checkpoint}")
        components.append({"name": name, "mode": mode, "checkpoint": str(checkpoint), "sha256": digest})
        networks[network].append(name)
    components.sort(key=lambda component: component["name"])
    networks = {network: sorted(names) for network, names in networks.items()}
    if set(average["components"]) != {component["name"] for component in components}:
        raise RuntimeError(f"{context}: the equal average holds other models than the gap stage")
    if float(average["threshold"]) != float(stage_summary["equal_average_threshold"]):
        raise RuntimeError(f"{context}: two different frozen equal-average thresholds")
    weights_by_name = dict(zip(average["components"], average["weights"]))
    return {
        "components": components,
        "networks": networks,
        "stage": file_record(repository, str(stage_dir / "frozen_gap_stage.json")),
        "threshold": float(stage["threshold"]),
        "validation_summary": file_record(repository, str(stage_dir / "summary.json")),
        "validation_predictions": file_record(repository, str(stage_dir / "validation_predictions.parquet")),
        "equal_average": {
            "weights": [float(weights_by_name[component["name"]]) for component in components],
            "threshold": float(average["threshold"]),
            "validation_summary": file_record(repository, str(average_dir / "summary.json")),
            "validation_predictions": file_record(repository, str(average_dir / "validation_predictions.parquet")),
        },
    }


def build_specification(repository: Path) -> dict:
    """Everything the manifest binds except the code revision and the time; shared by dry-run and build-manifest."""

    predecessor_policy = load_policy(repository / PREDECESSOR["policy"])
    frozen = predecessor_policy["final_locked_test"]["freeze_manifest"]
    if predecessor_policy["state"] != "final_consumed" or frozen is None:
        raise RuntimeError("the predecessor evaluation is not consumed")
    predecessor_manifest = repository / PREDECESSOR["manifest"]
    if sha256_file(predecessor_manifest) != frozen["sha256"]:
        raise RuntimeError("the predecessor manifest differs from the one its policy froze")
    pnas = json.loads(predecessor_manifest.read_text())
    selection_path = repository / SELECTION_POLICY
    selection = json.loads(selection_path.read_text())
    result = selection["development_selection_result"]
    if result["selected_family"] != FAMILY or result["networks"] != NETWORKS or int(result["seeds"]) != SEEDS:
        raise RuntimeError("the selection policy does not name this family")
    plan = selection["evaluation_plan"]
    contrasts = [list(pair) for pair in CONTRASTS]
    if plan["contrasts"] != contrasts or int(plan["bootstrap_replicates"]) != BOOTSTRAP_REPLICATES:
        raise RuntimeError("the selection policy's evaluation plan differs from this implementation")
    inventory = {name: file_record(repository, path) for name, path in FROZEN_INPUT_PATHS_V2.items()}
    return {
        "schema_version": MANIFEST_SCHEMA,
        "candidate_name": CANDIDATE_NAME,
        "upstream_revision": pnas["upstream_revision"],
        "benchmark": pnas["benchmark"],
        "selection_policy": SELECTION_POLICY,
        "selection_policy_sha256": sha256_file(selection_path),
        "predecessor": {
            "policy": PREDECESSOR["policy"],
            "policy_sha256": sha256_file(repository / PREDECESSOR["policy"]),
            "manifest": PREDECESSOR["manifest"],
            "manifest_sha256": frozen["sha256"],
            "candidate_name": pnas["candidate_name"],
            "results": PREDECESSOR["results"],
        },
        "development_protocol": {
            "grouping_unit": "pair_id",
            "development_folds": "five global pair-disjoint folds",
            "training_split": "published train pairs; expanded raw high/low sites",
            "selection_split": "published validation canonical rows",
            "checkpoint_selection": "maximum validation F1 over 1000 epochs",
            "threshold_selection": "published validation only",
            "stage_fit": "published train canonical rows (in-sample final-model scores; gate scheme)",
            "canonical_test_access_during_selection": False,
        },
        "candidate_specification": {
            "family": FAMILY,
            "networks": dict(NETWORKS),
            "seeds": SEEDS,
            "averaging": "each network averaged over its seeds in log-odds, then one stage",
            "stage": "frozen L2 logistic on (graph log-odds, CNN-alone log-odds, log1p genomic gap)",
            "gap": f"label-free lookup by pair sequence in {PAIR_GAPS}",
        },
        "evaluation": {
            "headline": HEADLINE,
            "comparators": [EQUAL_AVERAGE, PNAS, REFERENCE],
            "contrasts": contrasts,
            "metrics": list(BOOTSTRAP_METRICS),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": f"{BOOTSTRAP_SEED} + endpoint index in {list(CONTEXTS)}",
            "device": "cpu",
        },
        "frozen_inputs": inventory,
        "contexts": {context: context_specification(repository, context) for context in CONTEXTS},
    }


def validate_manifest_v2(manifest: dict) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("unsupported second-session manifest schema")
    for key in ("candidate_name", "code_revision", "selection_policy_sha256"):
        if not manifest.get(key):
            raise RuntimeError(f"manifest lacks {key}")
    protocol = manifest.get("development_protocol", {})
    if protocol.get("grouping_unit") != "pair_id" or protocol.get("canonical_test_access_during_selection") is not False:
        raise RuntimeError("manifest does not attest pair-grouped, sealed model selection")
    candidate = manifest.get("candidate_specification", {})
    if candidate.get("family") != FAMILY or candidate.get("networks") != NETWORKS or candidate.get("seeds") != SEEDS:
        raise RuntimeError("manifest names another candidate family")
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("contrasts") != [list(pair) for pair in CONTRASTS] or evaluation.get("headline") != HEADLINE:
        raise RuntimeError("manifest's evaluation plan differs from this implementation")
    predecessor = manifest.get("predecessor", {})
    if not all(predecessor.get(key) for key in ("policy", "policy_sha256", "manifest", "manifest_sha256")):
        raise RuntimeError("manifest lacks the predecessor evaluation")
    for name, artifact in manifest.get("frozen_inputs", {}).items():
        if not name or not artifact.get("path") or not artifact.get("sha256") or int(artifact.get("bytes", -1)) < 0:
            raise RuntimeError(f"invalid frozen input record: {name}")
    if set(manifest.get("frozen_inputs", {})) != set(FROZEN_INPUT_PATHS_V2):
        raise RuntimeError("manifest's frozen-input inventory is incomplete")
    if set(manifest.get("contexts", {})) != set(CONTEXTS):
        raise RuntimeError("manifest must freeze all five benchmark contexts")
    for context, specification in manifest["contexts"].items():
        components = specification.get("components", [])
        names = [component.get("name") for component in components]
        if len(components) != SEEDS * len(NETWORKS) or len(set(names)) != len(names) or not all(names):
            raise RuntimeError(f"{context}: expected {SEEDS * len(NETWORKS)} uniquely named components")
        modes = {component["name"]: component.get("mode") for component in components}
        for component in components:
            if component["mode"] not in FINAL_MODES or not component.get("checkpoint") or not component.get("sha256"):
                raise RuntimeError(f"{context}: incomplete or unsupported component {component.get('name')}")
        networks = specification.get("networks", {})
        if set(networks) != set(NETWORKS) or sorted(name for members in networks.values() for name in members) != sorted(names):
            raise RuntimeError(f"{context}: networks do not partition the components")
        for network, members in networks.items():
            if len(members) != SEEDS or {modes[name] for name in members} != {NETWORKS[network]}:
                raise RuntimeError(f"{context}: {network} needs {SEEDS} models of mode {NETWORKS[network]}")
        for key in ("stage", "validation_summary", "validation_predictions"):
            if not specification.get(key, {}).get("path") or not specification[key].get("sha256"):
                raise RuntimeError(f"{context}: manifest lacks {key}")
        average = specification.get("equal_average", {})
        weights = np.asarray(average.get("weights", []), dtype=float)
        if len(weights) != len(components) or not np.allclose(weights, 1 / len(components)):
            raise RuntimeError(f"{context}: the equal average must weight every component equally")
        for threshold in (specification.get("threshold", np.nan), average.get("threshold", np.nan)):
            if not 0 <= float(threshold) <= 1:
                raise RuntimeError(f"{context}: invalid frozen threshold")


def build_manifest(repository: Path, policy_path: Path, output: Path) -> dict:
    policy = load_policy(policy_path)
    if policy["state"] != "development_sealed" or "predecessor" not in policy:
        raise RuntimeError("build the manifest on a development_sealed successor policy")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite a final manifest: {output}")
    manifest = build_specification(repository)
    if manifest["predecessor"]["policy_sha256"] != policy["predecessor"]["policy_sha256"]:
        raise RuntimeError("the successor policy records another predecessor")
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["code_revision"] = git_revision(repository)
    validate_manifest_v2(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def git_revision(repository: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()


def load_development_table(repository: Path, context: str, split: str, baseline) -> pd.DataFrame:
    """``final_locked.load_canonical_table`` for a development split (through the quarantine's development guard)."""

    jsonl_path = development_file(repository, context, split, "jsonl")
    metadata_path = development_file(repository, context, split, "metadata.csv")
    parsed = []
    with jsonl_path.open() as handle:
        for index, line in enumerate(handle):
            sequence, structure, target, label = baseline.parse_openai_json(json.loads(line))
            if not sequence or not structure or len(sequence) != len(structure):
                raise RuntimeError(f"invalid {split} row {context}/{index}")
            parsed.append({"example_id": f"{context}:{split}:{index}", "context": context, "sequence": sequence,
                           "structure": structure, "local_position": target, "label": label})
    table = pd.DataFrame(parsed)
    metadata = pd.read_csv(metadata_path)
    if len(table) != len(metadata):
        raise RuntimeError(f"{split} JSONL/metadata length mismatch for {context}")
    if not np.array_equal(table["label"].to_numpy(), metadata["yes_no"].map({"no": 0, "yes": 1}).to_numpy()):
        raise RuntimeError(f"{split} label/metadata mismatch for {context}")
    table["pair_id"] = metadata["pair_id"].to_numpy()
    table["row_index"] = np.arange(len(table))
    return table


def score_context(repository: Path, context: str, specification: dict, table: pd.DataFrame, gaps: pd.DataFrame,
                  device: torch.device, batch_size: int) -> pd.DataFrame:
    """Every model of the session on one endpoint's rows (headline, equal average, AdarEdit); returns the table."""

    baseline_module = upstream_modules(repository)
    alphabet = set(baseline_module.BASE_MAP)
    graphs = [baseline_module.create_rna_graph(row.sequence, row.structure, int(row.local_position), int(row.label))
              for row in table.itertuples(index=False)]
    if any(graph is None for graph in graphs):
        raise RuntimeError(f"unbuildable graph for {context}")
    baseline_probability, baseline_threshold, baseline_info = baseline_predictions(
        repository, context, graphs, device, batch_size)
    if not np.array_equal(baseline_info["labels"], table["label"].to_numpy()):
        raise RuntimeError("AdarEdit baseline row mismatch")
    equal_probability, outputs = candidate_predictions(
        repository, {"components": specification["components"],
                     "weights": specification["equal_average"]["weights"]}, table, graphs, device, batch_size)
    averaged = {network: average_probabilities([outputs[name] for name in names])
                for network, names in specification["networks"].items()}
    found = lookup_gaps(table["sequence"], gaps, alphabet)
    stage = json.loads((repository / specification["stage"]["path"]).read_text())
    stage_probability = apply_frozen_stage(stage, averaged["graph"], averaged["cnn_alone"], found["gap"].to_numpy())
    if float(stage["threshold"]) != float(specification["threshold"]):
        raise RuntimeError(f"{context}: stage threshold differs from the manifest's")
    out = table.copy()
    out["gap_bed_row"] = found["bed_row"].to_numpy(dtype=int)
    out["gap_sequence_copies"] = found["sequence_copies"].to_numpy(dtype=int)
    out["gap_bp"], out["gap"] = found["gap_bp"].to_numpy(dtype=float), found["gap"].to_numpy(dtype=float)
    for name, values in outputs.items():
        out[f"{name}_probability"] = values
    out["graph_probability"], out["cnn_alone_probability"] = averaged["graph"], averaged["cnn_alone"]
    for name, probability, threshold in (
        (EQUAL_AVERAGE, equal_probability, specification["equal_average"]["threshold"]),
        ("baseline", baseline_probability, baseline_threshold),
        ("", stage_probability, specification["threshold"]),  # headline: PNAS's column names
    ):
        prefix = f"{name}_" if name else ""
        out[f"{prefix}probability"] = probability
        out[f"{prefix}prediction"] = (probability >= float(threshold)).astype(int)
        out[f"{prefix}threshold"] = float(threshold)
    for column in [c for c in out.columns if c.endswith("_probability") or c == "probability"]:
        if not np.isfinite(out[column].to_numpy(dtype=float)).all():
            raise RuntimeError(f"{context}: non-finite {column}")
    return out


def compare_probabilities(mine: pd.Series, saved: pd.Series, threshold: float) -> dict:
    """Max |difference| and prediction flips at the frozen threshold; a flip counts as boundary when both values lie
    within the tolerance of the threshold."""

    difference = (mine - saved).abs()
    flips = (mine >= threshold) != (saved >= threshold)
    boundary = flips & ((mine - threshold).abs() <= REPRODUCTION_TOLERANCE) & (
        (saved - threshold).abs() <= REPRODUCTION_TOLERANCE)
    return {"rows": len(mine), "max_abs_difference": float(difference.max()), "flips": int(flips.sum()),
            "boundary_flips": int(boundary.sum()),
            "passes": bool(difference.max() <= REPRODUCTION_TOLERANCE and flips.sum() == boundary.sum())}


def contrast_frame(table: pd.DataFrame, candidate: str, reference: str) -> pd.DataFrame:
    columns = {HEADLINE: ("probability", "prediction"), EQUAL_AVERAGE: ("equal_average_probability",
               "equal_average_prediction"), PNAS: ("pnas_probability", "pnas_prediction"),
               REFERENCE: ("baseline_probability", "baseline_prediction")}
    frame = table[["pair_id", "label"]].copy()
    frame["probability"], frame["prediction"] = (table[c] for c in columns[candidate])
    frame["baseline_probability"], frame["baseline_prediction"] = (table[c] for c in columns[reference])
    return frame


def model_rows(context: str, table: pd.DataFrame, candidate_name: str, models: list[str]) -> list[dict]:
    names = {HEADLINE: candidate_name, EQUAL_AVERAGE: EQUAL_AVERAGE, PNAS: PNAS, REFERENCE: BASELINE_MODEL_NAME}
    rows = []
    for model in models:
        frame = contrast_frame(table, model, model)
        rows.append({"context": context, "model": names[model],
                     "threshold": float(table[{HEADLINE: "threshold", EQUAL_AVERAGE: "equal_average_threshold",
                                               PNAS: "pnas_threshold", REFERENCE: "baseline_threshold"}[model]].iloc[0]),
                     "pair_count": int(table["pair_id"].nunique()),
                     **metrics(frame["label"], frame["probability"], frame["prediction"])})
    return rows


def contrast_outputs(context: str, table: pd.DataFrame, contrasts, replicates: int, output_dir: Path) -> pd.DataFrame:
    """Pair-clustered bootstrap of every contrast with PNAS's per-endpoint seed (identical resamples across
    contrasts). The headline-vs-AdarEdit draws are also written under PNAS's file names."""

    seed = BOOTSTRAP_SEED + CONTEXTS.index(context)
    rows = []
    for candidate, reference in contrasts:
        frame = contrast_frame(table, candidate, reference)
        draws = pair_bootstrap_delta(frame, replicates, seed)
        confidence = (draws.groupby("metric")["difference"].quantile([0.025, 0.5, 0.975]).unstack().reset_index()
                      .rename(columns={0.025: "lower", 0.5: "median", 0.975: "upper"}))
        if (candidate, reference) == (HEADLINE, REFERENCE):
            draws.to_parquet(output_dir / f"{context}_pair_bootstrap.parquet", index=False)
            confidence.to_csv(output_dir / f"{context}_pair_bootstrap_ci.csv", index=False)
        draws.to_parquet(output_dir / "contrasts" / f"{context}_{candidate}_vs_{reference}_pair_bootstrap.parquet",
                         index=False)
        point_candidate = metrics(frame["label"], frame["probability"], frame["prediction"])
        point_reference = metrics(frame["label"], frame["baseline_probability"], frame["baseline_prediction"])
        for record in confidence.to_dict("records"):
            rows.append({"context": context, "candidate": candidate, "reference": reference,
                         "metric": record["metric"],
                         "estimate": point_candidate[record["metric"]] - point_reference[record["metric"]],
                         "lower": record["lower"], "median": record["median"], "upper": record["upper"],
                         "replicates": replicates, "seed": seed})
    return pd.DataFrame(rows)


def dry_run(repository: Path, output_dir: Path, batch_size: int, replicates: int, device: torch.device) -> dict:
    """The session's scoring path on the published validation rows. Must reproduce step 11's frozen validation
    predictions model by model; the AdarEdit baseline must reproduce its checkpoint's validation metrics; the gap
    lookup must reproduce the development gaps. Writes only to ``output_dir``; registers nothing."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite {output_dir}")
    (output_dir / "contrasts").mkdir(parents=True, exist_ok=True)
    manifest = build_specification(repository)
    manifest["code_revision"] = git_revision(repository)
    validate_manifest_v2(manifest)
    (output_dir / "manifest_preview.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    gaps = pd.read_parquet(repository / PAIR_GAPS)
    master = pd.read_parquet(repository / "data/derived/site_master.parquet")
    baseline_module = upstream_modules(repository)
    checks, metric_rows, contrast_rows = {}, [], []
    for context in CONTEXTS:
        specification = manifest["contexts"][context]
        table = load_development_table(repository, context, "valid", baseline_module)
        development = master[(master["context"] == context) & (master["source_split"] == "valid")]
        development = development.set_index("source_row").loc[table["row_index"]]
        for column in ("pair_id", "label", "local_position", "sequence", "structure"):
            if not np.array_equal(development[column].to_numpy(), table[column].to_numpy()):
                raise RuntimeError(f"{context}: validation rows disagree with site_master on {column}")
        table["development_example_id"] = development["example_id"].to_numpy()
        scored = score_context(repository, context, specification, table, gaps, device, batch_size)
        keyed = scored.set_index("development_example_id")
        stage_saved = pd.read_parquet(repository / specification["validation_predictions"]["path"]).set_index(
            "example_id").loc[keyed.index]
        average_saved = pd.read_parquet(repository / specification["equal_average"]["validation_predictions"]["path"]
                                        ).set_index("example_id").loc[keyed.index]
        context_checks = {
            "gap_stage": compare_probabilities(keyed["probability"], stage_saved["gap_stage_probability"],
                                               specification["threshold"]),
            "equal_average": compare_probabilities(keyed["equal_average_probability"], average_saved["probability"],
                                                   specification["equal_average"]["threshold"]),
            "components": {component["name"]: compare_probabilities(
                keyed[f"{component['name']}_probability"], average_saved[f"{component['name']}_probability"],
                0.5)["max_abs_difference"] for component in specification["components"]},
        }
        development_gaps = pair_gap_table(master[master["context"] == context]).set_index("pair_id")
        context_checks["gap_mismatches"] = int(
            (scored["gap"].to_numpy() != development_gaps.loc[scored["pair_id"], "gap"].to_numpy()).sum())
        saved = torch_load(repository / "checkpoints" / f"baseline_{context}" / "best.pth", torch.device("cpu"))
        expected = saved["best"]["metrics"]
        observed = metrics(scored["label"], scored["baseline_probability"], scored["baseline_prediction"])
        pairs = {"auroc": "auc", "auprc": "auprc", "f1": "f1", "precision": "precision", "recall": "recall",
                 "specificity": "specificity", "n": "n"}
        context_checks["adaredit_validation_max_difference"] = max(
            abs(float(observed[mine]) - float(expected[theirs])) for mine, theirs in pairs.items())
        context_checks["adaredit_threshold_matches"] = float(scored["baseline_threshold"].iloc[0]) == float(
            expected["threshold"])
        checks[context] = context_checks
        models = [HEADLINE, EQUAL_AVERAGE, REFERENCE]
        metric_rows.extend(model_rows(context, scored, CANDIDATE_NAME, models))
        contrast_rows.append(contrast_outputs(context, scored, [c for c in CONTRASTS if PNAS not in c], replicates,
                                              output_dir))
        scored.drop(columns=["development_example_id"]).to_parquet(output_dir / f"{context}_predictions.parquet",
                                                                    index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(contrast_rows, ignore_index=True).to_csv(output_dir / "contrasts_pair_bootstrap_ci.csv", index=False)
    passes = all(
        c["gap_stage"]["passes"] and c["equal_average"]["passes"]
        and max(c["components"].values()) <= REPRODUCTION_TOLERANCE and c["gap_mismatches"] == 0
        and c["adaredit_validation_max_difference"] <= 1e-6 and c["adaredit_threshold_matches"]
        for c in checks.values())
    summary = {"scope": "dry run of the second-session scoring path on the published VALIDATION rows; its metrics "
                        "are a smoke test of the writer (validation chose the thresholds and the stage's penalty)",
               "candidate": CANDIDATE_NAME, "split": "valid", "bootstrap_replicates": replicates,
               "reproduction_tolerance": REPRODUCTION_TOLERANCE, "checks": checks, "passes": passes,
               "code_revision": manifest["code_revision"], "canonical_test_access": False}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not passes:
        raise RuntimeError(f"dry run does not reproduce the frozen validation predictions: {checks}")
    return summary


def pnas_predictions(repository: Path, context: str, table: pd.DataFrame, published: pd.DataFrame,
                     candidate_name: str) -> tuple[pd.DataFrame, dict]:
    """PNAS's selected predictor on the same canonical rows, from its consumed evaluation; checked against the rows
    and against PNAS's published aggregate before use."""

    path = repository / PREDECESSOR["results"] / f"{context}_predictions.parquet"
    saved = pd.read_parquet(path)
    for column in ("row_index", "pair_id", "label"):
        if not np.array_equal(saved[column].to_numpy(), table[column].to_numpy()):
            raise RuntimeError(f"{context}: PNAS's predictions are not on the same rows ({column})")
    baseline_difference = float(np.max(np.abs(saved["baseline_probability"].to_numpy()
                                              - table["baseline_probability"].to_numpy())))
    if baseline_difference > REPRODUCTION_TOLERANCE:
        raise RuntimeError(f"{context}: the AdarEdit baseline differs from PNAS's session by {baseline_difference}")
    row = published[(published["context"] == context) & (published["model"] == candidate_name)]
    if len(row) != 1:
        raise RuntimeError(f"{context}: PNAS's published aggregate is missing")
    observed = metrics(saved["label"], saved["probability"], saved["prediction"])
    worst = max(abs(float(observed[m]) - float(row.iloc[0][m])) for m in (*BOOTSTRAP_METRICS, "tp", "fp", "tn", "fn"))
    if worst > PNAS_METRIC_TOLERANCE:
        raise RuntimeError(f"{context}: PNAS's predictions do not reproduce its published metrics ({worst})")
    columns = pd.DataFrame({"pnas_probability": saved["probability"].to_numpy(),
                            "pnas_prediction": saved["prediction"].to_numpy().astype(int),
                            "pnas_threshold": float(row.iloc[0]["threshold"])}, index=table.index)
    return columns, {"sha256": sha256_file(path), "baseline_max_difference": baseline_difference,
                     "published_metric_max_difference": worst}


def preflight(repository: Path, policy_path: Path, manifest_path: Path, output_dir: Path) -> dict:
    """Every non-test check before the one-use authorization is spent."""

    policy = load_policy(policy_path)
    if policy["state"] != "final_frozen" or "predecessor" not in policy:
        raise QuarantineViolation("the successor policy is not frozen")
    frozen = policy["final_locked_test"]["freeze_manifest"]
    if frozen is None or sha256_file(manifest_path) != frozen["sha256"]:
        raise QuarantineViolation("provided manifest is not the frozen manifest")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest_v2(manifest)
    if manifest["code_revision"] != git_revision(repository):
        raise RuntimeError("working revision differs from the frozen manifest")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite an existing final-test result")
    if manifest["predecessor"]["policy_sha256"] != policy["predecessor"]["policy_sha256"]:
        raise RuntimeError("manifest and successor policy record different predecessors")
    if sha256_file(repository / manifest["predecessor"]["policy"]) != manifest["predecessor"]["policy_sha256"]:
        raise RuntimeError("the predecessor policy changed")
    verify_artifact_inventory(policy, repository)
    verify_frozen_inputs(manifest, repository)
    for context, specification in manifest["contexts"].items():
        for record in (specification["stage"], specification["validation_summary"],
                       specification["validation_predictions"], specification["equal_average"]["validation_summary"],
                       specification["equal_average"]["validation_predictions"]):
            if sha256_file(repository / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen file changed: {record['path']}")
        for component in specification["components"]:
            checkpoint = repository / component["checkpoint"]
            if sha256_file(checkpoint) != component["sha256"]:
                raise RuntimeError(f"frozen checkpoint hash mismatch: {checkpoint}")
            validate_checkpoint(torch_load(checkpoint, torch.device("cpu")), context, component["mode"])
        if not (repository / PREDECESSOR["results"] / f"{context}_predictions.parquet").is_file():
            raise RuntimeError(f"PNAS's {context} predictions are missing")
    return manifest


def run(repository: Path, policy_path: Path, manifest_path: Path, authorization: str, output_dir: Path,
        batch_size: int, device: torch.device) -> pd.DataFrame:
    manifest = preflight(repository, policy_path, manifest_path, output_dir)
    consume_final_authorization(policy_path, authorization)
    (output_dir / "contrasts").mkdir(parents=True, exist_ok=True)
    baseline_module = upstream_modules(repository)
    published_baseline = pd.read_csv(repository / "results/adarpro/baseline_reproduction.csv")
    published_pnas = pd.read_csv(repository / PREDECESSOR["results"] / "metrics.csv")
    gaps = pd.read_parquet(repository / PAIR_GAPS)
    mapping = pd.read_csv(repository / "data/derived/genomic_mapping/pair_mapping.csv")
    metric_rows, contrast_rows, checks = [], [], {}
    replicates = int(manifest["evaluation"]["bootstrap_replicates"])
    for context in CONTEXTS:
        specification = manifest["contexts"][context]
        table = load_canonical_table(repository, context, baseline_module)
        scored = score_context(repository, context, specification, table, gaps, device, batch_size)
        assert_published_baseline(context, metrics(scored["label"], scored["baseline_probability"],
                                                   scored["baseline_prediction"]), published_baseline)
        if set(scored["pair_id"]) & set(mapping["pair_id"]) or set(scored["gap_bed_row"]) & set(mapping["bed_row"]):
            raise RuntimeError(f"{context}: canonical pairs overlap development pairs or their BED rows")
        pnas_columns, pnas_check = pnas_predictions(repository, context, scored, published_pnas,
                                                    manifest["predecessor"]["candidate_name"])
        scored = pd.concat([scored, pnas_columns], axis=1)
        scored.to_parquet(output_dir / f"{context}_predictions.parquet", index=False)
        metric_rows.extend(model_rows(context, scored, manifest["candidate_name"], [REFERENCE, HEADLINE,
                                                                                 EQUAL_AVERAGE, PNAS]))
        contrast_rows.append(contrast_outputs(context, scored, CONTRASTS, replicates, output_dir))
        checks[context] = {"pnas_predictions": pnas_check, "rows": len(scored),
                           "pairs": int(scored["pair_id"].nunique()),
                           "pairs_sharing_a_sequence": int((scored["gap_sequence_copies"] > 1).sum())}
    metric_table = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "metrics.csv"
    metric_table.to_csv(metrics_path, index=False)
    pd.concat(contrast_rows, ignore_index=True).to_csv(output_dir / "contrasts_pair_bootstrap_ci.csv", index=False)
    summary = {
        "scope": "single consumed canonical-test evaluation (second session; the first is PNAS's, recorded in "
                 f"{manifest['predecessor']['policy']})",
        "candidate": manifest["candidate_name"],
        "manifest_sha256": sha256_file(manifest_path),
        "policy": str(policy_path.relative_to(repository)) if policy_path.is_relative_to(repository) else str(
            policy_path),
        "bootstrap_replicates": replicates,
        "contrasts": manifest["evaluation"]["contrasts"],
        "metrics_sha256": sha256_file(metrics_path),
        "checks": checks,
        "canonical_test_access": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    register(
        repository,
        {
            "run_id": f"final_locked_test_v2_{manifest['candidate_name']}",
            "experiment_type": "single_consumed_canonical_test_evaluation",
            "context": "all five canonical contexts",
            "input_hashes": {"frozen_manifest": sha256_file(manifest_path)},
            "fold_definition": "immutable published 64:16:20 pair split; test only",
            "seed": BOOTSTRAP_SEED,
            "features": manifest["candidate_name"],
            "hyperparameters": {"bootstrap_replicates": replicates, "contrasts": manifest["evaluation"]["contrasts"]},
            "artifact": str(metrics_path.relative_to(repository)),
            "metrics": summary,
            "canonical_test_access": True,
        },
    )
    return metric_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build-manifest")
    build.add_argument("--policy", type=Path, default=Path("provenance/test_quarantine_v2.json"))
    build.add_argument("--output", type=Path, default=Path("provenance/final_pipeline_manifest_v2.json"))
    dry = sub.add_parser("dry-run")
    dry.add_argument("--output-dir", type=Path, required=True)
    dry.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    session = sub.add_parser("run")
    session.add_argument("--policy", type=Path, default=Path("provenance/test_quarantine_v2.json"))
    session.add_argument("--manifest", type=Path, default=Path("provenance/final_pipeline_manifest_v2.json"))
    token = session.add_mutually_exclusive_group(required=True)
    token.add_argument("--authorization")
    token.add_argument("--authorization-file", type=Path)
    session.add_argument("--output-dir", type=Path, default=Path("results/adarpro/final_locked_test_v2"))
    for subparser in (dry, session):
        subparser.add_argument("--batch-size", type=int, default=256)
        subparser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    if args.action == "build-manifest":
        print(json.dumps(build_manifest(root, resolve(args.policy), resolve(args.output)), indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.action == "dry-run":
        print(json.dumps(dry_run(root, resolve(args.output_dir), args.batch_size, args.bootstrap_replicates, device),
                         indent=2))
        return
    authorization = args.authorization or resolve(args.authorization_file).read_text().strip()
    table = run(root, resolve(args.policy), resolve(args.manifest), authorization, resolve(args.output_dir),
                args.batch_size, device)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
