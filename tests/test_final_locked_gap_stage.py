import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from adarpro.evaluation import final_locked_gap_stage as session
from adarpro.evaluation.final_locked import load_canonical_table
from adarpro.models.residual import metrics
from adarpro.quarantine import CONTEXTS, QuarantineViolation


def _manifest():
    components = [{"name": f"{prefix}{seed}", "mode": mode, "checkpoint": f"models/{prefix}{seed}.pth", "sha256": "a"}
                  for prefix, mode in (("cnn", session.CNN_MODE), ("graph", session.GRAPH_MODE))
                  for seed in range(session.SEEDS)]
    record = {"path": "x", "sha256": "a"}
    context = {
        "components": components,
        "networks": {"graph": [f"graph{s}" for s in range(session.SEEDS)],
                     "cnn_alone": [f"cnn{s}" for s in range(session.SEEDS)]},
        "stage": record, "validation_summary": record, "validation_predictions": record, "threshold": 0.4,
        "equal_average": {"weights": [0.1] * 10, "threshold": 0.3, "validation_summary": record,
                          "validation_predictions": record},
    }
    return {
        "schema_version": session.MANIFEST_SCHEMA, "candidate_name": session.CANDIDATE_NAME, "code_revision": "abc",
        "selection_policy_sha256": "a",
        "development_protocol": {"grouping_unit": "pair_id", "canonical_test_access_during_selection": False},
        "candidate_specification": {"family": session.FAMILY, "networks": dict(session.NETWORKS),
                                    "seeds": session.SEEDS},
        "evaluation": {"headline": session.HEADLINE, "contrasts": [list(c) for c in session.CONTRASTS]},
        "predecessor": {"policy": "p", "policy_sha256": "a", "manifest": "m", "manifest_sha256": "a"},
        "frozen_inputs": {name: {"path": path, "bytes": 1, "sha256": "a"}
                          for name, path in session.FROZEN_INPUT_PATHS_V2.items()},
        "contexts": {c: copy.deepcopy(context) for c in CONTEXTS},
    }


def test_valid_manifest_passes():
    session.validate_manifest_v2(_manifest())


@pytest.mark.parametrize("edit, message", [
    (lambda m: m["evaluation"].update(contrasts=[["gap_stage", "adaredit_reference"]]), "evaluation plan"),
    (lambda m: m["contexts"]["Liver"]["networks"]["graph"].pop(), "partition"),
    (lambda m: m["contexts"]["Brain"]["equal_average"].update(weights=[0.2] * 5 + [0.0] * 5), "equally"),
    (lambda m: m["contexts"].pop("Combined"), "all five"),
    (lambda m: m["frozen_inputs"].pop("pair_gaps"), "inventory"),
    (lambda m: m["development_protocol"].update(canonical_test_access_during_selection=True), "sealed"),
    (lambda m: m["contexts"]["Artery"].update(threshold=1.5), "threshold"),
])
def test_manifest_defects_are_refused(edit, message):
    manifest = _manifest()
    edit(manifest)
    with pytest.raises(RuntimeError, match=message):
        session.validate_manifest_v2(manifest)


def test_reproduction_allows_only_boundary_flips():
    threshold = 0.5
    mine = pd.Series([0.1, 0.5000001, 0.9])
    saved = pd.Series([0.1, 0.4999999, 0.9])
    result = session.compare_probabilities(mine, saved, threshold)
    assert result["flips"] == 1 and result["boundary_flips"] == 1 and result["passes"]
    far = session.compare_probabilities(pd.Series([0.1, 0.6]), pd.Series([0.1, 0.4]), threshold)
    assert not far["passes"]


def _scored(pairs=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pair in range(pairs):
        for site in range(3):
            label = int(site < 2) if pair % 2 else int(site == 0)
            rows.append({"pair_id": pair, "label": label})
    table = pd.DataFrame(rows)
    for prefix in ("", "equal_average_", "baseline_", "pnas_"):
        probability = np.clip(table["label"] * 0.5 + rng.uniform(0, 0.5, len(table)), 0, 1)
        table[f"{prefix}probability"] = probability
        table[f"{prefix}prediction"] = (probability >= 0.5).astype(int)
        table[f"{prefix}threshold"] = 0.5
    return table


def test_contrasts_share_resamples_and_write_pnas_format(tmp_path):
    table = _scored()
    table["equal_average_probability"] = table["baseline_probability"]
    table["equal_average_prediction"] = table["baseline_prediction"]
    (tmp_path / "contrasts").mkdir()
    out = session.contrast_outputs("Liver", table, session.CONTRASTS, 20, tmp_path)
    reference = out[out["reference"] == session.REFERENCE].reset_index(drop=True)
    average = out[out["reference"] == session.EQUAL_AVERAGE].reset_index(drop=True)
    pd.testing.assert_frame_equal(reference.drop(columns="reference"), average.drop(columns="reference"))
    pnas_format = pd.read_csv(tmp_path / "Liver_pair_bootstrap_ci.csv")
    assert pnas_format.columns.tolist() == ["metric", "lower", "median", "upper"]
    assert set(pnas_format["metric"]) == set(session.BOOTSTRAP_METRICS)
    assert out["seed"].unique().tolist() == [session.BOOTSTRAP_SEED + CONTEXTS.index("Liver")]
    point = metrics(table["label"], table["probability"], table["prediction"])["auroc"] - metrics(
        table["label"], table["baseline_probability"], table["baseline_prediction"])["auroc"]
    assert np.isclose(reference.set_index("metric").loc["auroc", "estimate"], point)


def test_model_rows_use_pnas_names_for_the_headline_and_baseline():
    rows = session.model_rows("Brain", _scored(), "candidate", [session.REFERENCE, session.HEADLINE])
    assert [row["model"] for row in rows] == [session.BASELINE_MODEL_NAME, "candidate"]


def _fake_baseline():
    def parse(record):
        return record["sequence"], record["structure"], record["target"], record["label"]
    return SimpleNamespace(parse_openai_json=parse)


def test_development_loader_matches_pnas_canonical_loader(tmp_path):
    folder = tmp_path / "data" / "human" / "Liver"
    folder.mkdir(parents=True)
    lines = [json.dumps({"sequence": "ACGU", "structure": "(..)", "target": i, "label": i % 2}) for i in range(4)]
    metadata = "pair_id,yes_no\n" + "\n".join(f"{7 + i // 2},{'yes' if i % 2 else 'no'}" for i in range(4)) + "\n"
    for split in ("test", "valid"):
        (folder / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
        (folder / f"{split}.metadata.csv").write_text(metadata)
    mine = session.load_development_table(tmp_path, "Liver", "valid", _fake_baseline())
    theirs = load_canonical_table(tmp_path, "Liver", _fake_baseline())
    pd.testing.assert_frame_equal(mine.drop(columns="example_id"), theirs.drop(columns="example_id"))
    with pytest.raises(QuarantineViolation):
        session.load_development_table(tmp_path, "Liver", "test", _fake_baseline())


def _pnas_repository(tmp_path, table):
    """PNAS's saved session for Liver, and the two published aggregates it must reproduce."""
    folder = tmp_path / session.PREDECESSOR["results"]
    folder.mkdir(parents=True)
    saved = table[["row_index", "pair_id", "label", "baseline_probability", "baseline_prediction"]].copy()
    saved["probability"], saved["prediction"] = table["pnas_probability"], table["pnas_prediction"]
    saved.to_parquet(folder / "Liver_predictions.parquet", index=False)
    published_pnas = pd.DataFrame([{"context": "Liver", "model": "pnas", "threshold": 0.5,
                                    **metrics(saved["label"], saved["probability"], saved["prediction"])}])
    published_baseline = pd.DataFrame([{"context": "Liver", "variant": "baseline",
                                        **metrics(saved["label"], saved["baseline_probability"],
                                                  saved["baseline_prediction"])}])
    return published_pnas, published_baseline


def _session_table():
    table = _scored()
    table["row_index"] = np.arange(len(table))
    return table


def test_pnas_session_supplies_the_saved_adaredit_scores_and_keeps_ours_as_a_check(tmp_path):
    table = _session_table()
    published_pnas, published_baseline = _pnas_repository(tmp_path, table)
    assert ((table["baseline_probability"] - 0.5).abs() > 1e-3).all()  # fixture: no score at the threshold
    mine = table.copy()
    mine["baseline_probability"] = mine["baseline_probability"] + 5e-4  # cross-device noise within 1e-3
    mine["baseline_prediction"] = (mine["baseline_probability"] >= 0.5).astype(int)
    columns, check = session.pnas_session(tmp_path, "Liver", mine, published_pnas, published_baseline, "pnas")
    assert np.array_equal(columns["baseline_probability"], table["baseline_probability"])  # the saved scores
    assert np.array_equal(columns["baseline_cpu_probability"], mine["baseline_probability"])  # ours, kept
    assert np.array_equal(columns["pnas_probability"], table["pnas_probability"])
    assert check["adaredit_cpu_vs_saved"]["passes"]
    assert check["pnas_published_metric_max_difference"] == {"scores": 0.0, "counts": 0.0}


def test_pnas_session_refuses_misaligned_rows_and_large_disagreement(tmp_path):
    table = _session_table()
    published_pnas, published_baseline = _pnas_repository(tmp_path, table)
    shifted = table.copy()
    shifted["pair_id"] = shifted["pair_id"] + 1
    with pytest.raises(RuntimeError, match="same rows"):
        session.pnas_session(tmp_path, "Liver", shifted, published_pnas, published_baseline, "pnas")
    moved = table.copy()
    moved["baseline_probability"] = moved["baseline_probability"] + 0.01
    with pytest.raises(RuntimeError, match="CPU AdarEdit scores disagree"):
        session.pnas_session(tmp_path, "Liver", moved, published_pnas, published_baseline, "pnas")


def test_pnas_session_refuses_a_call_flip_away_from_the_threshold(tmp_path):
    table = _session_table()
    published_pnas, published_baseline = _pnas_repository(tmp_path, table)
    flipped = table.copy()
    row = int(np.argmax((flipped["baseline_probability"] - 0.5).abs().to_numpy()))
    flipped.loc[row, "baseline_prediction"] = 1 - flipped.loc[row, "baseline_prediction"]
    flipped.loc[row, "baseline_probability"] = 1.0 - flipped.loc[row, "baseline_probability"]
    with pytest.raises(RuntimeError, match="CPU AdarEdit scores disagree"):
        session.pnas_session(tmp_path, "Liver", flipped, published_pnas, published_baseline, "pnas")


def test_pnas_session_refuses_unreproduced_published_metrics(tmp_path):
    table = _session_table()
    published_pnas, published_baseline = _pnas_repository(tmp_path, table)
    wrong = published_pnas.copy()
    wrong["auroc"] = wrong["auroc"] - 1e-5
    with pytest.raises(RuntimeError, match="published metrics"):
        session.pnas_session(tmp_path, "Liver", table, wrong, published_baseline, "pnas")
    counts = published_pnas.copy()
    counts["tp"] = counts["tp"] + 1
    with pytest.raises(RuntimeError, match="published metrics"):
        session.pnas_session(tmp_path, "Liver", table, counts, published_baseline, "pnas")
    baseline = published_baseline.copy()
    baseline["f1"] = baseline["f1"] - 0.01
    with pytest.raises(RuntimeError, match="published baseline mismatch"):
        session.pnas_session(tmp_path, "Liver", table, published_pnas, baseline, "pnas")


def test_no_output_is_written_when_a_later_endpoint_fails_a_check(tmp_path, monkeypatch):
    written, prepared = [], []
    monkeypatch.setattr(session, "preflight", lambda *a: {"contexts": {}})
    monkeypatch.setattr(session, "consume_final_authorization", lambda *a: None)
    monkeypatch.setattr(session, "session_inputs", lambda *a: {})

    def prepare(repository, context, *rest):
        if context == CONTEXTS[-1]:
            raise RuntimeError("check failed on the last endpoint")
        prepared.append(context)
        return pd.DataFrame(), {}

    monkeypatch.setattr(session, "prepare_context", prepare)
    monkeypatch.setattr(session, "write_outputs", lambda *a: written.append(a))
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="last endpoint"):
        session.run(tmp_path, tmp_path / "p.json", tmp_path / "m.json", "token", output, 8, None)
    assert prepared == list(CONTEXTS[:-1]) and written == [] and not output.exists()
