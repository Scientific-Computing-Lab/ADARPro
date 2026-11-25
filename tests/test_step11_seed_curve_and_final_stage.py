"""Seed averaging in log-odds, the seed-curve CLI guards, and the frozen final gap stage.
Expected values are computed independently of the code under test where noted (numpy log-odds, a fresh sklearn fit)."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import adarpro.models.final_gap_stage as final_stage
from adarpro.evaluation.gap_stage_seed_curve import parse_runs
from adarpro.evaluation.gap_stage_seed_curve import run as curve_run
from adarpro.models.gap_stage import average_probabilities


def test_one_run_is_unchanged_and_several_average_in_log_odds():
    rng = np.random.default_rng(0)
    a, b, c = (rng.uniform(0.01, 0.99, 50) for _ in range(3))
    assert np.array_equal(average_probabilities([a]), a)
    logits = np.mean([np.log(x / (1 - x)) for x in (a, b, c)], axis=0)  # independent: plain numpy
    assert np.allclose(average_probabilities([a, b, c]), 1 / (1 + np.exp(-logits)), rtol=0, atol=1e-12)


def test_parse_runs_and_curve_guards(tmp_path):
    runs = parse_runs(["1=a", f"2={tmp_path}"], tmp_path)
    assert runs == {1: tmp_path / "a", 2: tmp_path}
    for bad in (["x=a"], ["1"], ["1=a", "1=b"]):
        with pytest.raises(ValueError):
            parse_runs(bad, tmp_path)
    absent = tmp_path / "absent.parquet"
    with pytest.raises(ValueError, match="run tag"):
        curve_run(tmp_path, absent, "Liver", runs, tmp_path, tmp_path / "out", None, 2, 10, 1, torch.device("cpu"),
                  run_tag="v-2")
    with pytest.raises(ValueError, match="at least two runs"):
        curve_run(tmp_path, absent, "Liver", {1: tmp_path}, tmp_path, tmp_path / "out", None, 2, 10, 1,
                  torch.device("cpu"))
    assert not (tmp_path / "out").exists()


def split_table(n_pairs: int = 300, sites: int = 5, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gap = rng.normal(0, 1, n_pairs)
    rows = []
    for p in range(n_pairs):
        for s in range(sites):
            score = rng.normal(0, 1.5)
            rows.append({"example_id": f"e{p}_{s}", "pair_id": f"p{p}", "source_split": "valid" if p % 5 == 0 else "train",
                         "graph_logit": score + rng.normal(0, 0.5), "cnn_logit": score + rng.normal(0, 0.5),
                         "gap": gap[p], "label": int(rng.random() < 1 / (1 + np.exp(-(score - gap[p]))))})
    return pd.DataFrame(rows)


def test_final_stage_fits_on_training_rows_only(monkeypatch):
    monkeypatch.setattr(final_stage, "CS", (0.1,))  # one penalty, so selection cannot move the fit
    table = split_table()
    model, selection = final_stage.fit_final_stage(table, 42)
    assert selection["train_rows"] == int((table.source_split == "train").sum())
    fresh = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, penalty="l2", solver="liblinear",
                                                               max_iter=5000, random_state=42))
    train = table[table.source_split == "train"]
    fresh.fit(train[final_stage.COLUMNS], train["label"])  # independent fit on the training rows alone
    assert np.allclose(model[-1].coef_, fresh[-1].coef_, rtol=0, atol=1e-12)
    leaked = table.copy()
    valid = leaked.source_split == "valid"
    leaked.loc[valid, "label"] = 1 - leaked.loc[valid, "label"]  # planted: validation labels flipped
    leaked.loc[valid, "gap"] = leaked.loc[valid, "gap"] * 50  # and validation features distorted (the scaler)
    leaked_model, _ = final_stage.fit_final_stage(leaked, 42)
    assert np.array_equal(leaked_model[-1].coef_, model[-1].coef_)
    assert np.array_equal(leaked_model[0].mean_, model[0].mean_)
    assert model[-1].coef_[0][2] < 0  # the planted negative gap effect is recovered


def test_frozen_stage_reproduces_the_fitted_model():
    table = split_table()
    model, selection = final_stage.fit_final_stage(table, 42)
    frozen = final_stage.freeze(model, selection)
    probability = lambda logit: 1 / (1 + np.exp(-logit))
    applied = final_stage.apply_frozen_stage(frozen, probability(table["graph_logit"].to_numpy()),
                                             probability(table["cnn_logit"].to_numpy()), table["gap"].to_numpy())
    assert np.allclose(applied, model.predict_proba(table[final_stage.COLUMNS])[:, 1], rtol=0, atol=1e-9)
    with pytest.raises(ValueError, match="unexpected frozen columns"):
        final_stage.apply_frozen_stage({**frozen, "columns": ["gap"]}, [0.5], [0.5], [1.0])


def test_final_stage_split_guards():
    table = split_table()
    leaky = table.copy()
    first_train_row = leaky.index[leaky["source_split"] == "train"][0]
    leaky.loc[first_train_row, "source_split"] = "valid"  # one site of a training pair moved: the pair is in both
    with pytest.raises(RuntimeError, match="pair leakage"):
        final_stage.fit_final_stage(leaky, 42)
    sealed = table.copy()
    sealed.loc[sealed.index[0], "source_split"] = "test"
    with pytest.raises(RuntimeError, match="exactly the published train and validation"):
        final_stage.fit_final_stage(sealed, 42)


def test_final_stage_run_refuses_published_folders_and_bad_input_before_work(tmp_path):
    absent = tmp_path / "absent.parquet"
    for published in ("models/frozen", "models/frozen_ensemble"):
        with pytest.raises(RuntimeError, match="published"):
            final_stage.run(tmp_path, absent, "Liver", [tmp_path], [tmp_path], tmp_path / published / "x", 2, 1,
                            torch.device("cpu"))
    with pytest.raises(ValueError, match="same, non-zero number"):
        final_stage.run(tmp_path, absent, "Liver", [tmp_path], [], tmp_path / "out", 2, 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="run tag"):
        final_stage.run(tmp_path, absent, "Liver", [tmp_path], [tmp_path], tmp_path / "out", 2, 1,
                        torch.device("cpu"), run_tag="v-2")
    assert not (tmp_path / "out").exists()


def test_final_ensemble_refuses_a_bad_run_tag_before_any_work(tmp_path):
    from adarpro.models.final_ensemble import run

    with pytest.raises(ValueError, match="run tag"):
        run(tmp_path, tmp_path / "absent.parquet", "Liver", [("a", "binary_pair_balanced", tmp_path),
                                                             ("b", "binary_pair_balanced_cnn", tmp_path)],
            tmp_path / "out", 2, 42, torch.device("cpu"), run_tag="v-2")
