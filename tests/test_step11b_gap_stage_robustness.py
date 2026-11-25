"""The gap stage at natural prevalence (guards) and the within-duplex gap_stage role. The strong check of
the natural-prevalence module is on real data: its refitted stage must reproduce the gap-stage run's predictions exactly."""

import numpy as np
import pandas as pd
import pytest

from adarpro.evaluation.gap_stage_natural_prevalence import natural_rows, run
from adarpro.evaluation.within_pair_discrimination import (
    MODEL_LABELS,
    parse_predictions,
)
from adarpro.models.gap_stage import ARMS, fit_arm, fit_arm_model


def natural_frame(n: int = 6, context: str = "Liver") -> pd.DataFrame:
    return pd.DataFrame({"example_id": [f"e{i}" for i in range(n)], "context": context,
                         "pair_id": [f"p{i // 2}" for i in range(n)], "pair_local_key": range(n),
                         "development_fold": [i % 5 for i in range(n)], "editing_index": 20.0, "label": [1, 0] * (n // 2),
                         "probability": np.linspace(0.1, 0.9, n), "threshold": 0.5, "prediction": 0,
                         "baseline_oof_probability": 0.5, "baseline_oof_threshold": 0.5, "baseline_oof_prediction": 1})


def test_natural_rows_guards(tmp_path):
    good = natural_frame()
    good.to_parquet(tmp_path / "g.parquet", index=False)
    good.to_parquet(tmp_path / "c.parquet", index=False)
    table = natural_rows(tmp_path / "g.parquet", tmp_path / "c.parquet", "Liver")
    assert list(table["graph_probability"]) == list(good["probability"])
    good.assign(context="Brain").to_parquet(tmp_path / "b.parquet", index=False)
    with pytest.raises(RuntimeError, match="rows differ"):
        natural_rows(tmp_path / "g.parquet", tmp_path / "b.parquet", "Liver")
    good.assign(baseline_oof_probability=0.4).to_parquet(tmp_path / "r.parquet", index=False)
    with pytest.raises(RuntimeError, match="development references"):
        natural_rows(tmp_path / "g.parquet", tmp_path / "r.parquet", "Liver")
    with pytest.raises(RuntimeError, match="not all from"):
        natural_rows(tmp_path / "g.parquet", tmp_path / "c.parquet", "Brain")


def test_bad_run_tag_is_refused_before_any_work(tmp_path):
    absent = tmp_path / "absent.parquet"
    with pytest.raises(ValueError, match="run tag"):
        run(tmp_path, absent, "Liver", absent, absent, absent, absent, tmp_path / "out", 1, run_tag="v-2")
    assert not (tmp_path / "out").exists()


def test_fit_arm_model_matches_fit_arm_and_returns_the_fitted_stage():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(200):
        gap = rng.normal()
        for s in range(5):
            score = rng.normal(0, 1.5)
            rows.append({"example_id": f"e{p}_{s}", "pair_id": f"p{p}", "development_fold": p % 5,
                         "graph_logit": score, "cnn_logit": score + rng.normal(0, .3), "gap": gap,
                         "label": int(rng.random() < 1 / (1 + np.exp(-(score - gap))))})
    table = pd.DataFrame(rows)
    table["expanded_reference_prediction"] = (table["graph_logit"] > 0).astype(int)
    outer_a, selection_a = fit_arm(table, ARMS["gap"], 1, 7)
    outer_b, selection_b, model = fit_arm_model(table, ARMS["gap"], 1, 7)
    assert outer_a["probability"].equals(outer_b["probability"]) and selection_a == selection_b
    assert np.allclose(model.predict_proba(outer_b[ARMS["gap"]])[:, 1], outer_b["probability"], rtol=0, atol=1e-12)


def test_within_pair_accepts_the_gap_stage_role(tmp_path):
    assert "gap_stage" in MODEL_LABELS
    parsed = parse_predictions(["Liver:gap_stage=x.parquet"], tmp_path)
    assert parsed == {"Liver": {"gap_stage": tmp_path / "x.parquet"}}


def test_fixed_ensemble_refuses_a_bad_run_tag_before_any_work(tmp_path):
    import torch

    from adarpro.models.fixed_checkpoint_ensemble import run_components

    with pytest.raises(ValueError, match="run tag"):
        run_components(tmp_path, tmp_path / "absent.parquet", [("a", "binary_pair_balanced", tmp_path),
                                                               ("b", "binary_pair_balanced_target_pool", tmp_path)],
                       "MuscleSkeletal", tmp_path / "out", 2, 1, torch.device("cpu"), run_tag="v-2")
    assert not (tmp_path / "out").exists()
