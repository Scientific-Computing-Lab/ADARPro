import numpy as np
import pandas as pd

from adarpro.evaluation.pair_balance_diagnostics import binary_log_loss


def test_binary_log_loss_matches_expected_values():
    labels = pd.Series([0, 1])
    probabilities = pd.Series([0.25, 0.75])
    observed = binary_log_loss(labels, probabilities)
    assert np.allclose(observed, [-np.log(0.75), -np.log(0.75)])


def test_binary_log_loss_clips_endpoint_probabilities():
    observed = binary_log_loss(pd.Series([0, 1]), pd.Series([1.0, 0.0]))
    assert np.isfinite(observed).all()


def _diagnostic_inputs(tmp_path):
    rows = []
    for pair in range(10):
        for site in range(pair + 1):
            label = int(site % 2 == 0)
            rows.append(
                {
                    "example_id": f"p{pair}_s{site}",
                    "context": "MuscleSkeletal",
                    "pair_id": f"p{pair}",
                    "pair_local_key": f"p{pair}:{site}",
                    "label": label,
                    "development_fold": pair % 5,
                    "probability": 0.7 if label else 0.3,
                    "prediction": label,
                    "baseline_oof_probability": 0.6 if label else 0.45,
                    "baseline_oof_prediction": 1,
                }
            )
    pair = pd.DataFrame(rows)
    site = pair.assign(probability=pair["probability"] * 0.9, prediction=0)
    raw = pair[["context", "pair_id"]].assign(binary_label=pair["label"].astype(float))
    paths = {name: tmp_path / f"{name}.parquet" for name in ("pair", "site", "raw")}
    pair.to_parquet(paths["pair"])
    site.to_parquet(paths["site"])
    raw.to_parquet(paths["raw"])
    return paths


def test_run_tag_suffixes_only_the_registry_run_id(tmp_path, monkeypatch):
    import adarpro.evaluation.pair_balance_diagnostics as module

    paths = _diagnostic_inputs(tmp_path)
    recorded = []
    monkeypatch.setattr(module, "register", lambda repository, record: recorded.append(record))
    for tag, folder in ((None, "published"), ("v2", "tagged")):
        module.run(tmp_path, paths["pair"], paths["site"], paths["raw"], tmp_path / folder, tag)
    assert [r["run_id"] for r in recorded] == [
        "pair_balance_diagnostic_MuscleSkeletal",
        "pair_balance_diagnostic_MuscleSkeletal_v2",
    ]
    for name in ("raw_site_count_quintiles.csv", "spearman_correlations.csv"):
        assert (tmp_path / "published" / name).read_text() == (tmp_path / "tagged" / name).read_text()


def test_bad_run_tag_is_refused_before_any_work(tmp_path):
    import pytest

    import adarpro.evaluation.pair_balance_diagnostics as module

    with pytest.raises(ValueError, match="run tag"):
        module.run(tmp_path, tmp_path / "missing.parquet", tmp_path / "missing.parquet",
                   tmp_path / "missing.parquet", tmp_path / "out", "v-2")
