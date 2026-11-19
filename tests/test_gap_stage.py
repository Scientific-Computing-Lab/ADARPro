"""Gap-only second stage (D39/D40). Reference values are computed independently of the code under test where noted:
the gap by plain coordinate arithmetic, the planted effect by construction of the synthetic labels."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adarpro.models.gap_stage import (
    ARMS,
    fit_arm,
    pair_gap_table,
    run,
    shuffled_gap_values,
    stage_table,
)

SITE_MASTER = Path(__file__).resolve().parents[1] / "data/derived/site_master.parquet"


def pairs_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        len1, len2 = int(rng.integers(250, 320)), int(rng.integers(250, 320))
        start1 = int(rng.integers(1_000, 1_000_000))
        end1 = start1 + len1 - 1
        gap = int(rng.integers(1, 5_000))
        start2 = end1 + gap
        end2 = start2 + len2 - 1
        sequence = "A" * (len1 + len2 + 10)
        rows.append({"pair_id": f"p{i}", "sequence": sequence, "structure": "." * len(sequence), "chr1": "chr1",
                     "start1": start1, "end1": end1, "chr2": "chr1", "start2": start2, "end2": end2,
                     "strand": "+" if i % 2 else "-", "true_gap": gap})
    return pd.DataFrame(rows)


def test_gap_equals_plain_arithmetic_on_both_strands():
    frame = pairs_frame()
    table = pair_gap_table(frame).set_index("pair_id")
    expected = frame.set_index("pair_id")
    assert np.array_equal(table.loc[expected.index, "gap_bp"].to_numpy(), expected["true_gap"].to_numpy())
    assert np.allclose(table.loc[expected.index, "gap"].to_numpy(), np.log1p(expected["true_gap"].to_numpy()))


@pytest.mark.skipif(not SITE_MASTER.exists(), reason="development site master not present")
def test_gap_on_real_development_pairs_matches_arithmetic():
    master = pd.read_parquet(SITE_MASTER)
    assert set(master["source_split"]) <= {"train", "valid"}
    canonical = master[master["context"] == "MuscleSkeletal"]
    table = pair_gap_table(canonical).set_index("pair_id")
    pairs = canonical.drop_duplicates("pair_id").set_index("pair_id")
    expected = np.log1p((pairs["start2"] - pairs["end1"]).to_numpy())  # independent: the coordinates alone
    assert np.allclose(table.loc[pairs.index, "gap"].to_numpy(), expected)
    assert len(table) == canonical["pair_id"].nunique()


def test_gap_guards():
    frame = pairs_frame(6)
    bad = frame.copy()
    bad.loc[0, "start2"] = bad.loc[0, "end1"] - 5  # overlapping arms -> negative gap
    with pytest.raises(RuntimeError, match="negative genomic gap"):
        pair_gap_table(bad)
    bad = frame.copy()
    bad.loc[0, "chr2"] = "chr2"
    with pytest.raises(RuntimeError, match="different chromosomes"):
        pair_gap_table(bad)
    bad = pd.concat([frame, frame.iloc[[0]].assign(start1=frame.loc[0, "start1"] + 1)], ignore_index=True)
    with pytest.raises(RuntimeError, match="more than one"):
        pair_gap_table(bad)


def test_shuffle_is_fixed_same_distribution_and_breaks_the_link():
    gaps = pair_gap_table(pairs_frame(200))
    first, hash_a, kept = shuffled_gap_values(gaps, 7)
    again, hash_b, _ = shuffled_gap_values(gaps.sample(frac=1, random_state=3), 7)  # input order must not matter
    _, hash_c, _ = shuffled_gap_values(gaps, 8)
    assert hash_a == hash_b and hash_a != hash_c
    assert np.array_equal(first["shuffled_gap"].to_numpy(), again["shuffled_gap"].to_numpy())
    assert np.array_equal(np.sort(first["shuffled_gap"]), np.sort(first["gap"]))
    assert kept < 0.05
    assert abs(np.corrcoef(first["gap"], first["shuffled_gap"])[0, 1]) < 0.3


def synthetic_stage(gap_effect: float, n_pairs: int = 400, sites: int = 6, seed: int = 1) -> pd.DataFrame:
    """Labels drawn from logit = score + gap_effect * standardized gap: the planted truth."""

    rng = np.random.default_rng(seed)
    pair_gap = rng.normal(0, 1, n_pairs)
    rows = []
    for p in range(n_pairs):
        for s in range(sites):
            score = rng.normal(0, 1.5)
            truth = score + gap_effect * pair_gap[p]
            rows.append({"example_id": f"e{p}_{s}", "context": "Synthetic", "pair_id": f"p{p}", "pair_local_key": s,
                         "development_fold": p % 5, "graph_logit": score + rng.normal(0, 0.5),
                         "cnn_logit": score + rng.normal(0, 0.5), "gap": pair_gap[p],
                         "shuffled_gap": pair_gap[(p * 7 + 3) % n_pairs],
                         "label": int(rng.random() < 1 / (1 + np.exp(-truth)))})
    table = pd.DataFrame(rows)
    table["expanded_reference_prediction"] = (table["graph_logit"] + table["cnn_logit"] > 0).astype(int)
    return table


def test_planted_gap_effect_is_recovered_and_absent_when_absent():
    with_effect = synthetic_stage(gap_effect=-1.0)
    _, selection = fit_arm(with_effect, ARMS["gap"], 0, 20260821)
    assert selection["weight_gap"] < -0.5  # planted negative effect, recovered in sign and rough size
    without = synthetic_stage(gap_effect=0.0)
    _, selection = fit_arm(without, ARMS["gap"], 0, 20260821)
    assert abs(selection["weight_gap"]) < 0.2


def test_no_fold_sees_its_own_rows():
    table = synthetic_stage(gap_effect=-1.0)
    outer, selection = fit_arm(table, ARMS["gap"], 2, 20260821)
    assert set(outer["development_fold"]) == {2}
    assert selection["train_rows"] == int((~table["development_fold"].isin([2, 3])).sum())
    leaked = table.copy()
    fold2 = leaked["development_fold"] == 2
    leaked.loc[fold2, "label"] = 1 - leaked.loc[fold2, "label"]  # planted leak: fold 2's labels flipped
    leaked.loc[fold2, "gap"] = leaked.loc[fold2, "gap"] * 50  # and fold 2's features distorted (the scaler)
    _, selection_leaked = fit_arm(leaked, ARMS["gap"], 2, 20260821)
    for key in ("C", "threshold", "weight_graph_logit", "weight_cnn_logit", "weight_gap", "intercept"):
        assert selection_leaked[key] == selection[key]  # nothing fitted for fold 2 depends on fold 2's rows


def test_stage_table_guards():
    frame = pairs_frame(4)
    gaps, _, _ = shuffled_gap_values(pair_gap_table(frame), 1)
    canonical = pd.DataFrame({"example_id": ["a", "b"], "context": ["Liver", "Liver"], "pair_id": ["p0", "p1"],
                              "pair_local_key": [0, 0], "label": [1, 0], "development_fold": [0, 1],
                              "baseline_oof_probability": [0.5, 0.5], "baseline_oof_prediction": [1, 0]})
    good = stage_table(canonical, "Liver", np.array([0.9, 0.1]), np.array([0.8, 0.2]), gaps)
    assert np.allclose(good["graph_logit"], np.log(np.array([0.9, 0.1]) / np.array([0.1, 0.9])))
    with pytest.raises(RuntimeError, match="differ from the canonical rows"):
        stage_table(canonical, "Liver", np.array([0.9]), np.array([0.8, 0.2]), gaps)
    with pytest.raises(RuntimeError, match="not all from"):
        stage_table(canonical, "Brain", np.array([0.9, 0.1]), np.array([0.8, 0.2]), gaps)
    with pytest.raises(RuntimeError, match="no gap for"):
        stage_table(canonical.assign(pair_id=["p0", "zz"]), "Liver", np.array([0.9, 0.1]), np.array([0.8, 0.2]),
                    gaps)


def test_arms_differ_only_in_the_gap_column():
    assert ARMS["no_gap"] == ["graph_logit", "cnn_logit"]
    assert ARMS["gap"] == ARMS["no_gap"] + ["gap"]
    assert ARMS["shuffled_gap"] == ARMS["no_gap"] + ["shuffled_gap"]


def test_bad_run_tag_is_refused_before_any_work(tmp_path):
    import torch

    with pytest.raises(ValueError, match="run tag"):
        run(tmp_path, tmp_path / "absent.parquet", tmp_path, tmp_path, tmp_path, "Liver", tmp_path / "out", 2, 10, 1,
            torch.device("cpu"), run_tag="v-2")
    assert not (tmp_path / "out").exists()
