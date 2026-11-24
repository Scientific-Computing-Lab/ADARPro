import numpy as np
import pandas as pd
import pytest

from adarpro.evaluation import reformer_error_screen as R


def test_bh_fdr_hand_example():
    q = R.bh_fdr(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_arm_first_from_sequence():
    seq = pd.Series(["AAAA" + "N" * 10 + "TTTT", "AAAA" + "N" * 10 + "TTTT"])
    pos = pd.Series([2, 16])
    assert R.arm_first_from_sequence(seq, pos).tolist() == [True, False]
    with pytest.raises(RuntimeError):
        R.arm_first_from_sequence(pd.Series(["AAAA"]), pd.Series([1]))


def test_err4():
    assert R.err4([1, 0, 0, 1], [1, 0, 1, 0]).tolist() == ["TP", "TN", "FP", "FN"]


def synthetic_frame(seed=1, n_pairs=60, per_pair=20):
    rng = np.random.default_rng(seed)
    n = n_pairs * per_pair
    pair = np.repeat(np.arange(n_pairs), per_pair)
    label = rng.integers(0, 2, n)
    prob = np.where(label == 1, rng.uniform(0.05, 1.0, n), rng.uniform(0.0, 0.95, n))
    pred = (prob >= 0.5).astype(int)
    frame = pd.DataFrame({
        "pair_id": pair, "label": label, "probability": prob, "prediction": pred,
        "editing_index": np.where(label == 1, rng.uniform(15, 80, n), rng.uniform(0, 1, n)),
        "arm_first": rng.integers(0, 2, n).astype(bool),
    })
    frame["err4"] = R.err4(frame.label, frame.prediction)
    for c in R.COMPOSITION:
        frame[c] = rng.normal(size=n)
    miss = (frame.err4 == "FN").to_numpy()
    frame["ILF3_HepG2__max25"] = rng.normal(size=n) + 1.5 * miss   # planted: higher in misses
    frame["ILF3_HepG2__focal"] = rng.normal(size=n)                # null
    frame["total_binding"] = rng.normal(size=n)
    for k in range(1, 6):
        frame[f"PC{k}"] = rng.normal(size=n)
    return frame


def test_screen_finds_planted_and_not_null():
    frame = synthetic_frame()
    feats = R.feature_columns(frame)
    assert feats[:2] == ["ILF3_HepG2__max25", "ILF3_HepG2__focal"]
    out = R.screen(frame, feats, seed=0)
    fn = out["fn"].set_index("feature")
    assert fn.loc["ILF3_HepG2__max25", "d_compCtrl"] > 0.8
    assert fn.loc["ILF3_HepG2__max25", "lo_compCtrl"] > 0
    assert fn.loc["ILF3_HepG2__focal", "lo_compCtrl"] < 0 < fn.loc["ILF3_HepG2__focal", "hi_compCtrl"]
    assert R.read_panel(out["fn"])["reading"] == "signal"
    # same seed, same numbers
    again = R.screen(frame, feats, seed=0)
    assert np.allclose(again["fn"]["d_compCtrl"].to_numpy(), out["fn"]["d_compCtrl"].to_numpy())
    assert np.allclose(again["sweep"].to_numpy(float), out["sweep"].to_numpy(float), equal_nan=True)


def test_read_panel_rule():
    def panel(d, lo, hi, q):
        return pd.DataFrame({"feature": ["total_binding", "PC1", "x", "placebo_permPC1"],
                             "visibility": ["invisible", "invisible", "invisible", "placebo"],
                             "d_compCtrl": d, "lo_compCtrl": lo, "hi_compCtrl": hi, "q_compCtrl": q})
    null = panel([0.01, 0.02, 0.05, 0.0], [-0.05, -0.04, -0.01, -0.05], [0.07, 0.08, 0.11, 0.05], [0.9, 0.9, 0.2, 0.9])
    assert R.read_panel(null)["reading"] == "no usable signal"
    wide = panel([0.01, 0.02, 0.05, 0.0], [-0.25, -0.04, -0.01, -0.05], [0.27, 0.08, 0.11, 0.05], [0.9, 0.9, 0.2, 0.9])
    assert R.read_panel(wide)["reading"] == "not shown either way"
    sig = panel([0.01, 0.02, 0.30, 0.0], [-0.05, -0.04, 0.2, -0.05], [0.07, 0.08, 0.4, 0.05], [0.9, 0.9, 0.001, 0.9])
    assert R.read_panel(sig)["reading"] == "signal"
