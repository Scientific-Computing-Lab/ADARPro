import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from adarpro.evaluation.nonhuman_transfer import (
    NB,
    SEED,
    analyse,
    bed_gap,
    best_f1,
    boot_weights,
    comparators,
    wauc,
    word,
)
from adarpro.features.genomic_context import pair_intrinsic_features


def _gap(start1, end1, start2, end2):
    features = pair_intrinsic_features(pd.Series({"structure": "((..))", "sequence": "ACGUAC", "strand": "+",
                                                  "start1": start1, "end1": end1, "start2": start2, "end2": end2}))
    return features["geometry__genomic_gap"]


def test_gap_counts_the_bases_strictly_between_the_arms():
    # 1-based inclusive arms 101-150 and 161-210: bases 151..160 lie between them, 10 bases
    bed = bed_gap(101, 150, 161, 210)
    assert bed == (100, 150, 160, 210)
    assert _gap(*bed) == 10.0
    # adjacent arms (150 then 151) leave no base between them
    assert _gap(*bed_gap(101, 150, 151, 200)) == 0.0


def test_gap_matches_the_human_bed_convention():
    # the human spacing stage reads start2 - end1 on BED half-open coordinates (pair_gap_table)
    assert _gap(100, 150, 160, 210) == 160 - 150


def test_weighted_auroc_with_unit_weights_equals_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)  # rounding creates ties
    assert wauc(s, y, np.ones((1, 300)))[0] == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_integer_weights_equal_duplicated_rows():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60)
    s = rng.normal(size=60) + y
    w = rng.integers(0, 4, 60)
    duplicated = roc_auc_score(np.repeat(y, w), np.repeat(s, w))
    assert wauc(s, y, w[None, :].astype(float))[0] == pytest.approx(duplicated, abs=1e-12)


def test_region_draws_are_reproducible_and_resample_whole_regions():
    regions = np.array(["r1", "r1", "r2", "r3", "r3", "r3"])
    first = boot_weights(regions, np.random.default_rng(SEED), replicates=50)
    second = boot_weights(regions, np.random.default_rng(SEED), replicates=50)
    assert np.array_equal(first, second)
    assert first.shape == (51, 6) and (first[0] == 1).all()
    for b in range(1, 51):  # every row of a region gets that region's draw count
        assert first[b, 0] == first[b, 1] and first[b, 3] == first[b, 4] == first[b, 5]
        assert first[b, [0, 2, 3]].sum() == 3  # three regions drawn with replacement
    assert NB == 2000


def test_words_follow_the_interval():
    assert word(0.001, 0.05) == "BEATS"
    assert word(-0.05, -0.001) == "WORSE"
    assert word(-0.01, 0.02).startswith("cannot be told apart")


def test_best_f1_is_the_ceiling_over_cut_offs():
    y = np.array([1, 1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    assert best_f1(y, s) == pytest.approx(2 * 3 / (4 + 3))  # call the top four: tp 3, called 4, positives 3


def test_comparator_is_the_checkpoint_with_the_higher_published_auroc(tmp_path):
    animals = ["Octopus", "Ptychodera", "Strongylocentrotus"]
    rows = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"] + animals
    (tmp_path / "results").mkdir()
    base = pd.DataFrame(0.80, index=rows, columns=animals)
    bio = pd.DataFrame(0.79, index=rows, columns=animals)
    bio.loc["Artery", "Ptychodera"] = 0.81
    base.to_csv(tmp_path / "results/matrix_baseline_auroc.csv")
    bio.to_csv(tmp_path / "results/matrix_bioaware_auroc.csv")
    table = comparators(tmp_path).set_index(["train", "eval"])
    assert table.loc[("Artery", "Ptychodera"), "column"] == "adaredit_bioaware_Artery"
    assert table.loc[("Artery", "Octopus"), "column"] == "adaredit_baseline_Artery"
    assert table.loc[("Ptychodera", "Ptychodera"), "kind"] == "own animal"
    assert len(table) == 3 * 6


def test_analyse_shares_one_region_draw_across_endpoints():
    from adarpro.evaluation.nonhuman_transfer import Checks
    from adarpro.quarantine import CONTEXTS

    rng = np.random.default_rng(2)
    animals = ["Octopus", "Ptychodera", "Strongylocentrotus"]
    sources = list(CONTEXTS) + animals
    frames = []
    for species in animals:
        n = 40
        y = rng.integers(0, 2, n)
        region = np.repeat([f"{species}{i}" for i in range(8)], 5)
        common = {f"adaredit_{v}_{src}": rng.random(n) for v in ("baseline", "bioaware") for src in sources}
        for endpoint in CONTEXTS:
            frame = pd.DataFrame({"endpoint": endpoint, "species": species, "line_no": np.arange(n),
                                  "region_id": region, "label": y, "gap_stage": rng.random(n) + y,
                                  "equal_average": rng.random(n) + y, "graph": rng.random(n),
                                  "cnn_alone": rng.random(n), "threshold_gap_stage": 0.5,
                                  "threshold_equal_average": 0.5, **common})
            frames.append(frame)
    D = pd.concat(frames, ignore_index=True)
    table = pd.DataFrame([{"train": src, "eval": sp, "column": f"adaredit_baseline_{src}"}
                          for sp in animals for src in list(CONTEXTS) + [sp]])
    thresholds = {v: pd.DataFrame(0.5, index=sources, columns=animals) for v in ("baseline", "bioaware")}
    check = Checks()
    M, C, F = analyse(D, table, thresholds, "label", check)
    check.require("synthetic")
    assert len(M) == 15 * 8 and len(C) == 15 * 7
    # an AdarEdit checkpoint shared by every endpoint gets the same AUROC whichever endpoint row it sits on
    own = M[M.model == "adaredit_animal_stronger"]
    assert own.groupby("species").auroc.nunique().eq(1).all()
    assert set(C.word.str.split(" ").str[0]) <= {"BEATS", "WORSE", "cannot"}


def test_gap_histogram_uses_one_set_of_bins_and_shares_of_all_sites():
    from adarpro.evaluation.nonhuman_transfer import ANIMALS, gap_histogram

    rng = np.random.default_rng(3)
    human = pd.DataFrame({"pair_id": np.arange(500), "gap": np.log1p(rng.integers(1, 5000, 500))})
    sites = pd.concat([pd.DataFrame({"species": sp, "split": split, "gap": np.log1p(rng.integers(3, 900, 50))})
                       for sp in ANIMALS for split in ("valid", "test")], ignore_index=True)
    table = gap_histogram(human, sites)
    edges = {g: tuple(t.bin_left) for g, t in table.groupby("group")}
    assert len(set(edges.values())) == 1 and len(edges) == 4
    shares = table.groupby("group").share_percent.sum()
    below = table.groupby("group").group_sites_below_range.first()
    total = table.groupby("group").group_sites.first()
    assert np.allclose(shares, 100 * (1 - below / total))
    assert total.loc["Octopus"] == 50  # test rows only
