import numpy as np
import pandas as pd
import pytest

from adarpro.mapping.bed_pair_gaps import (
    check_development,
    gap_table,
    lookup_gaps,
    normalize_sequence,
)

ALPHABET = {"A", "C", "G", "U", "T", "N"}


def _bed_sequences(rows):
    return pd.DataFrame(
        [{"bed_row": i, "chr1": c, "start1": s1, "end1": e1, "chr2": c, "start2": s2, "end2": e2, "strand": "+",
          "sequence": seq} for i, (c, s1, e1, s2, e2, seq) in enumerate(rows)]
    )


def test_normalize_follows_the_canonical_parser():
    assert normalize_sequence("acgRt", ALPHABET) == "ACGNT"


def test_gap_is_start2_minus_end1_with_log1p():
    table = gap_table(_bed_sequences([("chr1", 0, 99, 150, 249, "AAAC"), ("chr1", 0, 99, 99, 200, "AAAG")]),
                      ALPHABET)
    assert table["gap_bp"].tolist() == [51.0, 0.0]
    assert np.allclose(table["gap"], [np.log1p(51), 0.0])
    assert table["sequence_copies"].tolist() == [1, 1]


def test_shared_sequence_is_allowed_only_with_the_same_gap():
    same = gap_table(_bed_sequences([("chrX", 0, 9, 20, 29, "ACGT"), ("chrY", 0, 9, 20, 29, "ACGT")]), ALPHABET)
    assert same["sequence_copies"].tolist() == [2, 2]
    with pytest.raises(RuntimeError, match="share a sequence but not a gap"):
        gap_table(_bed_sequences([("chrX", 0, 9, 20, 29, "ACGT"), ("chrY", 0, 9, 40, 49, "ACGT")]), ALPHABET)


def test_negative_gap_is_refused():
    with pytest.raises(RuntimeError, match="negative genomic gap"):
        gap_table(_bed_sequences([("chr1", 0, 99, 50, 149, "ACGT")]), ALPHABET)


def test_lookup_maps_rows_by_sequence_and_refuses_unknown_sequences():
    gaps = gap_table(_bed_sequences([("chr1", 0, 9, 20, 29, "ACGT"), ("chrX", 0, 9, 30, 39, "GGGG"),
                                     ("chrY", 0, 9, 30, 39, "GGGG")]), ALPHABET)
    found = lookup_gaps(pd.Series(["acgt", "GGGG", "ACGT"], index=[5, 6, 7]), gaps, ALPHABET)
    assert found.index.tolist() == [5, 6, 7]
    assert found["gap_bp"].tolist() == [11.0, 21.0, 11.0]
    assert found["sequence_copies"].tolist() == [1, 2, 1]
    with pytest.raises(RuntimeError, match="match no BED pair"):
        lookup_gaps(pd.Series(["TTTT"]), gaps, ALPHABET)


def _site_master(sequence, end1=9):
    return pd.DataFrame([{"pair_id": 1, "sequence": sequence, "structure": "(..)", "chr1": "chr1", "start1": 0,
                          "end1": end1, "chr2": "chr1", "start2": 20, "end2": 29, "strand": "+"}])


def test_development_check_passes_and_catches_a_wrong_sequence_or_gap():
    gaps = gap_table(_bed_sequences([("chr1", 0, 9, 20, 29, "ACGT")]), ALPHABET)
    mapping = pd.DataFrame({"pair_id": [1], "bed_row": [0]})
    assert check_development(gaps, _site_master("ACGT"), mapping, ALPHABET)["sequence_mismatches"] == 0
    with pytest.raises(RuntimeError, match="do not reproduce"):
        check_development(gaps, _site_master("ACGA"), mapping, ALPHABET)
    with pytest.raises(RuntimeError, match="do not reproduce"):
        check_development(gaps, _site_master("ACGT", end1=8), mapping, ALPHABET)
