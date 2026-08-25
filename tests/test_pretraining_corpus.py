import pandas as pd

from adarpro.features.pretraining_corpus import (
    benchmark_overlap_mask,
    count_inverted_pairs,
    count_transcript_compatible_pairs,
    inverted_pair_spans,
)


def repeats():
    return pd.DataFrame(
        {
            "chromosome": ["chr1"] * 5,
            "start": [0, 200, 500, 900, 2_000],
            "end": [100, 300, 600, 1_000, 2_100],
            "strand": ["+", "-", "+", "-", "+"],
        }
    )


def test_benchmark_overlap_excludes_either_arm():
    pairs = pd.DataFrame(
        [
            {
                "chr1": "chr1",
                "start1": 10,
                "end1": 20,
                "chr2": "chr1",
                "start2": 920,
                "end2": 930,
                "strand": "+",
            }
        ]
    )
    assert benchmark_overlap_mask(repeats(), pairs).tolist() == [
        True,
        False,
        False,
        True,
        False,
    ]


def test_inverted_pair_count_respects_gap_windows_and_chromosome():
    observed, by_chromosome = count_inverted_pairs(repeats(), (100, 500, 2_000))
    assert observed == {100: 1, 500: 3, 2_000: 6}
    assert by_chromosome == {}


def test_transcript_compatibility_requires_full_pair_containment():
    spans = inverted_pair_spans(repeats(), 500)
    transcripts = {
        "chr1": (
            pd.Series([0, 450]).to_numpy(),
            pd.Series([550, 1_000]).to_numpy(),
        )
    }
    total, by_chromosome = count_transcript_compatible_pairs(spans, transcripts)
    assert total == 2
    assert by_chromosome == {"chr1": 2}
