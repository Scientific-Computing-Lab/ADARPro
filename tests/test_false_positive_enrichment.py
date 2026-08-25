import pandas as pd

from adarpro.evaluation.false_positive_enrichment import benjamini_hochberg


def test_benjamini_hochberg_is_monotone_in_rank():
    p_values = pd.Series([0.04, 0.001, 0.02, 0.8])
    adjusted = benjamini_hochberg(p_values)
    ranked = pd.DataFrame({"p": p_values, "q": adjusted}).sort_values("p")
    assert ranked["q"].is_monotonic_increasing
    assert (adjusted >= p_values).all()
    assert (adjusted <= 1).all()
