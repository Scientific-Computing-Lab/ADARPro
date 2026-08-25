import pandas as pd

from adarpro.models.pair_engagement_chromatin import run


def test_chromatin_module_exposes_run_entrypoint():
    assert callable(run)
    table = pd.DataFrame(
        {"chromatin__h3k36me3__pair__overlap__donor_mean": [0.0, 1.0]}
    )
    assert table.filter(like="__pair__").shape == (2, 1)
