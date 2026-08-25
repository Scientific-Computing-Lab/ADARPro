from pathlib import Path

import pandas as pd

from adarpro.evaluation.natural_prevalence import upsert_context_metrics


def test_upsert_context_metrics_preserves_other_contexts(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {"context": "MuscleSkeletal", "auroc": 0.8},
            {"context": "Liver", "auroc": 0.7},
        ]
    ).to_csv(path, index=False)

    upsert_context_metrics(path, [{"context": "Liver", "auroc": 0.9}])

    observed = pd.read_csv(path).set_index("context")["auroc"].to_dict()
    assert observed == {"Liver": 0.9, "MuscleSkeletal": 0.8}
