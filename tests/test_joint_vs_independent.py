import pandas as pd

from adarpro.evaluation.joint_vs_independent import observed_metrics


def test_observed_metrics_adds_macro_average():
    tables = {}
    for context in ("Artery", "Brain"):
        common = {
            "label": [0, 0, 1, 1],
            "pair_id": [1, 2, 3, 4],
        }
        candidate = pd.DataFrame(
            {**common, "probability": [0.1, 0.2, 0.8, 0.9], "prediction": [0, 0, 1, 1]}
        )
        reference = pd.DataFrame(
            {**common, "probability": [0.2, 0.3, 0.7, 0.8], "prediction": [0, 0, 1, 1]}
        )
        tables[context] = (candidate, reference)
    result = observed_metrics(tables)
    macro = result[result["context"].eq("macro")]
    assert set(macro["model"]) == {"joint", "independent"}
    assert (macro["f1"] == 1.0).all()
