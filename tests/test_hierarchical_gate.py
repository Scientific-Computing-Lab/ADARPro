import numpy as np
import pandas as pd

from adarpro.models.hierarchical_gate import select_pair_model


def test_pair_model_selection_excludes_nontraining_folds():
    table = pd.DataFrame(
        {
            "pair_id": np.arange(30),
            "development_fold": np.arange(30) % 5,
            "edited_fraction": np.linspace(0, 1, 30),
            "feature": np.linspace(0, 1, 30),
        }
    )
    model, parameter = select_pair_model(
        table, [0, 1, 2], ["feature"], seed=3, shuffled=False
    )
    assert parameter in {0.1, 1.0, 10.0, 100.0}
    prediction = model.predict(table[["feature"]])
    assert prediction.shape == (30,)
