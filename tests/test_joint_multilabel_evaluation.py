import pandas as pd

from adarpro.evaluation.joint_multilabel import TISSUES, bootstrap_joint


def test_joint_bootstrap_retains_all_tissues_and_macro():
    rows = []
    for pair_index in range(8):
        for context in TISSUES:
            label = pair_index % 2
            rows.append(
                {
                    "pair_id": f"p{pair_index}",
                    "context": context,
                    "label": label,
                    "probability": 0.8 if label else 0.2,
                    "prediction": label,
                    "baseline_oof_probability": 0.6 if label else 0.4,
                    "baseline_oof_prediction": label,
                }
            )
    result = bootstrap_joint(pd.DataFrame(rows), replicates=3, seed=7)
    assert set(result["context"]) == {*TISSUES, "macro"}
    assert result["replicate"].nunique() == 3
    assert len(result) == 3 * 5 * 9
