import pandas as pd

from adarpro.features.predicted_context import (
    INTERACTION_FEATURES,
    build_long,
    build_prismnet_long,
)


def test_predicted_interaction_is_binding_times_expression():
    examples = pd.DataFrame(
        {
            "example_id": ["x"],
            "context": ["Artery"],
            "pair_id": [1],
            "pair_local_key": ["1:2"],
            "site_id": ["chr1:3:+"],
        }
    )
    prediction = {
        "pair_id": 1,
        "pair_local_key": "1:2",
        "site_id": "chr1:3:+",
        "gene": "ILF3",
    }
    for feature in INTERACTION_FEATURES:
        prediction[feature] = 2.0
    expression = pd.DataFrame(
        {
            "context": ["Artery"],
            "gene": ["ILF3"],
            "tpm": [3.0],
            "log1p_tpm": [1.5],
            "expression_percentile_across_gtex": [0.5],
            "tpm_over_adar": [0.1],
            "tpm_over_adarb1": [0.2],
        }
    )
    result = build_long(examples, pd.DataFrame([prediction]), expression)
    assert result["target_focal_value_x_log1p_tpm"].iloc[0] == 3.0


def test_prismnet_interaction_uses_tissue_expression():
    examples = pd.DataFrame(
        {
            "example_id": ["x"],
            "context": ["Artery"],
            "pair_id": [1],
            "pair_local_key": ["1:2"],
            "site_id": ["chr1:3:+"],
        }
    )
    prediction = pd.DataFrame(
        {
            "pair_local_key": ["1:2"],
            "gene": ["ILF3"],
            "prismnet_focal_overlap_any": [1.0],
            "prismnet_focal_near_w25_any": [1.0],
            "prismnet_focal_max_score_w25": [0.8],
            "prismnet_partner_overlap_any": [0.0],
            "prismnet_duplex_peak_any": [1.0],
            "prismnet_duplex_coverage_fraction_max": [0.1],
            "prismnet_duplex_max_score": [0.9],
        }
    )
    expression = pd.DataFrame(
        {
            "context": ["Artery"],
            "gene": ["ILF3"],
            "tpm": [3.0],
            "log1p_tpm": [1.5],
            "expression_percentile_across_gtex": [0.5],
            "tpm_over_adar": [0.1],
            "tpm_over_adarb1": [0.2],
        }
    )
    result = build_prismnet_long(examples, prediction, expression)
    assert result["prismnet_focal_overlap_any_x_log1p_tpm"].iloc[0] == 1.5
