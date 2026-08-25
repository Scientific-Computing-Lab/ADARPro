import numpy as np
import pandas as pd

from adarpro.features.rbp_context import add_expression, aggregate_tracks


def test_track_aggregation_preserves_replicate_agreement():
    rows = []
    for overlap, accession in [(1, "a"), (0, "b")]:
        row = {
            "pair_local_key": "1:2",
            "gene": "ILF3",
            "file_accession": accession,
            "cell_line": accession,
            "focal_overlap": overlap,
            "partner_overlap": 0,
            "duplex_peak_count": overlap,
            "focal_arm_peak_count": overlap,
            "opposite_arm_peak_count": 0,
            "focal_nearest_distance": 0 if overlap else 10,
            "duplex_coverage_fraction": overlap * 0.1,
            "duplex_max_signal": overlap * 2,
            "focal_arm_coverage_fraction": overlap * 0.2,
            "opposite_arm_coverage_fraction": 0,
            "shifted_focal_overlap": 0,
            "shifted_partner_overlap": 0,
            "shifted_duplex_peak_count": 0,
            "shifted_duplex_coverage_fraction": 0,
        }
        for window in (5, 10, 25, 50, 100):
            row[f"focal_count_w{window}"] = overlap
            row[f"partner_count_w{window}"] = 0
            row[f"focal_max_signal_w{window}"] = overlap * 2
            row[f"shifted_focal_count_w{window}"] = 0
        rows.append(row)
    result = aggregate_tracks(pd.DataFrame(rows)).iloc[0]
    assert result["eclip_track_count"] == 2
    assert result["eclip_focal_overlap_any"] == 1
    assert result["eclip_focal_overlap_track_fraction"] == 0.5


def test_expression_is_not_fabricated_for_combined():
    site_gene = pd.DataFrame(
        {"pair_local_key": ["1:2"], "gene": ["ILF3"], "eclip_focal_overlap_any": [1]}
    )
    for column in [
        "eclip_focal_near_w25_any",
        "eclip_focal_near_w100_any",
        "eclip_partner_overlap_any",
        "eclip_duplex_peak_any",
        "eclip_duplex_coverage_fraction_max",
    ]:
        site_gene[column] = 0
    examples = pd.DataFrame(
        {
            "example_id": ["x", "y"],
            "context": ["Artery", "Combined"],
            "pair_id": [1, 1],
            "pair_local_key": ["1:2", "1:2"],
        }
    )
    expression = pd.DataFrame(
        {
            "context": ["Artery"],
            "gene": ["ILF3"],
            "tpm": [10.0],
            "log1p_tpm": [np.log1p(10)],
            "expression_percentile_across_gtex": [0.5],
            "tpm_over_adar": [0.1],
            "tpm_over_adarb1": [0.2],
        }
    )
    result = add_expression(site_gene, examples, expression)
    assert result.loc[result.context == "Combined", "tpm"].isna().all()
