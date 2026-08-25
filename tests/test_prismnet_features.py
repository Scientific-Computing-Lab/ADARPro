import pandas as pd

from adarpro.rbp.prismnet_features import aggregate_cells


def test_prismnet_cell_aggregation_keeps_cross_cell_consistency():
    rows = []
    for cell, overlap in [("K562", 1), ("HepG2", 0)]:
        row = {
            "pair_local_key": "1:2",
            "gene": "ILF3",
            "cell_line": cell,
            "focal_overlap": overlap,
            "partner_overlap": 0,
            "duplex_peak_count": overlap,
            "focal_nearest_distance": 0 if overlap else 20,
            "duplex_coverage_fraction": overlap * 0.1,
            "duplex_max_signal": overlap * 0.9,
            "focal_arm_coverage_fraction": overlap * 0.1,
            "opposite_arm_coverage_fraction": 0,
        }
        for window in (5, 10, 25, 50, 100):
            row[f"focal_count_w{window}"] = overlap
            row[f"focal_max_signal_w{window}"] = overlap * 0.9
            row[f"focal_mean_signal_w{window}"] = overlap * 0.9
        rows.append(row)
    result = aggregate_cells(pd.DataFrame(rows)).iloc[0]
    assert result["prismnet_focal_overlap_any"] == 1
    assert result["prismnet_focal_overlap_cell_fraction"] == 0.5
