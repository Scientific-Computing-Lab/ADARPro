import numpy as np

from adarpro.features.target_geometry import contact_features, pair_map, run_summary


def test_pair_map_and_helical_extent_features():
    structure = "(((...)))"
    partners = pair_map(structure)
    assert partners.tolist() == [8, 7, 6, -1, -1, -1, 2, 1, 0]
    values = contact_features("AAATTTTTT", structure, 1)
    assert values["target_geometry__target_paired"] == 1
    assert values["target_geometry__stem_extent_left"] == 1
    assert values["target_geometry__stem_extent_right"] == 1
    assert values["target_geometry__stem_extent_total"] == 3
    assert values["target_geometry__partner_base_T"] == 1


def test_run_summary_detects_unpaired_runs():
    values = run_summary(np.array([True, False, False, True, False]))
    assert np.isclose(values[0], 0.4)
    assert np.isclose(values[2], 2 / 5)
    assert np.isclose(values[3], 2 / 5)
