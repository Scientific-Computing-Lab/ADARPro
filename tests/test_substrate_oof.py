import numpy as np

from adarpro.models.substrate_oof import paired_positions, substrate_node_features


def test_paired_positions_and_node_features():
    partners = paired_positions("((..)).....")
    assert partners.tolist() == [5, 4, -1, -1, 1, 0, -1, -1, -1, -1, -1]
    features = substrate_node_features("AANNNNNNNNA", partners)
    assert features.shape == (11, 8)
    assert features[0, 0].item() == 1
    assert features[0, -1].item() == -1
    assert features[-1, -1].item() == 1
    assert np.isclose(features[5, -1].item(), 0)
