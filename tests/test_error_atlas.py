from adarpro.evaluation.error_atlas import local_fraction, paired_partner


def test_intrinsic_error_descriptors():
    assert paired_partner("((..))", 0) == 5
    assert paired_partner("((..))", 2) == -1
    assert local_fraction("AACGTA", 3, 2, {"G", "C"}) == 2 / 5
