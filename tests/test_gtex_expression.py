from adarpro.expression.gtex import TISSUE_MAP


def test_combined_is_not_assigned_a_gtex_tissue():
    assert set(TISSUE_MAP) == {"Artery", "Brain", "Liver", "MuscleSkeletal"}
    assert "Combined" not in TISSUE_MAP
