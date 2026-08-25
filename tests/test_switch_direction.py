from adarpro.models.switch_direction import randomized_order


def test_switch_order_is_deterministic_and_has_correct_label():
    first = randomized_order("chr1:1:+", "Brain", "Liver")
    second = randomized_order("chr1:1:+", "Brain", "Liver")
    assert first == second
    tissue_a, _, label = first
    assert (tissue_a == "Brain") == bool(label)
