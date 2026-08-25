from adarpro.cross_tissue import editing_state


def test_editing_state_thresholds() -> None:
    assert editing_state(0.999) == "nonedited"
    assert editing_state(1.0) == "intermediate"
    assert editing_state(14.999) == "intermediate"
    assert editing_state(15.0) == "edited"
