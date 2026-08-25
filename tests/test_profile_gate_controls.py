from adarpro.evaluation.profile_gate_controls import run


def test_profile_gate_control_entrypoint_is_importable():
    assert callable(run)
