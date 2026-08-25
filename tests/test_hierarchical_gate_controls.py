from adarpro.evaluation.hierarchical_gate_controls import run


def test_hierarchical_control_entrypoint_is_callable():
    assert callable(run)
