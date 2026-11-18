"""The shifted CNN alone (alignment control for the graph-free CNN), and the two small CLI additions used to rerun
the paper's development analyses on the selected design: ``within_pair_discrimination --prediction/--run-tag`` and
``unified_oof --run-tag``.

References are independent of the code under test: the shifted anchor is recomputed from the published shift rule
(``shifted_local_position``, unchanged) and compared window-for-window with ``local_window`` at that anchor.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from adarpro.models.final_training import FINAL_MODES
from adarpro.models.fixed_checkpoint_ensemble import example_kind
from adarpro.models.multitask_oof import (
    CNN_MODES,
    LOCAL_CONTROL_SHIFT,
    MODES,
    LocalCNN,
    canonical_examples_for_mode,
    create_model,
    local_window,
    make_training_graphs,
    mode_is_graph_free,
    mode_is_shifted,
    mode_uses_local_window,
    shifted_local_position,
)

SHIFTED_CNN = "binary_pair_balanced_shifted_cnn"
CNN = "binary_pair_balanced_cnn"


def table(length: int = 260, targets=(10, 130, 250)) -> pd.DataFrame:
    # a long stem-loop-stem so every anchor has real pairing context; targets near both ends and the middle
    half = (length - 20) // 2
    structure = "(" * half + "." * (length - 2 * half) + ")" * half
    sequence = ("ACGU" * length)[:length]
    return pd.DataFrame(
        {
            "example_id": [f"e{i}" for i in range(len(targets))],
            "sequence": [sequence] * len(targets),
            "structure": [structure] * len(targets),
            "local_position": list(targets),
            "label": [1, 0, 1][: len(targets)],
            "editing_index": [20.0, 0.0, 30.0][: len(targets)],
        }
    )


def test_the_shifted_cnn_mode_is_a_graph_free_shifted_control():
    assert SHIFTED_CNN in MODES and SHIFTED_CNN in CNN_MODES
    assert mode_is_graph_free(SHIFTED_CNN) and mode_uses_local_window(SHIFTED_CNN)
    assert mode_is_shifted(SHIFTED_CNN) and mode_is_shifted("binary_pair_balanced_shifted_local")
    assert not mode_is_shifted(CNN) and not mode_is_shifted("binary_pair_balanced_local")
    assert not mode_is_shifted("binary_pair_balanced")
    # a control is never a final model
    assert SHIFTED_CNN not in FINAL_MODES
    model, _ = create_model(torch.device("cpu"), SHIFTED_CNN)
    assert isinstance(model, LocalCNN)
    # the published modes are unchanged
    assert CNN in CNN_MODES and "binary_pair_balanced_shifted_local" in MODES


def test_canonical_windows_sit_at_the_published_shifted_anchor():
    frame = table()
    shifted = canonical_examples_for_mode(frame, SHIFTED_CNN, baseline=None, partner_lane="opposite")
    aligned = canonical_examples_for_mode(frame, CNN, baseline=None, partner_lane="opposite")
    for example, aligned_example, row in zip(shifted, aligned, frame.itertuples(index=False)):
        anchor = shifted_local_position(len(row.sequence), int(row.local_position))
        assert abs(anchor - int(row.local_position)) == LOCAL_CONTROL_SHIFT
        expected = local_window(row.sequence, row.structure, anchor, partner_lane="opposite")
        assert torch.equal(example.local_window, expected)
        assert torch.equal(
            aligned_example.local_window,
            local_window(row.sequence, row.structure, int(row.local_position), partner_lane="opposite"),
        )
        assert not torch.equal(example.local_window, aligned_example.local_window)
        assert example.edge_index is None and example.x is None  # no graph built


def test_raw_training_windows_match_the_canonical_rule():
    frame = table()
    raw = make_training_graphs(
        frame, None, include_local=True, shifted_local=mode_is_shifted(SHIFTED_CNN),
        partner_lane="opposite", include_graph=False,
    )
    canonical = canonical_examples_for_mode(frame, SHIFTED_CNN, baseline=None, partner_lane="opposite")
    for left, right in zip(raw, canonical):
        assert torch.equal(left.local_window, right.local_window)


def test_ensemble_components_get_the_right_example_set():
    assert example_kind("binary_pair_balanced") == "graph"
    assert example_kind("binary_pair_balanced_local") == "graph"
    assert example_kind(CNN) == "window"
    assert example_kind(SHIFTED_CNN) == "shifted_window"


def test_ensemble_still_refuses_the_shifted_hybrid(tmp_path):
    from adarpro.models.fixed_checkpoint_ensemble import run_components

    components = [("graph", "binary_pair_balanced", tmp_path), ("shifted", "binary_pair_balanced_shifted_local", tmp_path)]
    with pytest.raises(ValueError, match="shifted-local controls"):
        run_components(tmp_path, tmp_path / "absent.parquet", components, "MuscleSkeletal", tmp_path, 2, 1,
                       torch.device("cpu"))


def test_final_ensemble_refuses_the_shifted_cnn(tmp_path):
    from adarpro.models.final_ensemble import run

    components = [("graph", "binary_pair_balanced", tmp_path / "a.pth"), ("shifted", SHIFTED_CNN, tmp_path / "b.pth")]
    with pytest.raises(ValueError, match="unsupported frozen ensemble component"):
        run(tmp_path, tmp_path / "absent.parquet", "MuscleSkeletal", components, tmp_path / "out", 2, 42,
            torch.device("cpu"))


# --- within_pair_discrimination --prediction / --run-tag ---
def test_prediction_specifications_parse_and_refuse_bad_input(tmp_path):
    from adarpro.evaluation.within_pair_discrimination import parse_predictions

    parsed = parse_predictions(["Liver:pair_first=a.parquet", "Liver:target_local=/abs/b.parquet"], tmp_path)
    assert parsed == {"Liver": {"pair_first": tmp_path / "a.parquet", "target_local": Path("/abs/b.parquet")}}
    for bad in ("Liver=x", "Liver:pair_first", "Lung:pair_first=x", "Liver:cnn=x", "Liver:development_reference=x"):
        with pytest.raises(ValueError):
            parse_predictions([bad], tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        parse_predictions(["Liver:pair_first=a", "Liver:pair_first=b"], tmp_path)


def test_within_pair_override_never_falls_back_to_published_files(tmp_path):
    from adarpro.evaluation.within_pair_discrimination import run

    # an override that omits a context must fail (no silent mix with the published paths)
    with pytest.raises(FileNotFoundError, match="pair-first"):
        run(tmp_path, tmp_path / "out", tmp_path / "src", 10, 1, predictions={}, run_tag="v2")
    with pytest.raises(FileNotFoundError, match="missing --prediction"):
        run(tmp_path, tmp_path / "out", tmp_path / "src", 10, 1,
            predictions={"Artery": {"pair_first": tmp_path / "nope.parquet"}}, run_tag="v2")


def test_bad_run_tags_are_refused_before_any_work(tmp_path):
    from adarpro.evaluation.within_pair_discrimination import run as within_run
    from adarpro.models.unified_oof import run as unified_run

    with pytest.raises(ValueError, match="run tag"):
        within_run(tmp_path, tmp_path / "out", tmp_path / "src", 10, 1, run_tag="v-2")
    with pytest.raises(ValueError, match="run tag"):
        unified_run(tmp_path, tmp_path / "absent.parquet", tmp_path / "absent.parquet", "MuscleSkeletal",
                    "unified_target_local_interaction_aux_v1", tmp_path, 1, 2, 1, 1, torch.device("cpu"),
                    run_tag="v-2")
    assert not (tmp_path / "out").exists()
