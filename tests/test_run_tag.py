"""Registry run tag: an optional ``_<tag>`` suffix that keeps a rerun of a published configuration (same mode, seed and
epochs) from colliding with the published registry row; no tag leaves every published run_id unchanged."""

from pathlib import Path

import pytest
import torch

from adarpro.models import final_training, multitask_oof
from adarpro.models.multitask_oof import partner_lane_run_suffix, run_tag_suffix


def test_no_tag_is_empty_and_tag_appends():
    assert run_tag_suffix(None) == ""
    assert run_tag_suffix("v2") == "_v2"


def test_published_graph_run_id_unchanged_without_tag():
    name = (
        "binary_pair_balanced_oof_Artery_seed20260821_epochs1000"
        + partner_lane_run_suffix("opposite", uses_local=False)
        + run_tag_suffix(None)
    )
    assert name == "binary_pair_balanced_oof_Artery_seed20260821_epochs1000"


def test_tag_follows_the_lane_suffix():
    name = (
        "binary_pair_balanced_local_oof_Liver_seed20260821_epochs1000"
        + partner_lane_run_suffix("opposite", uses_local=True)
        + run_tag_suffix("v2")
    )
    assert name.endswith("_partner_opposite_v2")


@pytest.mark.parametrize("bad", ["", "v-2", "v 2", "v_2", "v2/..", "vé"])
def test_bad_tags_are_refused(bad):
    with pytest.raises(ValueError, match="run tag"):
        run_tag_suffix(bad)


def test_oof_run_refuses_a_bad_tag_before_any_work():
    missing = Path("/nonexistent")
    with pytest.raises(ValueError, match="run tag"):
        multitask_oof.run(missing, missing, missing, "Artery", "binary_pair_balanced", missing, 1, 8, 1, 1, 0,
                          torch.device("cpu"), run_tag="v-2")


def test_final_training_refuses_a_bad_tag_before_any_work():
    missing = Path("/nonexistent")
    with pytest.raises(ValueError, match="run tag"):
        final_training.run(missing, missing, missing, "Artery", "binary_pair_balanced", missing, 1, 8, 1, 0,
                           torch.device("cpu"), run_tag="v-2")
