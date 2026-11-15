"""Opposite-base partner lane for unpaired local-window anchors.

Known answers below were worked by hand from the dot-bracket strings (the comment on each case gives the
derivation); they do not come from running the code under test.
"""

import pytest

from adarpro.models.multitask_oof import (
    LOCAL_RADIUS,
    PARTNER_LANES,
    common_partner_lane,
    local_window,
    opposite_base,
    partner_lane_run_suffix,
    saved_partner_lane,
    structure_partners,
)

CENTRE = LOCAL_RADIUS


def opposite(sequence: str, structure: str, position: int) -> int:
    return opposite_base(sequence, structure_partners(structure), position)


@pytest.mark.parametrize(
    ("sequence", "structure", "position", "expected"),
    [
        # symmetric 1x1 loop, 5' strand: flanks 0-10 and 2-8, one unpaired base each side -> 10 - (1 - 0) = 9
        ("GAGAAAAACAC", "(.(.....).)", 1, 9),
        # the same loop with a linker N opposite -> none
        ("GAGAAAAACNC", "(.(.....).)", 1, -1),
        # hairpin letter inside 2-8: the flanking pairs pair each other, not an internal loop -> none
        ("GAGAAAAACAC", "(.(.....).)", 4, -1),
        # symmetric 1x1 loop, 3' strand: flanks 9-3 and 11-1, base 2 unpaired opposite -> 3 - (10 - 9) = 2
        ("GGAGAAAAACACC", "((.(.....).))", 10, 2),
        # ... and back from the 5' strand: flanks 1-11 and 3-9 -> 11 - (2 - 1) = 10
        ("GGAGAAAAACACC", "((.(.....).))", 2, 10),
        # 1x1 loop between 1-9 and 3-7 -> 9 - (2 - 1) = 8
        ("AAAAAAAAAAA", "((.(...).))", 2, 8),
        # symmetric 2x2 loop between 1-12 and 4-9: 2 -> 12 - 1 = 11, 3 -> 12 - 2 = 10 (antiparallel)
        ("AAAAAAAAAAAAAA", "((..(....)..))", 2, 11),
        ("AAAAAAAAAAAAAA", "((..(....)..))", 3, 10),
        ("AAAAAAAAAAAAAA", "((..(....)..))", 11, 2),
        ("AAAAAAAAAAAAAA", "((..(....)..))", 10, 3),
        # bulge: flanks 1-9 and 3-8, nothing unpaired opposite -> none
        ("AAAAAAAAAAA", "((.(....)))", 2, -1),
        # asymmetric loop: one unpaired base on this side (2), two opposite (8, 9) -> none
        ("AAAAAAAAAAAA", "((.(...)..))", 2, -1),
        # multi-branch loop: two helices (2-5, 7-10) inside the closing pair 0-12 -> none
        ("AAAAAAAAAAAAA", "(.(..).(..).)", 1, -1),
        # exterior loop between two hairpins -> none
        ("AAAAAAAAAAAAAAA", "((...)).((...))", 7, -1),
        # unpaired 5' end with no left flank -> none
        ("AAAAAAAAA", "..((...))", 0, -1),
        # a paired position has no "opposite base" here (its partner is used directly)
        ("AAAAAAA", "((...))", 1, -1),
    ],
)
def test_opposite_base_known_answers(sequence, structure, position, expected):
    assert len(sequence) == len(structure)
    assert opposite(sequence, structure, position) == expected


@pytest.mark.parametrize("loop", [1, 2, 3, 4])
def test_opposite_of_opposite_is_the_target(loop):
    # helix 4 bp, symmetric loop of `loop` bases on each strand, inner hairpin
    structure = "((((" + "." * loop + "((...))" + "." * loop + "))))"
    sequence = "A" * len(structure)
    partners = structure_partners(structure)
    targets = [index for index in range(len(structure)) if partners[index] < 0]
    mismatched = [t for t in targets if opposite_base(sequence, partners, t) >= 0]
    assert len(mismatched) == 2 * loop
    for target in mismatched:
        across = opposite_base(sequence, partners, target)
        assert partners[across] < 0
        assert opposite_base(sequence, partners, across) == target


def test_opposite_lane_is_centred_on_the_opposite_base():
    sequence, structure = "GAGAAAAACAC", "(.(.....).)"
    fixed = local_window(sequence, structure, 1, partner_lane="opposite")
    legacy = local_window(sequence, structure, 1, partner_lane="legacy")
    # target lane is identical under both rules
    assert (fixed[0, :6] == legacy[0, :6]).all()
    # legacy: unpaired anchor -> partner lane is all N (channel 10) with no paired flag
    assert (legacy[0, 10] == 1).all() and (legacy[0, 6:10] == 0).all() and (legacy[0, 11] == 0).all()
    # opposite: centre column reads base 9 ('A', channel 6), itself unpaired (channel 11 = 0)
    assert fixed[0, 6, CENTRE] == 1 and fixed[0, 10, CENTRE] == 0 and fixed[0, 11, CENTRE] == 0
    # one step 3' on the target strand is one step 5' on the opposite strand: +1 -> base 8 ('C', paired),
    # -1 -> base 10 ('C', paired); two steps -> 7 ('A', unpaired) and off the end (N)
    assert fixed[0, 8, CENTRE + 1] == 1 and fixed[0, 11, CENTRE + 1] == 1
    assert fixed[0, 8, CENTRE - 1] == 1 and fixed[0, 11, CENTRE - 1] == 1
    assert fixed[0, 6, CENTRE + 2] == 1 and fixed[0, 11, CENTRE + 2] == 0
    assert fixed[0, 10, CENTRE - 2] == 1


@pytest.mark.parametrize(
    ("sequence", "structure", "target"),
    [
        ("AACGUU", "((..))", 1),  # paired target
        ("AAAAAAAAAAA", "((.(....)))", 2),  # bulge
        ("AAAAAAAAAAAA", "((.(...)..))", 2),  # asymmetric loop
        ("GAGAAAAACAC", "(.(.....).)", 4),  # hairpin
        ("GAGAAAAACNC", "(.(.....).)", 1),  # linker N opposite
        ("AGC", ".()", 0),  # unpaired end
    ],
)
def test_rules_agree_wherever_there_is_no_opposite_base(sequence, structure, target):
    assert (
        local_window(sequence, structure, target, partner_lane="opposite")
        == local_window(sequence, structure, target, partner_lane="legacy")
    ).all()


def test_unknown_partner_lane_is_rejected():
    with pytest.raises(ValueError):
        local_window("AACGUU", "((..))", 1, partner_lane="mirror")


def test_saved_configs_without_the_key_are_legacy():
    assert PARTNER_LANES == ("opposite", "legacy")
    assert saved_partner_lane({}) == "legacy"
    assert saved_partner_lane({"config": {"mode": "binary_pair_balanced_local"}}) == "legacy"
    assert saved_partner_lane({"config": {"partner_lane": "opposite"}}) == "opposite"
    with pytest.raises(RuntimeError):
        saved_partner_lane({"config": {"partner_lane": "mirror"}})


def test_components_must_share_one_rule():
    assert common_partner_lane(["opposite", "opposite"]) == "opposite"
    with pytest.raises(RuntimeError):
        common_partner_lane(["opposite", "legacy"])


def test_fixed_runs_get_distinct_registry_ids():
    assert partner_lane_run_suffix("legacy") == ""
    assert partner_lane_run_suffix("opposite") == "_partner_opposite"
    assert partner_lane_run_suffix("opposite", uses_local=False) == ""
