import pytest

from adarpro.models.final_ensemble import validate_checkpoint


def checkpoint(context="Liver", mode="binary_pair_balanced"):
    return {
        "config": {
            "context": context,
            "mode": mode,
            "training_split": "published train pairs; expanded raw high/low sites",
            "selection_split": "published validation canonical rows",
            "canonical_test_access": False,
        }
    }


def test_validate_checkpoint_accepts_sealed_final_component():
    validate_checkpoint(checkpoint(), "Liver", "binary_pair_balanced")


def test_validate_checkpoint_rejects_context_or_split_mismatch():
    with pytest.raises(RuntimeError, match="context/mode"):
        validate_checkpoint(checkpoint(), "Brain", "binary_pair_balanced")
    contaminated = checkpoint()
    contaminated["config"]["canonical_test_access"] = True
    with pytest.raises(RuntimeError, match="sealed-development"):
        validate_checkpoint(contaminated, "Liver", "binary_pair_balanced")
