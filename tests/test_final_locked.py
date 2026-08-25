import numpy as np
import pandas as pd
import pytest

from adarpro.evaluation.final_locked import (
    pair_bootstrap_delta,
    validate_manifest,
    verify_frozen_inputs,
    weighted_logit_average,
)
from adarpro.quarantine import CONTEXTS, sha256_file


def manifest():
    specification = {
        "validation_summary": "summary.json",
        "validation_summary_sha256": "abc",
        "components": [
            {
                "name": "pair",
                "mode": "binary_pair_balanced",
                "checkpoint": "models/frozen/example.pth",
                "sha256": "abc",
            }
        ],
        "weights": [1.0],
        "threshold": 0.4,
    }
    return {
        "schema_version": 1,
        "candidate_name": "ADARpro",
        "code_revision": "abc",
        "selection_policy_sha256": "abc",
        "development_protocol": {
            "grouping_unit": "pair_id",
            "canonical_test_access_during_selection": False,
        },
        "candidate_specification": {
            "family": "test",
            "components": {"pair": "binary_pair_balanced"},
            "weights": [1.0],
        },
        "frozen_inputs": {
            "test": {"path": "input", "bytes": 1, "sha256": "abc"}
        },
        "contexts": {context: specification.copy() for context in CONTEXTS},
    }


def test_validate_manifest_requires_all_contexts_and_valid_weights():
    payload = manifest()
    validate_manifest(payload)
    payload["contexts"].pop("Combined")
    with pytest.raises(RuntimeError, match="all five"):
        validate_manifest(payload)


def test_validate_manifest_rejects_unfrozen_model_mode():
    payload = manifest()
    payload["contexts"]["Liver"]["components"][0]["mode"] = "binary_natural"
    with pytest.raises(RuntimeError, match="unsupported final component"):
        validate_manifest(payload)


def test_verify_frozen_inputs_rejects_changed_content(tmp_path):
    source = tmp_path / "policy.json"
    source.write_bytes(b"frozen")
    payload = manifest()
    payload["selection_policy_sha256"] = sha256_file(source)
    payload["frozen_inputs"] = {
        "model_selection_policy": {
            "path": "policy.json",
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    }
    verify_frozen_inputs(payload, tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="frozen input"):
        verify_frozen_inputs(payload, tmp_path)


def test_weighted_logit_average_returns_single_component_unchanged():
    probability = np.array([0.1, 0.5, 0.9])
    assert np.array_equal(weighted_logit_average([probability], np.array([1.0])), probability)


def test_pair_bootstrap_is_clustered_and_reproducible():
    table = pd.DataFrame(
        {
            "pair_id": [1, 1, 2, 2],
            "label": [0, 1, 0, 1],
            "probability": [0.1, 0.8, 0.2, 0.9],
            "prediction": [0, 1, 0, 1],
            "baseline_probability": [0.2, 0.7, 0.3, 0.8],
            "baseline_prediction": [0, 1, 0, 1],
        }
    )
    first = pair_bootstrap_delta(table, 4, 7)
    second = pair_bootstrap_delta(table, 4, 7)
    pd.testing.assert_frame_equal(first, second)
