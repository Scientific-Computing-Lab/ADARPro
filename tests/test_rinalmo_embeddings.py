import numpy as np
import pytest

from adarpro.features.rinalmo_embeddings import pool_representation


def test_pool_representation_aligns_sequence_after_cls_token():
    representation = np.arange(7 * 3, dtype=float).reshape(7, 3)
    target, global_mean, local_mean = pool_representation(
        representation, sequence_length=5, target_position=2, local_radius=1
    )
    assert np.array_equal(target, representation[3])
    assert np.array_equal(global_mean, representation[1:6].mean(axis=0))
    assert np.array_equal(local_mean, representation[2:5].mean(axis=0))


def test_pool_representation_clips_local_window_and_rejects_invalid_target():
    representation = np.arange(6 * 2, dtype=float).reshape(6, 2)
    _, _, local_mean = pool_representation(
        representation, sequence_length=4, target_position=0, local_radius=2
    )
    assert np.array_equal(local_mean, representation[1:4].mean(axis=0))
    with pytest.raises(ValueError, match="outside"):
        pool_representation(representation, 4, 4, 2)
