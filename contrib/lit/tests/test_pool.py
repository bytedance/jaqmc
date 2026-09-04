# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jaqmc_contrib_lit.continuation_policy import _source_pool_target_digest
from jaqmc_contrib_lit.pool import (
    _batched_data_chunks,
    _load_batched_pool,
    _require_pool_walker_count,
    _save_batched_pool,
    _shuffled_batched_data_chunk,
    _shuffled_batched_data_chunk_index,
)
from jax import numpy as jnp
from upath import UPath

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.data import BatchedData


def _pool(walkers: int = 12) -> BatchedData[MoleculeData]:
    return BatchedData(
        data=MoleculeData(
            electrons=jnp.arange(3 * walkers, dtype=jnp.float32).reshape(walkers, 1, 3),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.ones((1,), dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )


def test_source_pool_round_trip_checks_identity_and_exact_walker_count(tmp_path):
    path = UPath(str(tmp_path / "pool.npz"))
    pool = _pool(8)
    metadata = {
        "axis": 2,
        "source_center": 0.125,
        "target_sha256": "ground-and-geometry-a",
    }
    _save_batched_pool(path, pool, metadata=metadata)

    restored = _load_batched_pool(path, _pool(4), metadata=metadata)
    _require_pool_walker_count(restored, expected_walkers=8, split="training")
    np.testing.assert_array_equal(restored.data.electrons, pool.data.electrons)

    with pytest.raises(ValueError, match=r"target_sha256.*mismatch"):
        _load_batched_pool(
            path,
            _pool(4),
            metadata={**metadata, "target_sha256": "ground-and-geometry-b"},
        )
    with pytest.raises(ValueError, match="expected exactly 4"):
        _require_pool_walker_count(restored, expected_walkers=4, split="training")


def test_source_pool_chunks_cover_all_walkers_and_shuffle_each_epoch():
    pool = _pool(12)
    chunks = list(_batched_data_chunks(pool, 5))
    assert [chunk.batch_size for chunk in chunks] == [5, 5, 2]
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(chunk.data.electrons) for chunk in chunks]),
        np.asarray(pool.data.electrons),
    )

    first_epoch = [
        _shuffled_batched_data_chunk_index(12, 3, step, seed=1701) for step in range(4)
    ]
    second_epoch = [
        _shuffled_batched_data_chunk_index(12, 3, step, seed=1701)
        for step in range(4, 8)
    ]
    assert sorted(first_epoch) == [0, 1, 2, 3]
    assert sorted(second_epoch) == [0, 1, 2, 3]
    selected = _shuffled_batched_data_chunk(pool, 3, 0, seed=1701)
    expected = pool.data.electrons[first_epoch[0] * 3 : (first_epoch[0] + 1) * 3]
    np.testing.assert_array_equal(selected.data.electrons, expected)


def test_training_shuffle_rejects_a_partial_walker_tail():
    with pytest.raises(ValueError, match="must be divisible"):
        _shuffled_batched_data_chunk_index(10, 4, 0, seed=9)


def test_source_pool_digest_binds_ground_and_static_system_only():
    data = MoleculeData(
        electrons=jnp.zeros((4, 2, 3)),
        atoms=jnp.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]),
        charges=jnp.asarray([1.0, 1.0]),
    )
    params = {"orbital": jnp.asarray([1.0, 2.0])}
    reference = _source_pool_target_digest(params, data)

    assert reference == _source_pool_target_digest(
        params,
        data.merge({"electrons": jnp.ones_like(data.electrons)}),
    )
    assert reference != _source_pool_target_digest(
        {"orbital": jnp.asarray([1.0, 2.1])},
        data,
    )
    assert reference != _source_pool_target_digest(
        params,
        data.merge({"atoms": data.atoms.at[1, 2].set(0.8)}),
    )
