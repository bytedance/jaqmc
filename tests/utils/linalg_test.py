# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.utils.linalg import slogdet_blocks


@pytest.mark.parametrize(
    ("blocks", "expected_sign", "expected_logdet"),
    [
        # Real: signs multiply, log dets add.
        (([[2.0, 0.0], [0.0, -3.0]], [[5.0]]), -1.0, np.log(30.0)),
        # Complex: phases multiply, log dets add.
        (([[1j]], [[1 + 1j]]), (-1 + 1j) / np.sqrt(2), np.log(np.sqrt(2))),
    ],
)
def test_slogdet_blocks_combines_block_determinants(
    blocks, expected_sign, expected_logdet
):
    sign, logdet = slogdet_blocks(*(jnp.asarray(b) for b in blocks))

    np.testing.assert_allclose(sign, expected_sign)
    np.testing.assert_allclose(logdet, expected_logdet)


def test_slogdet_blocks_treats_empty_blocks_as_identity():
    empty = jnp.zeros((2, 0, 0))
    occupied = jnp.stack([jnp.eye(1), jnp.eye(1)])

    sign, logdet = slogdet_blocks(occupied, empty)

    np.testing.assert_allclose(sign, jnp.ones(2))
    np.testing.assert_allclose(logdet, jnp.zeros(2))


@pytest.mark.parametrize(
    "blocks",
    [(), (jnp.zeros((2, 3)),), (jnp.zeros(2),)],
    ids=["no_blocks", "non_square", "not_a_matrix"],
)
def test_slogdet_blocks_rejects_invalid_blocks(blocks):
    with pytest.raises(ValueError):
        slogdet_blocks(*blocks)
