# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from jaqmc.wavefunction.output.logdet import LogDet


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_all_singular_determinants_return_log_zero(dtype):
    matrices = jnp.zeros((2, 2, 2), dtype=dtype)

    result = cast(Mapping[str, jax.Array], LogDet().apply({}, matrices))

    assert jnp.isneginf(result["logpsi"].real)
    assert result["logpsi"].imag == 0
    assert jnp.all(result["sign_logdets"] == 0)
    assert jnp.all(jnp.isneginf(result["abs_logdets"]))
    if "sign_logpsi" in result:
        assert result["sign_logpsi"] == 0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_singular_determinants_do_not_change_nonzero_sum(dtype):
    matrices = jnp.stack([jnp.zeros((2, 2), dtype=dtype), jnp.eye(2, dtype=dtype)])

    result = cast(Mapping[str, jax.Array], LogDet().apply({}, matrices))

    assert result["logpsi"] == 0
    if "sign_logpsi" in result:
        assert result["sign_logpsi"] == 1
