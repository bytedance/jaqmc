# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
from jax import numpy as jnp

from jaqmc_legacy.dmc.utils import agg_mean, agg_sum


def test_single_host_agg_mean() -> None:
    x = jnp.arange(1, 10)
    assert float(agg_mean(x)) == pytest.approx(5.0)


def test_single_host_agg_mean_weighted() -> None:
    x = jnp.arange(1, 10)
    weights = jnp.array([1.0] * 5 + [0.0] * 4)
    val = agg_mean(x, weights=weights)
    assert float(val) == pytest.approx(3.0)


def test_single_host_agg_sum() -> None:
    x = jnp.arange(1, 101)
    assert float(agg_sum(x)) == pytest.approx(5050.0)
