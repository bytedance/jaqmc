# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Multi-device equivalence tests for streaming VMC gradients."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.data import BatchedData, Data
from jaqmc.estimator import LossAndGrad, StreamingLossAndGrad
from jaqmc.utils import parallel_jax


class GradientData(Data):
    x: jax.Array


def _logpsi(params, data):
    real = params["a"] * data.x + params["b"] * data.x**2
    return real + 1j * params["b"] * data.x


def _finalize(estimator, params, data, loss):
    walker_stats, _ = estimator.evaluate_batch_walkers(
        params, data, {"energy": loss}, None, jax.random.key(0)
    )
    reduced = estimator.reduce(walker_stats)
    final = estimator.finalize_stats(
        jax.tree.map(lambda x: x[None], reduced), None
    )
    return reduced, final


def test_streaming_loss_grad_matches_global_reference_across_devices():
    if jax.device_count() < 2:
        pytest.skip("At least two JAX devices are required")

    device_count = jax.device_count()
    batch_size = 4 * device_count
    x = jnp.linspace(-1.2, 1.3, batch_size)
    loss = jnp.linspace(-0.7, 1.8, batch_size) + 0.2j * x
    params = {"a": jnp.array(0.6), "b": jnp.array(-0.3)}
    data = BatchedData(GradientData(x=x), ["x"])

    reference = LossAndGrad(
        loss_key="energy", clip_method="none", f_log_psi=_logpsi
    )
    expected_reduced, expected_final = _finalize(
        reference, params, data, loss
    )

    streaming = StreamingLossAndGrad(
        loss_key="energy",
        vmap_chunk_size=1,
        clip_method="none",
        f_log_psi=_logpsi,
    )

    def distributed(p, d, local_loss):
        return _finalize(streaming, p, d, local_loss)

    distributed = parallel_jax.shard_map(
        distributed,
        mesh=parallel_jax.make_mesh(),
        in_specs=(
            parallel_jax.SHARE_PARTITION,
            data.partition_spec,
            parallel_jax.DATA_PARTITION,
        ),
        out_specs=parallel_jax.SHARE_PARTITION,
    )
    actual_reduced, actual_final = distributed(params, data, loss)

    for key in ("grad_logpsi", "grad_logpsi_and_loss"):
        for actual, expected in zip(
            jax.tree.leaves(actual_reduced[key]),
            jax.tree.leaves(expected_reduced[key]),
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_reduced["loss"], expected_reduced["loss"], rtol=1e-6
    )
    for actual, expected in zip(
        jax.tree.leaves(actual_final["grads"]),
        jax.tree.leaves(expected_final["grads"]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
