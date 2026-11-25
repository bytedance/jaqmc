# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Haldane-sphere angular-momentum estimator."""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.data import BatchedData
from jaqmc.estimator.angular_momentum import (
    AngularMomentumLoss,
    SphericalAngularMomentum,
)
from jaqmc.estimator.loss_grad import LossGrad
from jaqmc.utils.func_transform import transform_maybe_complex


def _mock_all_gather_identity(monkeypatch):
    monkeypatch.setattr(
        lax,
        "all_gather",
        lambda x, axis_name, tiled=True: x,
    )


@pytest.mark.x64_modes
@pytest.mark.parametrize("u_power", [0, 2], ids=["singlet", "highest_weight"])
@pytest.mark.parametrize(
    "first_theta",
    [
        pytest.param(None, id="generic"),
        pytest.param(np.pi / 2, id="chart_seam"),
        pytest.param(np.pi / 2 - 1e-6, id="beside_seam"),
        pytest.param(1e-3, id="near_pole"),
        pytest.param(1e-7, id="at_pole"),
    ],
)
def test_eigenstates_are_stable_near_chart_boundaries(
    x64_mode: bool,
    u_power: int,
    first_theta: float | None,
):
    """Known eigenstates retain their exact values near chart boundaries."""
    nelec = 5
    pair_power = 3
    electrons = jnp.asarray(
        [
            [1.1, 0.2],
            [2.0, -1.3],
            [0.8, 2.2],
            [2.4, 0.9],
            [1.6, -2.5],
        ],
        dtype=jnp.float64 if x64_mode else jnp.float32,
    )
    if first_theta is not None:
        electrons = electrons.at[0, 0].set(first_theta)

    def log_psi(params, u, v, *, precision: str | None = None):
        del params, precision
        pair = u[:, None] * v[None, :] - v[:, None] * u[None, :]
        upper = jnp.triu(jnp.ones((nelec, nelec), dtype=bool), k=1)
        value = pair_power * jnp.sum(jnp.log(jnp.where(upper, pair, 1.0)))
        return value + u_power * jnp.sum(jnp.log(u))

    lz = nelec * u_power / 2
    estimator = SphericalAngularMomentum(
        f_log_psi_from_spinor=log_psi,
    )
    stats, _ = estimator.evaluate_single_walker(
        params={},
        data=HallData(electrons=electrons),
        prev_walker_stats={},
        state=None,
        rngs=jax.random.key(0),
    )
    atol = 1e-8 if x64_mode else 1e-4

    np.testing.assert_allclose(stats["angular_momentum_z"], lz, rtol=0, atol=atol)
    np.testing.assert_allclose(
        stats["angular_momentum_z_square"], lz**2, rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        stats["angular_momentum_square"], lz * (lz + 1), rtol=0, atol=atol
    )


def test_spherical_angular_momentum_penalty_emissions(monkeypatch):
    """SphericalAngularMomentum emits real stats and complex-aware penalty grads."""
    _mock_all_gather_identity(monkeypatch)
    # 4 walkers of 1 electron on the sphere
    walkers = jnp.array(
        [
            [[0.7, 0.2]],
            [[1.3, -1.1]],
            [[2.1, 0.8]],
            [[2.5, -2.0]],
        ]
    )

    def log_psi_from_spinor(params, u, v, *, precision: str | None = None):
        del precision
        u0, v0 = u[0], v[0]
        w = params["w_r"] + 1j * params["w_i"]
        # Mix Lz (m=1 and m=0) and L^2 sectors with complex amplitude and phase
        return jnp.log(u0**2 + w * u0 * v0) + (params["c_r"] + 1j * params["c_i"]) * (
            jnp.abs(u0) ** 2
        )

    def log_psi(params, data, *, precision: str | None = None):
        from jaqmc.geometry.sphere import spinor_coordinates_from_angles

        u, v = spinor_coordinates_from_angles(data.electrons)
        return log_psi_from_spinor(params, u, v, precision=precision)

    weights = {
        "angular_momentum_z": -1.0,
        "angular_momentum_z_square": 2.0,
        "angular_momentum_square": 0.5,
    }
    estimator = SphericalAngularMomentum(
        f_log_psi_from_spinor=log_psi_from_spinor,
        loss_weight=AngularMomentumLoss(
            lz=weights["angular_momentum_z"],
            lz_square=weights["angular_momentum_z_square"],
            l_square=weights["angular_momentum_square"],
        ),
    )
    params = {
        "w_r": jnp.array(0.4),
        "w_i": jnp.array(-0.3),
        "c_r": jnp.array(0.2),
        "c_i": jnp.array(0.5),
    }
    rng = jax.random.key(0)
    estimator.init(HallData(electrons=walkers[0]), rng)

    batched_data = BatchedData(HallData(electrons=walkers), ["electrons"])
    stats, _ = estimator.evaluate_batch_walkers(params, batched_data, {}, None, rng)

    reduced = estimator.reduce(stats)
    ref_grad_logpsi = jax.vmap(
        transform_maybe_complex(log_psi, jax.grad), in_axes=(None, 0)
    )(params, HallData(electrons=walkers))

    for key, w in weights.items():
        local_c = stats[key]
        assert jnp.iscomplexobj(local_c)
        assert jnp.isrealobj(reduced[key])
        np.testing.assert_allclose(
            jnp.mean(stats[f"aux:loss:{key}"]), w * jnp.mean(local_c.real)
        )

        loss_grad = LossGrad(clip_method="none", f_log_psi=log_psi)
        grad_walkers, _ = loss_grad.evaluate_batch_walkers(
            params,
            batched_data,
            {
                f"aux:loss:{key}": stats[f"aux:loss:{key}"],
                f"aux:dloss_dlogpsi:{key}": stats[f"aux:dloss_dlogpsi:{key}"],
            },
            None,
            rng,
        )
        term_reduced = loss_grad.reduce(grad_walkers)
        np.testing.assert_allclose(
            term_reduced[f"loss:{key}"], w * jnp.mean(local_c.real)
        )
        term_grads = term_reduced["grads"]
        assert isinstance(term_grads, dict)
        centered = local_c - jnp.mean(local_c)
        real_only_grads = {}
        for param_key, g in ref_grad_logpsi.items():
            expected_grad = w * 2.0 * jnp.mean(jnp.real(jnp.conj(g) * centered))
            real_only_grads[param_key] = (
                w * 2.0 * jnp.mean(jnp.real(jnp.conj(g) * centered.real))
            )
            np.testing.assert_allclose(
                term_grads[param_key],
                expected_grad,
                rtol=1e-5,
                atol=1e-6,
            )
        # Verify the imaginary part of the local observable contributes to
        # the parameter gradient.
        assert any(not np.allclose(term_grads[p], real_only_grads[p]) for p in params)
