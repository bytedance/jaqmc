# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for rotational angular-momentum projection."""

from operator import itemgetter

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.estimator.angular_momentum.spherical import (
    SphericalAngularMomentumProjection,
    _angular_momentum_characters,
)


def _singlet_log_psi(_params, u, v):
    return jnp.log(u[0] * v[1] - v[0] * u[1])


def _polarized_log_psi(_params, u, _v):
    return 2 * jnp.log(u[0])


def _data() -> HallData:
    return HallData(electrons=jnp.asarray([[0.8, 0.3], [2.0, -1.1]]))


def test_angular_momentum_characters_support_integer_and_half_integer_sectors():
    half_traces = jnp.asarray([-0.5, 0.0, 0.5, 1.0])

    characters = _angular_momentum_characters(half_traces, (0, 1, 2))

    np.testing.assert_allclose(characters[0], 1)
    np.testing.assert_allclose(characters[1], 2 * half_traces)
    np.testing.assert_allclose(characters[2], 4 * half_traces**2 - 1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"angular_momenta": ()}, "at least one"),
        ({"angular_momenta": (-1.0,)}, "nonnegative"),
        ({"angular_momenta": (float("nan"),)}, "finite"),
        ({"angular_momenta": (float("inf"),)}, "finite"),
        ({"angular_momenta": (0.25,)}, "half-integer"),
        ({"angular_momenta": (1.0, 1.0)}, "duplicates"),
        ({"n_rotations": 0}, "positive"),
    ],
)
def test_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SphericalAngularMomentumProjection(
            f_log_psi_from_spinor=_singlet_log_psi,
            **kwargs,
        )


def test_integer_angular_momenta_are_normalized():
    estimator = SphericalAngularMomentumProjection(
        angular_momenta=(0, 1),
        f_log_psi_from_spinor=_singlet_log_psi,
    )

    assert estimator._twice_angular_momenta == (0, 2)


def test_singlet_is_invariant_under_sampled_su2_rotations():
    estimator = SphericalAngularMomentumProjection(
        angular_momenta=(0.0,),
        n_rotations=32,
        f_log_psi_from_spinor=_singlet_log_psi,
    )

    stats, _ = estimator.evaluate_single_walker(
        params={},
        data=_data(),
        prev_walker_stats={"total_energy": 2.5},
        state=None,
        rngs=jax.random.key(2),
    )

    np.testing.assert_allclose(stats["angular_projection:weight"], [1], atol=2e-6)
    np.testing.assert_allclose(
        stats["angular_projection:energy_weight"], [2.5], atol=5e-6
    )

    final = estimator.finalize_stats(jax.tree.map(itemgetter(None), stats), None)
    np.testing.assert_allclose(final["angular_projection:weight"], [1], atol=2e-6)
    np.testing.assert_allclose(final["angular_projection:energy"], [2.5], atol=5e-6)


def test_hamiltonian_kernel_conjugates_local_energy():
    estimator = SphericalAngularMomentumProjection(
        n_rotations=4,
        f_log_psi_from_spinor=_singlet_log_psi,
    )

    stats, _ = estimator.evaluate_single_walker(
        params={},
        data=_data(),
        prev_walker_stats={"total_energy": 2.0 + 3.0j},
        state=None,
        rngs=jax.random.key(4),
    )

    np.testing.assert_allclose(
        stats["angular_projection:energy_weight"],
        [2.0 - 3.0j],
        atol=5e-6,
    )


def test_rotations_are_resampled_from_each_evaluation_rng():
    estimator = SphericalAngularMomentumProjection(
        angular_momenta=(0.0, 1.0),
        n_rotations=8,
        f_log_psi_from_spinor=_polarized_log_psi,
    )
    data = HallData(electrons=_data().electrons[:1])

    first, _ = estimator.evaluate_single_walker(
        {}, data, {"total_energy": 2.0}, None, jax.random.key(5)
    )
    repeated, _ = estimator.evaluate_single_walker(
        {}, data, {"total_energy": 2.0}, None, jax.random.key(5)
    )
    second, _ = estimator.evaluate_single_walker(
        {}, data, {"total_energy": 2.0}, None, jax.random.key(6)
    )

    np.testing.assert_allclose(
        first["angular_projection:weight"],
        repeated["angular_projection:weight"],
    )
    assert not np.allclose(
        first["angular_projection:weight"],
        second["angular_projection:weight"],
    )


def test_fresh_rotation_average_resolves_pure_l1_state():
    estimator = SphericalAngularMomentumProjection(
        angular_momenta=(0.0, 1.0),
        n_rotations=16_384,
        f_log_psi_from_spinor=_polarized_log_psi,
    )

    stats, _ = estimator.evaluate_single_walker(
        params={},
        data=HallData(electrons=_data().electrons[:1]),
        prev_walker_stats={"total_energy": 2.0},
        state=None,
        rngs=jax.random.key(7),
    )

    np.testing.assert_allclose(
        stats["angular_projection:weight"],
        [0.0, 1.0],
        atol=0.04,
    )


def test_finalize_forms_energy_from_accumulated_sector_weights():
    estimator = SphericalAngularMomentumProjection(
        angular_momenta=(0.0, 1.0),
        n_rotations=2,
        f_log_psi_from_spinor=_singlet_log_psi,
    )
    batched_stats = {
        "angular_projection:weight": jnp.asarray([[0.0, 0.25], [0.0, 0.75]]),
        "angular_projection:energy_weight": jnp.asarray([[0.0, 0.75], [0.0, 2.25]]),
    }

    final = estimator.finalize_stats(batched_stats, None)

    np.testing.assert_allclose(final["angular_projection:weight"], [0, 0.5], atol=1e-7)
    np.testing.assert_allclose(final["angular_projection:energy"][1], 3, atol=1e-7)
