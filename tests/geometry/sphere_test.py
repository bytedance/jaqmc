# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for spherical geometry and rotation sampling."""

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.geometry.sphere import (
    cartesian_from_spinor,
    sample_haar_so3,
    sample_haar_su2,
    spinor_coordinates_from_angles,
)


def test_haar_su2_samples_are_unitary_with_unit_determinant():
    rotations = sample_haar_su2(jax.random.key(0), 16)

    identity = rotations @ jnp.swapaxes(jnp.conj(rotations), -1, -2)
    expected_identity = jnp.broadcast_to(
        jnp.eye(2, dtype=rotations.dtype), rotations.shape
    )
    np.testing.assert_allclose(identity, expected_identity, atol=1e-6)
    np.testing.assert_allclose(jnp.linalg.det(rotations), 1, atol=1e-6)


def test_haar_so3_samples_are_orthogonal_with_unit_determinant():
    rotations = sample_haar_so3(jax.random.key(1), 16)

    identity = rotations @ jnp.swapaxes(rotations, -1, -2)
    expected_identity = jnp.broadcast_to(
        jnp.eye(3, dtype=rotations.dtype), rotations.shape
    )
    np.testing.assert_allclose(identity, expected_identity, atol=1e-6)
    np.testing.assert_allclose(jnp.linalg.det(rotations), 1, atol=1e-6)


def test_haar_rotation_samplers_support_empty_batches():
    key = jax.random.key(2)

    assert sample_haar_su2(key, 0).shape == (0, 2, 2)
    assert sample_haar_so3(key, 0).shape == (0, 3, 3)


def test_cartesian_from_spinors_is_gauge_invariant():
    electrons = jnp.asarray([[0.3, -1.2], [1.7, 2.4], [jnp.pi, 0.5]])
    u, v = spinor_coordinates_from_angles(electrons)
    phase = jnp.exp(0.7j)

    cartesian = cartesian_from_spinor(u, v)
    reconstructed = cartesian_from_spinor(u * phase, v * phase)

    np.testing.assert_allclose(reconstructed, cartesian, atol=1e-6)
