# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Direct math tests for the ``standard`` PH derivative backend helpers."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.estimator.ph._standard import (
    build_second_order_matrix,
    evaluate_ph_first_order_vector,
)


def test_second_order_matrix_matches_bennett_formula_with_multi_center_accumulation():
    r"""Anchor the PH mass tensor to Bennett et al. (2022) on multiple atoms.

    Asserts ``M = 0.5 I + sum_a l2(r_a) (r_a I - r_a r_a^T / |r_a|)`` and
    symmetry of the assembled mass matrix on a two-atom geometry. The
    multi-atom geometry subsumes the single-atom special case for this
    closed-form formula.
    """
    electron = jnp.array([0.5, 0.1, -0.2])
    atoms = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    l2_values = jnp.array([0.03, 0.02])

    mass = build_second_order_matrix(electron, atoms, l2_values)

    rel = np.asarray(electron) - np.asarray(atoms)
    r = np.linalg.norm(rel, axis=-1)
    expected = 0.5 * np.eye(3)
    for rel_vec, dist, l2 in zip(rel, r, np.asarray(l2_values), strict=True):
        expected += l2 * (dist * np.eye(3) - np.outer(rel_vec, rel_vec) / dist)

    np.testing.assert_allclose(mass, expected, atol=1e-8)
    np.testing.assert_allclose(mass, mass.T, atol=1e-8)


def test_second_order_matrix_stays_finite_for_coincident_geometry():
    """Keep the second-order tensor finite at singular electron-atom geometry."""
    electron = jnp.array([0.0, 0.0, 0.0])
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    l2_values = jnp.array([0.03])

    mass = build_second_order_matrix(electron, atoms, l2_values)

    np.testing.assert_allclose(mass, 0.5 * np.eye(3), atol=1e-8)
    assert np.isfinite(np.asarray(mass)).all()


def test_first_order_vector_matches_bennett_identity_with_multi_center_sum():
    r"""Anchor ``_standard``'s jacfwd path to Bennett et al. (2022)'s analytic identity.

    Bennett's identity for the PH first-order vector is

    .. math::

        b = -\nabla \cdot (M - \tfrac{1}{2} I)
          = \sum_a 2\, v_{L^2}(r_a)\,(x - R_a)
          = \sum_a 2\, \ell_2(r_a) / r_a \cdot (x - R_a)

    in the XML :math:`\ell_2 = r\,v_{L^2}` convention. The ``_standard``
    backend computes the left-hand side via :func:`jax.jacfwd` of
    :math:`M - I/2`; the ``forward_laplacian`` backend uses the analytic
    right-hand side directly. This test pins the equality between the two
    paths *within* the ``_standard`` backend on a two-atom geometry, so a
    sign or factor-of-2 regression in the jacfwd path is caught
    independently of the end-to-end FL-vs-standard parity test. The
    multi-atom geometry subsumes the single-atom case for this identity.
    """
    electron = jnp.array([0.5, 0.1, -0.2])
    atoms = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    l2_values = jnp.array([0.03, 0.02])

    vec = evaluate_ph_first_order_vector(electron, atoms, l2_values)

    rel = np.asarray(electron) - np.asarray(atoms)
    r = np.linalg.norm(rel, axis=-1)
    expected = np.zeros(3)
    for rel_vec, dist, l2 in zip(rel, r, np.asarray(l2_values), strict=True):
        expected += l2 * 2.0 * rel_vec / dist

    np.testing.assert_allclose(vec, expected, atol=1e-8)


def test_first_order_vector_supports_traced_batched_electrons_under_vmap():
    """Protect first-order vector evaluation under traced per-electron PH tables."""
    electrons = jnp.array(
        [[0.1, 0.2, 0.3], [0.4, -0.2, 0.5], [0.3, 0.3, -0.1]],
        dtype=jnp.float32,
    )
    ph_atoms = jnp.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=jnp.float32)
    l2_values = jnp.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.5]], dtype=jnp.float32)

    expected = np.stack(
        [
            evaluate_ph_first_order_vector(electron, ph_atoms, l2_row)
            for electron, l2_row in zip(electrons, l2_values, strict=True)
        ]
    )
    batched = jax.vmap(evaluate_ph_first_order_vector, in_axes=(0, None, 0))(
        electrons, ph_atoms, l2_values
    )

    np.testing.assert_allclose(batched, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    "helper",
    [
        build_second_order_matrix,
        evaluate_ph_first_order_vector,
    ],
)
def test_operator_helpers_reject_direct_batched_electron_inputs(helper):
    """Keep public PH operator helpers focused on one formula instance."""
    electrons = jnp.array([[1.0, 0.0, 0.0]])
    atoms = jnp.array([[0.0, 0.0, 0.0]])
    radial_values = jnp.array([0.03])

    with pytest.raises(ValueError, match="electron must have shape \\(3,\\)"):
        helper(electrons, atoms, radial_values)
