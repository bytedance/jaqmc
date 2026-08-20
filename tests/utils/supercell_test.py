# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.utils.supercell import (
    fold_to_reciprocal_voronoi,
    get_primitive_kpts_for_supercell,
    get_reciprocal_vectors,
    get_supercell_copies,
    get_supercell_kpts_in_first_bz,
    supercell_fractional_kpts,
)


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    return values[np.lexsort(values.T[::-1])]


@pytest.mark.parametrize(
    ("supercell_matrix", "twist", "expected_fractional"),
    [
        # Diagonal supercell, Gamma-centered mesh.
        (
            [[2, 0], [0, 3]],
            None,
            [[x, y] for x in [0.0, 0.5] for y in [0.0, 1 / 3, 2 / 3]],
        ),
        # A twist shifts the mesh in supercell reciprocal fractional coordinates.
        (
            [[2, 0], [0, 3]],
            [0.25, 0.5],
            [[x, y] for x in [0.125, 0.625] for y in [1 / 6, 0.5, 5 / 6]],
        ),
        # Non-diagonal supercell: skewed enumeration, twist in supercell basis.
        (
            [[2, 1], [0, 2]],
            [0.25, 0.5],
            [[0.0, 0.25], [0.25, 0.75], [0.5, 0.25], [0.75, 0.75]],
        ),
    ],
    ids=["gamma_mesh", "twist_shift", "non_diagonal_with_twist"],
)
def test_get_primitive_kpts_for_supercell(supercell_matrix, twist, expected_fractional):
    reciprocal = 2 * jnp.pi * jnp.eye(2)

    kpts = get_primitive_kpts_for_supercell(
        jnp.asarray(supercell_matrix),
        reciprocal,
        None if twist is None else jnp.asarray(twist),
    )

    np.testing.assert_allclose(
        _sorted_rows(np.asarray(kpts)),
        _sorted_rows(2 * np.pi * np.asarray(expected_fractional)),
        atol=1e-6,
    )


def test_get_reciprocal_vectors_satisfies_orthogonality():
    lattice = jnp.asarray([[4.0, 0.5], [0.0, 3.0]])

    reciprocal = get_reciprocal_vectors(lattice)

    np.testing.assert_allclose(
        np.asarray(lattice @ reciprocal.T), 2 * np.pi * np.eye(2), atol=1e-6
    )


def test_supercell_fractional_kpts_enumerates_det_points_in_unit_cube():
    fractional = supercell_fractional_kpts(jnp.asarray([[2, 1], [0, 2]]))

    assert fractional.shape == (4, 2)
    assert bool(jnp.all(fractional >= 0.0))
    assert bool(jnp.all(fractional < 1.0))


def test_fold_to_reciprocal_voronoi_picks_nearest_image():
    reciprocal = 2 * jnp.pi * jnp.eye(2)
    kpts = jnp.asarray([[3 * jnp.pi, 0.0], [0.25 * jnp.pi, 1.5 * jnp.pi]])

    folded = fold_to_reciprocal_voronoi(kpts, reciprocal)

    np.testing.assert_allclose(
        np.asarray(folded),
        [[np.pi, 0.0], [0.25 * np.pi, -0.5 * np.pi]],
        atol=1e-6,
    )


def test_get_supercell_kpts_in_first_bz_folds_mesh_into_voronoi_cell():
    primitive = jnp.eye(2)

    kpts = get_supercell_kpts_in_first_bz(jnp.asarray([[3, 0], [0, 3]]), primitive)

    third = 2 * np.pi / 3
    expected = np.asarray(
        [[x, y] for x in [0.0, third, -third] for y in [0.0, third, -third]]
    )
    np.testing.assert_allclose(
        _sorted_rows(np.asarray(kpts)), _sorted_rows(expected), atol=1e-6
    )


def test_get_supercell_copies_tiles_supercell_with_primitive_translations():
    latvec = jnp.asarray([[2.0, 0.0], [0.0, 1.0]])

    copies = get_supercell_copies(latvec, jnp.asarray([[2, 0], [0, 3]]))

    expected = np.asarray([[2 * i, j] for i in [0, 1] for j in [0, 1, 2]], dtype=float)
    np.testing.assert_allclose(
        _sorted_rows(np.asarray(copies)), _sorted_rows(expected), atol=1e-6
    )
