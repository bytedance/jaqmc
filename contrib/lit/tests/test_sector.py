# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from jaqmc_contrib_lit.sector import (
    SourceSector,
    discover_source_sector,
    transform_molecule_data,
)
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData


def _assert_labeled_geometry_is_preserved(
    atoms: np.ndarray,
    charges: np.ndarray,
    sector: SourceSector,
) -> None:
    centered = atoms - np.asarray(sector.center)
    for operation_tuple in sector.operations:
        operation = np.asarray(operation_tuple)
        np.testing.assert_allclose(operation.T @ operation, np.eye(3), atol=1e-10)
        transformed = centered @ operation.T
        for charge in np.unique(charges):
            original_species = centered[charges == charge]
            transformed_species = transformed[charges == charge]
            original_distances = np.sort(
                np.linalg.norm(
                    original_species[:, None] - original_species[None],
                    axis=-1,
                ),
                axis=1,
            )
            cross_distances = np.min(
                np.linalg.norm(
                    transformed_species[:, None] - original_species[None],
                    axis=-1,
                ),
                axis=1,
            )
            assert np.max(cross_distances) < 1e-8
            assert original_distances.shape[0] == transformed_species.shape[0]


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def test_atom_uses_full_octahedral_subgroup_with_identity_first():
    atoms = np.asarray([[1.2, -0.4, 0.7]])
    charges = np.asarray([2.0])

    sector = discover_source_sector(atoms, charges)

    assert sector.label == "atom_Oh"
    assert sector.order == 48
    np.testing.assert_array_equal(np.asarray(sector.operations[0]), np.eye(3))
    assert any(
        np.array_equal(np.asarray(operation), -np.eye(3))
        for operation in sector.operations
    )
    _assert_labeled_geometry_is_preserved(atoms, charges, sector)


def test_homonuclear_and_heteronuclear_linear_subgroups_differ_by_axis_reversal():
    h2_atoms = np.asarray([[0.2, -0.1, -0.7], [0.2, -0.1, 0.9]])
    co_atoms = h2_atoms.copy()

    h2 = discover_source_sector(h2_atoms, np.asarray([1.0, 1.0]))
    co = discover_source_sector(co_atoms, np.asarray([6.0, 8.0]))

    assert h2.label == "linear_D4h"
    assert h2.order == 16
    assert co.label == "linear_C4v"
    assert co.order == 8
    _assert_labeled_geometry_is_preserved(h2_atoms, np.asarray([1.0, 1.0]), h2)
    _assert_labeled_geometry_is_preserved(co_atoms, np.asarray([6.0, 8.0]), co)


def test_water_discovers_all_four_c2v_operations():
    atoms = np.asarray([[0.0, 0.0, 0.0], [0.0, -0.757, 0.587], [0.0, 0.757, 0.587]])
    charges = np.asarray([8.0, 1.0, 1.0])

    sector = discover_source_sector(atoms, charges)

    assert sector.order == 4
    _assert_labeled_geometry_is_preserved(atoms, charges, sector)


def test_generic_noncoplanar_perturbation_falls_back_to_c1():
    atoms = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.1],
            [0.1, 1.3, 0.3],
            [0.2, 0.4, 1.7],
        ]
    )
    charges = np.asarray([1.0, 2.0, 3.0, 4.0])

    sector = discover_source_sector(atoms, charges)

    assert sector.label == "C1"
    assert sector.is_trivial
    np.testing.assert_array_equal(np.asarray(sector.operations[0]), np.eye(3))


def test_discovery_is_equivariant_under_rigid_rotation_and_translation():
    atoms = np.asarray([[0.0, 0.0, 0.0], [0.0, -0.757, 0.587], [0.0, 0.757, 0.587]])
    charges = np.asarray([8.0, 1.0, 1.0])
    rotation = _axis_angle_rotation(np.asarray([0.3, -0.7, 0.2]), 0.731)
    translation = np.asarray([1.4, -0.3, 0.9])
    moved_atoms = atoms @ rotation.T + translation

    original = discover_source_sector(atoms, charges)
    moved = discover_source_sector(moved_atoms, charges)

    assert moved.order == original.order == 4
    np.testing.assert_allclose(
        moved.center,
        np.mean(atoms, axis=0) @ rotation.T + translation,
        atol=1e-12,
    )
    for operation in original.operations:
        conjugated = rotation @ np.asarray(operation) @ rotation.T
        assert any(
            np.allclose(conjugated, np.asarray(candidate), atol=1e-9)
            for candidate in moved.operations
        )
    _assert_labeled_geometry_is_preserved(moved_atoms, charges, moved)


def test_transform_molecule_data_supports_unbatched_and_batched_electrons():
    operation = np.diag([-1.0, 1.0, -1.0])
    center = np.asarray([0.2, -0.3, 0.7])
    electrons = jnp.asarray([[0.4, 0.1, -0.2], [-0.8, 0.5, 1.1]])
    data = MoleculeData(
        electrons=electrons,
        atoms=jnp.asarray([[0.2, -0.3, 0.7]]),
        charges=jnp.asarray([2.0]),
    )

    transformed = transform_molecule_data(data, operation, center)
    expected = (np.asarray(electrons) - center) @ operation.T + center
    np.testing.assert_allclose(transformed.electrons, expected, atol=1e-7)
    assert transformed.atoms is data.atoms
    assert transformed.charges is data.charges

    batched = data.merge({"electrons": jnp.stack((electrons, electrons + 0.3))})
    transformed_batched = transform_molecule_data(batched, operation, center)
    expected_batched = (np.asarray(batched.electrons) - center) @ operation.T + center
    np.testing.assert_allclose(
        transformed_batched.electrons,
        expected_batched,
        atol=1e-7,
    )
