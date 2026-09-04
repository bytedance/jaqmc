# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: DOC201,DOC501

r"""Host-side discovery of clamped-nuclei spatial symmetry operations.

A :class:`SourceSector` stores only immutable Python tuples, so the discovered
geometry can safely be captured by JAX-compiled parity checks without
introducing NumPy work into the compiled path.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import permutations, product

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc_contrib_lit.common import (
    _ATOM_HARD_EVEN_SECTOR_LABEL,
    _ATOM_HARD_ODD_SECTOR_LABEL,
    _ATOM_PARITY_PENDING_SECTOR_LABEL,
)

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)

type OrthogonalOperation = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True)
class SourceSector:
    """A finite molecular symmetry subgroup for a dipole-vector response.

    Attributes:
        center: Fixed point about which spatial operations act.
        operations: Orthogonal 3-by-3 matrices.  The identity is always first.
        label: Human-readable description of the discovered subgroup.

    The tuple representation is intentional: instances are immutable, hashable,
    and safe to read from a JAX closure.  Convert one operation with
    ``jnp.asarray(sector.operations[index])`` inside numerical code.
    """

    center: tuple[float, float, float]
    operations: tuple[OrthogonalOperation, ...]
    label: str = "C1"

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            msg = f"center must be a finite length-3 vector, got shape {center.shape}."
            raise ValueError(msg)
        operations = _canonicalize_operations(self.operations, tolerance=1e-10)
        if not operations:
            msg = "A source sector must contain at least the identity operation."
            raise ValueError(msg)
        object.__setattr__(self, "center", _vector_tuple(center))
        object.__setattr__(self, "operations", operations)

    @property
    def order(self) -> int:
        """Number of operations in the finite subgroup."""
        return len(self.operations)

    @property
    def is_trivial(self) -> bool:
        """Whether the discovered subgroup contains only the identity."""
        return self.order == 1


def discover_source_sector(
    atoms: np.ndarray | jnp.ndarray,
    charges: np.ndarray | jnp.ndarray,
    *,
    tolerance: float = 1e-5,
    axial_order: int = 4,
) -> SourceSector:
    """Discover a finite source-vector symmetry subgroup from nuclear geometry.

    The returned operations map every nucleus to a nucleus with the same charge.
    General finite geometries are handled by enumerating the possible images of
    a linearly independent nuclear frame and validating each candidate against
    the complete labeled point set.  Continuous-symmetry cases use inexpensive
    finite subgroups:

    * an atom (or a rank-zero point set) uses the 48-element full octahedral
      group, including inversion;
    * a linear molecule uses a finite axial ``C_nv`` subgroup of order
      ``2 * axial_order`` and includes axis reversal when the labeled nuclear
      geometry permits it;
    * a generic molecule uses its discovered finite point group, with ``C1`` as
      the guaranteed fallback.

    Args:
        atoms: Nuclear coordinates with shape ``(n_atoms, 3)``.
        charges: Nuclear charges/species labels with shape ``(n_atoms,)``.
        tolerance: Relative geometric matching tolerance.
        axial_order: Number of sampled rotations around a linear molecular axis.
    """
    atoms_array = np.asarray(atoms, dtype=np.float64)
    charges_array = np.asarray(charges, dtype=np.float64)
    if atoms_array.ndim != 2 or atoms_array.shape[1:] != (3,):
        msg = f"atoms must have shape (n_atoms, 3), got {atoms_array.shape}."
        raise ValueError(msg)
    if atoms_array.shape[0] == 0:
        raise ValueError("At least one atom is required to discover a source sector.")
    if charges_array.shape != (atoms_array.shape[0],):
        msg = (
            "charges must have shape (n_atoms,), got "
            f"{charges_array.shape} for {atoms_array.shape[0]} atoms."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(atoms_array)) or not np.all(np.isfinite(charges_array)):
        raise ValueError("atoms and charges must contain only finite values.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance!r}.")
    if int(axial_order) != axial_order or int(axial_order) < 2:
        raise ValueError(f"axial_order must be an integer >= 2, got {axial_order!r}.")

    # Every species-preserving permutation preserves the unweighted centroid.
    # It is therefore a valid symmetry center without relying on atomic masses.
    center = np.mean(atoms_array, axis=0)
    centered = atoms_array - center
    scale = max(1.0, float(np.max(np.linalg.norm(centered, axis=1))))
    singular_values = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.sum(singular_values > float(tolerance) * scale))

    if rank == 0:
        operations = _octahedral_operations()
        label = "atom_Oh"
    elif rank == 1:
        operations, reversible = _linear_operations(
            centered,
            charges_array,
            tolerance=float(tolerance),
            axial_order=int(axial_order),
        )
        label = (
            f"linear_D{int(axial_order)}h"
            if reversible
            else f"linear_C{int(axial_order)}v"
        )
    else:
        operations = _discover_finite_operations(
            centered,
            charges_array,
            rank=rank,
            tolerance=float(tolerance),
        )
        label = "C1" if len(operations) == 1 else f"finite_O3_{len(operations)}"

    return SourceSector(
        center=_vector_tuple(center),
        operations=tuple(_matrix_tuple(operation) for operation in operations),
        label=label,
    )


def transform_molecule_data(
    data: MoleculeData,
    operation: Sequence[Sequence[float]] | np.ndarray | jnp.ndarray,
    center: Sequence[float] | np.ndarray | jnp.ndarray,
) -> MoleculeData:
    """Apply a nuclear symmetry operation to electronic coordinates.

    Coordinates use the column-vector convention ``r' = G (r - c) + c``.
    ``electrons`` may be unbatched ``(n_electrons, 3)`` or have arbitrary
    leading batch dimensions.  The fixed nuclear arrays are deliberately left
    unchanged: a valid operation permutes the labeled nuclear *set*, while this
    function evaluates the electronic wavefunction at ``gX`` in that same
    clamped-nuclei problem.
    """
    matrix = jnp.asarray(operation, dtype=data.electrons.dtype)
    center_array = jnp.asarray(center, dtype=data.electrons.dtype)
    transformed = (
        jnp.einsum(
            "ij,...j->...i",
            matrix,
            data.electrons - center_array,
            precision=jax.lax.Precision.HIGHEST,
        )
        + center_array
    )
    return data.merge({"electrons": transformed})


def _discover_finite_operations(
    centered: np.ndarray,
    charges: np.ndarray,
    *,
    rank: int,
    tolerance: float,
) -> tuple[np.ndarray, ...]:
    anchor_indices = _independent_anchor_indices(centered, rank, tolerance)
    source_frame = centered[np.asarray(anchor_indices)].T
    source_gram = source_frame.T @ source_frame
    candidate_indices = [
        tuple(
            int(index)
            for index in np.flatnonzero(
                np.isclose(charges, charges[anchor], rtol=0.0, atol=1e-8)
            )
        )
        for anchor in anchor_indices
    ]
    scale = max(1.0, float(np.max(np.linalg.norm(centered, axis=1))))
    gram_tolerance = 8.0 * tolerance * scale**2
    candidates: list[np.ndarray] = [np.eye(3)]

    for target_indices in product(*candidate_indices):
        if len(set(target_indices)) != rank:
            continue
        target_frame = centered[np.asarray(target_indices)].T
        if not np.allclose(
            target_frame.T @ target_frame,
            source_gram,
            rtol=8.0 * tolerance,
            atol=gram_tolerance,
        ):
            continue
        possible: tuple[np.ndarray, ...]
        if rank == 3:
            possible = (target_frame @ np.linalg.inv(source_frame),)
        else:
            source_normal = _unit(np.cross(source_frame[:, 0], source_frame[:, 1]))
            target_normal = _unit(np.cross(target_frame[:, 0], target_frame[:, 1]))
            plane_map = target_frame @ np.linalg.pinv(source_frame)
            possible = (
                plane_map + np.outer(target_normal, source_normal),
                plane_map - np.outer(target_normal, source_normal),
            )
        for raw_operation in possible:
            operation = _nearest_orthogonal(raw_operation)
            if _valid_operation(
                operation,
                centered,
                charges,
                tolerance=tolerance,
            ):
                candidates.append(operation)
    return tuple(
        np.asarray(operation)
        for operation in _canonicalize_operations(
            candidates,
            tolerance=20.0 * tolerance,
        )
    )


def _linear_operations(
    centered: np.ndarray,
    charges: np.ndarray,
    *,
    tolerance: float,
    axial_order: int,
) -> tuple[tuple[np.ndarray, ...], bool]:
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = _canonical_axis(vh[0])
    transverse_x = _perpendicular_unit(axis)
    transverse_y = _unit(np.cross(axis, transverse_x))
    basis = np.stack((transverse_x, transverse_y, axis), axis=1)
    local_operations: list[np.ndarray] = []
    for step in range(axial_order):
        angle = 2.0 * np.pi * float(step) / float(axial_order)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        local_operations.extend((rotation, rotation @ np.diag([1.0, -1.0, 1.0])))

    axial_operations = [basis @ operation @ basis.T for operation in local_operations]
    axis_reversal = basis @ np.diag([1.0, 1.0, -1.0]) @ basis.T
    reversible = _valid_operation(
        axis_reversal,
        centered,
        charges,
        tolerance=tolerance,
    )
    if reversible:
        axial_operations.extend(
            axis_reversal @ operation for operation in axial_operations.copy()
        )
    operations = _canonicalize_operations(axial_operations, tolerance=20.0 * tolerance)
    return tuple(np.asarray(operation) for operation in operations), reversible


def _octahedral_operations() -> tuple[np.ndarray, ...]:
    operations = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            operation = np.zeros((3, 3), dtype=np.float64)
            operation[np.arange(3), np.asarray(permutation)] = np.asarray(signs)
            operations.append(operation)
    return tuple(
        np.asarray(operation)
        for operation in _canonicalize_operations(operations, tolerance=1e-12)
    )


def _independent_anchor_indices(
    centered: np.ndarray,
    rank: int,
    tolerance: float,
) -> tuple[int, ...]:
    selected: list[int] = []
    current_rank = 0
    scale = max(1.0, float(np.max(np.linalg.norm(centered, axis=1))))
    threshold = tolerance * scale
    # Long vectors give better-conditioned frames and reproducible candidates.
    order = sorted(
        range(centered.shape[0]),
        key=lambda index: (-float(np.linalg.norm(centered[index])), index),
    )
    for index in order:
        proposed = [*selected, index]
        singular_values = np.linalg.svd(
            centered[np.asarray(proposed)],
            compute_uv=False,
        )
        proposed_rank = int(np.sum(singular_values > threshold))
        if proposed_rank > current_rank:
            selected.append(index)
            current_rank = proposed_rank
        if current_rank == rank:
            return tuple(selected)
    msg = f"Could not construct a rank-{rank} nuclear frame."
    raise ValueError(msg)


def _valid_operation(
    operation: np.ndarray,
    centered: np.ndarray,
    charges: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    scale = max(1.0, float(np.max(np.linalg.norm(centered, axis=1))))
    coordinate_tolerance = 20.0 * tolerance * scale
    if not np.allclose(
        operation.T @ operation,
        np.eye(3),
        rtol=20.0 * tolerance,
        atol=20.0 * tolerance,
    ):
        return False
    transformed = centered @ operation.T
    adjacency = (
        np.linalg.norm(transformed[:, None, :] - centered[None, :, :], axis=-1)
        <= coordinate_tolerance
    ) & np.isclose(charges[:, None], charges[None, :], rtol=0.0, atol=1e-8)
    return _has_perfect_matching(adjacency)


def _has_perfect_matching(adjacency: np.ndarray) -> bool:
    """Return whether a small boolean bipartite graph has a perfect matching."""
    target_to_source = np.full(adjacency.shape[1], -1, dtype=np.int64)

    def augment(source: int, visited: np.ndarray) -> bool:
        for target in np.flatnonzero(adjacency[source]):
            if visited[target]:
                continue
            visited[target] = True
            previous = int(target_to_source[target])
            if previous < 0 or augment(previous, visited):
                target_to_source[target] = source
                return True
        return False

    return all(
        augment(source, np.zeros(adjacency.shape[1], dtype=bool))
        for source in range(adjacency.shape[0])
    )


def _canonicalize_operations(
    operations: Sequence[Sequence[Sequence[float]] | np.ndarray],
    *,
    tolerance: float,
) -> tuple[OrthogonalOperation, ...]:
    unique: list[np.ndarray] = []
    for raw_operation in operations:
        operation = np.asarray(raw_operation, dtype=np.float64)
        if operation.shape != (3, 3) or not np.all(np.isfinite(operation)):
            msg = (
                "Every operation must be a finite 3-by-3 matrix, got "
                f"{operation.shape}."
            )
            raise ValueError(msg)
        if not np.allclose(
            operation.T @ operation,
            np.eye(3),
            rtol=max(tolerance, 1e-10),
            atol=max(tolerance, 1e-10),
        ):
            msg = "Every source-sector operation must be orthogonal."
            raise ValueError(msg)
        operation = _nearest_orthogonal(operation)
        if not any(np.max(np.abs(operation - known)) <= tolerance for known in unique):
            unique.append(operation)
    identity_index = next(
        (
            index
            for index, operation in enumerate(unique)
            if np.allclose(
                operation,
                np.eye(3),
                atol=tolerance,
                rtol=0.0,
            )
        ),
        None,
    )
    if identity_index is None:
        unique.append(np.eye(3))
    else:
        unique[identity_index] = np.eye(3)

    decimals = max(0, min(14, int(np.ceil(-np.log10(max(tolerance, 1e-14)))) + 1))
    unique.sort(
        key=lambda operation: (
            0 if np.array_equal(operation, np.eye(3)) else 1,
            tuple(float(value) for value in np.round(operation.ravel(), decimals)),
        )
    )
    return tuple(_matrix_tuple(operation) for operation in unique)


def _nearest_orthogonal(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix)
    return left @ right


def _canonical_axis(vector: np.ndarray) -> np.ndarray:
    axis = _unit(vector)
    pivot = int(np.argmax(np.abs(axis)))
    return axis if axis[pivot] >= 0.0 else -axis


def _perpendicular_unit(axis: np.ndarray) -> np.ndarray:
    reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
    return _unit(reference - np.dot(reference, axis) * axis)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a zero vector.")
    return np.asarray(vector, dtype=np.float64) / norm


def _vector_tuple(vector: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.asarray(vector))  # type: ignore[return-value]


def _matrix_tuple(matrix: np.ndarray) -> OrthogonalOperation:
    rows = tuple(tuple(float(value) for value in row) for row in np.asarray(matrix))
    return rows  # type: ignore[return-value]


def _is_identity_operation(operation, *, tolerance: float = 1e-10) -> bool:
    return bool(
        np.allclose(
            np.asarray(operation),
            np.eye(3),
            rtol=0.0,
            atol=tolerance,
        )
    )


def _is_atom_parity_sector(sector: SourceSector) -> bool:
    """Return whether a sector denotes pending or resolved atomic parity."""
    if (
        sector.label
        not in {
            _ATOM_PARITY_PENDING_SECTOR_LABEL,
            _ATOM_HARD_ODD_SECTOR_LABEL,
            _ATOM_HARD_EVEN_SECTOR_LABEL,
        }
        or sector.order != 2
    ):
        return False
    return any(
        np.allclose(
            np.asarray(operation),
            -np.eye(3),
            rtol=0.0,
            atol=1e-10,
        )
        for operation in sector.operations
    )


def _is_atom_hard_parity_sector(sector: SourceSector) -> bool:
    """Return whether an atomic sector has a diagnosed response parity."""
    return _is_atom_parity_sector(sector) and sector.label in {
        _ATOM_HARD_ODD_SECTOR_LABEL,
        _ATOM_HARD_EVEN_SECTOR_LABEL,
    }


def _response_parity_character(sector: SourceSector) -> int:
    """Return +1 for hard even, -1 for hard odd, or zero if unresolved/C1."""
    if sector.label == _ATOM_HARD_EVEN_SECTOR_LABEL:
        return 1
    if sector.label == _ATOM_HARD_ODD_SECTOR_LABEL:
        return -1
    return 0


def _resolve_atomic_parity_sector(
    sector: SourceSector,
    response_parity: int,
) -> SourceSector:
    """Attach one diagnosed response parity to a pending atomic sector.

    Returns:
        The resolved hard-parity atomic sector, or the unchanged C1 sector.

    Raises:
        ValueError: If the supplied character is invalid for the sector.
    """
    if sector.label == _ATOM_PARITY_PENDING_SECTOR_LABEL:
        if response_parity not in (-1, 1):
            msg = (
                "A pending atomic sector requires response parity +1 or -1, "
                f"got {response_parity!r}."
            )
            raise ValueError(msg)
        label = (
            _ATOM_HARD_EVEN_SECTOR_LABEL
            if response_parity == 1
            else _ATOM_HARD_ODD_SECTOR_LABEL
        )
        return replace(sector, label=label)
    if _is_atom_hard_parity_sector(sector):
        expected = _response_parity_character(sector)
        if response_parity != expected:
            msg = (
                f"Resolved atomic sector requires parity {expected:+d}, got "
                f"{response_parity!r}."
            )
            raise ValueError(msg)
        return sector
    if response_parity != 0:
        msg = f"C1 sector cannot resolve atomic parity {response_parity!r}."
        raise ValueError(msg)
    return sector


def _response_symmetry_policy(sector: SourceSector, response_parity: int) -> str:
    """Return the persisted runtime policy name for one resolved geometry.

    Raises:
        ValueError: If the supplied character is invalid for the sector.
    """
    if sector.label == _ATOM_PARITY_PENDING_SECTOR_LABEL:
        if response_parity != 0:
            msg = "Pending atomic parity cannot have a resolved character."
            raise ValueError(msg)
        return "atom_hard_auto"
    if _is_atom_hard_parity_sector(sector):
        expected = _response_parity_character(sector)
        if response_parity != expected:
            msg = (
                f"Atomic response policy requires parity {expected:+d}, got "
                f"{response_parity!r}."
            )
            raise ValueError(msg)
        return "atom_hard_even" if response_parity == 1 else "atom_hard_odd"
    if response_parity != 0:
        msg = f"C1 response policy cannot use parity {response_parity!r}."
        raise ValueError(msg)
    return "molecule_c1"


def _project_source_center_to_invariant_subspace(
    source_center,
    source_sector: SourceSector | None,
    *,
    electron_count: int,
    tolerance: float = 1e-10,
) -> np.ndarray:
    r"""Project an affine dipole center into the sector's invariant subspace.

    For a spatial operation about ``c``, the electronic dipole transforms as
    ``D(gX) = g (D(X) + N_e c) - N_e c``. Consequently ``D-D0`` is a
    Cartesian vector precisely when ``q = D0 + N_e c`` is fixed by every
    configured operation. The joint nullspace of the stacked ``g-I``
    constraints gives the closest symmetry-consistent center.

    Returns:
        A float64 Cartesian center. A missing or trivial sector is returned
        unchanged so callers of the previous private estimator API retain C1
        behavior.

    Raises:
        ValueError: If ``source_center`` is not a Cartesian vector.
    """
    center = np.asarray(source_center, dtype=np.float64)
    if center.shape != (3,):
        msg = f"source_center must have shape (3,), got {center.shape}."
        raise ValueError(msg)
    if source_sector is None or source_sector.is_trivial:
        return np.array(center, copy=True)

    symmetry_center = np.asarray(source_sector.center, dtype=np.float64)
    q = center + int(electron_count) * symmetry_center
    constraints = np.concatenate(
        [
            np.asarray(operation, dtype=np.float64) - np.eye(3)
            for operation in source_sector.operations
            if not _is_identity_operation(operation)
        ],
        axis=0,
    )
    _, singular_values, right_vectors = np.linalg.svd(
        constraints,
        full_matrices=True,
    )
    largest = float(singular_values[0]) if singular_values.size else 0.0
    numerical_cutoff = (
        64.0 * max(constraints.shape) * np.finfo(np.float64).eps * max(1.0, largest)
    )
    cutoff = max(numerical_cutoff, float(tolerance) * max(1.0, largest))
    rank = int(np.sum(singular_values > cutoff))
    null_basis = right_vectors[rank:].T
    if null_basis.shape[1] == 0:
        projected_q = np.zeros(3, dtype=np.float64)
    else:
        projected_q = null_basis @ (null_basis.T @ q)
    return projected_q - int(electron_count) * symmetry_center


def _log_source_center_projection(
    unprojected_center,
    projected_center,
    source_sector: SourceSector | None,
) -> None:
    """Log the Cartesian correction made by a nontrivial source sector."""
    if source_sector is None or source_sector.is_trivial:
        return
    correction = np.asarray(projected_center) - np.asarray(unprojected_center)
    logger.info(
        "LIT source-center projection sector=%s correction="
        "(%.8e, %.8e, %.8e) correction_norm=%.8e",
        source_sector.label,
        float(correction[0]),
        float(correction[1]),
        float(correction[2]),
        float(np.linalg.norm(correction)),
    )
