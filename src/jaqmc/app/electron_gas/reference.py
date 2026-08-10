# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Analytic free-electron orbitals used to bootstrap HEG training."""

import numpy as np
from jax import numpy as jnp

from jaqmc.utils.supercell import get_reciprocal_vectors


def _occupied_kpoints(
    count: int,
    lattice: np.ndarray,
    twist: tuple[float, float, float],
) -> jnp.ndarray:
    r"""Fill the ``count`` lowest-energy plane waves deterministically.

    Candidate momenta are :math:`(\mathbf{n}+\mathbf{t})\mathbf{B}`, where
    :math:`\mathbf{n}\in\mathbb{Z}^3`, ``t`` is the canonical fractional
    twist, and the rows of ``B`` are reciprocal lattice vectors. ``lattice`` is
    assumed to be the simple-cubic cell supplied by ``ElectronGasConfig``.

    Returns:
        Cartesian momenta ordered by increasing kinetic energy.

    Raises:
        ValueError: If ``count`` is negative.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative. Got {count}.")
    if count == 0:
        return jnp.empty((0, 3), dtype=jnp.asarray(lattice).dtype)

    canonical = (np.asarray(twist, dtype=float) + 0.5) % 1.0 - 0.5
    # This cube contains the lowest ``count`` momenta for any canonical twist,
    # while keeping the number of candidates linear in ``count``.
    radius = int(np.ceil(np.cbrt(count)))
    grid = range(-radius, radius + 1)
    integer_points = np.stack(
        np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    fractional_k = integer_points + canonical
    squared_norm = np.einsum("ij,ij->i", fractional_k, fractional_k)
    order = np.lexsort(
        (
            integer_points[:, 2],
            integer_points[:, 1],
            integer_points[:, 0],
            squared_norm,
        )
    )
    selected = order[:count]

    reciprocal = np.asarray(get_reciprocal_vectors(jnp.asarray(lattice)))
    return jnp.asarray(fractional_k[selected] @ reciprocal)


class FreeElectronReference:
    """Spin-separated plane-wave Slater reference for a finite HEG cell."""

    def __init__(
        self,
        nspins: tuple[int, int],
        lattice: np.ndarray,
        twist: tuple[float, float, float],
    ) -> None:
        self.nspins = nspins
        self.lattice = jnp.asarray(lattice)
        self.spin_kpoints = tuple(
            _occupied_kpoints(count, lattice, twist) for count in nspins
        )

    def get_orbital_kpoints(self) -> jnp.ndarray:
        """Return alpha then beta orbital momenta for the neural ansatz."""
        return jnp.concatenate(self.spin_kpoints, axis=0)

    def eval_orbitals(
        self, pos: jnp.ndarray, nspins: tuple[int, int]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate normalized spin-separated plane-wave orbital matrices.

        Returns:
            Alpha and beta orbital matrices.

        Raises:
            ValueError: If ``nspins`` differs from the configured system.
        """
        if nspins != self.nspins:
            raise ValueError(f"Expected nspins={self.nspins}. Got {nspins}.")

        volume = jnp.abs(jnp.linalg.det(self.lattice))
        normalization = volume**-0.5
        split_positions = jnp.split(pos, [nspins[0]], axis=-2)

        matrices = []
        for positions, kpoints, count in zip(
            split_positions, self.spin_kpoints, nspins, strict=True
        ):
            if count == 0:
                complex_dtype = jnp.result_type(pos.dtype, jnp.complex64)
                matrices.append(jnp.empty((*pos.shape[:-2], 0, 0), dtype=complex_dtype))
            else:
                phase = jnp.einsum("...id,jd->...ij", positions, kpoints)
                matrices.append(normalization * jnp.exp(1j * phase))
        return matrices[0], matrices[1]

    def eval_slater(self, pos: jnp.ndarray, nspins: tuple[int, int]) -> jnp.ndarray:
        """Evaluate the complex logarithm of the free-electron Slater product.

        Returns:
            Complex log wavefunction for each leading input batch element.
        """
        orbitals = self.eval_orbitals(pos, nspins)
        logpsi = jnp.zeros(
            pos.shape[:-2], dtype=jnp.result_type(pos.dtype, jnp.complex64)
        )
        for matrix, count in zip(orbitals, nspins, strict=True):
            if count:
                sign, logabs = jnp.linalg.slogdet(matrix)
                logpsi = logpsi + logabs + jnp.log(sign)
        return logpsi
