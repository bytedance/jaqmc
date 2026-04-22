# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""ECP energy estimator for molecular QMC calculations.

Computes local (:math:`l=0`) and nonlocal (:math:`l>0`) effective core
potential contributions to the energy.

.. seealso:: :doc:`/guide/estimators/ecp` for background, formulas,
   and implementation notes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import LocalEstimator
from jaqmc.geometry.pbc import build_distance_fn
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import WavefunctionEvaluate

from .nonlocal_integral import make_nonlocal_integral
from .quadrature import get_quadrature


@configurable_dataclass
class ECPEnergy(LocalEstimator):
    r"""ECP energy estimator.

    Computes both local and nonlocal effective core potential contributions.
    Added automatically when ``ecp`` is set in the system configuration.

    - Local (:math:`l=0`): Direct potential energy from the :math:`l=0` channel
    - Nonlocal (:math:`l>0`): Angular integral weighted by :math:`V_l(r)`

    The estimator outputs ``energy:ecp`` which is included in the
    ``total_energy`` sum automatically.

    .. seealso:: :doc:`/guide/estimators/ecp` for the local/nonlocal
       decomposition and quadrature details.

    Args:
        max_core: Maximum number of nearest ECP atoms to consider per
            electron when evaluating nonlocal integrals. Only the
            closest ``max_core`` ECP atoms contribute per electron;
            the rest are skipped. Increase this if your system has
            many ECP atoms in close proximity.
        quadrature_id: Spherical quadrature rule used to evaluate
            nonlocal ECP integrals. When ``None``, a default rule
            is selected automatically.
        electrons_field: Name of electron position field in data.
        atoms_field: Name of atom position field in data.
        phase_logpsi: Wavefunction ratio function (runtime dep).
        ecp_coefficients: PySCF ECP dict (runtime dep).
        atom_symbols: List of element symbols, e.g. ``["Li", "H"]`` (runtime dep).
        lattice: Lattice vectors for PBC (runtime dep, optional).
        twist: Twist angle for PBC (runtime dep, optional).

    Raises:
        ValueError: If no atoms have ECP coefficients.
    """

    max_core: int = 2
    quadrature_id: str | None = None
    electrons_field: str = "electrons"
    atoms_field: str = "atoms"

    phase_logpsi: WavefunctionEvaluate = runtime_dep()
    ecp_coefficients: dict[str, Any] = runtime_dep()
    atom_symbols: list[str] = runtime_dep()
    lattice: jnp.ndarray | None = runtime_dep(default=None)
    twist: jnp.ndarray | None = runtime_dep(default=None)

    def init(self, data: Data, rngs: PRNGKey) -> None:
        """Compute derived state from config + runtime deps.

        Raises:
            ValueError: If no atoms have ECP coefficients.
        """
        # PySCF uses l=-1 for local, l=0,1,... for semi-local.
        # We map l → l+1 in from_pyscf, so num_channels = max_l + 2.
        num_channels = max(
            max(ang_l for ang_l, _ in self.ecp_coefficients[sym][1]) + 2
            for sym in self.ecp_coefficients
        )
        quadrature = get_quadrature(self.quadrature_id)

        self._ecp_atom_indices = [
            i for i, sym in enumerate(self.atom_symbols) if sym in self.ecp_coefficients
        ]
        if not self._ecp_atom_indices:
            raise ValueError(
                f"No atoms in {self.atom_symbols} have ECP coefficients. "
                f"Available ECP elements: {list(self.ecp_coefficients.keys())}"
            )
        self._ecp_radial = ECPRadial.from_pyscf(
            self.ecp_coefficients,
            self.atom_symbols,
            self._ecp_atom_indices,
            num_channels,
        )
        self._nonlocal_integral = make_nonlocal_integral(
            num_channels, quadrature, self.lattice, self.twist
        )
        self._distance_fn = (
            build_distance_fn(self.lattice) if self.lattice is not None else None
        )
        return None

    def evaluate_local(
        self,
        params: Params,
        data: Data,
        prev_local_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_local_stats

        electrons = data[self.electrons_field]
        atoms = data[self.atoms_field]

        ecp_atoms = atoms[jnp.array(self._ecp_atom_indices)]

        if self._distance_fn is not None:
            r_ea_vectors, r_ea_distances = self._distance_fn(electrons, ecp_atoms)
        else:
            r_ea_vectors = electrons[:, None, :] - ecp_atoms[None, :, :]
            r_ea_distances = jnp.linalg.norm(r_ea_vectors, axis=-1)

        radial_values = self._ecp_radial(r_ea_distances)
        local_energy = jnp.sum(radial_values[..., 0])

        radial_values, r_ea_vectors, _ = _select_nearest_cores(
            radial_values, r_ea_vectors, r_ea_distances, self.max_core
        )

        atom_positions = electrons[:, None, :] - r_ea_vectors

        integrals = self._nonlocal_integral(
            lambda x: self.phase_logpsi(params, data.merge({self.electrons_field: x})),
            electrons,
            atom_positions,
            rngs,
        )
        nonlocal_energy = jnp.sum(integrals * radial_values[..., 1:])

        return {"energy:ecp": local_energy + nonlocal_energy}, state


@dataclass(frozen=True)
class ECPRadial:
    r"""Preprocessed ECP radial potential in JAX-friendly format.

    All arrays have shape ``(n_ecp_atoms, num_channels, max_terms)`` where:

    - ``num_channels`` equals :math:`l_{\max} + 2`. Channel 0 is
      the local potential (PySCF ``l=-1``); channels 1+ are nonlocal
      (PySCF ``l=0, 1, \ldots``).
    - ``max_terms`` is the maximum number of terms :math:`k` in the sum across
      all atoms and channels. Unused terms are padded with zeros (which
      contribute zero to the sum).

    The radial potential formula is:

    .. math::

        V_l(r) = \sum_k \text{coeffs}_k \cdot r^{\text{powers}_k}
                 \cdot \exp(-\text{alphas}_k \cdot r^2)

    Args:
        alphas: Gaussian exponents :math:`\alpha_k`, indexed by :math:`k`.
        coeffs: Coefficients :math:`c_k`, indexed by :math:`k`.
        powers: Powers :math:`n_k - 2`, indexed by :math:`k` (using PySCF
            convention where ``power_idx`` corresponds to :math:`n_k`).
    """

    alphas: jnp.ndarray
    coeffs: jnp.ndarray
    powers: jnp.ndarray

    @classmethod
    def from_pyscf(
        cls,
        ecp_coefficients: dict[str, Any],
        atom_symbols: list[str],
        ecp_atom_indices: list[int],
        num_channels: int,
    ) -> Self:
        """Convert PySCF ECP format to internal padded format.

        PySCF ECP structure (``mol._ecp``)::

            {element: [n_core_electrons, [
                [l, [                         # channel for angular momentum l
                    [[alpha, coeff], ...],    # power_idx=0: r^(-2) terms
                    [[alpha, coeff], ...],    # power_idx=1: r^(-1) terms
                    [[alpha, coeff], ...],    # power_idx=2: r^0 terms
                    ...
                ]],
                ...  # more channels
            ]]}

        PySCF uses ``l=-1`` for the local channel and ``l=0, 1, 2, ...``
        for semi-local channels.  We map ``l → l+1`` so that channel 0 is
        always local and channels 1+ are nonlocal (s, p, d, ...).
        We flatten the ``power_idx`` and ``(alpha, coeff)`` levels into
        a single ``term_idx`` dimension.

        Args:
            ecp_coefficients: PySCF ECP dict (``mol._ecp``).
            atom_symbols: Element symbols for all atoms.
            ecp_atom_indices: Indices of atoms that have ECP.
            num_channels: Number of angular momentum channels.

        Returns:
            ECPRadial with padded arrays.
        """
        # Count max terms per channel across all atoms
        max_terms = 0
        for atom_idx in ecp_atom_indices:
            sym = atom_symbols[atom_idx]
            for _, radial_terms in ecp_coefficients[sym][1]:
                n_terms = sum(len(power_group) for power_group in radial_terms)
                max_terms = max(max_terms, n_terms)

        # Build padded arrays
        n_atoms = len(ecp_atom_indices)
        alphas = np.zeros((n_atoms, num_channels, max_terms))
        coeffs = np.zeros((n_atoms, num_channels, max_terms))
        powers = np.zeros((n_atoms, num_channels, max_terms))

        # PySCF groups terms by power_idx, then by (alpha, coeff) pairs.
        # We flatten these two levels into a single term_idx (the k index in Σ_k).
        #
        # PySCF uses l=-1 for the local channel and l=0,1,2,... for
        # semi-local channels.  We map l → channel_idx = l + 1 so that
        # channel 0 is always the local potential (used by evaluate_local
        # as `radial_values[..., 0]`) and channels 1+ are nonlocal.
        for i, atom_idx in enumerate(ecp_atom_indices):
            sym = atom_symbols[atom_idx]
            for pyscf_l, radial_terms in ecp_coefficients[sym][1]:
                channel_idx = pyscf_l + 1
                term_idx = 0
                for power_idx, power_group in enumerate(radial_terms):
                    for alpha, coeff in power_group:
                        alphas[i, channel_idx, term_idx] = alpha
                        coeffs[i, channel_idx, term_idx] = coeff
                        powers[i, channel_idx, term_idx] = power_idx - 2
                        term_idx += 1

        return cls(
            alphas=jnp.array(alphas),
            coeffs=jnp.array(coeffs),
            powers=jnp.array(powers),
        )

    def __call__(self, r_ea_distances: jnp.ndarray) -> jnp.ndarray:
        r"""Compute radial ECP potential :math:`V_l(r)` for all electron-atom pairs.

        Args:
            r_ea_distances: Electron-atom distances,
                shape (n_electrons, n_ecp_atoms).

        Returns:
            Radial potential values :math:`V_l(r)`,
            shape (n_electrons, n_ecp_atoms, num_channels).
        """
        r = r_ea_distances[:, :, None, None]
        terms = self.coeffs * (r**self.powers) * jnp.exp(-self.alphas * r**2)
        return jnp.sum(terms, axis=-1)


def _select_nearest_cores(
    radial_values: jnp.ndarray,
    r_ea_vectors: jnp.ndarray,
    r_ea_distances: jnp.ndarray,
    max_core: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Select the nearest ECP atoms for each electron.

    This is an optimization to limit the number of nonlocal integrals
    computed per electron.

    Args:
        radial_values: Radial potential values,
            shape (n_electrons, n_ecp_atoms, num_channels).
        r_ea_vectors: Electron-atom displacement vectors,
            shape (n_electrons, n_ecp_atoms, 3).
        r_ea_distances: Electron-atom distances,
            shape (n_electrons, n_ecp_atoms).
        max_core: Maximum number of nearest atoms to consider per electron.

    Returns:
        Tuple of selected (radial_values, r_ea_vectors, r_ea_distances),
        each truncated to at most max_core atoms per electron.
    """
    n_ecp_atoms = r_ea_distances.shape[1]
    if n_ecp_atoms <= max_core:
        return radial_values, r_ea_vectors, r_ea_distances

    # Sort by distance and select nearest
    sort_indices = jnp.argsort(r_ea_distances, axis=1)[:, :max_core]

    # Gather using advanced indexing
    n_elec = r_ea_distances.shape[0]
    elec_indices = jnp.arange(n_elec)[:, None]

    selected_distances = r_ea_distances[elec_indices, sort_indices]
    selected_vectors = r_ea_vectors[elec_indices, sort_indices, :]
    selected_values = radial_values[elec_indices, sort_indices, :]

    return selected_values, selected_vectors, selected_distances
