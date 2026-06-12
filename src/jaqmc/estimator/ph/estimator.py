# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Pseudo-Hamiltonian energy estimator.

.. seealso:: :doc:`/guide/estimators/ph` for background, formulas, and
   implementation notes.
"""

from collections.abc import Mapping
from typing import Any, Literal

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.utils.atomic.pp import SUPPORTED_PH_ELEMENTS
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

from . import _forward_laplacian, _standard
from .data import load_ph_element_data

__all__ = ["PHEnergy"]


@configurable_dataclass
class PHEnergy(PerWalkerEstimator):
    r"""Pseudo-Hamiltonian energy correction estimator.

    Replaces the standalone kinetic estimator on systems with runtime PH
    support. Added automatically when any element in the system
    configuration is set to ``pp: ph``. Workflows should not add a separate
    kinetic estimator alongside :class:`PHEnergy`.

    The estimator outputs two keys:

    - ``energy:kinetic`` — the PH-modified kinetic contribution (combined
      derivative operator from Bennett et al. 2022). Replaces the
      Euclidean kinetic energy used on non-PH runs.
    - ``energy:ph`` — the short-range local PH residual, summed over PH
      atoms and added on top of the bare electron-nucleus Coulomb that
      ``potential_energy`` supplies for every atom.

    Both contributions are included in the ``total_energy`` sum
    automatically.

    .. seealso:: :doc:`/guide/estimators/ph` for the operator
       decomposition, definitions of the mass matrix and first-order
       vector, the additive composition with ``potential_energy``, and
       per-backend implementation notes.

    Args:
        f_log_psi: Log-wavefunction evaluate function.
        atom_symbols: Element symbols in geometry order.
        ph: Element symbols configured for PH treatment.
        kinetic_backend: PH derivative implementation. ``forward_laplacian``
            (default) uses a single ``folx.forward_laplacian`` pass over a
            Cholesky-shifted coordinate map; ``standard`` uses direct
            gradient/Hessian transforms with explicit einsum contractions
            and is kept as the math reference exercised by the parity
            test. Validated at ``init()`` time.
        electrons_field: Name of the electron-coordinate field in ``Data``.
        atoms_field: Name of the atom-coordinate field in ``Data``.
    """

    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()
    atom_symbols: list[str] = runtime_dep()
    ph: list[str] = runtime_dep()
    kinetic_backend: Literal["standard", "forward_laplacian"] = "forward_laplacian"
    electrons_field: str = "electrons"
    atoms_field: str = "atoms"

    def init(self, data: Data, rngs: PRNGKey) -> None:
        del rngs

        self._validate_backend()
        self._ph_atom_indices = self._resolve_ph_atom_indices()
        self._validate_data_shapes(data)

        loc_tables = []
        l2_tables = []
        for atom_index in self._ph_atom_indices:
            loc_data, l2_data = load_ph_element_data(self.atom_symbols[atom_index])
            loc_tables.append(jnp.asarray(loc_data))
            l2_tables.append(jnp.asarray(l2_data))

        self._loc_tables = jnp.stack(loc_tables, axis=0)
        self._l2_tables = jnp.stack(l2_tables, axis=0)
        self._radial_grid = jnp.linspace(0.0, 10.0, self._loc_tables.shape[-1])
        return None

    def _validate_backend(self) -> None:
        if self.kinetic_backend not in ("standard", "forward_laplacian"):
            raise ValueError(
                "unknown PH kinetic_backend: "
                f"{self.kinetic_backend!r}; expected 'standard' or "
                "'forward_laplacian'"
            )

    def _resolve_ph_atom_indices(self) -> list[int]:
        ph_symbols = frozenset(self.ph)
        if not ph_symbols:
            raise ValueError(
                "No PH element is requested while invoking PH calculations"
            )

        unsupported = sorted(ph_symbols - SUPPORTED_PH_ELEMENTS)
        if unsupported:
            raise ValueError("unsupported PH element(s): " + ", ".join(unsupported))
        indices = [
            index
            for index, symbol in enumerate(self.atom_symbols)
            if symbol in ph_symbols
        ]
        if ph_symbols and not indices:
            raise ValueError(
                "ph requested element(s) not present in atom_symbols: "
                f"ph={sorted(ph_symbols)} atom_symbols={list(self.atom_symbols)}"
            )
        return indices

    def _validate_data_shapes(self, data: Data) -> None:
        electrons = data[self.electrons_field]
        if electrons.ndim != 2 or electrons.shape[-1] != 3:
            raise ValueError(
                f"data[{self.electrons_field!r}] must have shape "
                f"(n_electrons, 3); got {electrons.shape}"
            )
        atoms = data[self.atoms_field]
        if atoms.ndim != 2 or atoms.shape[-1] != 3:
            raise ValueError(
                f"data[{self.atoms_field!r}] must have shape "
                f"(n_atoms, 3); got {atoms.shape}"
            )
        charges = data["charges"]
        if charges.ndim != 1 or charges.shape[0] != atoms.shape[0]:
            raise ValueError(
                'data["charges"] must have shape (n_atoms,) matching '
                f"data[{self.atoms_field!r}]; got {charges.shape} vs "
                f"{atoms.shape}"
            )

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs

        electrons = data[self.electrons_field]
        atoms = jnp.asarray(data[self.atoms_field])
        ph_atom_indices = jnp.asarray(self._ph_atom_indices)
        ph_atoms = atoms[ph_atom_indices]

        loc_values, l2_values = self._interpolate_radial_values(electrons, ph_atoms)
        derivative = self._evaluate_derivative_energy(params, data, ph_atoms, l2_values)
        zero_order = self._evaluate_zero_order_term(electrons, ph_atoms, loc_values)

        return {
            "energy:kinetic": derivative,
            "energy:ph": zero_order,
        }, state

    def _interpolate_radial_values(
        self, electrons: jnp.ndarray, ph_atoms: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        distances = jnp.linalg.norm(
            electrons[:, None, :] - ph_atoms[None, :, :], axis=-1
        )
        return (
            self._interp_tables(distances, self._loc_tables),
            self._interp_tables(distances, self._l2_tables),
        )

    def _interp_tables(
        self, distances: jnp.ndarray, tables: jnp.ndarray
    ) -> jnp.ndarray:
        if tables.shape[0] == 0:
            return jnp.zeros(distances.shape, dtype=distances.dtype)
        return jax.vmap(
            lambda r_atom, table: jnp.interp(r_atom, self._radial_grid, table),
            in_axes=(1, 0),
            out_axes=1,
        )(distances, tables)

    def _evaluate_derivative_energy(
        self,
        params: Params,
        data: Data,
        ph_atoms: jnp.ndarray,
        l2_values: jnp.ndarray,
    ) -> jnp.ndarray:
        # Dispatch to the selected backend. Both backends compute the same
        # Bennett et al. (2022) PH derivative term
        #     E_PH,i = -Tr(M_i H_i) - g_i^T M_i g_i + b_i^T g_i;
        # `_forward_laplacian` uses a Cholesky-shifted forward-Laplacian
        # pass (Fu et al. 2025), `_standard` differentiates the operator
        # definition directly via jax.grad/jax.hessian/jax.jacfwd.
        backend = (
            _forward_laplacian
            if self.kinetic_backend == "forward_laplacian"
            else _standard
        )
        return backend.compute_derivative_energy(
            self.f_log_psi,
            params,
            data,
            ph_atoms,
            l2_values,
            electrons_field=self.electrons_field,
        )

    def _evaluate_zero_order_term(
        self,
        electrons: jnp.ndarray,
        ph_atoms: jnp.ndarray,
        loc_values: jnp.ndarray,
    ) -> jnp.ndarray:
        # Short-range residual local channel:
        #     E_PH,0 = sum_e sum_a (tilde_v_loc(r_ea) + Z_a / r_ea).
        # The PH data layer stores the local table as
        #     loc_data(r) = r * tilde_v_loc(r) + Z_a
        # (see ``load_ph_element_data``'s docstring for the loader-side
        # contract), so the residual is recovered as ``loc_data(r) / r``
        # without re-introducing ``Z`` at evaluate time. The bare
        # ``-Z_a / r`` is supplied by ``potential_energy``, so the sum of
        # ``energy:potential`` and ``energy:ph`` recovers the paper form
        # ``sum_e sum_a tilde_v_loc(r_ea)`` of Bennett et al. (2022)
        # without any masking between the two estimators.
        rel = electrons[:, None, :] - ph_atoms[None, :, :]
        r = jnp.linalg.norm(rel, axis=-1)
        return jnp.sum(loc_values / r)
