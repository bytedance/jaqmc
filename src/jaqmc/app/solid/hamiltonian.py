# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.estimator import LocalEstimator
from jaqmc.estimator.ewald import EwaldSum
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

from .data import SolidData


@configurable_dataclass
class PotentialEnergy(LocalEstimator):
    """Potential energy estimator for solid systems using Ewald summation.

    Args:
        supercell_lattice: Lattice vectors for Ewald summation (runtime dep).
    """

    supercell_lattice: jnp.ndarray = runtime_dep()

    def init(self, data: SolidData, rngs: PRNGKey) -> None:
        self.ewald = EwaldSum(self.supercell_lattice)
        return None

    def evaluate_local(
        self,
        params: Params,
        data: SolidData,
        prev_local_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del params, rngs, prev_local_stats

        electrons = data.electrons.reshape(-1, 3)
        atoms = data.atoms.reshape(-1, 3)
        nelec = electrons.shape[0]

        # Concatenate all particles
        # Electrons have charge -1
        electron_charges = -jnp.ones(nelec)
        atom_charges = data.charges

        all_coords = jnp.concatenate([electrons, atoms], axis=0)
        all_charges = jnp.concatenate([electron_charges, atom_charges], axis=0)

        potential = self.ewald.energy(all_coords, all_charges)

        return {"energy:potential": potential}, state
