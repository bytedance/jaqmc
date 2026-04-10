# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jax import numpy as jnp

from jaqmc.utils.config import configurable_dataclass

from . import elements


@configurable_dataclass(kw_only=False)
class Atom:
    """A single atom in a molecular or periodic system.

    Args:
        symbol: Chemical element symbol (e.g. ``"H"``, ``"Fe"``).
        coords: 3D Cartesian coordinates in Bohr.
        atomic_number: Nuclear charge. Derived from ``symbol`` if not given.
        charge: Effective charge seen by valence electrons. Equals
            ``atomic_number`` by default; reduced when using ECPs.
    """

    symbol: str
    coords: list[float]
    atomic_number: int = -1
    charge: float = -1

    def __post_init__(self):
        if len(self.coords) != 3:
            raise ValueError(
                f"Expected three-dimensional atom. Got {len(self.coords)} in {self}."
            )
        if self.atomic_number == -1:
            self.atomic_number = elements.from_symbol[self.symbol].atomic_number
        if self.charge == -1:
            self.charge = self.atomic_number

    @property
    def coords_array(self) -> jnp.ndarray:
        """Returns the coordinates JAX array."""
        return jnp.array(self.coords)

    @property
    def spin_config(self) -> tuple[int, int]:
        """Per-atom spin configuration ``(n_alpha, n_beta)``.

        Derives electron count from ``charge`` rather than
        ``atomic_number`` so that ECP atoms (whose ``charge`` is set
        to the valence electron count) return the correct result.

        Note:
            For fractional charges, per-atom rounding may not preserve
            the system total (e.g., four atoms with ``charge=1.25``
            each round to 1, giving 4 instead of 5).
        """
        element = elements.from_symbol[self.symbol]
        electrons = round(self.charge)
        unpaired = element.unpaired_electron
        return (electrons + unpaired) // 2, (electrons - unpaired) // 2
