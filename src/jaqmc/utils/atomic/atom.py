# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from jax import numpy as jnp

from . import elements
from .pp import PP_PH, core_electrons_by_pp


@dataclass(init=False)
class Atom:
    """A single atom in a molecular or periodic system.

    Args:
        symbol: Chemical element symbol (e.g. ``"H"``, ``"Fe"``).
        coords: 3D Cartesian coordinates in Bohr.
        charge: Effective charge seen by valence electrons. Resolved
            later from system pseudopotential context when omitted.
        pp: Optional pseudopotential applied to this atom. ``"ph"`` selects
            PH treatment; any other string is interpreted as a PySCF ECP name.
    """

    symbol: str
    coords: list[float]
    charge: int
    core_electrons: int
    pp: str | None

    def __init__(
        self,
        symbol: str,
        coords: list[float],
        charge: int | None = None,
        pp: str | None = None,
    ):
        self.symbol = symbol
        self.coords = coords
        self.pp = pp
        self.core_electrons = core_electrons_by_pp(symbol, pp)
        if charge is not None:
            if charge + self.core_electrons != self.atomic_number and pp == PP_PH:
                raise ValueError(
                    f"Atom {symbol} cannot use PH treatment with a custom charge, "
                    f"because that would break PH residual cancellation."
                )
            self.charge = charge
        else:
            self.charge = self.atomic_number - self.core_electrons

    @property
    def atomic_number(self) -> int:
        """Nuclear charge derived from ``symbol``."""
        return elements.from_symbol[self.symbol].atomic_number

    @property
    def unpaired_electron(self) -> int:
        """Default number of unpaired electrons for the neutral atom."""
        return elements.from_symbol[self.symbol].unpaired_electron

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
        This remains the chemical default when no initialization hint
        is specified.
        """
        electrons = self.charge
        unpaired = self.unpaired_electron
        return (electrons + unpaired) // 2, (electrons - unpaired) // 2
