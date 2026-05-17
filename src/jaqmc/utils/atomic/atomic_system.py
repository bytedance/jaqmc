# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jaqmc.utils.atomic.atom import Atom
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class AtomicSystemConfig:
    """Configuration for an atomic system.

    Args:
        atoms: List of atoms in the system.
        electron_spins: Tuple of two integers representing the number of
            explicitly simulated up and down electrons. In all-electron systems
            this is the full electron count; with ECPs it is the valence count
            after core electrons are replaced by the pseudopotential.
        ecp: Effective core potential specification. Can be None (no ECP),
            a string (same ECP for all atoms, e.g., "ccecp"), or a dict
            mapping element symbols to ECP names (e.g., {"Fe": "ccecp"}).
            Elements not in the dict use all-electron treatment.
        fixed_spins_per_atom: Optional list of fixed spin configurations
            per atom.
        electron_init_width: Width of the Gaussian distribution for
            initializing electron positions.
    """

    atoms: list[Atom]
    ecp: str | dict[str, str] | None = None
    electron_spins: tuple[int, int]
    fixed_spins_per_atom: list[tuple[int, int]] | None = None
    electron_init_width: float = 1.0

    def __post_init__(self) -> None:
        if len(self.electron_spins) != 2:
            raise ValueError(
                "Only support two channels of spins (up and down). Got electron_spins "
                f"with length {len(self.electron_spins)}: {self.electron_spins}."
            )
        if self.fixed_spins_per_atom is not None and not all(
            len(spins) == 2 for spins in self.fixed_spins_per_atom
        ):
            raise ValueError(
                f"Malformed fixed_spins_per_atom: {self.fixed_spins_per_atom}."
            )
