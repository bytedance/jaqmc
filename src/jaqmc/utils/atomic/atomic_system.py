# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jaqmc.utils.atomic.atom import Atom
from jaqmc.utils.atomic.pp import resolve_atom_treatments
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class AtomicSystemConfig:
    """Configuration for an atomic system.

    Args:
        atoms: List of atoms in the system.
        electron_spins: Tuple of two integers representing the
            number of up and down electrons.
        basis: The basis set for Hartree-Fock pretrain. Can be a string
            (e.g., "sto-3g", "ccecpccpvdz") or a dict mapping element
            symbols to basis names (e.g., {"Fe": "ccecpccpvdz", "O": "cc-pvdz"}).
        pp: Pseudopotential specification. Can be None (no ECP nor PH),
            a string (same PP for all atoms, e.g., "ccecp", "ph"), or a dict
            mapping element symbols to ECP names (e.g., {"Fe": "ccecp", "Cu": "ph"}).
            Elements not in the dict use all-electron treatment.
        fixed_spins_per_atom: Optional list of fixed spin configurations
            per atom.
        electron_init_width: Width of the Gaussian distribution for
            initializing electron positions.
    """

    atoms: list[Atom]
    basis: str | dict[str, str] = "sto-3g"
    pp: str | dict[str, str] | None = None
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
        resolve_atom_treatments(self.atoms, self.pp)
