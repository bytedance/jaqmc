# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Mapping
from typing import Any

from pyscf import gto

from jaqmc.utils.atomic.pp import PP_PH
from jaqmc.utils.config import configurable_dataclass

from .atom import Atom


@configurable_dataclass
class AtomInitialization:
    """Initialization parameters for atoms.

    Args:
        local_s_z: Optional local spin along the z direction.
            Namely, ``(n_alpha - n_beta) / 2``.
        local_charge: Optional local charge around the atom.
            Positive values cause fewer electrons to be initialized near the atom.
    """

    local_s_z: float | None = None
    local_charge: int = 0

    def __post_init__(self):
        if self.local_s_z is not None and not math.isclose(
            2 * self.local_s_z, round(2 * self.local_s_z)
        ):
            raise ValueError(
                "local_s_z must be a half integer when provided. "
                f"Got {self.local_s_z!r}."
            )
        if not isinstance(self.local_charge, int):
            raise ValueError(
                "local_charge must be an integer since it is "
                "used for initialization of electrons."
            )

    @property
    def spin_imbalance(self) -> int | None:
        """Returns ``n_alpha - n_beta``."""
        return round(2 * self.local_s_z) if self.local_s_z is not None else None


@configurable_dataclass
class AtomicSystemConfig:
    """Shared configuration for atomic systems.

    Args:
        total_charge: Net system charge. The simulated electron count is
            ``sum(resolved_atom_charge) - total_charge`` after pseudopotential treatment
            is resolved.
        s_z: Total spin along the z direction for the explicitly simulated electrons.
            Namely, ``(n_alpha - n_beta) / 2``.
        pp: Pseudopotential specification. Can be None (no ECP nor PH),
            a string (same PP for all atoms, e.g., "ccecp", "ph"), or a dict
            mapping element symbols to pseudopotential names
            (e.g., {"Fe": "ccecp", "Cu": "ph"}).
            Elements not in the dict use all-electron treatment.
        electron_init_width: Width of the Gaussian distribution for
            initializing electron positions.
    """

    pp: str | dict[str, str] | None = None
    total_charge: int = 0
    s_z: float = 0
    electron_init_width: float = 1.0

    def __post_init__(self) -> None:
        if len(self.atoms) != len(self.per_atom_init):
            raise ValueError(
                "Resolved atoms and initialization hints must have the same length."
            )

        if isinstance(self.pp, Mapping) and (
            unused_pp := ({s for s in self.pp} - {a.symbol for a in self.atoms})
        ):
            raise ValueError(
                f"Pseudopotential is specified for elements {unused_pp} but not used."
            )

        for atom, init in zip(self.atoms, self.per_atom_init):
            local_electron_count = atom.charge + init.local_charge
            if local_electron_count < 0:
                raise ValueError(
                    f"Atom-local initialization would place {local_electron_count} "
                    f"electrons near atom {atom.symbol!r}. The local electron count "
                    "must be non-negative."
                )
            spin_imbalance = init.spin_imbalance
            if spin_imbalance is not None and (
                abs(spin_imbalance) > local_electron_count
                or (local_electron_count + spin_imbalance) % 2 != 0
            ):
                raise ValueError(
                    f"Atom-local initialization requests s_z={init.local_s_z} near "
                    f"atom {atom.symbol!r}, but that is not possible with "
                    f"{local_electron_count} electrons near that atom."
                )

        total_electron_count = (
            sum(atom.charge for atom in self.atoms) - self.total_charge
        )
        if total_electron_count < 0:
            raise ValueError(
                "Atomic system configuration must satisfy total_electron_count >= 0; "
                f"got total_electron_count={total_electron_count}, "
                f"total_charge={self.total_charge}."
            )
        if not math.isclose(2 * self.s_z, round(2 * self.s_z)):
            raise ValueError(f"s_z must be a half integer. Got {self.s_z!r}.")
        if (
            abs(2 * self.s_z) > total_electron_count
            or round(total_electron_count + 2 * self.s_z) % 2 != 0
        ):
            raise ValueError(
                f"Impossible s_z={self.s_z} for {total_electron_count} electrons."
            )

    @property
    def atoms(self) -> list[Atom]:
        raise NotImplementedError()

    @property
    def per_atom_init(self) -> list[AtomInitialization]:
        raise NotImplementedError()

    @property
    def spin_imbalance(self) -> int:
        """Return the total spin imbalance ``n_alpha - n_beta``."""
        return round(2 * self.s_z)

    @property
    def electron_spins(self) -> tuple[int, int]:
        """Return the resolved system-wide ``(n_alpha, n_beta)`` electron counts."""
        nelec = sum(atom.charge for atom in self.atoms) - self.total_charge
        return (nelec + self.spin_imbalance) // 2, (nelec - self.spin_imbalance) // 2

    @property
    def ecp_coefficients(self) -> dict[str, Any]:
        """Return PySCF ECP coefficients for non-PH pseudopotential atoms."""
        return {
            atom.symbol: gto.basis.load_ecp(atom.pp, atom.symbol)
            for atom in self.atoms
            if atom.pp is not None and atom.pp != PP_PH
        }

    @property
    def ph_elements(self) -> set[str]:
        """Return the set of element symbols treated with PH pseudopotentials."""
        return {atom.symbol for atom in self.atoms if atom.pp == PP_PH}
