# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Factory helpers for system-layer atomic objects.

Sits above :mod:`jaqmc.utils.atomic.atom` and :mod:`jaqmc.utils.atomic.pp`.
:func:`make_atom` translates the user-facing unified ``pp`` spec used in
:class:`~jaqmc.utils.atomic.atomic_system.AtomicSystemConfig` into the
per-atom primitives consumed by :class:`~jaqmc.utils.atomic.atom.Atom`,
dispatching on whether the per-atom value selects all-electron, PH, or
ECP treatment.
"""

from __future__ import annotations

from collections.abc import Sequence

from .atom import Atom
from .pp import (
    PP_PH,
    SUPPORTED_PH_ELEMENTS,
    AtomPseudopotentialKind,
    get_valence_spin_config,
)

__all__ = ["electron_spins_from_total", "make_atom"]


def make_atom(
    symbol: str,
    coords: Sequence[float],
    *,
    pp: str | None = None,
) -> Atom:
    """Build an :class:`Atom` honoring its per-atom pseudopotential value.

    For all-electron atoms (``pp=None``) the returned atom uses the
    standard nuclear charge derived from ``symbol``. Otherwise ``charge``
    is set to the valence-electron count implied by ``pp``.

    Args:
        symbol: Element symbol (e.g., ``"H"``, ``"Fe"``).
        coords: 3D Cartesian coordinates in Bohr.
        pp: Per-atom pseudopotential value. ``None`` selects all-electron
            treatment. The reserved literal ``"ph"`` selects PH treatment
            (only valid for elements in
            :data:`~jaqmc.utils.atomic.pp.SUPPORTED_PH_ELEMENTS`). Any
            other string is interpreted as an ECP name (e.g.,
            ``"ccecp"``).

    Returns:
        A new :class:`Atom` with the appropriate ``charge``.

    Raises:
        ValueError: If ``pp == "ph"`` for a symbol not in
            :data:`~jaqmc.utils.atomic.pp.SUPPORTED_PH_ELEMENTS`.
    """
    coord_list = list(coords)
    if pp is None:
        return Atom(symbol=symbol, coords=coord_list)
    if pp == PP_PH:
        if symbol not in SUPPORTED_PH_ELEMENTS:
            raise ValueError(
                f"make_atom(symbol={symbol!r}): pp={PP_PH!r} requested but "
                f"{symbol!r} is not in SUPPORTED_PH_ELEMENTS "
                f"({sorted(SUPPORTED_PH_ELEMENTS)})."
            )
        valence = sum(
            get_valence_spin_config(symbol, pp_kind=AtomPseudopotentialKind.ph)
        )
    else:
        valence = sum(
            get_valence_spin_config(symbol, pp_kind=AtomPseudopotentialKind.ecp, ecp=pp)
        )
    return Atom(symbol=symbol, coords=coord_list, charge=valence)


def electron_spins_from_total(
    total_electrons: int,
    spin: int,
) -> tuple[int, int]:
    """Compute ``(n_alpha, n_beta)`` from total electron count and spin.

    Args:
        total_electrons: Total number of electrons in the system.
        spin: Total spin (number of unpaired electrons).

    Returns:
        Tuple ``(n_alpha, n_beta)``.

    Raises:
        ValueError: If ``total_electrons`` and ``spin`` have different
            parity.
    """
    if (total_electrons + spin) % 2 != 0:
        raise ValueError(
            f"Total electrons ({total_electrons}) and spin ({spin}) "
            f"must have the same parity."
        )
    return (total_electrons + spin) // 2, (total_electrons - spin) // 2
