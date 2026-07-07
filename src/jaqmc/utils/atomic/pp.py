# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Pseudopotential vocabulary and helpers.

Owns the per-atom pseudopotential kind enum, the resolution of mixed
ECP / PH / all-electron systems, and helpers for valence-electron
counting. PH-specific *runtime* concerns (XML parameter files, the PH
operator implementation) live in :mod:`jaqmc.estimator.ph`.

The user-facing pseudopotential spec is a single ``pp`` argument:

- ``None``: all-electron for every atom.
- ``str``: same pseudopotential name applied to every atom. The reserved
  literal ``"ph"`` selects PH treatment (Bennett et al. 2022); any other
  string is interpreted as an ECP name (e.g. ``"ccecp"``) resolved by
  PySCF.
- ``Mapping[str, str]``: per-element mapping such as
  ``{"Fe": "ccecp", "Cu": "ph"}``. Elements not present in the mapping
  use all-electron treatment.

The reserved ``"ph"`` literal is a deliberate simplification while there
is exactly one bundled PH library. A future second PH library would
require a namespaced spelling (e.g. ``"ph:<name>"``); that migration is
deferred until the need actually arises.
"""

from pyscf import gto

__all__ = [
    "PH_SURROGATE_ECP",
    "PP_PH",
    "SUPPORTED_PH_ELEMENTS",
]

#: Reserved value in the unified ``pp`` spec that selects PH treatment.
PP_PH = "ph"


# PH valence electron counts as defined by Bennett et al. (2022).
_PH_NEON_CORE_ELEMENTS = frozenset(
    {"Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "P", "S", "Cl"}
)
_PH_CORE_COUNTS = {symbol: 10 for symbol in _PH_NEON_CORE_ELEMENTS}

SUPPORTED_PH_ELEMENTS: frozenset[str] = _PH_NEON_CORE_ELEMENTS

# ECP used to bootstrap the SCF pretrain for each PH-treated element.
PH_SURROGATE_ECP = {symbol: "ccecp" for symbol in _PH_NEON_CORE_ELEMENTS}


def core_electrons_by_pp(symbol: str, pp: str | None = None):
    """Return the number of core electrons implied by a pseudopotential.

    Args:
        symbol: Chemical element symbol.
        pp: Pseudopotential name. ``None`` means all-electron treatment.

    Returns:
        Number of electrons removed from explicit simulation by ``pp``.

    Raises:
        ValueError: If PH is requested for an unsupported element, or if PySCF
            does not provide the requested ECP for ``symbol``.
    """
    if pp is None:
        return 0
    if pp == PP_PH:
        if symbol not in _PH_CORE_COUNTS:
            raise ValueError(f"PH is not available for element {symbol}.")
        return _PH_CORE_COUNTS[symbol]
    unsupported_ecp = f"ECP {pp!r} is not supported for element {symbol}."
    try:
        ecp = gto.basis.load_ecp(pp, symbol)
    except Exception as exc:
        raise ValueError(unsupported_ecp) from exc
    if not isinstance(ecp, tuple | list) or not ecp:
        raise ValueError(unsupported_ecp)
    return ecp[0]
