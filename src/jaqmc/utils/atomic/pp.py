# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence

from . import elements
from .atom import Atom

__all__ = [
    "PH_SURROGATE_ECP",
    "PP_PH",
    "SUPPORTED_PH_ELEMENTS",
    "AtomPseudopotentialKind",
    "ResolvedPseudopotentialConfig",
    "get_core_electrons",
    "get_ph_effective_charge",
    "get_ph_supported_elements",
    "get_valence_spin_config",
    "resolve_atom_pp",
    "resolve_atom_treatments",
    "resolve_pseudopotential_config",
]

#: Reserved value in the unified ``pp`` spec that selects PH treatment.
PP_PH = "ph"


class AtomPseudopotentialKind(enum.StrEnum):
    """Per-atom treatment kind for pseudopotential selection."""

    all_electron = "all_electron"
    ecp = "ecp"
    ph = "ph"


@dataclasses.dataclass(frozen=True)
class ResolvedPseudopotentialConfig:
    """Resolved per-symbol pseudopotential treatment for a system."""

    treatment_by_symbol: dict[str, AtomPseudopotentialKind]
    core_electrons: dict[str, int]
    scf_ecp: dict[str, str]
    runtime_ecp_symbols: tuple[str, ...]
    runtime_ph_symbols: tuple[str, ...]

    def uses_runtime_ecp(self) -> bool:
        return bool(self.runtime_ecp_symbols)

    def uses_runtime_ph(self) -> bool:
        return bool(self.runtime_ph_symbols)


# PH valence electron counts as defined by Bennett et al. (2022).
_PH_VALENCE_COUNTS: Mapping[str, int] = {
    "Cr": 14,
    "Mn": 15,
    "Fe": 16,
    "Co": 17,
    "Ni": 18,
    "Cu": 19,
    "Zn": 20,
    "S": 6,
}

SUPPORTED_PH_ELEMENTS: frozenset[str] = frozenset(_PH_VALENCE_COUNTS)

# ECP used to bootstrap the SCF pretrain for each PH-treated element.
PH_SURROGATE_ECP: Mapping[str, str] = {
    "Cr": "ccecp",
    "Mn": "ccecp",
    "Fe": "ccecp",
    "Co": "ccecp",
    "Ni": "ccecp",
    "Cu": "ccecp",
    "Zn": "ccecp",
    "S": "ccecp",
}


def get_ph_supported_elements() -> frozenset[str]:
    """Return the set of element symbols with PH parameterizations."""
    return SUPPORTED_PH_ELEMENTS


def get_ph_effective_charge(symbol: str) -> int:
    """Return the PH valence electron count for a supported element.

    Raises:
        ValueError: If ``symbol`` is not one of the supported PH elements.
    """
    try:
        return _PH_VALENCE_COUNTS[symbol]
    except KeyError as err:
        raise ValueError(f"unsupported PH element: {symbol}") from err


def get_valence_spin_config(
    symbol: str,
    *,
    pp_kind: AtomPseudopotentialKind,
    ecp: str | Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Return ``(n_alpha, n_beta)`` valence electrons for a pseudized atom.

    Operates on the *resolved* per-atom treatment (``pp_kind`` plus, for
    ECP, an explicit ``ecp`` name). User-facing code should usually go
    through :func:`make_atom` or :func:`resolve_pseudopotential_config`,
    which decode the unified ``pp`` spec into this resolved form.

    Args:
        symbol: Element symbol (e.g., ``"Fe"``).
        pp_kind: Pseudopotential kind. Must be ``ecp`` or ``ph``.
        ecp: ECP name (e.g., ``"ccecp"``) or per-element mapping.
            Required when ``pp_kind == ecp`` and must be ``None``
            otherwise.

    Returns:
        Tuple of (n_alpha, n_beta) valence electrons.

    Raises:
        ValueError: If ``pp_kind`` is not ``ecp`` or ``ph``, or if
            ``ecp`` is inconsistent with ``pp_kind``.
    """
    if pp_kind == AtomPseudopotentialKind.ecp:
        if ecp is None:
            raise ValueError(
                "get_valence_spin_config: pp_kind=AtomPseudopotentialKind.ecp "
                "requires an ecp name (e.g. ecp='ccecp'), but got ecp=None."
            )
        return _ecp_valence_spin_config(symbol, ecp)
    if pp_kind == AtomPseudopotentialKind.ph:
        if ecp is not None:
            raise ValueError(
                f"get_valence_spin_config: pp_kind=AtomPseudopotentialKind.ph "
                f"does not use an ecp name, but got ecp={ecp!r}."
            )
        valence = get_ph_effective_charge(symbol)
        unpaired = elements.from_symbol[symbol].unpaired_electron
        return (valence + unpaired) // 2, (valence - unpaired) // 2
    raise ValueError(
        f"get_valence_spin_config: pp_kind={pp_kind!r} is not supported; "
        "expected AtomPseudopotentialKind.ecp or AtomPseudopotentialKind.ph."
    )


def _ecp_valence_spin_config(
    symbol: str, ecp: str | Mapping[str, str]
) -> tuple[int, int]:
    import pyscf.gto

    mol = pyscf.gto.Mole(atom=[[symbol, [0.0, 0.0, 0.0]]], unit="bohr")
    mol.ecp = ecp
    mol.spin = elements.from_symbol[symbol].unpaired_electron
    mol.build()
    return mol.nelec


def resolve_atom_pp(
    symbol: str,
    pp: str | Mapping[str, str] | None,
) -> str | None:
    """Return the per-atom ``pp`` value implied by a molecule-level spec.

    Args:
        symbol: Element symbol of the atom being resolved.
        pp: Unified pseudopotential spec. ``None`` (all-electron), a
            string applied to every atom, or a per-element mapping.

    Returns:
        The per-atom ``pp`` value to feed to :func:`make_atom`. ``None``
        for all-electron, ``"ph"`` for PH treatment, or an ECP name.

    Notes:
        This helper does no element-vs-spec validation; it only routes
        the spec to the per-atom value. Element validity (e.g., that a
        PH-treated symbol is in :data:`SUPPORTED_PH_ELEMENTS`) is
        enforced in :func:`make_atom` and
        :func:`resolve_atom_treatments`.
    """
    if pp is None:
        return None
    if isinstance(pp, str):
        return pp
    return pp.get(symbol)


def _split_pp_value(pp_value: str | None) -> tuple[AtomPseudopotentialKind, str | None]:
    """Decode a per-atom ``pp`` value into ``(pp_kind, ecp_name)``.

    Returns:
        Tuple ``(pp_kind, ecp_name)`` describing the resolved treatment.
        ``ecp_name`` is ``None`` for PH and all-electron atoms.
    """
    if pp_value is None:
        return AtomPseudopotentialKind.all_electron, None
    if pp_value == PP_PH:
        return AtomPseudopotentialKind.ph, None
    return AtomPseudopotentialKind.ecp, pp_value


def get_core_electrons(
    atoms: Sequence[Atom],
    pp: str | Mapping[str, str] | None,
) -> dict[str, int]:
    """Get the number of core electrons removed for each ECP-treated element.

    Only ECP atoms contribute to the result. PH atoms are accounted for
    separately via :func:`resolve_pseudopotential_config`.

    Args:
        atoms: List of Atom objects.
        pp: Unified pseudopotential spec.

    Returns:
        Mapping from element symbol to number of core electrons removed.
    """
    if pp is None:
        return {}

    core: dict[str, int] = {}
    for symbol in {atom.symbol for atom in atoms}:
        pp_value = resolve_atom_pp(symbol, pp)
        pp_kind, ecp_name = _split_pp_value(pp_value)
        if pp_kind is not AtomPseudopotentialKind.ecp:
            continue
        assert ecp_name is not None
        total = elements.from_symbol[symbol].atomic_number
        valence = sum(
            get_valence_spin_config(
                symbol, pp_kind=AtomPseudopotentialKind.ecp, ecp=ecp_name
            )
        )
        if valence < total:
            core[symbol] = total - valence

    return core


def resolve_atom_treatments(
    atoms: Sequence[Atom],
    pp: str | Mapping[str, str] | None,
) -> list[AtomPseudopotentialKind]:
    """Resolve the configured treatment kind for each atom.

    Args:
        atoms: System atoms in geometry order.
        pp: Unified pseudopotential spec (see module docstring).

    Returns:
        A list of per-atom treatment kinds aligned with ``atoms``.

    Raises:
        ValueError: If any symbol selected for PH treatment is not in
            :data:`SUPPORTED_PH_ELEMENTS`. This includes symbols only
            present as dict keys (unused entries are still validated).
    """
    if isinstance(pp, Mapping):
        bad = sorted(
            symbol
            for symbol, value in pp.items()
            if value == PP_PH and symbol not in SUPPORTED_PH_ELEMENTS
        )
        if bad:
            raise ValueError("unsupported PH element(s): " + ", ".join(bad))
    elif pp == PP_PH:
        bad = sorted({a.symbol for a in atoms} - SUPPORTED_PH_ELEMENTS)
        if bad:
            raise ValueError(
                f"pp={PP_PH!r} requested but the following atom symbols are "
                "not supported by PH: " + ", ".join(bad)
            )

    treatments: list[AtomPseudopotentialKind] = []
    for atom in atoms:
        pp_kind, _ = _split_pp_value(resolve_atom_pp(atom.symbol, pp))
        treatments.append(pp_kind)
    return treatments


def resolve_pseudopotential_config(
    atoms: Sequence[Atom],
    pp: str | Mapping[str, str] | None,
) -> ResolvedPseudopotentialConfig:
    """Resolve the system-wide pseudopotential configuration.

    Args:
        atoms: System atoms in geometry order.
        pp: Unified pseudopotential spec (see module docstring).

    Returns:
        A :class:`ResolvedPseudopotentialConfig` with per-symbol
        treatment, derived core electron counts, the SCF surrogate ECP
        map, and the runtime ECP/PH symbol tuples.
    """
    treatments = resolve_atom_treatments(atoms, pp)
    symbols = tuple(dict.fromkeys(atom.symbol for atom in atoms))

    treatment_by_symbol: dict[str, AtomPseudopotentialKind] = {}
    for symbol in symbols:
        for atom, treatment in zip(atoms, treatments, strict=True):
            if atom.symbol == symbol:
                treatment_by_symbol[symbol] = treatment
                break

    runtime_ecp_symbols = tuple(
        symbol
        for symbol, treatment in treatment_by_symbol.items()
        if treatment is AtomPseudopotentialKind.ecp
    )
    runtime_ph_symbols = tuple(
        symbol
        for symbol, treatment in treatment_by_symbol.items()
        if treatment is AtomPseudopotentialKind.ph
    )

    return ResolvedPseudopotentialConfig(
        treatment_by_symbol=treatment_by_symbol,
        core_electrons=_build_core_electrons(atoms, pp),
        scf_ecp=_build_scf_ecp(symbols, treatment_by_symbol, pp),
        runtime_ecp_symbols=runtime_ecp_symbols,
        runtime_ph_symbols=runtime_ph_symbols,
    )


def _build_scf_ecp(
    symbols: Sequence[str],
    treatment_by_symbol: Mapping[str, AtomPseudopotentialKind],
    pp: str | Mapping[str, str] | None,
) -> dict[str, str]:
    scf_ecp: dict[str, str] = {}
    for symbol in symbols:
        treatment = treatment_by_symbol[symbol]
        if treatment is AtomPseudopotentialKind.ecp:
            pp_value = resolve_atom_pp(symbol, pp)
            assert pp_value is not None and pp_value != PP_PH
            scf_ecp[symbol] = pp_value
        elif treatment is AtomPseudopotentialKind.ph:
            scf_ecp[symbol] = PH_SURROGATE_ECP[symbol]
    return scf_ecp


def _build_core_electrons(
    atoms: Sequence[Atom],
    pp: str | Mapping[str, str] | None,
) -> dict[str, int]:
    core = get_core_electrons(atoms, pp)
    if pp is None:
        return core
    ph_symbols = {
        atom.symbol for atom in atoms if resolve_atom_pp(atom.symbol, pp) == PP_PH
    }
    for symbol in ph_symbols:
        total = elements.from_symbol[symbol].atomic_number
        valence = get_ph_effective_charge(symbol)
        if valence < total:
            core[symbol] = total - valence
    return core
