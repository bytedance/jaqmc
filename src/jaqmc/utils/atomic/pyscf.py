# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared PySCF-to-JaQMC atomic conversion policies."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pyscf import gto

from .atom import Atom
from .pp import PP_PH, SUPPORTED_PH_ELEMENTS

__all__ = [
    "PySCFPseudopotentials",
    "automatic_double_zeta_basis",
    "occupied_orbital_masks",
    "resolve_pyscf_basis",
    "translate_pp_to_pyscf",
]


@dataclass(frozen=True)
class PySCFPseudopotentials:
    """Pseudopotential maps accepted by PySCF molecular and periodic cells."""

    ecp: dict[str, str]
    pseudo: dict[str, str]


_PH_SURROGATE_ECP = {symbol: "ccecp" for symbol in SUPPORTED_PH_ELEMENTS}


def automatic_double_zeta_basis(atoms: Sequence[Atom]) -> dict[str, str]:
    """Choose the automatic PySCF double-zeta basis for each element.

    All-electron atoms use ``cc-pVDZ``. Atoms using ``ccecp`` or molecular
    PH treatment use ``ccecpccpvdz`` because PH is represented by its
    ``ccecp`` surrogate in PySCF.

    Returns:
        A per-element PySCF basis-name mapping.

    Raises:
        ValueError: If an atom uses an ECP family without an automatic basis
            policy.
    """
    basis: dict[str, str] = {}
    for atom in atoms:
        if atom.pp is None:
            basis[atom.symbol] = "cc-pVDZ"
        elif atom.pp in {"ccecp", PP_PH}:
            basis[atom.symbol] = "ccecpccpvdz"
        else:
            raise ValueError(
                f"Automatic double-zeta basis selection does not define a "
                f"basis for ECP {atom.pp!r} on element {atom.symbol}; provide "
                "an explicit solver.basis or a prepared reference."
            )
    return basis


def resolve_pyscf_basis(
    atoms: Sequence[Atom], basis: str | Mapping[str, str] | None
) -> str | dict[str, str]:
    """Resolve the PySCF basis, overlaying a mapping onto the automatic policy.

    A string is used as-is for every element. ``None`` or an empty mapping
    uses :func:`automatic_double_zeta_basis` for every element. A non-empty
    mapping overrides selected elements and fills unspecified elements from
    that automatic policy.

    Returns:
        A PySCF basis name or per-element mapping.

    Raises:
        ValueError: If the mapping names unused elements, or an unspecified
            element uses an ECP family without an automatic basis.
    """
    if isinstance(basis, str):
        return basis
    specified = {} if not basis else dict(basis)
    symbols = {atom.symbol for atom in atoms}
    if unused := specified.keys() - symbols:
        raise ValueError(
            f"solver.basis is specified for elements {unused} but not used."
        )
    unspecified = [atom for atom in atoms if atom.symbol not in specified]
    automatic = automatic_double_zeta_basis(unspecified) if unspecified else {}
    return automatic | specified


def occupied_orbital_masks(
    occupations: np.ndarray,
    *,
    unrestricted: bool,
    nelec: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate PySCF occupations and return alpha and beta occupied masks.

    Restricted occupations encode an empty, singly alpha-occupied, or doubly
    occupied spatial orbital as 0, 1, or 2. Unrestricted occupations encode
    each spin channel separately as 0 or 1. The returned masks preserve every
    non-spin dimension of ``occupations``.

    When ``nelec`` is given as the system's ``(n_alpha, n_beta)`` electron
    counts, the occupied counts at every leading (e.g. k-point) index must
    equal it.

    Returns:
        Boolean masks for occupied alpha and beta orbital columns.

    Raises:
        ValueError: If any occupation is fractional, or if ``nelec`` is given
            and the occupied counts do not match it.
    """
    allowed = (0.0, 1.0) if unrestricted else (0.0, 1.0, 2.0)
    if not np.all(
        np.any(np.isclose(occupations[..., None], allowed, atol=1e-8), axis=-1)
    ):
        raise ValueError(
            "PySCF occupations are fractional and cannot define a QMC "
            "reference determinant."
        )
    if unrestricted:
        alpha_masks, beta_masks = occupations[0] > 0.9, occupations[1] > 0.9
    else:
        alpha_masks, beta_masks = occupations > 0.9, occupations > 1.1
    if nelec is not None and (
        np.any(alpha_masks.sum(axis=-1) != nelec[0])
        or np.any(beta_masks.sum(axis=-1) != nelec[1])
    ):
        raise ValueError(
            "PySCF occupied orbital counts do not match the electron count "
            f"nelec={nelec}."
        )
    return alpha_masks, beta_masks


def _pyscf_pp_core_electrons(atom: Atom, name: str) -> tuple[bool, int]:
    """Return whether ``name`` is an ECP and its PySCF core-electron count.

    Raises:
        ValueError: If PySCF cannot load ``name`` for ``atom``.
    """
    try:
        ecp = gto.basis.load_ecp(name, atom.symbol)
    except Exception:
        ecp = None
    if isinstance(ecp, tuple | list) and ecp:
        return True, int(ecp[0])

    try:
        pseudo = gto.basis.load_pseudo(name, atom.symbol)
        valence_electrons = sum(pseudo[0])
    except Exception as exc:
        raise ValueError(
            f"Pseudopotential {name!r} is not a supported PySCF ECP or GTH "
            f"pseudo for element {atom.symbol}."
        ) from exc
    return False, atom.atomic_number - valence_electrons


def translate_pp_to_pyscf(
    atoms: Sequence[Atom], solver_pp: str | Mapping[str, str]
) -> PySCFPseudopotentials:
    """Translate JaQMC atom treatments and PySCF overrides into PySCF maps.

    A string ``solver_pp`` applies one PySCF pseudopotential to every atom. A
    mapping overrides selected elements; omitted elements retain their JaQMC
    treatment, with PH translated to its PySCF surrogate ECP. Explicit
    ``solver_pp`` values are interpreted solely as PySCF ECP or GTH pseudo
    names.

    Returns:
        Pseudopotential maps for PySCF's ``ecp`` and ``pseudo`` inputs.

    Raises:
        ValueError: If a mapping ``solver_pp`` contains unused elements, a
            pseudopotential cannot be resolved by PySCF, or an override
            changes an atom's valence-electron count.
    """
    symbols = {atom.symbol for atom in atoms}
    overrides = (
        {atom.symbol: solver_pp for atom in atoms}
        if isinstance(solver_pp, str)
        else solver_pp
    )
    if unused := overrides.keys() - symbols:
        raise ValueError(f"solver.pp is specified for elements {unused} but not used.")

    ecp: dict[str, str] = {}
    pseudo: dict[str, str] = {}
    for atom in atoms:
        if atom.symbol not in overrides:
            if atom.pp is not None:
                ecp[atom.symbol] = (
                    _PH_SURROGATE_ECP[atom.symbol] if atom.pp == PP_PH else atom.pp
                )
            continue

        name = overrides[atom.symbol]
        is_ecp, core_electrons = _pyscf_pp_core_electrons(atom, name)
        if core_electrons != atom.core_electrons:
            raise ValueError(
                f"solver.pp {name!r} for element {atom.symbol} removes "
                f"{core_electrons} core electrons, but system.pp removes "
                f"{atom.core_electrons}. Reference solver pseudopotentials "
                "must match the valence count used by the JaQMC system."
            )
        (ecp if is_ecp else pseudo)[atom.symbol] = name

    return PySCFPseudopotentials(ecp=ecp, pseudo=pseudo)
