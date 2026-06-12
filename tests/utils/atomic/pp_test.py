# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from jaqmc.app.molecule.config.diatomic import diatomic_config
from jaqmc.utils.atomic import Atom
from jaqmc.utils.atomic.pp import (
    AtomPseudopotentialKind,
    ResolvedPseudopotentialConfig,
    resolve_atom_pp,
    resolve_pseudopotential_config,
)


def test_resolve_pseudopotential_config_mixes_ph_ecp_and_all_electron():
    """Protect integrated PH/ECP/all-electron routing and SCF surrogate ECP."""
    resolved = resolve_pseudopotential_config(
        atoms=[
            Atom("Fe", [0.0, 0.0, 0.0]),
            Atom("Li", [1.0, 0.0, 0.0]),
            Atom("H", [2.0, 0.0, 0.0]),
        ],
        pp={"Fe": "ph", "Li": "ccecp"},
    )

    assert isinstance(resolved, ResolvedPseudopotentialConfig)
    assert resolved.treatment_by_symbol == {
        "Fe": AtomPseudopotentialKind.ph,
        "Li": AtomPseudopotentialKind.ecp,
        "H": AtomPseudopotentialKind.all_electron,
    }
    assert resolved.core_electrons == {"Fe": 10, "Li": 2}
    assert resolved.scf_ecp == {"Fe": "ccecp", "Li": "ccecp"}
    assert resolved.runtime_ph_symbols == ("Fe",)
    assert resolved.runtime_ecp_symbols == ("Li",)


def test_resolve_pseudopotential_config_rejects_unsupported_ph_symbol():
    """Reject unsupported PH elements at the resolver boundary."""
    with pytest.raises(ValueError, match="unsupported PH element"):
        resolve_pseudopotential_config(
            atoms=[Atom("O", [0.0, 0.0, 0.0])],
            pp={"O": "ph"},
        )


def test_resolve_pseudopotential_config_rejects_unsupported_ph_even_when_unused():
    """Reject ``"ph"`` dict entries for unsupported symbols even when unused."""
    with pytest.raises(ValueError, match="unsupported PH element"):
        resolve_pseudopotential_config(
            atoms=[Atom("Fe", [0.0, 0.0, 0.0])],
            pp={"Fe": "ph", "Q": "ph"},
        )


def test_resolve_pseudopotential_config_rejects_string_ph_with_unsupported_atom():
    """``pp="ph"`` is invalid when any atom is not a supported PH element."""
    with pytest.raises(ValueError, match="not supported by PH"):
        resolve_pseudopotential_config(
            atoms=[Atom("Fe", [0.0, 0.0, 0.0]), Atom("H", [1.0, 0.0, 0.0])],
            pp="ph",
        )


def test_resolve_pseudopotential_config_supports_ph_only_without_ecp():
    """Ensure pure PH systems use SCF surrogate ECP without runtime ECP terms."""
    resolved = resolve_pseudopotential_config(
        atoms=[Atom("Fe", [0.0, 0.0, 0.0]), Atom("S", [1.0, 0.0, 0.0])],
        pp={"Fe": "ph", "S": "ph"},
    )

    assert resolved.scf_ecp == {"Fe": "ccecp", "S": "ccecp"}
    assert resolved.runtime_ecp_symbols == ()
    assert resolved.runtime_ph_symbols == ("Fe", "S")


def test_resolve_pseudopotential_config_accepts_string_ph_for_all_supported():
    """``pp="ph"`` resolves to PH for every (supported) atom."""
    resolved = resolve_pseudopotential_config(
        atoms=[Atom("Fe", [0.0, 0.0, 0.0]), Atom("S", [1.0, 0.0, 0.0])],
        pp="ph",
    )
    assert resolved.runtime_ph_symbols == ("Fe", "S")
    assert resolved.runtime_ecp_symbols == ()


def test_resolve_atom_pp_returns_none_for_symbol_outside_dict():
    """Per-atom lookup returns ``None`` when the symbol is missing from the dict."""
    assert resolve_atom_pp("H", pp={"Fe": "ph"}) is None
    assert resolve_atom_pp("H", pp=None) is None


def test_resolve_atom_pp_returns_string_value_for_all_symbols():
    """A string ``pp`` applies the same value to every symbol."""
    assert resolve_atom_pp("Li", pp="ccecp") == "ccecp"
    assert resolve_atom_pp("Fe", pp="ph") == "ph"


def test_resolve_atom_pp_resolves_dict_per_symbol():
    """Dict ``pp`` returns the per-symbol value when present, ``None`` otherwise."""
    assert resolve_atom_pp("Li", pp={"Li": "ccecp"}) == "ccecp"
    assert resolve_atom_pp("Cu", pp={"Fe": "ph", "Cu": "ph"}) == "ph"
    assert resolve_atom_pp("H", pp={"Li": "ccecp"}) is None


def test_diatomic_config_supports_ph_treatment():
    """Guard PH valence charge propagation in mixed molecule config factories."""
    cfg = diatomic_config(formula="FeH", basis="sto-3g", pp={"Fe": "ph"}, spin=1)

    assert cfg.pp == {"Fe": "ph"}
    assert cfg.atoms[0].charge == 16
    assert cfg.atoms[1].charge == 1
    assert sum(cfg.electron_spins) == 17
