# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import jax
import pytest

from jaqmc.app.molecule.config import AtomConfig, MoleculeConfig
from jaqmc.utils.atomic import Atom, AtomInitialization, distribute_spins
from jaqmc.utils.atomic.initialization import _atom_initial_spin_config


def _resolved_atom_with_init(
    symbol: str,
    *,
    charge: int | None = None,
    pp: str | None = None,
    local_s_z: float | None = None,
    local_charge: int = 0,
) -> tuple[Atom, AtomInitialization]:
    return (
        Atom(symbol, [0.0, 0.0, 0.0], charge=charge, pp=pp),
        AtomInitialization(local_s_z=local_s_z, local_charge=local_charge),
    )


def _split_atoms_and_inits(
    entries: list[tuple[Atom, AtomInitialization]],
) -> tuple[list[Atom], list[AtomInitialization]]:
    atoms, inits = zip(*entries, strict=True)
    return list(atoms), list(inits)


def _atom_config(
    symbol: str,
    *,
    charge: int | None = None,
    local_s_z: float | None = None,
    local_charge: int = 0,
) -> AtomConfig:
    return AtomConfig(
        symbol=symbol,
        coords=[0.0, 0.0, 0.0],
        charge=charge,
        initialization=AtomInitialization(
            local_s_z=local_s_z,
            local_charge=local_charge,
        ),
    )


def test_distribute_spins_rebalances_unspecified_atoms_to_match_total():
    atoms, per_atom_init = _split_atoms_and_inits(
        [_resolved_atom_with_init("H"), _resolved_atom_with_init("H")]
    )

    spins_per_atom = distribute_spins(
        jax.random.PRNGKey(0), atoms, per_atom_init, (0, 2)
    )

    assert spins_per_atom == [(0, 1), (0, 1)]


def test_distribute_spins_can_increase_alpha_when_needed():
    atoms, per_atom_init = _split_atoms_and_inits(
        [
            _resolved_atom_with_init("H", local_s_z=-0.5),
            _resolved_atom_with_init("Li"),
        ]
    )

    spins_per_atom = distribute_spins(
        jax.random.PRNGKey(0), atoms, per_atom_init, (3, 1)
    )

    assert spins_per_atom == [(0, 1), (3, 0)]


def test_distribute_spins_preserves_explicit_overrides():
    atoms, per_atom_init = _split_atoms_and_inits(
        [
            _resolved_atom_with_init("H", local_s_z=0.5),
            _resolved_atom_with_init("Li"),
        ]
    )

    spins_per_atom = distribute_spins(
        jax.random.PRNGKey(0), atoms, per_atom_init, (2, 2)
    )

    assert spins_per_atom == [(1, 0), (1, 2)]


def test_distribute_spins_returns_all_explicit_overrides_when_consistent():
    atoms, per_atom_init = _split_atoms_and_inits(
        [
            _resolved_atom_with_init("H", local_s_z=0.5),
            _resolved_atom_with_init("H", local_s_z=-0.5),
        ]
    )

    spins_per_atom = distribute_spins(
        jax.random.PRNGKey(0), atoms, per_atom_init, (1, 1)
    )

    assert spins_per_atom == [(1, 0), (0, 1)]


def test_distribute_spins_rejects_inconsistent_explicit_overrides():
    atoms, per_atom_init = _split_atoms_and_inits(
        [
            _resolved_atom_with_init("H", local_s_z=0.5),
            _resolved_atom_with_init("H", local_s_z=0.5),
        ]
    )

    with pytest.raises(ValueError, match="Expected total spins"):
        distribute_spins(jax.random.PRNGKey(0), atoms, per_atom_init, (1, 1))


def test_distribute_spins_rejects_total_electron_mismatch_without_overrides():
    atoms, per_atom_init = _split_atoms_and_inits([_resolved_atom_with_init("H")])

    with pytest.raises(
        ValueError,
        match="After applying the explicit initialization hints",
    ):
        distribute_spins(jax.random.PRNGKey(0), atoms, per_atom_init, (0, 0))


def test_distribute_spins_rejects_override_electron_mismatch():
    atoms, per_atom_init = _split_atoms_and_inits(
        [
            _resolved_atom_with_init("H", local_charge=1, local_s_z=0),
            _resolved_atom_with_init("H"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="per-atom charge offsets must sum to zero",
    ):
        distribute_spins(jax.random.PRNGKey(0), atoms, per_atom_init, (1, 1))


def test_initial_spin_config_tracks_pp_resolved_charge():
    all_electron = Atom("C", [0.0, 0.0, 0.0])
    ecp = Atom("C", [0.0, 0.0, 0.0], pp="ccecp")

    assert sum(all_electron.spin_config) == 6
    assert sum(ecp.spin_config) == 4


def test_atom_spin_config_uses_eager_all_electron_charge_resolution():
    atom = Atom("H", [0.0, 0.0, 0.0])

    assert atom.charge == 1
    assert atom.spin_config == (1, 0)


def test_atom_initialization_rejects_non_integer_local_charge():
    invalid_local_charge: Any = 0.5
    with pytest.raises(ValueError, match="local_charge must be an integer"):
        AtomInitialization(local_charge=invalid_local_charge)


def test_atom_with_initialization_rejects_spin_parity_mismatch():
    with pytest.raises(ValueError, match=r"Atom-local initialization requests s_z=0"):
        MoleculeConfig(atom_configs=[_atom_config("H", local_s_z=0)])


def test_atom_with_initialization_rejects_spin_magnitude_mismatch():
    with pytest.raises(ValueError, match=r"Atom-local initialization requests s_z=1.5"):
        MoleculeConfig(atom_configs=[_atom_config("H", local_s_z=1.5)])


def test_atom_with_initialization_rejects_negative_local_electron_count():
    with pytest.raises(ValueError, match="local electron count must be non-negative"):
        MoleculeConfig(atom_configs=[_atom_config("H", local_charge=-2)])


def test_atomic_system_rejects_spin_magnitude_mismatch():
    with pytest.raises(ValueError, match=r"Impossible s_z=1.5"):
        MoleculeConfig(
            atom_configs=[AtomConfig(symbol="H", coords=[0.0, 0.0, 0.0])],
            s_z=1.5,
        )


def test_atomic_system_rejects_negative_total_electron_count():
    with pytest.raises(ValueError, match="total_electron_count >= 0"):
        MoleculeConfig(
            atom_configs=[AtomConfig(symbol="H", coords=[0.0, 0.0, 0.0])],
            total_charge=2,
        )


def test_atomic_system_accepts_fully_polarized_one_electron_state():
    cfg = MoleculeConfig(
        atom_configs=[AtomConfig(symbol="H", coords=[0.0, 0.0, 0.0])],
        s_z=0.5,
    )

    assert cfg.electron_spins == (1, 0)


def test_atomic_system_accepts_zero_electron_state():
    cfg = MoleculeConfig(
        atom_configs=[AtomConfig(symbol="H", coords=[0.0, 0.0, 0.0])],
        total_charge=1,
        s_z=0,
    )

    assert cfg.electron_spins == (0, 0)


@pytest.mark.parametrize(
    ("atom_configs", "pp", "s_z", "expected_charges", "expected_spins"),
    [
        (
            [
                AtomConfig(
                    symbol="Cu",
                    coords=[0.0, 0.0, 0.0],
                    initialization=AtomInitialization(local_s_z=0.5),
                ),
                AtomConfig(
                    symbol="S",
                    coords=[0.0, 0.0, 0.0],
                    initialization=AtomInitialization(local_s_z=0),
                ),
            ],
            "ph",
            0.5,
            [19, 6],
            (13, 12),
        ),
        (
            [AtomConfig(symbol="Li", coords=[0.0, 0.0, 0.0])],
            "ccecp",
            0.5,
            [1],
            (1, 0),
        ),
    ],
)
def test_atomic_system_resolves_implicit_charge_from_system_pp(
    atom_configs, pp, s_z, expected_charges, expected_spins
):
    cfg = MoleculeConfig(
        atom_configs=atom_configs,
        pp=pp,
        s_z=s_z,
    )

    assert [atom.charge for atom in cfg.atoms] == expected_charges
    assert cfg.electron_spins == expected_spins


def test_atomic_system_rejects_explicit_charge_conflicting_with_system_pp():
    with pytest.raises(
        ValueError,
        match="cannot use PH treatment with a custom charge",
    ):
        MoleculeConfig(
            atom_configs=[_atom_config("Cu", charge=29, local_s_z=0.5)],
            pp="ph",
            s_z=0.5,
        )


def test_atom_initial_spin_config_rejects_negative_per_spin_occupancy():
    atom = Atom("H", [0.0, 0.0, 0.0], charge=1)
    initialization = AtomInitialization(local_charge=2)

    with pytest.raises(ValueError, match="non-negative per-spin occupancies"):
        _atom_initial_spin_config(atom, initialization)
