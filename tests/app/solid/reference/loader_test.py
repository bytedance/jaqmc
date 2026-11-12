# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from jaqmc.app.solid.reference import load_reference
from tests.app.solid.reference.helpers import _solid_system
from tests.utils.reference_fixtures import make_dummy_solid_reference


def test_load_validates_the_configured_solid(tmp_path, caplog):
    system = _solid_system(
        supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        twist=[0.25, 0.0, 0.0],
    )
    path = tmp_path / "reference.npz"
    base = make_dummy_solid_reference(system)
    reference = replace(
        base,
        symbols=("He1",),
        atom_coords=base.atom_coords + 5e-5,
    )
    reference.save(path)

    assert load_reference(path, system).nspins == tuple(
        value * system.scale for value in system.electron_spins
    )
    assert "Reference atoms" in caplog.text

    with pytest.raises(ValueError, match="k-points"):
        load_reference(
            path,
            _solid_system(
                supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
            ),
        )

    replace(
        base,
        atom_coords=np.asarray([[0.0, 0.0, 1.0]]),
    ).save(path)
    with pytest.raises(ValueError, match="geometry"):
        load_reference(path, system)


def test_load_rejects_same_scale_supercell_mismatch(tmp_path):
    prepared = _solid_system(supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    used = _solid_system(supercell_matrix=[[1, 0, 0], [0, 2, 0], [0, 0, 1]])
    assert prepared.scale == used.scale
    path = tmp_path / "reference.npz"
    replace(
        make_dummy_solid_reference(prepared),
        alpha_counts=np.asarray([1, 1]),
        beta_counts=np.asarray([1, 1]),
    ).save(path)

    with pytest.raises(ValueError, match="k-points"):
        load_reference(path, used)


def test_load_accepts_permuted_and_g_wrapped_kpoints(tmp_path):
    system = _solid_system(supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    reference = make_dummy_solid_reference(system)
    reciprocal = 2 * np.pi * np.linalg.inv(reference.lattice).T
    kpoints = np.concatenate(
        [reference.kpoints[1:], reference.kpoints[:1] + reciprocal[0]]
    )
    wrapped = replace(
        reference,
        kpoints=kpoints,
        alpha_counts=np.asarray([1, 1]),
        beta_counts=np.asarray([1, 1]),
    )
    path = tmp_path / "reference.npz"
    wrapped.save(path)

    load_reference(path, system)


def test_load_accepts_occupied_subset_and_ignores_empty_kpoints(tmp_path):
    system = _solid_system(
        supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        twist=[0.25, -0.1, 0.2],
    )
    full = make_dummy_solid_reference(system)
    selected = replace(
        full,
        kpoints=np.concatenate([full.kpoints[:1], [[0.123, 0.456, 0.789]]]),
        alpha_counts=np.asarray([full.nspins[0], 0]),
        beta_counts=np.asarray([full.nspins[1], 0]),
    )
    path = tmp_path / "reference.npz"
    selected.save(path)

    assert load_reference(path, system).nspins == full.nspins
