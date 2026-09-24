# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from jaqmc.app.solid.reference import load_reference
from tests.app.solid.reference.helpers import _solid_system
from tests.utils.reference_fixtures import make_dummy_solid_reference


def test_load_validates_the_configured_solid(tmp_path):
    system = _solid_system(
        supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        twist=[0.25, 0.0, 0.0],
    )
    path = tmp_path / "reference.npz"
    make_dummy_solid_reference(system).save(path)

    assert load_reference(path, system).nspins == tuple(
        value * system.scale for value in system.electron_spins
    )
    with pytest.raises(ValueError, match="k-points"):
        load_reference(
            path,
            _solid_system(
                supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
            ),
        )


def test_load_rejects_same_scale_supercell_mismatch(tmp_path):
    prepared = _solid_system(supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    used = _solid_system(supercell_matrix=[[1, 0, 0], [0, 2, 0], [0, 0, 1]])
    assert prepared.scale == used.scale
    path = tmp_path / "reference.npz"
    make_dummy_solid_reference(prepared).save(path)

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
        alpha_counts=reference.alpha_counts[::-1],
        beta_counts=reference.beta_counts[::-1],
    )
    path = tmp_path / "reference.npz"
    wrapped.save(path)

    load_reference(path, system)


def test_load_rejects_symmetry_reduced_kpoint_subset(tmp_path):
    system = _solid_system(supercell_matrix=[[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    full = make_dummy_solid_reference(system)
    reduced = replace(
        full,
        kpoints=full.kpoints[:1],
        alpha_counts=np.asarray([full.nspins[0]]),
        beta_counts=np.asarray([full.nspins[1]]),
    )
    path = tmp_path / "reference.npz"
    reduced.save(path)

    with pytest.raises(ValueError, match="k-points"):
        load_reference(path, system)
