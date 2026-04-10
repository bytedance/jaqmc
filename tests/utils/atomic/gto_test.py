# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/google-deepmind/ferminet/blob/main/LICENSE.
#
# This modified file is released under the same license.

import jax
import numpy as np
import pyscf.gto
import pyscf.lib
import pyscf.pbc
import pytest

from jaqmc.utils.atomic.gto import AtomicOrbitalEvaluator, PBCAtomicOrbitalEvaluator


@pytest.fixture(autouse=True)
def no_temp_file():
    pyscf.lib.param.TMPDIR = None


def test_eval_gto():
    mol = pyscf.gto.M(atom="Na 0 0 -1; F 0 0 1", basis="def2-qzvp", unit="bohr")

    coords = np.random.default_rng(seed=42).uniform(-2, 2, size=(10, 3))
    aos_pyscf = mol.eval_gto("GTOval_sph", coords)

    eval_aos = AtomicOrbitalEvaluator.from_pyscf(mol)
    aos_jax = jax.jit(eval_aos)(coords)

    # Loose tolerances due to float32
    np.testing.assert_allclose(aos_pyscf, aos_jax, atol=4.0e-4, rtol=2.0e-4)


@pytest.mark.parametrize("twist", [np.zeros(3), np.array([0.5, 0.5, 0])])
def test_eval_gto_cell(twist):
    cell = pyscf.pbc.gto.Cell(atom="Li 0 0 0; H 1 1 1", a=np.eye(3) * 2, basis="ccpvdz")
    cell.build()
    coords = np.random.default_rng(seed=42).uniform(0, 2, size=(10, 3))
    scale = [2, 2, 2]
    twist_ks = np.dot(np.linalg.inv(cell.a), np.mod(twist, 1.0)) * 2 * np.pi
    kpts = cell.make_kpts(scale) + twist_ks

    aos_pyscf = cell.eval_gto("PBCGTOval_sph", coords, kpts=kpts)

    eval_aos = PBCAtomicOrbitalEvaluator.from_pyscf(cell)
    aos_jax = jax.jit(eval_aos)(coords, kpts=kpts)

    # Loose tolerances due to float32
    np.testing.assert_allclose(aos_jax, aos_pyscf, atol=4.0e-4, rtol=2.0e-4)
