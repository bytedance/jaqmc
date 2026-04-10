# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for SCF classes (MolecularSCF and PeriodicSCF)."""

import jax
import numpy as np
import pyscf.lib
import pytest
from jax import numpy as jnp

from jaqmc.utils.atomic.atom import Atom
from jaqmc.utils.atomic.scf import PeriodicSCF


@pytest.fixture(autouse=True)
def no_temp_file():
    pyscf.lib.param.TMPDIR = None


# k-point configurations: (name, fractional coordinates)
KPOINT_CONFIGS = {
    "gamma": np.array([[0.0, 0.0, 0.0]]),
    "kx_quarter": np.array([[0.25, 0.0, 0.0]]),
    "kxyz": np.array([[0.1, 0.2, 0.3]]),
}


@pytest.fixture(params=list(KPOINT_CONFIGS.values()), ids=list(KPOINT_CONFIGS.keys()))
def hydrogen_pbc_scf(request):
    atoms = [Atom("H", (0.0, 0.0, 0.0))]
    nspins = (1, 0)
    latvec = np.eye(3) * 5.0
    recip_vecs = 2 * np.pi * np.linalg.inv(latvec).T
    kpts = request.param @ recip_vecs

    pbc_scf = PeriodicSCF(
        atoms=atoms,
        nelectrons=nspins,
        basis="sto-3g",
        lattice_vectors=latvec,
        kpts=kpts,
        restricted=False,  # UHF for odd electron system
    )
    pbc_scf.run()
    return pbc_scf, nspins, latvec, kpts


def test_bloch_periodicity(hydrogen_pbc_scf):
    """Bloch theorem: eval_slater(r + R) = e^(ik·R) * eval_slater(r)."""
    pbc_scf, nspins, latvec, kpts = hydrogen_pbc_scf
    kpt = kpts[0]
    n_electrons = sum(nspins)

    rng = np.random.default_rng(seed=123)
    eval_slater = jax.vmap(lambda x: pbc_scf.eval_slater(x, nspins))
    pos = rng.uniform(0, 5, size=(5, 3))
    logdet1 = eval_slater(pos)

    for i in range(3):
        pos_shifted = pos + latvec[i]
        logdet2 = eval_slater(pos_shifted)

        # Expected phase: e^(i k·R) per electron -> e^(i n_e k·R) for determinant
        expected_phase = n_electrons * np.dot(kpt, latvec[i])
        phase_diff = (logdet2 - logdet1).imag

        # Phase difference should match (mod 2π)
        np.testing.assert_allclose(
            np.exp(1j * phase_diff),
            np.exp(1j * expected_phase),
            atol=1e-6,
            err_msg=f"Bloch phase violated for lattice vector {i}",
        )

        # Magnitude (real part of log) should be unchanged
        np.testing.assert_allclose(
            logdet1.real,
            logdet2.real,
            atol=1e-6,
            err_msg=f"Magnitude changed for lattice vector {i}",
        )


def test_density_matches_pyscf_get_rho(hydrogen_pbc_scf):
    """Compare density against PySCF reference (Gamma point only)."""
    import pyscf.pbc.dft

    pbc_scf, nspins, *_ = hydrogen_pbc_scf

    grids = pyscf.pbc.dft.gen_grid.UniformGrids(pbc_scf._cell)
    grids.build()
    coords = grids.coords
    rho_pyscf = pbc_scf.mean_field.get_rho(grids=grids)
    # Use logdet.real to get log|det| (phase is in imaginary part for complex matrices)
    eval_density = jax.vmap(lambda x: jnp.exp(2 * pbc_scf.eval_slater(x, nspins).real))
    rho_ours = eval_density(coords)
    np.testing.assert_allclose(rho_ours, rho_pyscf, atol=1e-5, rtol=1e-4)


def test_get_orbital_kpoints_h2_supercell():
    """Test that get_orbital_kpoints returns each k-point twice for H2 2x1x1 cell.

    For H2 with a 2x1x1 supercell:
    - 2 electrons per primitive cell (1 up, 1 down)
    - 2 k-points from the supercell folding
    - Each k-point contributes 1 alpha and 1 beta orbital
    - Result should have shape (4, 3) with k-points as [k0, k1, k0, k1]
    """
    from jaqmc.utils.supercell import get_reciprocal_vectors, get_supercell_kpts

    # H2 molecule in a box
    atoms = [
        Atom("H", (0.0, 0.0, 0.0)),
        Atom("H", (1.4, 0.0, 0.0)),
    ]
    nspins = (1, 1)  # 1 up, 1 down per primitive cell
    latvec = np.diag([10.0, 10.0, 10.0])

    # 2x1x1 supercell gives 2 k-points
    supercell_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    recip_vecs = get_reciprocal_vectors(jnp.asarray(latvec))
    kpts = np.asarray(get_supercell_kpts(jnp.asarray(supercell_matrix), recip_vecs))

    pbc_scf = PeriodicSCF(
        atoms=atoms,
        nelectrons=nspins,
        basis="sto-3g",
        lattice_vectors=latvec,
        kpts=kpts,
        restricted=True,
    )
    pbc_scf.run()

    orbital_kpts = pbc_scf.get_orbital_kpoints()

    # Should have 4 orbitals total: 2 alpha + 2 beta
    assert orbital_kpts.shape == (4, 3), (
        f"Expected shape (4, 3), got {orbital_kpts.shape}"
    )

    # First 2 rows should be alpha orbitals (k0, k1)
    # Last 2 rows should be beta orbitals (k0, k1)
    np.testing.assert_allclose(
        orbital_kpts[:2], kpts, atol=1e-10, err_msg="Alpha k-points mismatch"
    )
    np.testing.assert_allclose(
        orbital_kpts[2:], kpts, atol=1e-10, err_msg="Beta k-points mismatch"
    )
