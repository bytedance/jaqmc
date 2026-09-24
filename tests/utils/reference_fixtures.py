# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for prepared orbital-reference files."""

from pathlib import Path

import numpy as np
import pyscf.gto as pyscf_gto
import pyscf.scf as pyscf_scf
from jax import numpy as jnp

from jaqmc.app.molecule.reference import pyscf as reference_pyscf
from jaqmc.app.solid.config import SolidConfig
from jaqmc.app.solid.reference import PlaneWaveSolidReference
from jaqmc.utils.supercell import (
    get_primitive_kpts_for_supercell,
    get_reciprocal_vectors,
)


def write_molecule_pyscf_reference(
    path: Path,
    *,
    atom: str,
    spin: int,
    method: str = "RHF",
    basis: str = "sto-3g",
    unit: str = "Bohr",
) -> None:
    """Write a ``reference.npz`` converted from a short PySCF mean-field job."""
    mol = pyscf_gto.M(atom=atom, basis=basis, spin=spin, unit=unit, verbose=0)
    mean_field = getattr(pyscf_scf, method)(mol).set(verbose=0)
    chkfile = path.with_name(f"{path.stem}.chk")
    mean_field.chkfile = str(chkfile)
    mean_field.kernel()
    reference_pyscf.convert_checkpoint(chkfile, path)


def make_dummy_solid_reference(system: SolidConfig) -> PlaneWaveSolidReference:
    """Create a geometry-matching dummy solid reference for tests."""
    n_alpha = system.electron_spins[0] * system.scale
    n_beta = system.electron_spins[1] * system.scale
    kpoints = np.asarray(
        get_primitive_kpts_for_supercell(
            jnp.asarray(system.supercell_matrix),
            get_reciprocal_vectors(jnp.asarray(system.lattice_vectors)),
            jnp.asarray(system.twist),
        )
    )
    nk = len(kpoints)
    alpha_counts = np.zeros(nk, dtype=int)
    beta_counts = np.zeros(nk, dtype=int)
    alpha_counts[0] = n_alpha
    beta_counts[0] = n_beta
    return PlaneWaveSolidReference(
        symbols=tuple(atom.symbol for atom in system.atoms),
        atom_coords=np.asarray([atom.coords for atom in system.atoms], dtype=float),
        lattice=np.asarray(system.lattice_vectors, dtype=float),
        kpoints=kpoints,
        alpha_counts=alpha_counts,
        beta_counts=beta_counts,
        alpha_coeffs=np.ones((1, n_alpha), dtype=complex),
        beta_coeffs=np.ones((1, n_beta), dtype=complex),
        alpha_g_vectors=np.zeros((1, 3)),
        beta_g_vectors=np.zeros((1, 3)),
    )
