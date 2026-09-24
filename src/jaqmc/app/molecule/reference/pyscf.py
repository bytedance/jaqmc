# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""PySCF preparation and conversion for molecular references."""

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyscf.gto
import pyscf.scf.chkfile
from upath import UPath

from jaqmc.app.molecule.config import MoleculeConfig, MoleculeSolverConfig
from jaqmc.utils.atomic.gto import AtomicOrbitalEvaluator
from jaqmc.utils.atomic.pyscf import (
    occupied_orbital_masks,
    resolve_pyscf_basis,
    translate_pp_to_pyscf,
)

from .model import MoleculeReference

logger = logging.getLogger(__name__)

__all__ = ["MoleculePySCFReferenceJob", "convert_checkpoint"]


def _solver_data(
    system: MoleculeConfig, solver: MoleculeSolverConfig
) -> dict[str, Any]:
    atoms = [
        {"symbol": atom.symbol, "coords": atom.coords, "charge": atom.charge}
        for atom in system.atoms
    ]
    for atom in system.atoms:
        if atom.charge != atom.atomic_number - atom.core_electrons:
            raise ValueError(
                "PySCF reference preparation does not support custom per-atom "
                f"charges ({atom.symbol} has charge={atom.charge})."
            )
    pseudopotentials = translate_pp_to_pyscf(system.atoms, solver.pp)
    return {
        "atoms": atoms,
        "total_charge": system.total_charge,
        "spin": system.spin_imbalance,
        "basis": resolve_pyscf_basis(system.atoms, solver.basis),
        "ecp": pseudopotentials.ecp,
        "pseudo": pseudopotentials.pseudo,
        "method": solver.method,
        "xc": solver.xc,
        "checkpoint": solver.checkpoint,
        "verbose": solver.verbose,
        "extra": solver.extra,
    }


def _script(data: dict[str, Any]) -> str:
    payload = json.dumps(data, indent=2, default=lambda value: value.tolist())
    return f'''"""Generated PySCF molecular reference job."""
import json
import pyscf.dft
import pyscf.gto
import pyscf.scf

config = json.loads(r"""
{payload}
""")
atoms = [[item["symbol"], item["coords"]] for item in config["atoms"]]
mol = pyscf.gto.Mole(
    atom=atoms,
    unit="Bohr",
    basis=config["basis"],
    ecp=config["ecp"] or None,
    pseudo=config["pseudo"] or None,
    charge=config["total_charge"],
    spin=config["spin"],
    cart=False,
    verbose=config["verbose"],
)
mol.build()
method = config["method"]
if method in ("RHF", "UHF"):
    mf = getattr(pyscf.scf, method)(mol)
else:
    mf = getattr(pyscf.dft, method)(mol)
    mf.xc = config["xc"]
mf.chkfile = config["checkpoint"]
for key, value in config["extra"].items():
    if not hasattr(mf, key):
        raise ValueError(f"Unknown PySCF mean-field setting: {{key}}")
    setattr(mf, key, value)
mf.kernel()
if not mf.converged:
    raise RuntimeError("PySCF did not converge.")
'''


@dataclass(frozen=True)
class MoleculePySCFReferenceJob:
    """PySCF reference job for a molecular system and solver configuration."""

    system: MoleculeConfig
    solver: MoleculeSolverConfig

    def prepare(self, job_dir: Path) -> None:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input.py").write_text(
            _script(_solver_data(self.system, self.solver)),
            encoding="utf-8",
        )

    @property
    def command(self) -> list[str]:
        return [sys.executable, "input.py"]

    @property
    def output_path(self) -> Path:
        return Path(self.solver.checkpoint)

    def run(self, job_dir: Path, *, wait: bool = True) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(self.command, cwd=job_dir)
        if wait and (returncode := process.wait()):
            raise RuntimeError(f"PySCF job failed with exit status {returncode}.")
        return process

    def finalize_npz(self, job_dir: Path, reference_path: UPath) -> None:
        convert_checkpoint(job_dir / self.output_path, reference_path)


def _occupied_coefficients(
    coeff: Any, occ: Any, nelec: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    coeff_array = np.asarray(coeff)
    occ_array = np.asarray(occ)
    unrestricted = isinstance(coeff, (tuple, list)) or (
        coeff_array.ndim == 3 and coeff_array.shape[0] == 2
    )
    if unrestricted:
        if coeff_array.shape[0] != 2 or occ_array.shape[0] != 2:
            raise ValueError("Unsupported spin-orbital layout in PySCF checkpoint.")
    elif occ_array.shape != (coeff_array.shape[1],):
        raise ValueError("Unsupported spin-orbital layout in PySCF checkpoint.")
    alpha_mask, beta_mask = occupied_orbital_masks(
        occ_array, unrestricted=unrestricted, nelec=nelec
    )
    return (
        coeff_array[0][:, alpha_mask] if unrestricted else coeff_array[:, alpha_mask],
        coeff_array[1][:, beta_mask] if unrestricted else coeff_array[:, beta_mask],
    )


def convert_checkpoint(
    input_path: str | Path, output_path: str | Path | UPath
) -> MoleculeReference:
    """Convert a spherical Gaussian PySCF checkpoint to ``reference.npz``.

    Returns:
        The converted reference.

    Raises:
        ValueError: If the checkpoint uses an unsupported layout.
    """
    input_path = Path(input_path)
    mol, values = pyscf.scf.chkfile.load_scf(str(input_path))
    if not isinstance(values, dict):
        raise ValueError(
            f"Checkpoint does not contain a supported SCF result: {input_path}"
        )
    if mol.cart:
        raise ValueError(
            "Molecular reference conversion supports spherical Gaussian orbitals only."
        )
    coeff = values.get("mo_coeff")
    occ = values.get("mo_occ")
    if coeff is None or occ is None:
        raise ValueError(
            "PySCF checkpoint is missing molecular orbital coefficients or occupations."
        )
    # Restricted RHF/ROHF encodes spatial-orbital occupations as 2 (doubly
    # occupied), 1 (singly occupied alpha), or 0. Unrestricted occupations
    # are 0 or 1.
    # Occupied columns are selected by those occupations, not leading-column order.
    alpha_coeffs, beta_coeffs = _occupied_coefficients(coeff, occ, mol.nelec)
    nspins = (alpha_coeffs.shape[1], beta_coeffs.shape[1])
    evaluator = AtomicOrbitalEvaluator.from_pyscf(mol)
    atoms = pyscf.gto.format_atom(mol.atom, unit="Bohr")
    reference = MoleculeReference(
        symbols=tuple(atom[0] for atom in atoms),
        coords=np.asarray([atom[1] for atom in atoms]),
        # PySCF atom_charge / atom_charges already return Z_eff for ECP atoms.
        charges=np.asarray(mol.atom_charges(), dtype=int),
        nspins=nspins,
        basis=evaluator.basis_dict,
        alpha_coeffs=alpha_coeffs,
        beta_coeffs=beta_coeffs,
    )
    reference.save(output_path)
    return reference
