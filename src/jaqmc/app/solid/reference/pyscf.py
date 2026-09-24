# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""PySCF preparation and conversion for solid orbital references."""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyscf.pbc.gto
import pyscf.pbc.scf.chkfile
from jax import numpy as jnp
from upath import UPath

from jaqmc.app.solid.config import SolidConfig, SolidPySCFSolverConfig
from jaqmc.utils.atomic.gto import PBCAtomicOrbitalEvaluator
from jaqmc.utils.atomic.pyscf import (
    occupied_orbital_masks,
    resolve_pyscf_basis,
    translate_pp_to_pyscf,
)
from jaqmc.utils.supercell import (
    get_primitive_kpts_for_supercell,
    get_reciprocal_vectors,
)

from .model import GaussianSolidReference

__all__ = ["SolidPySCFReferenceJob", "convert_checkpoint"]


def _kpoints(system: SolidConfig) -> np.ndarray:
    return np.asarray(
        get_primitive_kpts_for_supercell(
            jnp.asarray(system.supercell_matrix),
            get_reciprocal_vectors(jnp.asarray(system.lattice_vectors)),
            jnp.asarray(system.twist),
        )
    )


def _solver_data(system: SolidConfig, solver: SolidPySCFSolverConfig) -> dict[str, Any]:
    if solver.method not in {"KRHF", "KUHF", "KRKS", "KUKS"}:
        raise ValueError(f"Unsupported PySCF solid method: {solver.method}.")
    for atom in system.atoms:
        if atom.charge != atom.atomic_number - atom.core_electrons:
            raise ValueError(
                "PySCF reference preparation does not support custom per-atom "
                f"charges ({atom.symbol} has charge={atom.charge})."
            )
    pseudopotentials = translate_pp_to_pyscf(system.atoms, solver.pp)
    return {
        "atoms": [
            {"symbol": atom.symbol, "coords": atom.coords} for atom in system.atoms
        ],
        "lattice": system.lattice_vectors,
        "kpoints": _kpoints(system),
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
    return f'''"""Generated PySCF periodic reference job."""
import json
import numpy as np
import pyscf.dft
import pyscf.gto
import pyscf.pbc.dft
import pyscf.pbc.gto
import pyscf.pbc.scf

config = json.loads(r"""
{payload}
""")
atoms = [[item["symbol"], item["coords"]] for item in config["atoms"]]
cell = pyscf.pbc.gto.Cell(
    atom=atoms,
    a=np.asarray(config["lattice"]),
    unit="Bohr",
    basis=config["basis"],
    ecp=config["ecp"] or None,
    pseudo=config["pseudo"] or None,
    charge=config["total_charge"],
    spin=config["spin"],
    verbose=config["verbose"],
)
cell.cart = False
cell.build()
method = config["method"]
if method in ("KRHF", "KUHF"):
    mf = getattr(pyscf.pbc.scf, method)(cell, kpts=np.asarray(config["kpoints"]))
else:
    mf = getattr(pyscf.pbc.dft, method)(cell, kpts=np.asarray(config["kpoints"]))
    mf.xc = config["xc"]
mf = mf.density_fit()
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
class SolidPySCFReferenceJob:
    """PySCF reference job for a periodic system and solver configuration."""

    system: SolidConfig
    solver: SolidPySCFSolverConfig

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


def _gamma_kpoints_and_orbitals(
    values: dict[str, Any], coeff: Any, occ: Any
) -> tuple[np.ndarray, Any, Any]:
    kpoint = np.asarray(values.get("kpt"))
    if kpoint.shape != (3,):
        raise ValueError("PySCF Gamma-point checkpoint has an invalid k-point.")
    coeff_array = np.asarray(coeff)
    occ_array = np.asarray(occ)
    if coeff_array.ndim == 2 and occ_array.shape == (coeff_array.shape[1],):
        return kpoint[None, :], coeff_array[None, ...], occ_array[None, ...]
    if (
        coeff_array.ndim == 3
        and coeff_array.shape[0] == 2
        and occ_array.shape == (2, coeff_array.shape[2])
    ):
        return kpoint[None, :], coeff_array[:, None, ...], occ_array[:, None, ...]
    raise ValueError(
        "PySCF Gamma-point checkpoint must contain RHF/RKS arrays shaped "
        "(nao, nmo) and (nmo,), or UHF/UKS arrays shaped "
        "(2, nao, nmo) and (2, nmo)."
    )


def convert_checkpoint(
    input_path: str | Path, output_path: str | Path | UPath
) -> GaussianSolidReference:
    """Convert a spherical Gaussian periodic PySCF checkpoint.

    Returns:
        The converted reference.

    Raises:
        ValueError: If the checkpoint uses an unsupported layout.
    """
    cell, values = pyscf.pbc.scf.chkfile.load_scf(str(input_path))
    if not isinstance(values, dict):
        raise ValueError(
            f"Checkpoint does not contain a supported SCF result: {input_path}"
        )
    if cell.cart:
        raise ValueError(
            "Solid reference conversion supports spherical Gaussian orbitals only."
        )
    coeff = values.get("mo_coeff")
    occ = values.get("mo_occ")
    if coeff is None or occ is None:
        raise ValueError(
            "PySCF checkpoint is missing orbital coefficients or occupations."
        )
    kpoints = values.get("kpts")
    if kpoints is None:
        kpoints, coeff, occ = _gamma_kpoints_and_orbitals(values, coeff, occ)
    kpoints = np.asarray(kpoints)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError("Periodic checkpoint is missing Cartesian k-points.")
    nk = len(kpoints)
    coeff_array = np.asarray(coeff)
    occ_array = np.asarray(occ)
    # PySCF's chkfile loader stores unrestricted (alpha, beta) results as either
    # a single (2, nk, nao, nmo) ndarray or a nested [alpha_by_k, beta_by_k]
    # list, depending on which SCF class produced the checkpoint. Both encode
    # the same physical result; the restricted `else` branch below is the only
    # genuinely different case (alpha and beta share the same coefficients).
    is_ndarray_unrestricted = coeff_array.ndim == 4 and coeff_array.shape[:2] == (2, nk)
    is_nested_unrestricted = (
        isinstance(coeff, (tuple, list))
        and len(coeff) == 2
        and all(isinstance(channel, (tuple, list)) for channel in coeff)
    )
    unrestricted = is_ndarray_unrestricted or is_nested_unrestricted
    if is_ndarray_unrestricted:
        alpha_coeffs_by_k = coeff_array[0]
        beta_coeffs_by_k = coeff_array[1]
    elif is_nested_unrestricted:
        alpha_coeffs_by_k = coeff[0]
        beta_coeffs_by_k = coeff[1]
    else:
        alpha_coeffs_by_k = coeff
        beta_coeffs_by_k = coeff
    alpha_masks, beta_masks = occupied_orbital_masks(
        occ_array, unrestricted=unrestricted, nelec=cell.nelec
    )
    alpha_blocks: list[np.ndarray] = []
    beta_blocks: list[np.ndarray] = []
    alpha_counts = []
    beta_counts = []
    for k in range(nk):
        alpha_mask = alpha_masks[k]
        beta_mask = beta_masks[k]
        alpha_blocks.append(np.asarray(alpha_coeffs_by_k[k])[:, alpha_mask])
        beta_blocks.append(np.asarray(beta_coeffs_by_k[k])[:, beta_mask])
        alpha_counts.append(int(alpha_mask.sum()))
        beta_counts.append(int(beta_mask.sum()))
    evaluator = PBCAtomicOrbitalEvaluator.from_pyscf(cell)
    atoms = pyscf.gto.format_atom(cell.atom, unit="Bohr")
    reference = GaussianSolidReference(
        symbols=tuple(atom[0] for atom in atoms),
        atom_coords=np.asarray([atom[1] for atom in atoms]),
        lattice=np.asarray(cell.a),
        kpoints=kpoints,
        alpha_counts=np.asarray(alpha_counts),
        beta_counts=np.asarray(beta_counts),
        alpha_coeffs=np.concatenate(alpha_blocks, axis=1),
        beta_coeffs=np.concatenate(beta_blocks, axis=1),
        basis=evaluator.eval_aos.basis_dict,
        image_translation_vectors=np.asarray(evaluator.image_translation_vectors),
    )
    reference.save(output_path)
    return reference
