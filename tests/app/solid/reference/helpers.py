# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import h5py
import numpy as np
import pyscf.pbc.gto
import pyscf.pbc.scf

from jaqmc.app.solid.config import SolidAtomConfig, SolidConfig
from jaqmc.app.solid.config.base import LatticeParams


def _solid_system(
    *,
    symbol="He",
    pp=None,
    s_z=0,
    total_charge=0,
    supercell_matrix=None,
    twist=None,
):
    kwargs: dict[str, Any] = {
        "lattice": LatticeParams(a=4, b=4, c=4),
        "atom_configs": [SolidAtomConfig(symbol=symbol, frac_coords=[0, 0, 0])],
        "pp": pp,
        "s_z": s_z,
        "total_charge": total_charge,
    }
    if supercell_matrix is not None:
        kwargs["supercell_matrix"] = supercell_matrix
    if twist is not None:
        kwargs["twist"] = twist
    return SolidConfig(**kwargs)


def _write_qe_xml(
    path,
    kpoints: list[tuple[str, str]],
    *,
    lsda: bool = False,
    packed_spin: bool = False,
    spin_bands: tuple[int, int] | None = None,
    lattice: float = 4.0,
    nelec: float | None = None,
    symbol: str = "He",
    converged: bool | None = None,
    weights: list[float] | None = None,
    eigenvalues: list[str] | None = None,
    symmetry_flags: tuple[bool, bool] | None = (True, True),
    include_input: bool = True,
) -> None:
    if packed_spin:
        assert lsda
        assert spin_bands is not None
    nkpoints = (
        len(kpoints) if packed_spin else (len(kpoints) // 2 if lsda else len(kpoints))
    )
    if weights is None:
        weights = [1 / nkpoints] * len(kpoints)
    assert len(weights) == len(kpoints)
    if nelec is None:
        nelec = sum(
            weight * sum(float(value) for value in occupations.split())
            for (_, occupations), weight in zip(kpoints, weights, strict=True)
        )
    if eigenvalues is None:
        eigenvalues = [
            " ".join(str(index) for index, _ in enumerate(occupations.split()))
            for _, occupations in kpoints
        ]
    assert len(eigenvalues) == len(kpoints)
    energies = "\n".join(
        f"""      <ks_energies>
        <k_point weight="{weight}">{kpoint}</k_point>
        <eigenvalues>{energy}</eigenvalues>
        <occupations>{occupations}</occupations>
      </ks_energies>"""
        for (kpoint, occupations), weight, energy in zip(
            kpoints, weights, eigenvalues, strict=True
        )
    )
    convergence = ""
    if converged is not None:
        convergence = (
            "<convergence_info>"
            f"<convergence_achieved>{str(converged).lower()}</convergence_achieved>"
            "</convergence_info>"
        )
    spin_metadata = ""
    if packed_spin:
        assert spin_bands is not None
        spin_metadata = (
            f"<nks>{nkpoints}</nks>"
            f"<nbnd_up>{spin_bands[0]}</nbnd_up>"
            f"<nbnd_dw>{spin_bands[1]}</nbnd_dw>"
        )
    symmetry_metadata = ""
    if include_input:
        symmetry_metadata = "<input>"
        if symmetry_flags is not None:
            nosym, noinv = symmetry_flags
            symmetry_metadata += (
                "<symmetry_flags>"
                f"<nosym>{str(nosym).lower()}</nosym>"
                f"<noinv>{str(noinv).lower()}</noinv>"
                "</symmetry_flags>"
            )
        symmetry_metadata += "</input>"
    path.write_text(
        f"""<qes:espresso xmlns:qes="qes">
  {symmetry_metadata}
  <output>
    {convergence}
    <dft><functional>PBE</functional></dft>
    <atomic_structure alat="{lattice}">
      <cell>
        <a1>{lattice} 0 0</a1>
        <a2>0 {lattice} 0</a2>
        <a3>0 0 {lattice}</a3>
      </cell>
      <atomic_positions units="bohr">
        <atom name="{symbol}">0 0 0</atom>
      </atomic_positions>
    </atomic_structure>
    <band_structure>
      <noncolin>false</noncolin><lsda>{str(lsda).lower()}</lsda>
      <nelec>{nelec}</nelec>
      {spin_metadata}
{energies}
    </band_structure>
  </output>
</qes:espresso>""",
        encoding="utf-8",
    )


def _write_gamma_checkpoint(
    tmp_path,
    filename,
    coefficients: np.ndarray,
    occupations: np.ndarray,
):
    cell = pyscf.pbc.gto.Cell(
        atom="He 0 0 0",
        a=np.eye(3) * 4,
        basis="sto-3g",
        verbose=0,
    )
    cell.build()
    checkpoint = tmp_path / filename
    mean_field = (
        pyscf.pbc.scf.UHF(cell) if coefficients.ndim == 3 else pyscf.pbc.scf.RHF(cell)
    )
    mean_field.mo_coeff = coefficients
    mean_field.mo_occ = occupations
    mean_field.mo_energy = np.zeros_like(occupations)
    mean_field.e_tot = 0.0
    mean_field.dump_chk(str(checkpoint))
    return checkpoint


def _write_qe_wfc(
    path,
    miller,
    coefficients,
    *,
    xk=(0.0, 0.0, 0.0),
    ik=1,
    ispin=1,
    gamma_only=False,
) -> None:
    miller = np.asarray(miller)
    coefficients = np.asarray(coefficients)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("MillerIndices", data=miller)
        handle.create_dataset("evc", data=coefficients)
        handle.attrs["igwx"] = len(miller)
        handle.attrs["nbnd"] = len(coefficients)
        handle.attrs["npol"] = 1
        handle.attrs["ik"] = ik
        handle.attrs["xk"] = xk
        handle.attrs["ispin"] = ispin
        handle.attrs["gamma_only"] = ".TRUE." if gamma_only else ".FALSE."
        handle.attrs["scale_factor"] = 1.0
