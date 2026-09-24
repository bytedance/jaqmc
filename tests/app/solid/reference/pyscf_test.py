# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any

import numpy as np
import pyscf.pbc.gto
import pyscf.pbc.scf.chkfile
import pytest

from jaqmc.app.solid.config import SolidPySCFSolverConfig
from jaqmc.app.solid.reference import SolidReference
from jaqmc.app.solid.reference import pyscf as reference_pyscf
from tests.app.solid.reference.helpers import _solid_system, _write_gamma_checkpoint


@pytest.mark.parametrize("layout", ["restricted", "nested", "ndarray"])
def test_pyscf_convert_flattens_occupied_multik_layout(tmp_path, monkeypatch, layout):
    cell = pyscf.pbc.gto.Cell(
        atom="He 0 0 0",
        a=np.eye(3) * 4,
        basis="sto-3g",
        spin=0,
        verbose=0,
    )
    cell.build()
    kpoints = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    # Each k-point must account for the full unit-cell electron count
    # (cell.nelec == (1, 1) for He); per-k occupied columns vary instead.
    if layout == "restricted":
        coefficients: Any = [
            np.asarray([[1.0, 10.0]]),
            np.asarray([[2.0, 20.0]]),
        ]
        occupations: Any = [np.asarray([2.0, 0.0]), np.asarray([0.0, 2.0])]
        expected_alpha = [[1.0, 20.0]]
        expected_beta = [[1.0, 20.0]]
    else:
        coefficients = [
            [np.asarray([[1.0, 10.0]]), np.asarray([[2.0, 20.0]])],
            [np.asarray([[3.0, 30.0]]), np.asarray([[4.0, 40.0]])],
        ]
        occupations = [
            [np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])],
            [np.asarray([0.0, 1.0]), np.asarray([1.0, 0.0])],
        ]
        if layout == "ndarray":
            coefficients = np.asarray(coefficients)
            occupations = np.asarray(occupations)
        expected_alpha = [[1.0, 20.0]]
        expected_beta = [[30.0, 4.0]]
    values: dict[str, Any] = {
        "kpts": kpoints,
        "mo_coeff": coefficients,
        "mo_occ": occupations,
    }
    monkeypatch.setattr(
        pyscf.pbc.scf.chkfile,
        "load_scf",
        lambda _path: (cell, values),
    )

    reference = reference_pyscf.convert_checkpoint(
        tmp_path / "foreign.chk", tmp_path / "reference.npz"
    )
    loaded = SolidReference.load(tmp_path / "reference.npz")

    assert reference.nspins == (2, 2)
    assert loaded.alpha_counts.tolist() == [1, 1]
    assert loaded.beta_counts.tolist() == [1, 1]
    np.testing.assert_allclose(loaded.kpoints, kpoints)
    np.testing.assert_allclose(loaded.alpha_coeffs, expected_alpha)
    np.testing.assert_allclose(loaded.beta_coeffs, expected_beta)
    assert loaded.get_kpoint_occupancies()[1][1:] == (
        1,
        1,
    )


def test_pyscf_prepare_rejects_custom_effective_charge(tmp_path):
    system = _solid_system()
    system.atom_configs[0].charge = 1
    job = reference_pyscf.SolidPySCFReferenceJob(
        system, SolidPySCFSolverConfig(verbose=0)
    )

    with pytest.raises(
        ValueError,
        match=(
            r"PySCF reference preparation does not support custom per-atom "
            r"charges \(He has charge=1\)"
        ),
    ):
        job.prepare(tmp_path / "job")


def test_pyscf_prepare_imports_periodic_dft_module(tmp_path):
    job_dir = tmp_path / "job"
    reference_pyscf.SolidPySCFReferenceJob(
        _solid_system(),
        SolidPySCFSolverConfig(method="KRKS", verbose=0),
    ).prepare(job_dir)

    assert "import pyscf.pbc.dft" in (job_dir / "input.py").read_text()


def test_pyscf_prepare_uses_solver_pp_override_for_gth(tmp_path):
    job_dir = tmp_path / "job"
    reference_pyscf.SolidPySCFReferenceJob(
        _solid_system(pp="ccecp", symbol="F", s_z=0.5),
        SolidPySCFSolverConfig(
            basis="gth-dzv",
            pp={"F": "gth-pbe-q7"},
            method="KUHF",
            verbose=0,
        ),
    ).prepare(job_dir)

    payload = (job_dir / "input.py").read_text()
    assert '"pseudo": {' in payload
    assert '"F": "gth-pbe-q7"' in payload
    assert '"ecp": {}' in payload


def test_pyscf_prepare_uses_automatic_double_zeta_basis_when_omitted(tmp_path):
    ae_dir = tmp_path / "ae"
    ecp_dir = tmp_path / "ecp"
    reference_pyscf.SolidPySCFReferenceJob(
        _solid_system(), SolidPySCFSolverConfig(verbose=0)
    ).prepare(ae_dir)
    reference_pyscf.SolidPySCFReferenceJob(
        _solid_system(pp="ccecp", symbol="F", s_z=0.5),
        SolidPySCFSolverConfig(verbose=0),
    ).prepare(ecp_dir)

    assert '"He": "cc-pVDZ"' in (ae_dir / "input.py").read_text()
    assert '"F": "ccecpccpvdz"' in (ecp_dir / "input.py").read_text()


def test_pyscf_run_executes_input_in_job_directory(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    calls = []
    job = reference_pyscf.SolidPySCFReferenceJob(
        _solid_system(), SolidPySCFSolverConfig()
    )

    class Process:
        def wait(self):
            return 0

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return Process()

    monkeypatch.setattr(reference_pyscf.subprocess, "Popen", fake_popen)

    job.run(job_dir)

    assert calls[0][0] == ([sys.executable, "input.py"],)
    assert calls[0][1]["cwd"] == job_dir
    assert "stderr" not in calls[0][1]
    assert "stdout" not in calls[0][1]


def test_pyscf_convert_accepts_gamma_point_rhf_checkpoint(tmp_path):
    checkpoint = _write_gamma_checkpoint(
        tmp_path, "gamma.chk", np.ones((1, 1)), np.array([2.0])
    )

    reference = reference_pyscf.convert_checkpoint(
        checkpoint, tmp_path / "reference.npz"
    )

    np.testing.assert_allclose(reference.kpoints, [[0.0, 0.0, 0.0]])
    assert reference.nspins == (1, 1)


def test_pyscf_convert_accepts_gamma_point_uhf_checkpoint(tmp_path):
    checkpoint = _write_gamma_checkpoint(
        tmp_path,
        "gamma-uhf.chk",
        np.array([[[2.0]], [[3.0]]]),
        np.ones((2, 1)),
    )

    reference = reference_pyscf.convert_checkpoint(
        checkpoint, tmp_path / "reference.npz"
    )

    np.testing.assert_allclose(reference.alpha_coeffs, [[2.0]])
    np.testing.assert_allclose(reference.beta_coeffs, [[3.0]])
    assert reference.nspins == (1, 1)


def test_pyscf_convert_rejects_fractional_occupations(tmp_path):
    checkpoint = _write_gamma_checkpoint(
        tmp_path, "fractional.chk", np.ones((1, 1)), np.array([1.5])
    )

    with pytest.raises(ValueError, match="occupations are fractional"):
        reference_pyscf.convert_checkpoint(checkpoint, tmp_path / "reference.npz")


def test_pyscf_convert_rejects_cartesian_checkpoint(tmp_path, monkeypatch):
    cell = pyscf.pbc.gto.Cell(
        atom="He 0 0 0", a=np.eye(3) * 4, basis="sto-3g", verbose=0
    )
    cell.cart = True
    cell.build()
    monkeypatch.setattr(
        pyscf.pbc.scf.chkfile,
        "load_scf",
        lambda _path: (cell, {}),
    )

    with pytest.raises(ValueError, match="spherical Gaussian"):
        reference_pyscf.convert_checkpoint(
            tmp_path / "cartesian.chk", tmp_path / "reference.npz"
        )


def _spin_down_hydrogen_cell() -> pyscf.pbc.gto.Cell:
    """One-electron H cell with s_z=-0.5, i.e. nelec=(0, 1)."""
    cell = pyscf.pbc.gto.Cell(
        atom="H 0 0 0",
        a=np.eye(3) * 8,
        basis="cc-pvdz",
        spin=-1,
        verbose=0,
    )
    cell.build()
    assert cell.nelec == (0, 1)
    return cell


def test_pyscf_convert_rejects_occupations_inconsistent_with_cell(
    tmp_path, monkeypatch
):
    # Regression test for issue #100: PBC-KUHF with cell.spin=-1 can report
    # every alpha orbital occupied although the cell has zero alpha electrons.
    cell = _spin_down_hydrogen_cell()
    nmo = cell.nao_nr()
    values: dict[str, Any] = {
        "kpts": np.zeros((1, 3)),
        "mo_coeff": [[np.eye(nmo)], [np.eye(nmo)]],
        "mo_occ": [
            [np.ones(nmo)],
            [np.concatenate(([1.0], np.zeros(nmo - 1)))],
        ],
    }
    monkeypatch.setattr(
        pyscf.pbc.scf.chkfile,
        "load_scf",
        lambda _path: (cell, values),
    )

    with pytest.raises(ValueError, match="do not match the electron count"):
        reference_pyscf.convert_checkpoint(
            tmp_path / "inconsistent.chk", tmp_path / "reference.npz"
        )


def test_pyscf_convert_accepts_zero_alpha_electron_checkpoint(tmp_path, monkeypatch):
    cell = _spin_down_hydrogen_cell()
    nmo = cell.nao_nr()
    values: dict[str, Any] = {
        "kpts": np.zeros((1, 3)),
        "mo_coeff": [[np.eye(nmo)], [np.eye(nmo)]],
        "mo_occ": [
            [np.zeros(nmo)],
            [np.concatenate(([1.0], np.zeros(nmo - 1)))],
        ],
    }
    monkeypatch.setattr(
        pyscf.pbc.scf.chkfile,
        "load_scf",
        lambda _path: (cell, values),
    )

    reference = reference_pyscf.convert_checkpoint(
        tmp_path / "spin-down.chk", tmp_path / "reference.npz"
    )

    assert reference.nspins == (0, 1)
    assert reference.alpha_coeffs.shape == (nmo, 0)
    assert reference.beta_coeffs.shape == (nmo, 1)
