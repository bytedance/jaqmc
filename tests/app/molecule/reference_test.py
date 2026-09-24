# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyscf.gto
import pyscf.lib
import pyscf.scf
import pytest
from click.testing import CliRunner
from jax import numpy as jnp
from upath import UPath

from jaqmc.app.cli import cli
from jaqmc.app.molecule.config import AtomConfig, MoleculeConfig, MoleculeSolverConfig
from jaqmc.app.molecule.reference import (
    MoleculeReference,
    load_reference,
)
from jaqmc.app.molecule.reference import (
    pyscf as reference_pyscf,
)
from jaqmc.app.molecule.workflow import MoleculeTrainWorkflow
from jaqmc.utils.config import ConfigManager


@pytest.fixture(autouse=True)
def no_pyscf_temp_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pyscf.lib.param, "TMPDIR", None)


def _checkpoint(
    path,
    *,
    method: str,
    atom: str,
    spin: int,
):
    mol = pyscf.gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    mean_field = getattr(pyscf.scf, method)(mol).set(verbose=0)
    mean_field.chkfile = str(path)
    mean_field.kernel()


@pytest.mark.parametrize(
    ("method", "atom", "spin", "nspins"),
    [
        ("RHF", "H 0 0 0; H 0 0 1.4", 0, (1, 1)),
        ("RHF", "H 0 0 0", 1, (1, 0)),
        ("UHF", "H 0 0 0", 1, (1, 0)),
    ],
)
@pytest.mark.parametrize(
    "output_uri",
    ["{tmp_path}/{method}.npz", "memory://jaqmc-molecule-model/{method}.npz"],
)
def test_convert_accepts_restricted_and_unrestricted_checkpoints(
    tmp_path, method, atom, spin, nspins, output_uri
):
    checkpoint = tmp_path / f"{method.lower()}.chk"
    output = UPath(output_uri.format(tmp_path=tmp_path, method=method.lower()))
    _checkpoint(checkpoint, method=method, atom=atom, spin=spin)

    reference_pyscf.convert_checkpoint(checkpoint, output)
    loaded = MoleculeReference.load(output)

    assert output.exists()
    assert loaded.nspins == nspins
    assert loaded.alpha_coeffs.shape == (loaded.alpha_coeffs.shape[0], nspins[0])
    assert loaded.beta_coeffs.shape == (loaded.beta_coeffs.shape[0], nspins[1])


def _molecule_reference(
    *,
    symbols: tuple[str, ...] = ("H",),
    coords: np.ndarray | None = None,
    charges: np.ndarray | None = None,
    nspins: tuple[int, int] = (1, 0),
) -> MoleculeReference:
    return MoleculeReference(
        symbols=symbols,
        coords=np.zeros((1, 3)) if coords is None else coords,
        charges=np.ones(1, dtype=int) if charges is None else charges,
        nspins=nspins,
        basis={},
        alpha_coeffs=np.ones((1, nspins[0])),
        beta_coeffs=np.ones((1, nspins[1])),
    )


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        (_molecule_reference(symbols=("He",)), "atoms"),
        (_molecule_reference(coords=np.asarray([[0.0, 0.0, 1.0]])), "geometry"),
        (_molecule_reference(charges=np.zeros(1, dtype=int)), "charges"),
        (_molecule_reference(nspins=(0, 1)), "electron spins"),
    ],
)
def test_load_rejects_mismatched_configured_molecule(tmp_path, reference, error):
    system = MoleculeConfig(
        atom_configs=[AtomConfig(symbol="H", coords=[0.0, 0.0, 0.0])],
        s_z=0.5,
    )
    path = tmp_path / "reference.npz"
    _molecule_reference().save(path)

    assert load_reference(path, system).nspins == (1, 0)
    reference.save(path)

    with pytest.raises(ValueError, match=error):
        load_reference(path, system)


def test_prepare_run_and_convert_is_solver_only(tmp_path):
    system = MoleculeConfig(
        atom_configs=[AtomConfig(symbol="H", coords=[0.0, 0.0, 0.0])],
        s_z=0.5,
    )
    solver = MoleculeSolverConfig(basis="sto-3g", method="UHF", verbose=0)
    job_dir = tmp_path / "job"
    job = reference_pyscf.MoleculePySCFReferenceJob(system, solver)
    job.prepare(job_dir)

    assert "jaqmc" not in (job_dir / "input.py").read_text().lower()

    job.run(job_dir)
    job.finalize_npz(job_dir, UPath(job_dir / "reference.npz"))
    mol, _ = pyscf.scf.chkfile.load_scf(str(job_dir / solver.checkpoint))
    reference = MoleculeReference.load(job_dir / "reference.npz")

    assert not mol.cart
    assert reference.nspins == (1, 0)

    workflow = MoleculeTrainWorkflow(
        ConfigManager(
            {
                "system": {
                    "s_z": 0.5,
                    "atoms": [{"symbol": "H", "coords": [0.0, 0.0, 0.0]}],
                },
                "reference": str(job_dir / "reference.npz"),
                "pretrain": {"run": {"iterations": 1}},
                "train": {"run": {"iterations": 0}},
                "wf": {
                    "hidden_dims_single": [2, 2],
                    "hidden_dims_double": [2, 2],
                },
            }
        )
    )
    workflow.prepare(dry_run=True)
    assert workflow.pretrain_stage is not None


def test_molecule_prepare_cli_unwraps_system_factory(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    job_dir = tmp_path / "job"
    caplog.set_level("INFO")
    result = CliRunner().invoke(
        cli,
        [
            "molecule",
            "reference",
            "prepare",
            "--output",
            str(job_dir),
            "system.module=atom",
            "system.symbol=H",
            "solver.basis=sto-3g",
            "solver.verbose=0",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = (job_dir / "input.py").read_text()
    assert '"symbol": "H"' in payload
    assert '"spin": 1' in payload
    assert "Wrote solver job to" in caplog.text
    assert "Resolved configurations:" in caplog.text


def test_molecule_prepare_cli_runs_and_finalizes_selected_job(tmp_path, monkeypatch):
    events = []

    def run(self, job_dir, *, wait=True):
        events.append(("run", job_dir, wait))
        return object()

    def finalize(self, job_dir, reference_path):
        events.append(("finalize", job_dir, reference_path))

    monkeypatch.setattr(reference_pyscf.MoleculePySCFReferenceJob, "run", run)
    monkeypatch.setattr(
        reference_pyscf.MoleculePySCFReferenceJob, "finalize_npz", finalize
    )
    job_dir = tmp_path / "job"
    result = CliRunner().invoke(
        cli,
        [
            "molecule",
            "reference",
            "prepare",
            "--output",
            str(job_dir),
            "--run",
            "--reference-output",
            "custom.npz",
            "system.module=atom",
            "system.symbol=H",
            "solver.basis=sto-3g",
            "solver.verbose=0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [event[0] for event in events] == ["run", "finalize"]
    assert Path(events[1][2]).name == "custom.npz"


def test_molecule_prepare_cli_reads_system_only_yaml(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    yml = tmp_path / "atom.yml"
    yml.write_text(
        "system:\n  module: atom\n  symbol: H\n",
        encoding="utf-8",
    )
    job_dir = tmp_path / "job"
    caplog.set_level("INFO")
    result = CliRunner().invoke(
        cli,
        [
            "molecule",
            "reference",
            "prepare",
            "--yml",
            str(yml),
            "--output",
            str(job_dir),
            "solver.basis=sto-3g",
            "solver.verbose=0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"symbol": "H"' in (job_dir / "input.py").read_text()
    assert "Resolved configurations:" in caplog.text
    assert "Wrote solver job to" in caplog.text


def test_molecule_prepare_cli_dry_run_prints_config_without_writing(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    job_dir = tmp_path / "job"
    caplog.set_level("INFO")
    result = CliRunner().invoke(
        cli,
        [
            "molecule",
            "reference",
            "prepare",
            "--output",
            str(job_dir),
            "--dry-run",
            "system.module=atom",
            "system.symbol=H",
            "solver.basis=sto-3g",
        ],
    )

    assert result.exit_code == 0, result.output
    assert not job_dir.exists()
    assert "Resolved configurations:" in caplog.text
    assert "Dry run: would write solver job" in caplog.text
    assert "symbol: H" in caplog.text


def test_prepare_runs_from_job_directory(tmp_path, monkeypatch):
    system = MoleculeConfig(
        atom_configs=[AtomConfig(symbol="H", coords=[0.0, 0.0, 0.0])],
        s_z=0.5,
    )
    solver = MoleculeSolverConfig(basis="sto-3g", method="UHF", verbose=0)
    job_dir = tmp_path / "job"
    job = reference_pyscf.MoleculePySCFReferenceJob(system, solver)
    job.prepare(job_dir)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    job.run(job_dir)

    assert (job_dir / solver.checkpoint).exists()
    assert not (elsewhere / solver.checkpoint).exists()


@pytest.mark.parametrize(
    "output_uri",
    ["{tmp_path}/reference.npz", "memory://jaqmc-molecule-reference/reference.npz"],
)
def test_molecule_convert_cli_writes_explicit_output(
    tmp_path, caplog: pytest.LogCaptureFixture, output_uri: str
):
    checkpoint = tmp_path / "foreign.chk"
    _checkpoint(
        checkpoint,
        method="RHF",
        atom="H 0 0 0; H 0 0 1.4",
        spin=0,
    )
    output = UPath(output_uri.format(tmp_path=tmp_path))
    caplog.set_level("INFO")

    result = CliRunner().invoke(
        cli,
        [
            "molecule",
            "reference",
            "convert",
            "--input",
            str(checkpoint),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert MoleculeReference.load(output).nspins == (1, 1)
    assert "Wrote reference to" in caplog.text


def test_prepare_rejects_inconsistent_explicit_charge(tmp_path):
    system = MoleculeConfig(
        atom_configs=[
            AtomConfig(symbol="H", coords=[0.0, 0.0, 0.0], charge=2),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"PySCF reference preparation does not support custom per-atom "
            r"charges \(H has charge=2\)"
        ),
    ):
        reference_pyscf.MoleculePySCFReferenceJob(
            system, MoleculeSolverConfig(basis="sto-3g")
        ).prepare(tmp_path / "job")


def test_prepare_uses_solver_pp_override_for_gth(tmp_path):
    system = MoleculeConfig(
        atom_configs=[AtomConfig(symbol="F", coords=[0.0, 0.0, 0.0])],
        pp="ccecp",
        s_z=0.5,
    )
    job_dir = tmp_path / "job"
    job = reference_pyscf.MoleculePySCFReferenceJob(
        system,
        MoleculeSolverConfig(
            basis="gth-dzv",
            pp={"F": "gth-pbe-q7"},
            method="UHF",
            verbose=0,
        ),
    )
    job.prepare(job_dir)

    payload = (job_dir / "input.py").read_text()
    assert '"pseudo": {' in payload
    assert '"F": "gth-pbe-q7"' in payload
    assert '"ecp": {}' in payload


def test_prepare_uses_automatic_double_zeta_basis_when_omitted(tmp_path):
    system = MoleculeConfig(
        atom_configs=[
            AtomConfig(symbol="H", coords=[0.0, 0.0, 0.0]),
            AtomConfig(symbol="Li", coords=[0.0, 0.0, 1.0]),
            AtomConfig(symbol="Fe", coords=[0.0, 0.0, 2.0]),
        ],
        pp={"Li": "ccecp", "Fe": "ph"},
    )
    job_dir = tmp_path / "job"
    reference_pyscf.MoleculePySCFReferenceJob(
        system, MoleculeSolverConfig(verbose=0)
    ).prepare(job_dir)

    payload = (job_dir / "input.py").read_text()
    assert '"H": "cc-pVDZ"' in payload
    assert '"Li": "ccecpccpvdz"' in payload
    assert '"Fe": "ccecpccpvdz"' in payload


def test_convert_stores_pyscf_effective_charges_for_ecp(tmp_path):
    checkpoint = tmp_path / "c.chk"
    mol = pyscf.gto.M(atom="C 0 0 0", basis="sto-3g", ecp="ccecp", spin=0, verbose=0)
    mean_field = pyscf.scf.RHF(mol).set(verbose=0)
    mean_field.chkfile = str(checkpoint)
    mean_field.kernel()
    system = MoleculeConfig(
        atom_configs=[AtomConfig(symbol="C", coords=[0.0, 0.0, 0.0])],
        pp="ccecp",
    )

    reference = reference_pyscf.convert_checkpoint(
        checkpoint, tmp_path / "reference.npz"
    )

    assert reference.charges.tolist() == [system.atoms[0].charge]
    assert reference.nspins == system.electron_spins


def test_molecule_train_skips_reference_when_pretraining_is_disabled():
    cfg = ConfigManager(
        {
            "system": {"module": "atom", "symbol": "H"},
            "pretrain": {"run": {"iterations": 0}},
            "train": {"run": {"iterations": 0}},
            "wf": {
                "hidden_dims_single": [2, 2],
                "hidden_dims_double": [2, 2],
            },
        }
    )

    workflow = MoleculeTrainWorkflow(cfg)

    assert workflow.pretrain_stage is None
    assert workflow.reference is None


def test_convert_matches_occupied_pyscf_orbital(tmp_path):
    checkpoint = tmp_path / "scf.chk"
    mol = pyscf.gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, unit="Bohr", verbose=0)
    mean_field = pyscf.scf.UHF(mol).set(verbose=0)
    mean_field.chkfile = str(checkpoint)
    mean_field.kernel()
    output = tmp_path / "reference.npz"
    reference_pyscf.convert_checkpoint(checkpoint, output)
    reference = MoleculeReference.load(output)
    position = np.array([[0.2, 0.0, 0.0]])

    actual, empty = reference.eval_orbitals(jnp.asarray(position), (1, 0))
    expected = mol.eval_gto("GTOval_sph", position) @ mean_field.mo_coeff[0][:, :1]

    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert empty.shape == (0, 0)


def test_convert_masks_occupied_columns_not_leading_orbitals(tmp_path):
    checkpoint = tmp_path / "hole.chk"
    mol = pyscf.gto.M(
        atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", spin=0, unit="Bohr", verbose=0
    )
    mean_field = pyscf.scf.RHF(mol).set(verbose=0)
    mean_field.chkfile = str(checkpoint)
    mean_field.kernel()
    mean_field.mo_occ = np.array([0.0, 2.0])
    mean_field.dump_chk(str(checkpoint))

    reference = reference_pyscf.convert_checkpoint(
        checkpoint, tmp_path / "reference.npz"
    )
    position = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 1.3]])

    actual, beta = reference.eval_orbitals(jnp.asarray(position), (1, 1))
    occupied = mol.eval_gto("GTOval_sph", position) @ mean_field.mo_coeff[:, 1:2]

    np.testing.assert_allclose(actual, occupied[:1], atol=1e-6)
    np.testing.assert_allclose(beta, occupied[1:], atol=1e-6)


def test_convert_rejects_fractional_occupations(tmp_path):
    checkpoint = tmp_path / "fractional.chk"
    mol = pyscf.gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    mean_field = pyscf.scf.RHF(mol).set(verbose=0)
    mean_field.mo_coeff = np.ones((mol.nao_nr(), 1))
    mean_field.mo_occ = np.array([1.5])
    mean_field.mo_energy = np.array([0.0])
    mean_field.e_tot = 0.0
    mean_field.dump_chk(str(checkpoint))

    with pytest.raises(ValueError, match="occupations are fractional"):
        reference_pyscf.convert_checkpoint(checkpoint, tmp_path / "reference.npz")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("basis", None),
        ("coords", np.zeros((2, 3))),
        ("charges", np.ones((1, 1), dtype=int)),
        ("alpha_coeffs", np.ones((2, 1))),
        ("beta_coeffs", np.ones((1, 1))),
    ],
)
def test_load_rejects_incomplete_or_malformed_reference_file(tmp_path, field, value):
    path = tmp_path / "invalid.npz"
    archive = {
        "metadata": np.asarray(
            json.dumps(
                {
                    "format": "jaqmc.molecule.reference",
                    "version": 1,
                    "symbols": ["H"],
                    "nspins": [1, 0],
                    "basis": {},
                }
            )
        ),
        "coords": np.zeros((1, 3)),
        "charges": np.ones(1, dtype=int),
        "alpha_coeffs": np.ones((1, 1)),
        "beta_coeffs": np.zeros((1, 0)),
    }
    if value is None:
        if field == "basis":
            metadata = json.loads(str(archive["metadata"].item()))
            del metadata[field]
            archive["metadata"] = np.asarray(json.dumps(metadata))
        else:
            del archive[field]
    else:
        archive[field] = value
    np.savez(path, **cast(Any, archive))

    with pytest.raises(ValueError, match="Invalid molecular reference file"):
        MoleculeReference.load(path)


def test_convert_rejects_cartesian_basis(tmp_path):
    checkpoint = tmp_path / "cartesian.chk"
    mol = pyscf.gto.M(
        atom="Li 0 0 0",
        basis="sto-3g",
        cart=True,
        spin=1,
        unit="Bohr",
        verbose=0,
    )
    mean_field = pyscf.scf.UHF(mol).set(verbose=0)
    mean_field.chkfile = str(checkpoint)
    mean_field.kernel()

    with pytest.raises(ValueError, match="spherical Gaussian"):
        reference_pyscf.convert_checkpoint(checkpoint, tmp_path / "reference.npz")


def test_prepared_job_rejects_unknown_pyscf_setting(tmp_path, capfd):
    system = MoleculeConfig(
        atom_configs=[AtomConfig(symbol="H", coords=[0.0, 0.0, 0.0])],
        s_z=0.5,
    )
    job_dir = tmp_path / "job"
    job = reference_pyscf.MoleculePySCFReferenceJob(
        system,
        MoleculeSolverConfig(basis="sto-3g", extra={"not_a_pyscf_attr": 1}),
    )
    job.prepare(job_dir)

    with pytest.raises(RuntimeError, match="PySCF job failed"):
        job.run(job_dir)

    assert (
        "Unknown PySCF mean-field setting: not_a_pyscf_attr" in capfd.readouterr().err
    )
