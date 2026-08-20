# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from jaqmc.app.solid.config import SolidAtomConfig, SolidConfig, SolidQESolverConfig
from jaqmc.app.solid.config.base import LatticeParams
from jaqmc.app.solid.reference import qe
from jaqmc.app.solid.reference.qe import job, validate_upf_header
from jaqmc.utils.atomic import AtomInitialization
from tests.app.solid.reference.helpers import _solid_system


def _charged_fe_system() -> SolidConfig:
    return SolidConfig(
        lattice=LatticeParams(a=4, b=4, c=4),
        atom_configs=[
            SolidAtomConfig(
                symbol="Fe",
                frac_coords=[0, 0, 0],
                initialization=AtomInitialization(local_charge=2),
            )
        ],
        pp="ph",
        s_z=1,
        total_charge=2,
    )


def test_qe_prepare_checks_upf_and_writes_solver_only_input(tmp_path):
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Fe.ccECP.upf").write_text(
        '<UPF z_valence="16.0" generated="test"/>',
        encoding="utf-8",
    )
    system = _charged_fe_system()
    solver = SolidQESolverConfig(
        pseudo_dir=str(pseudo_dir),
        prefix="test",
        pw_cutoff=40,
        rho_cutoff=160,
        pseudo_file={"Fe": "Fe.ccECP.upf"},
    )

    job_dir = tmp_path / "job"
    qe.SolidQEReferenceJob(system, solver).prepare(job_dir)
    qe_input = (job_dir / "qe.in").read_text()

    assert validate_upf_header(pseudo_dir / "Fe.ccECP.upf", 16) == 16
    assert "occupations = 'fixed'" in qe_input
    assert "smearing =" not in qe_input
    assert "degauss =" not in qe_input
    assert "nosym = .true." in qe_input
    assert "noinv = .true." in qe_input
    assert "tot_charge = 2" in qe_input
    assert "tot_magnetization = 2" in qe_input
    assert "Fe 26" in qe_input
    assert "Fe.ccECP.upf" in qe_input
    assert f"pseudo_dir = '{pseudo_dir}'" in qe_input
    assert "outdir = '.'" in qe_input


def test_qe_run_executes_pw_in_job_directory(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    calls = []
    reference_job = qe.SolidQEReferenceJob(_solid_system(), SolidQESolverConfig())

    class Process:
        def wait(self):
            return 0

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return Process()

    monkeypatch.setattr(job.subprocess, "Popen", fake_popen)

    reference_job.run(job_dir)

    assert calls[0][0] == (["pw.x", "-in", "qe.in"],)
    assert calls[0][1]["cwd"] == job_dir
    assert "stderr" not in calls[0][1]
    assert "stdout" not in calls[0][1]


def test_qe_prepare_writes_smearing_settings(tmp_path):
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Fe.ccECP.upf").write_text(
        '<UPF z_valence="16.0" generated="test"/>',
        encoding="utf-8",
    )

    job_dir = tmp_path / "job"
    qe.SolidQEReferenceJob(
        _solid_system(symbol="Fe", pp="ph"),
        SolidQESolverConfig(
            pseudo_dir=str(pseudo_dir),
            smearing="mv",
            degauss=0.02,
            pseudo_file={"Fe": "Fe.ccECP.upf"},
        ),
    ).prepare(job_dir)

    qe_input = (job_dir / "qe.in").read_text()
    assert "occupations = 'smearing'" in qe_input
    assert "smearing = 'mv'" in qe_input
    assert "degauss = 0.02" in qe_input


def test_qe_prepare_requires_explicit_pseudo_filename(tmp_path):
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Fe.ccECP.upf").write_text(
        '<UPF z_valence="16.0" generated="test"/>',
        encoding="utf-8",
    )
    system = _solid_system(symbol="Fe", pp="ph", s_z=1)

    with pytest.raises(ValueError, match=r"solver\.pseudo_file\.Fe"):
        qe.SolidQEReferenceJob(
            system,
            SolidQESolverConfig(pseudo_dir=str(pseudo_dir)),
        ).prepare(tmp_path / "missing")

    job_dir = tmp_path / "job"
    qe.SolidQEReferenceJob(
        system,
        SolidQESolverConfig(
            pseudo_dir=str(pseudo_dir),
            pseudo_file={"Fe": "Fe.ccECP.upf"},
        ),
    ).prepare(job_dir)

    assert "Fe.ccECP.upf" in (job_dir / "qe.in").read_text()


def test_validate_upf_header_reads_v1_z_valence(tmp_path):
    path = tmp_path / "Fe.ccECP.upf"
    path.write_text("16.0000      Z valence\n", encoding="utf-8")
    assert validate_upf_header(path, 16) == 16

    path.write_text("Z valence 16.0\n", encoding="utf-8")
    assert validate_upf_header(path, 16) == 16

    path.write_text("15.0000      Z valence\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 16"):
        validate_upf_header(path, 16)
