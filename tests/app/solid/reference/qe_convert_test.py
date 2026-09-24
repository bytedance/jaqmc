# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.solid.reference import PlaneWaveSolidReference, SolidReference, qe
from tests.app.solid.reference.helpers import (
    _solid_system,
    _write_qe_wfc,
    _write_qe_xml,
)


@pytest.mark.parametrize(
    ("xml_options", "match"),
    [
        ({"symmetry_flags": (False, True)}, "nosym=.true."),
        ({"symmetry_flags": (True, False)}, "noinv=.true."),
        ({"symmetry_flags": None}, "<symmetry_flags> is missing"),
        ({"include_input": False}, "<input> is missing"),
    ],
    ids=["nosym", "noinv", "missing-symmetry-flags", "missing-input"],
)
def test_qe_convert_rejects_unproven_mesh(tmp_path, xml_options, match):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(save_dir / "data-file-schema.xml", [("0 0 0", "2")], **xml_options)
    _write_qe_wfc(save_dir / "wfc1.hdf5", [[0, 0, 0]], [[1.0, 0.0]])

    with pytest.raises(ValueError, match=match):
        qe.convert_save_directory(save_dir, tmp_path / "reference.npz")


def test_qe_convert_accepts_fixed_integer_occupations(tmp_path):
    system = _solid_system(symbol="Fe", pp="ph")
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", " ".join(["2"] * 8 + ["0"] * 8))],
        symbol="Fe",
    )
    coefficients = np.arange(64, dtype=float).reshape(16, 4)
    _write_qe_wfc(
        save_dir / "wfc1.hdf5",
        [[0, 0, 0], [1, 0, 0]],
        coefficients,
    )

    output_path = tmp_path / "reference.npz"
    reference = qe.convert_save_directory(save_dir, output_path)
    loaded = cast(PlaneWaveSolidReference, SolidReference.load(output_path))

    assert output_path.is_file()
    assert loaded.symbols == reference.symbols == ("Fe",)
    np.testing.assert_allclose(loaded.lattice, np.eye(3) * 4)
    assert loaded.nspins == reference.nspins == system.electron_spins == (8, 8)
    assert loaded.alpha_counts.tolist() == [8]
    assert loaded.beta_counts.tolist() == [8]
    expected_coefficients = (coefficients[:8, 0::2] + 1j * coefficients[:8, 1::2]).T
    np.testing.assert_allclose(loaded.alpha_coeffs, expected_coefficients)
    np.testing.assert_allclose(loaded.beta_coeffs, expected_coefficients)
    np.testing.assert_allclose(loaded.kpoints, reference.kpoints)
    np.testing.assert_allclose(loaded.atom_coords, reference.atom_coords)
    np.testing.assert_allclose(loaded.alpha_coeffs, reference.alpha_coeffs)
    np.testing.assert_allclose(loaded.beta_coeffs, reference.beta_coeffs)
    np.testing.assert_allclose(loaded.alpha_g_vectors, reference.alpha_g_vectors)
    np.testing.assert_allclose(loaded.beta_g_vectors, reference.beta_g_vectors)
    assert loaded.alpha_g_vectors.shape == (1, 2, 3)
    assert loaded.beta_g_vectors.shape == (1, 2, 3)


def test_qe_convert_halves_unpolarized_smeared_occupations(tmp_path, caplog):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "1.7 1.3 0.8 0.2")],
        eigenvalues=["3 1 0 2"],
        nelec=4,
    )
    _write_qe_wfc(
        save_dir / "wfc1.hdf5",
        [[0, 0, 0]],
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
    )

    reference = qe.convert_save_directory(save_dir, tmp_path / "reference.npz")

    assert reference.alpha_counts.tolist() == [2]
    assert reference.beta_counts.tolist() == [2]
    np.testing.assert_allclose(reference.alpha_coeffs, [[3.0, 2.0]])
    np.testing.assert_allclose(reference.beta_coeffs, [[3.0, 2.0]])
    assert "fractional occupations were converted" in caplog.text


def test_qe_convert_integerizes_fractional_occupations_by_energy(tmp_path, caplog):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "1.5 0.5 0")],
        eigenvalues=["1 0 2"],
        nelec=2,
    )
    _write_qe_wfc(
        save_dir / "wfc1.hdf5",
        [[0, 0, 0]],
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
    )

    reference = qe.convert_save_directory(save_dir, tmp_path / "reference.npz")

    assert reference.alpha_counts.tolist() == [1]
    assert reference.beta_counts.tolist() == [1]
    np.testing.assert_allclose(reference.alpha_coeffs, [[2.0]])
    np.testing.assert_allclose(reference.beta_coeffs, [[2.0]])
    assert "fractional occupations were converted" in caplog.text


def test_qe_convert_selects_fractional_bands_globally_across_kpoints(tmp_path):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "1.5 0.5 0"), ("0.5 0 0", "1.5 0.5 0")],
        eigenvalues=["0 1 2", "10 11 12"],
        nelec=2,
    )
    _write_qe_wfc(
        save_dir / "wfc1.hdf5", [[0, 0, 0]], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    _write_qe_wfc(
        save_dir / "wfc2.hdf5",
        [[0, 0, 0]],
        [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
        xk=(np.pi / 4, 0.0, 0.0),
        ik=2,
    )

    reference = qe.convert_save_directory(save_dir, tmp_path / "reference.npz")

    assert reference.alpha_counts.tolist() == [2, 0]
    assert reference.beta_counts.tolist() == [2, 0]
    np.testing.assert_allclose(reference.alpha_coeffs, [[1.0, 2.0]])


def test_qe_convert_rejects_unpolarized_odd_electron_count(tmp_path):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "1 0")],
        nelec=1,
    )
    _write_qe_wfc(
        save_dir / "wfc1.hdf5",
        [[0, 0, 0]],
        [[1.0, 0.0], [2.0, 0.0]],
    )

    with pytest.raises(ValueError, match="unpolarized spin channel"):
        qe.convert_save_directory(save_dir, tmp_path / "reference.npz")


def test_qe_convert_validates_nonuniform_kpoint_weights(tmp_path):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "2"), ("0.5 0 0", "2")],
        nelec=2.0,
        weights=[0.25, 0.75],
    )
    _write_qe_wfc(
        save_dir / "wfc1.hdf5",
        [[0, 0, 0]],
        [[1.0, 0.0]],
    )
    _write_qe_wfc(
        save_dir / "wfc2.hdf5",
        [[0, 0, 0]],
        [[1.0, 0.0]],
        xk=(np.pi / 4, 0.0, 0.0),
        ik=2,
    )

    reference = qe.convert_save_directory(save_dir, tmp_path / "reference.npz")

    assert reference.alpha_counts.tolist() == [1, 1]
    assert reference.beta_counts.tolist() == [1, 1]


def test_qe_convert_matches_files_by_index_not_filename(tmp_path):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "2"), ("0.5 0 0", "2")],
    )
    _write_qe_wfc(
        save_dir / "wfc1.hdf5",
        [[1, 0, 0]],
        [[3.0, 0.0]],
        xk=(np.pi / 4, 0.0, 0.0),
        ik=2,
    )
    _write_qe_wfc(
        save_dir / "wfc2.hdf5",
        [[0, 0, 0], [1, 0, 0]],
        [[1.0, 0.0, 2.0, 0.0]],
    )

    reference = qe.convert_save_directory(save_dir, tmp_path / "reference.npz")

    assert reference.alpha_counts.tolist() == [1, 1]
    assert reference.beta_counts.tolist() == [1, 1]
    np.testing.assert_allclose(reference.alpha_coeffs, [[1, 3], [2, 0]])
    assert reference.alpha_g_vectors.shape == (2, 2, 3)
    assert reference.beta_g_vectors.shape == (2, 2, 3)
    reciprocal = 2 * np.pi / 4
    np.testing.assert_allclose(
        reference.alpha_g_vectors[0],
        [[0.0, 0.0, 0.0], [reciprocal, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        reference.alpha_g_vectors[1],
        [[reciprocal, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )


@pytest.mark.parametrize(
    ("ik", "xk"),
    [
        (1, (np.pi / 4, 0.0, 0.0)),
        (2, (0.0, 0.0, 0.0)),
    ],
    ids=["duplicate-index", "inconsistent-kpoint"],
)
def test_qe_convert_rejects_inconsistent_wavefunction_identity(tmp_path, ik, xk):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "2"), ("0.5 0 0", "2")],
    )
    _write_qe_wfc(save_dir / "wfc1.hdf5", [[0, 0, 0]], [[1.0, 0.0]])
    _write_qe_wfc(
        save_dir / "wfc2.hdf5",
        [[0, 0, 0]],
        [[1.0, 0.0]],
        xk=xk,
        ik=ik,
    )

    with pytest.raises(ValueError, match="k-points or spin labels"):
        qe.convert_save_directory(save_dir, tmp_path / "reference.npz")


def test_qe_convert_reads_qe7_xml_occupations_and_real_evc(tmp_path):
    system = _solid_system(symbol="Fe", pp="ph")
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", " ".join(["2"] * 8 + ["0"] * 8))],
        symbol="Fe",
    )
    evc = np.zeros((16, 4))
    evc[:, 0::2] = 1.0
    _write_qe_wfc(
        save_dir / "wfc1.hdf5",
        np.asarray([[0, 0, 0], [1, 0, 0]]),
        evc,
    )

    reference = qe.convert_save_directory(save_dir, tmp_path / "reference.npz")

    assert reference.nspins == system.electron_spins == (8, 8)
    assert reference.alpha_coeffs.shape == (2, 8)
    np.testing.assert_allclose(reference.alpha_coeffs, 1.0 + 0.0j)


def test_qe_convert_reads_spin_polarized_ispin_and_paired_kpoints(tmp_path):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    miller = np.asarray([[0, 0, 0], [1, 0, 0]])
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "1 1"), ("0 0 0", "0 0")],
        lsda=True,
    )
    _write_qe_wfc(
        save_dir / "wfcup1.hdf5",
        miller,
        np.ones((2, 4)),
        ispin=1,
    )
    _write_qe_wfc(
        save_dir / "wfcdw1.hdf5",
        miller,
        np.ones((2, 4)),
        ispin=2,
    )

    reference = qe.convert_save_directory(save_dir, tmp_path / "reference.npz")

    assert reference.alpha_counts.tolist() == [2]
    assert reference.beta_counts.tolist() == [0]
    assert reference.beta_coeffs.shape == (2, 0)
    assert reference.alpha_g_vectors.shape == (1, 2, 3)
    assert reference.beta_g_vectors.shape == (1, 2, 3)
    np.testing.assert_allclose(reference.kpoints, np.zeros((1, 3)))


def test_qe_convert_reads_packed_spin_polarized_kpoints(tmp_path):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "1 1 0"), ("0.5 0 0", "0 1 1")],
        lsda=True,
        packed_spin=True,
        spin_bands=(2, 1),
    )
    miller = np.asarray([[0, 0, 0]])
    _write_qe_wfc(
        save_dir / "wfcup1.hdf5",
        miller,
        [[1.0, 0.0], [2.0, 0.0]],
        ispin=1,
    )
    _write_qe_wfc(
        save_dir / "wfcup2.hdf5",
        miller,
        [[3.0, 0.0], [4.0, 0.0]],
        xk=(np.pi / 4, 0.0, 0.0),
        ispin=1,
        ik=2,
    )
    _write_qe_wfc(
        save_dir / "wfcdw1.hdf5",
        miller,
        [[5.0, 0.0]],
        ispin=2,
    )
    _write_qe_wfc(
        save_dir / "wfcdw2.hdf5",
        miller,
        [[6.0, 0.0]],
        xk=(np.pi / 4, 0.0, 0.0),
        ispin=2,
        ik=2,
    )

    reference = qe.convert_save_directory(save_dir, tmp_path / "reference.npz")

    assert reference.alpha_counts.tolist() == [2, 1]
    assert reference.beta_counts.tolist() == [0, 1]
    np.testing.assert_allclose(
        reference.alpha_coeffs,
        [[1.0, 2.0, 4.0]],
    )
    np.testing.assert_allclose(reference.beta_coeffs, [[6.0]])
    np.testing.assert_allclose(
        reference.kpoints,
        [[0.0, 0.0, 0.0], [np.pi / 4, 0.0, 0.0]],
    )


def test_qe_convert_restores_gamma_only_plane_waves(tmp_path):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "2")],
        lattice=1.0,
    )
    _write_qe_wfc(
        save_dir / "wfc1.hdf5",
        [[0, 0, 0], [1, 0, 0]],
        [[1.0, 0.0, 1.0, 0.0]],
        gamma_only=True,
    )

    reference = qe.convert_save_directory(save_dir, tmp_path / "reference.npz")
    alpha, beta = reference.eval_orbitals(
        jnp.array([[0.125, 0.0, 0.0], [0.375, 0.0, 0.0]]),
        (1, 1),
    )

    np.testing.assert_allclose(alpha, [[1 + np.sqrt(2)]], atol=1e-6)
    np.testing.assert_allclose(beta, [[1 - np.sqrt(2)]], atol=1e-6)


def test_qe_convert_evaluates_non_gamma_bloch_orbital(tmp_path):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0.5 0 0", "2")],
        lattice=1.0,
    )
    _write_qe_wfc(
        save_dir / "wfc1.hdf5",
        [[1, 0, 0]],
        [[1.0, 0.0]],
        xk=(np.pi, 0.0, 0.0),
    )

    reference = qe.convert_save_directory(save_dir, tmp_path / "reference.npz")
    alpha, beta = reference.eval_orbitals(
        jnp.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        (1, 1),
    )

    np.testing.assert_allclose(alpha, [[-1j]], atol=1e-6)
    np.testing.assert_allclose(beta, [[1.0]], atol=1e-6)


def test_qe_convert_rejects_inconsistent_electron_count(tmp_path):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "2")],
        nelec=1,
    )
    _write_qe_wfc(save_dir / "wfc1.hdf5", [[0, 0, 0]], [[1.0, 0.0]])

    with pytest.raises(ValueError, match="reported electron count"):
        qe.convert_save_directory(save_dir, tmp_path / "reference.npz")


def test_qe_convert_rejects_unconverged_scf(tmp_path):
    save_dir = tmp_path / "test.save"
    save_dir.mkdir()
    _write_qe_xml(
        save_dir / "data-file-schema.xml",
        [("0 0 0", "2")],
        converged=False,
    )
    _write_qe_wfc(save_dir / "wfc1.hdf5", [[0, 0, 0]], [[1.0, 0.0]])

    with pytest.raises(ValueError, match="did not converge"):
        qe.convert_save_directory(save_dir, tmp_path / "reference.npz")
