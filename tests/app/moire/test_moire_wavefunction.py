# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.moire.data import MoireData
from jaqmc.app.moire.wavefunction import MoireWavefunction
from jaqmc.utils.supercell import (
    get_reciprocal_vectors,
    get_supercell_copies,
    get_supercell_kpts_in_first_bz,
)


def _diag_supercell(kmesh: tuple[int, int]) -> jnp.ndarray:
    return jnp.asarray([[kmesh[0], 0], [0, kmesh[1]]])


def _wavefunction_kwargs(
    *,
    nspins=(4, 0),
    primitive_lattice=None,
    simulation_lattice=None,
    kmesh=(2, 2),
    phase_mode="singlephase",
    k_com=(0.0, 0.0),
    ndets=1,
):
    primitive = (
        jnp.asarray([[1.13, 0.17], [0.29, 0.91]])
        if primitive_lattice is None
        else primitive_lattice
    )
    simulation = primitive if simulation_lattice is None else simulation_lattice
    klist = get_supercell_kpts_in_first_bz(_diag_supercell(kmesh), primitive)
    kwargs = {
        "nspins": nspins,
        "simulation_lattice": simulation,
        "primitive_lattice": primitive,
        "twist": jnp.asarray([0.0, 0.0]),
        "hidden_dims_single": [8],
        "hidden_dims_double": [4],
        "ndets": ndets,
        "phase_mode": phase_mode,
    }
    if phase_mode == "singlephase":
        original_nspins = tuple(n // (kmesh[0] * kmesh[1]) for n in nspins)
        kwargs["singlephase_klists"] = tuple(
            jnp.tile(klist, (nspin, 1)) for nspin in original_nspins
        )
    else:
        kwargs["k_com"] = k_com
        kwargs["multiphase_klist"] = klist
        kwargs["translation_vectors"] = get_supercell_copies(
            primitive, jnp.asarray([[kmesh[0], 0], [0, kmesh[1]]])
        )
    return kwargs


def _make_data(seed: int = 0) -> MoireData:
    key = jax.random.PRNGKey(seed)
    key_pos, key_spin = jax.random.split(key)
    return MoireData(
        positions=jax.random.uniform(key_pos, (4, 2)),
        spin_coords=jax.random.uniform(key_spin, (4,), maxval=2.0 * jnp.pi),
    )


def _twist_row_phase(
    positions: jnp.ndarray, twist: jnp.ndarray, lattice: jnp.ndarray
) -> jnp.ndarray:
    k_twist = twist @ get_reciprocal_vectors(lattice)
    return jnp.exp(1j * (positions @ k_twist))[..., None, :, None]


@pytest.mark.parametrize("phase_mode", ["singlephase", "multiphase"])
def test_twist_phase_is_applied_to_phase_modes(phase_mode):
    lattice = jnp.asarray([[1.13, 0.17], [0.29, 0.91]])
    twist = jnp.asarray([0.2, -0.125])
    kwargs = _wavefunction_kwargs(
        primitive_lattice=lattice,
        simulation_lattice=lattice,
        phase_mode=phase_mode,
    )
    wf_0 = MoireWavefunction(**kwargs)
    wf_twist = MoireWavefunction(**{**kwargs, "twist": twist})
    data = _make_data(seed=7)

    params = wf_0.init_params(data, jax.random.PRNGKey(6))

    orbital_0 = wf_0.orbitals(params, data)
    orbital_twist = wf_twist.orbitals(params, data)
    if phase_mode == "singlephase":
        expected = orbital_0 * _twist_row_phase(data.positions, twist, lattice)
    else:
        translations = kwargs["translation_vectors"]
        shifted_positions = data.positions[None, ...] - translations[:, None, :]
        phases = _twist_row_phase(shifted_positions, twist, lattice)
        by_translation = orbital_0.reshape(translations.shape[0], -1, 4, 4)
        expected = (by_translation * phases).reshape(orbital_0.shape)
    np.testing.assert_allclose(
        np.asarray(orbital_twist), np.asarray(expected), atol=1e-5
    )


def test_singlephase_gamma_centered_kmesh_for_square3():
    primitive = jnp.eye(2)
    klist = get_supercell_kpts_in_first_bz(_diag_supercell((3, 3)), primitive)
    expected_axis = 2.0 * np.pi * np.asarray([0.0, 1.0 / 3.0, -1.0 / 3.0])
    expected = np.asarray([[x, y] for x in expected_axis for y in expected_axis])
    np.testing.assert_allclose(np.asarray(klist), expected, atol=1e-6)


def test_singlephase_rejects_k_com():
    wf = MoireWavefunction(
        k_com=(0.0, 0.0),
        **_wavefunction_kwargs(
            primitive_lattice=jnp.eye(2),
            simulation_lattice=2.0 * jnp.eye(2),
        ),
    )

    with pytest.raises(
        ValueError, match="k_com must be None when phase_mode='singlephase'"
    ):
        wf.init_params(_make_data(), jax.random.PRNGKey(8))


def test_multiphase_rejects_missing_k_com():
    kwargs = _wavefunction_kwargs(phase_mode="multiphase")
    kwargs.pop("k_com")
    wf = MoireWavefunction(**kwargs)

    with pytest.raises(
        ValueError, match="k_com must be provided when phase_mode='multiphase'"
    ):
        wf.init_params(_make_data(), jax.random.PRNGKey(10))


@pytest.mark.parametrize("dependency", ["multiphase_klist", "translation_vectors"])
def test_multiphase_rejects_missing_runtime_dependency(dependency):
    kwargs = _wavefunction_kwargs(phase_mode="multiphase")
    kwargs[dependency] = None
    wf = MoireWavefunction(**kwargs)

    with pytest.raises(
        ValueError,
        match=rf"{dependency} must be provided when phase_mode='multiphase'",
    ):
        wf.init_params(_make_data(), jax.random.PRNGKey(10))


@pytest.mark.parametrize(
    ("nlayers", "exponents"),
    [(3, (1, 0, -1)), (4, (2, 1, -1, -2))],
)
def test_nlayers_amplitude_matches_symmetric_phases(nlayers, exponents):
    wf = MoireWavefunction(**{**_wavefunction_kwargs(), "nlayers": nlayers})
    data = _make_data()
    params = wf.init_params(data, jax.random.PRNGKey(20 + nlayers))

    layer_orbitals, _, _ = wf.layer_components(params, data)
    expected = sum(
        phi * np.exp(1j * m * np.asarray(data.spin_coords))[None, :, None]
        for phi, m in zip(layer_orbitals, exponents)
    )

    np.testing.assert_allclose(
        np.asarray(wf.orbitals(params, data)), np.asarray(expected), atol=1e-6
    )


def test_non_symmetric_supercell_kpts_fold_to_supercell_gamma():
    primitive = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    supercell_matrix = jnp.asarray([[2, 1], [0, 2]])

    kpts = get_supercell_kpts_in_first_bz(supercell_matrix, primitive)
    reciprocal = get_reciprocal_vectors(primitive)
    frac = kpts @ jnp.linalg.inv(reciprocal)
    supercell_frac = frac @ supercell_matrix.T

    assert kpts.shape == (4, 2)
    np.testing.assert_allclose(
        np.asarray(supercell_frac),
        np.round(np.asarray(supercell_frac)),
        atol=1e-6,
    )
