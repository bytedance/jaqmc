# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="arg-type, call-overload, var-annotated"

import jax
import numpy as np
import pytest
from jaqmc_contrib_lit.pool import (
    _add_source_sums,
    _signed_lit_jackknife_pseudovalues,
)
from jaqmc_contrib_lit.response import (
    LITResponseFermiNet,
    local_action_ratio,
    parity_project_log_amplitude,
    restore_params_from_checkpoint,
    source_sampled_stats,
    source_sampled_sums,
    stats_from_source_sums,
)
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.data import BatchedData
from jaqmc.utils.checkpoint import NumPyCheckpointManager


def _hydrogen_1s_logpsi(params, data: MoleculeData):
    del params
    return -jnp.linalg.norm(data.electrons[0] - data.atoms[0])


def _hydrogen_2pz_logpsi(params, data: MoleculeData):
    del params
    rel = data.electrons[0] - data.atoms[0]
    sign_phase = jnp.where(rel[2] < 0.0, jnp.pi, 0.0)
    return jnp.log(jnp.abs(rel[2])) - 0.5 * jnp.linalg.norm(rel) + 1j * sign_phase


def _scaled_hydrogen_2pz_logpsi(params, data: MoleculeData):
    return params["scale"] * _hydrogen_2pz_logpsi({}, data)


def _h_batch() -> BatchedData[MoleculeData]:
    return BatchedData(
        data=MoleculeData(
            electrons=jnp.asarray(
                [
                    [[0.2, 0.1, 0.8]],
                    [[-0.3, 0.0, 0.7]],
                    [[0.1, -0.2, 0.9]],
                    [[-0.1, 0.2, 0.6]],
                    [[0.2, -0.2, 0.4]],
                ],
                dtype=jnp.float32,
            ),
            atoms=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
            charges=jnp.asarray([1.0], dtype=jnp.float32),
        ),
        fields_with_batch=["electrons"],
    )


def test_hydrogen_2pz_response_local_action_is_exact():
    point = MoleculeData(
        electrons=jnp.asarray([[0.3, -0.2, 0.7]], dtype=jnp.float32),
        atoms=jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charges=jnp.asarray([1.0], dtype=jnp.float32),
    )

    action, response_ratio, local_energy = local_action_ratio(
        _hydrogen_2pz_logpsi,
        {},
        _hydrogen_1s_logpsi,
        {},
        point,
        ground_energy=-0.5,
        omega=0.4,
        eta=0.02,
    )

    expected = (0.375 - 0.4 - 0.02j) * response_ratio
    np.testing.assert_allclose(np.asarray(action), np.asarray(expected), rtol=2e-6)
    np.testing.assert_allclose(float(jnp.real(local_energy)), -0.125, rtol=2e-6)


def test_full_response_source_sampled_hydrogen_stats_are_finite():
    batch = _h_batch()

    stats = source_sampled_stats(
        _hydrogen_2pz_logpsi,
        {},
        _hydrogen_1s_logpsi,
        {},
        batch,
        axis=2,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=-0.5,
        omega=0.375,
        eta=0.02,
        source_floor=1e-4,
    )

    assert np.isfinite(float(stats.loss))
    assert 0.0 <= float(stats.fidelity) <= 1.0
    assert np.isfinite(float(stats.reverse_kl))
    assert float(stats.reverse_kl) >= 0.0
    assert np.isfinite(float(stats.signed_lit))
    np.testing.assert_allclose(
        float(stats.broadened),
        0.02 * float(stats.signed_lit) / np.pi,
    )
    assert float(stats.reweight_ess) > 0.0
    assert 0.0 < float(stats.reweight_ess_fraction) <= 1.0
    assert np.isfinite(float(stats.error_d))
    assert np.isfinite(float(stats.equation_relative_residual))
    assert float(stats.equation_relative_residual) >= 0.0
    np.testing.assert_allclose(float(stats.invalid_sample_fraction), 0.0)
    np.testing.assert_allclose(float(stats.source_norm), 1.0)


def test_signed_lit_jackknife_uses_leave_one_out_ratio_of_sums():
    batch = _h_batch()

    def sliced(start: int, stop: int) -> BatchedData[MoleculeData]:
        return BatchedData(
            data=MoleculeData(
                electrons=batch.data.electrons[start:stop],
                atoms=batch.data.atoms,
                charges=batch.data.charges,
            ),
            fields_with_batch=["electrons"],
        )

    common = dict(
        response_apply=_hydrogen_2pz_logpsi,
        response_params={},
        ground_logpsi=_hydrogen_1s_logpsi,
        ground_params={},
        axis=2,
        source_center=0.0,
        ground_energy=-0.5,
        omega=0.375,
        eta=0.02,
        source_floor=1e-4,
    )
    block_sums = tuple(
        source_sampled_sums(
            batched_data=block,
            **common,
        )
        for block in (sliced(0, 2), sliced(2, 5))
    )
    full_sums = _add_source_sums(*block_sums)
    full_stats = stats_from_source_sums(
        full_sums,
        source_norm=1.0,
        omega=0.375,
        eta=0.02,
    )

    actual = _signed_lit_jackknife_pseudovalues(
        full_stats,
        block_sums,
        source_norm=1.0,
        omega=0.375,
        eta=0.02,
    )
    leave_one_out = np.asarray(
        [
            float(
                stats_from_source_sums(
                    block_sums[1 - index],
                    source_norm=1.0,
                    omega=0.375,
                    eta=0.02,
                ).signed_lit
            )
            for index in range(2)
        ]
    )
    expected = 2.0 * float(full_stats.signed_lit) - leave_one_out
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_response_ferminet_returns_finite_complex_log_amplitude():
    batch = _h_batch()
    data = MoleculeData(
        electrons=batch.data.electrons[0],
        atoms=batch.data.atoms,
        charges=batch.data.charges,
    )
    response = LITResponseFermiNet(
        nspins=(1, 0),
        ndets=2,
        hidden_dims_single=(4,),
        hidden_dims_double=(2,),
    )
    params = response.init(jax.random.PRNGKey(1), batch.unbatched_example())
    value = response.apply(params, data)

    assert jnp.iscomplexobj(value)
    assert np.isfinite(np.asarray(value)).all()


@pytest.mark.parametrize("parity", [-1, 1])
def test_parity_projection_survives_extreme_common_log_scales(parity):
    base = jnp.asarray(0.4 + 0.3j, dtype=jnp.complex64)
    inverted = jnp.asarray(-0.7 - 0.2j, dtype=jnp.complex64)
    common_shifts = jnp.asarray([1.0e3, -1.0e3], dtype=jnp.float32)

    projected = parity_project_log_amplitude(
        base + common_shifts,
        inverted + common_shifts,
        parity,
    )
    scaled_amplitudes = jnp.exp(projected - common_shifts)
    expected = 0.5 * (jnp.exp(base) + parity * jnp.exp(inverted))

    assert np.all(np.isfinite(np.asarray(projected)))
    np.testing.assert_allclose(
        np.asarray(scaled_amplitudes),
        np.broadcast_to(np.asarray(expected), (2,)),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize("parity", [-1, 1])
def test_parity_projection_has_finite_gradient_and_hessian(parity):
    def objective(coordinates):
        x, y = coordinates
        first = 0.3 * x - 0.2 * y**2 + 1j * (0.4 * y + 0.1 * x**2)
        second = -0.7 + 0.1 * x * y + 1j * (-0.2 * x + 0.3 * y)
        projected = parity_project_log_amplitude(first, second, parity)
        return jnp.real(projected) + 0.25 * jnp.imag(projected)

    coordinates = jnp.asarray([0.4, -0.2], dtype=jnp.float32)
    assert np.all(np.isfinite(np.asarray(jax.grad(objective)(coordinates))))
    assert np.all(np.isfinite(np.asarray(jax.hessian(objective)(coordinates))))


def test_ground_stage_checkpoint_restores_parameter_subtree(tmp_path):
    fallback = {"params": {"w": jnp.asarray([0.0, 0.0])}}
    restored = {"params": {"w": jnp.asarray([1.0, 2.0])}}
    NumPyCheckpointManager(tmp_path, prefix="train").save(
        7,
        {"params": restored},
    )

    step, params = restore_params_from_checkpoint(tmp_path, fallback)

    assert step == 7
    np.testing.assert_allclose(np.asarray(params["params"]["w"]), [1.0, 2.0])
