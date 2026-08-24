# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.data import Data
from jaqmc.wavefunction.determinant_state import (
    DeterminantStateWavefunction,
    SubspaceSpec,
)


class ToyData(Data):
    electrons: jax.Array
    fixed: jax.Array


class ToyWavefunction:
    def init_params(self, data, rngs):
        del data
        return {"slope": jax.random.uniform(rngs, (), minval=0.5, maxval=1.5)}

    def phase_logpsi(self, params, data):
        value = 1 + params["slope"] * data.electrons.sum()
        return jnp.sign(value), jnp.log(jnp.abs(value))


def test_m1_reduces_to_component_amplitude_and_gradient():
    wf = DeterminantStateWavefunction(ToyWavefunction(), SubspaceSpec(1))
    data = ToyData(electrons=jnp.array([[[0.2], [0.3]]]), fixed=jnp.array(1.0))
    params = {"slope": jnp.array([0.7])}
    physical = ToyData(electrons=data.electrons[0], fixed=data.fixed)
    _, expected_logabs = ToyWavefunction().phase_logpsi(
        {"slope": params["slope"][0]}, physical
    )

    actual = wf.logpsi(params, data)
    actual_grad = jax.grad(lambda p: wf.logpsi(p, data).real)(params)
    expected_grad = jax.grad(
        lambda p: ToyWavefunction().phase_logpsi(p, physical)[1]
    )({"slope": params["slope"][0]})

    np.testing.assert_allclose(actual.real, expected_logabs, rtol=1e-6)
    np.testing.assert_allclose(actual_grad["slope"][0], expected_grad["slope"])


def test_m1_initialization_uses_the_native_key_exactly():
    base = ToyWavefunction()
    wf = DeterminantStateWavefunction(base, SubspaceSpec(1))
    physical = ToyData(electrons=jnp.array([[0.2]]), fixed=jnp.array(1.0))
    replica = ToyData(
        electrons=physical.electrons[None, ...], fixed=physical.fixed
    )
    key = jax.random.key(11)

    native = base.init_params(physical, key)
    determinant = wf.init_params(replica, key)

    np.testing.assert_array_equal(determinant["slope"][0], native["slope"])


def test_component_matrix_matches_explicit_loop_and_permutation_invariance():
    wf = DeterminantStateWavefunction(ToyWavefunction(), SubspaceSpec(2))
    data = ToyData(
        electrons=jnp.array([[[0.1]], [[0.8]]]), fixed=jnp.array(1.0)
    )
    params = {"slope": jnp.array([0.4, 1.2])}

    logs = wf.component_logpsi_matrix(params, data)
    expected = np.empty((2, 2), dtype=np.complex64)
    for r in range(2):
        for s in range(2):
            phase, logabs = ToyWavefunction().phase_logpsi(
                {"slope": params["slope"][s]},
                ToyData(electrons=data.electrons[r], fixed=data.fixed),
            )
            expected[r, s] = np.asarray(logabs + 1j * jnp.angle(phase + 0j))
    np.testing.assert_allclose(logs, expected, rtol=1e-6)

    original = wf.logpsi(params, data)
    permuted_params = {"slope": params["slope"][::-1]}
    permuted = wf.logpsi(permuted_params, data)
    np.testing.assert_allclose(original.real, permuted.real, atol=1e-6)
    np.testing.assert_allclose(
        jnp.exp(2 * original.real), jnp.exp(2 * permuted.real), rtol=1e-6
    )
