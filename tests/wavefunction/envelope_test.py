# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the envelope system."""

import jax
import jax.numpy as jnp
import pytest

from jaqmc.wavefunction.output.envelope import (
    Envelope,
    EnvelopeType,
)

TEST_KEY = jax.random.PRNGKey(42)


def make_test_inputs(n_elec: int, n_atoms: int, key: jax.Array = TEST_KEY):
    k1, k2 = jax.random.split(key)
    electrons = jax.random.normal(k1, (n_elec, 3))
    atoms = jax.random.normal(k2, (n_atoms, 3))
    ae_vectors = electrons[:, None, :] - atoms[None, :, :]
    r_ae = jnp.linalg.norm(ae_vectors, axis=-1)
    return ae_vectors, r_ae


class TestEnvelope:
    """Tests for Envelope Flax module."""

    @pytest.mark.parametrize("orbitals_spin_split", [False, True])
    @pytest.mark.parametrize(
        "envelope_type",
        [
            EnvelopeType.isotropic,
            EnvelopeType.abs_isotropic,
            EnvelopeType.diagonal,
        ],
    )
    def test_output_shape(self, envelope_type, orbitals_spin_split):
        nspins = (2, 1)
        ndets = 4
        n_atoms = 2
        n_elec = sum(nspins)

        module = Envelope(
            envelope_type=envelope_type,
            ndets=ndets,
            nspins=nspins,
            orbitals_spin_split=orbitals_spin_split,
        )
        ae_vectors, r_ae = make_test_inputs(n_elec, n_atoms)

        params = module.init(TEST_KEY, ae_vectors, r_ae)
        output = module.apply(params, ae_vectors, r_ae)

        assert output.shape == (ndets, n_elec, n_elec)
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.parametrize("orbitals_spin_split", [False, True])
    def test_null_envelope_returns_ones(self, orbitals_spin_split):
        nspins = (2, 1)
        ndets = 4
        n_atoms = 2
        n_elec = sum(nspins)

        module = Envelope(
            envelope_type=EnvelopeType.null,
            ndets=ndets,
            nspins=nspins,
            orbitals_spin_split=orbitals_spin_split,
        )
        ae_vectors, r_ae = make_test_inputs(n_elec, n_atoms)

        params = module.init(TEST_KEY, ae_vectors, r_ae)
        output = module.apply(params, ae_vectors, r_ae)

        assert output.shape == (ndets, n_elec, n_elec)
        assert jnp.allclose(output, 1.0)

    @pytest.mark.parametrize(
        "envelope_type",
        [
            EnvelopeType.isotropic,
            EnvelopeType.abs_isotropic,
            EnvelopeType.diagonal,
        ],
    )
    def test_envelope_varies_with_position(self, envelope_type):
        """Non-null envelopes produce different values for different positions."""
        nspins = (2, 1)
        ndets = 4
        n_atoms = 2
        n_elec = sum(nspins)

        module = Envelope(
            envelope_type=envelope_type,
            ndets=ndets,
            nspins=nspins,
            orbitals_spin_split=False,
        )
        ae_vectors_a, r_ae_a = make_test_inputs(n_elec, n_atoms, jax.random.PRNGKey(0))
        ae_vectors_b, r_ae_b = make_test_inputs(n_elec, n_atoms, jax.random.PRNGKey(99))

        params = module.init(TEST_KEY, ae_vectors_a, r_ae_a)
        out_a = module.apply(params, ae_vectors_a, r_ae_a)
        out_b = module.apply(params, ae_vectors_b, r_ae_b)

        # Different positions must produce different envelope values
        assert not jnp.allclose(out_a, out_b)
