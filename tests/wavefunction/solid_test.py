# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

from jaqmc.geometry.pbc import DistanceType, SymmetryType
from jaqmc.wavefunction.input.atomic import SolidFeatures
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType

# Mock data
NDIM = 3
NELEC = 4
NATOM = 2
NDETS = 3
LATTICE = jnp.eye(NDIM) * 10.0  # Cubic lattice
PRIMITIVE_LATTICE = jnp.eye(NDIM) * 5.0


class TestSolidModules:
    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.key = jax.random.PRNGKey(0)
        self.electrons = jax.random.normal(self.key, (NELEC, NDIM))
        self.atoms = jax.random.normal(self.key, (NATOM, NDIM))

    @pytest.mark.parametrize("distance_type", [DistanceType.nu, DistanceType.tri])
    def test_solid_features_shape(self, distance_type):
        features_layer = SolidFeatures(
            simulation_lattice=LATTICE,
            primitive_lattice=PRIMITIVE_LATTICE,
            distance_type=distance_type,
            sym_type=SymmetryType.minimal,
        )

        variables = features_layer.init(self.key, self.electrons, self.atoms)
        embedding = features_layer.apply(variables, self.electrons, self.atoms)

        # Check AE features shape
        # AE features are reshaped to (nelec, natom * feature_dim)
        # feature_dim is (1 + ndim) for 'nu' and (1 + 2*ndim) for 'tri'
        ndim_feat = NDIM if distance_type == DistanceType.nu else 2 * NDIM
        feature_dim = 1 + ndim_feat
        expected_ae_dim = NATOM * feature_dim
        assert embedding["ae_features"].shape == (NELEC, expected_ae_dim)

        # Check EE features shape
        # EE features are (nelec, nelec, feature_dim)
        assert embedding["ee_features"].shape == (NELEC, NELEC, feature_dim)

        # Check envelope distances shape
        assert embedding["r_ae"].shape == (NELEC, NATOM)
        assert embedding["ae_vec"].shape == (NELEC, NATOM, ndim_feat)

        # Check values are finite
        assert jnp.all(jnp.isfinite(embedding["ae_features"]))
        assert jnp.all(jnp.isfinite(embedding["ee_features"]))
        assert jnp.all(jnp.isfinite(embedding["r_ae"]))
        assert jnp.all(jnp.isfinite(embedding["ae_vec"]))

    @pytest.mark.parametrize(
        "distance_type, envelope_type",
        [
            (DistanceType.nu, EnvelopeType.isotropic),
            (DistanceType.nu, EnvelopeType.abs_isotropic),
            (DistanceType.nu, EnvelopeType.diagonal),
            (DistanceType.tri, EnvelopeType.isotropic),
            (DistanceType.tri, EnvelopeType.abs_isotropic),
            (DistanceType.tri, EnvelopeType.diagonal),
        ],
    )
    def test_envelope_with_pbc_distances(self, distance_type, envelope_type):
        """Test Envelope with periodic distances from SolidFeatures."""
        nspins = (NELEC // 2, NELEC - NELEC // 2)
        envelope_layer = Envelope(
            envelope_type=envelope_type,
            ndets=NDETS,
            nspins=nspins,
            orbitals_spin_split=False,
        )

        # Use SolidFeatures to compute periodic distances
        features_layer = SolidFeatures(
            simulation_lattice=LATTICE,
            primitive_lattice=PRIMITIVE_LATTICE,
            distance_type=distance_type,
            sym_type=SymmetryType.minimal,
        )
        feat_vars = features_layer.init(self.key, self.electrons, self.atoms)
        embedding = features_layer.apply(feat_vars, self.electrons, self.atoms)

        variables = envelope_layer.init(
            self.key, embedding["ae_vec"], embedding["r_ae"]
        )
        envelope_val = envelope_layer.apply(
            variables, embedding["ae_vec"], embedding["r_ae"]
        )

        # Expected shape: (ndets, nelec, norbital)
        assert envelope_val.shape == (NDETS, NELEC, NELEC)

        # Check values are finite
        assert jnp.all(jnp.isfinite(envelope_val))

    @pytest.mark.parametrize("distance_type", [DistanceType.nu, DistanceType.tri])
    def test_solid_features_periodic(self, distance_type):
        """Features must be invariant under lattice translation (PBC)."""
        features_layer = SolidFeatures(
            simulation_lattice=LATTICE,
            primitive_lattice=PRIMITIVE_LATTICE,
            distance_type=distance_type,
            sym_type=SymmetryType.minimal,
        )

        variables = features_layer.init(self.key, self.electrons, self.atoms)
        embedding_orig = features_layer.apply(variables, self.electrons, self.atoms)

        # Shift all electrons by a lattice vector
        lattice_vector = LATTICE[0]  # [10, 0, 0]
        shifted = self.electrons + lattice_vector
        embedding_shifted = features_layer.apply(variables, shifted, self.atoms)

        for key in ("ae_features", "ee_features", "r_ae"):
            assert jnp.allclose(
                embedding_orig[key], embedding_shifted[key], atol=1e-5
            ), f"{key} not invariant under lattice translation"

    def test_envelope_grad_with_pbc_distances(self):
        """Test that we can take gradients through the envelope with PBC distances."""
        nspins = (NELEC // 2, NELEC - NELEC // 2)
        envelope_layer = Envelope(
            envelope_type=EnvelopeType.isotropic,
            ndets=NDETS,
            nspins=nspins,
            orbitals_spin_split=False,
        )

        features_layer = SolidFeatures(
            simulation_lattice=LATTICE,
            primitive_lattice=PRIMITIVE_LATTICE,
            distance_type=DistanceType.nu,
        )
        feat_vars = features_layer.init(self.key, self.electrons, self.atoms)

        embedding = features_layer.apply(feat_vars, self.electrons, self.atoms)
        env_vars = envelope_layer.init(self.key, embedding["ae_vec"], embedding["r_ae"])

        def loss_fn(electrons):
            emb = features_layer.apply(feat_vars, electrons, self.atoms)
            val = envelope_layer.apply(env_vars, emb["ae_vec"], emb["r_ae"])
            return jnp.sum(val**2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(self.electrons)

        assert grads.shape == self.electrons.shape
        assert jnp.all(jnp.isfinite(grads))
