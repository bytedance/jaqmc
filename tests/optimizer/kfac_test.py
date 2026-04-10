# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import flax.linen as nn
import jax
import jax.numpy as jnp
import kfac_jax
import pytest

from jaqmc.optimizer.kfac.tag_registration import make_graph_patterns
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType


def get_tags(func, params, *args):
    """Retrieves KFAC tags for the given function and parameters.

    Args:
        func: The function to inspect.
        params: The parameters of the function.
        *args: Additional arguments to the function.

    Returns:
        A PyTree of tags with the same structure as `params`.
        Each leaf contains the tag string (or None if no tag found).
    """

    def _wrapped_func(p, *a):
        out = func(p, *a)
        # We need to register a distribution for KFAC to trace the graph
        kfac_jax.register_normal_predictive_distribution(out.ravel()[:, None])
        return out

    tagged_func = kfac_jax.tag_graph_matcher.auto_register_tags(
        _wrapped_func, (params, *args), graph_patterns=make_graph_patterns()
    )
    labels = [
        "|".join(tagged_func._param_labels.get(p, ["Orphan"]))
        for p in tagged_func._func_graph.params_vars
    ]
    return jax.tree_util.tree_unflatten(tagged_func._func_graph.params_tree, labels)


def test_single_dense_layer_registration():
    model = nn.Dense(5)
    x = jnp.ones((1, 3))
    params = model.init(jax.random.PRNGKey(0), x)

    tags = get_tags(model.apply, params, x)
    dense_tags = tags["params"]

    assert "tag_variant=dense(0)" in dense_tags["kernel"]
    assert "match_type=flax_dense_with_bias" in dense_tags["kernel"]
    assert "tag_variant=dense(0)" in dense_tags["bias"]


def test_repeated_dense_layer_registration():
    model = nn.Dense(5)
    x = jnp.ones((1, 2, 3))
    params = model.init(jax.random.PRNGKey(0), x)

    tags = get_tags(model.apply, params, x)
    dense_tags = tags["params"]

    assert "tag_variant=repeated_dense(0)" in dense_tags["kernel"]
    assert "match_type=repeated[1]_dense_with_bias_with_reshape" in dense_tags["kernel"]
    assert "tag_variant=repeated_dense(0)" in dense_tags["bias"]


@pytest.mark.parametrize(
    "envelope_type",
    [EnvelopeType.isotropic, EnvelopeType.abs_isotropic, EnvelopeType.diagonal],
)
@pytest.mark.parametrize(
    "orbitals_spin_split,env_keys",
    [
        (False, ["_env"]),
        (True, ["_env_up", "_env_down"]),
    ],
)
def test_envelope_layer_registration(envelope_type, orbitals_spin_split, env_keys):
    n_elec = 3
    n_atoms = 2
    model = Envelope(
        envelope_type=envelope_type,
        ndets=4,
        nspins=(1, 2),
        orbitals_spin_split=orbitals_spin_split,
    )
    # Use 3D ae_vectors for compatibility with all envelope types
    ae_vectors = jnp.zeros((n_elec, n_atoms, 3))
    r_ae = jnp.zeros((n_elec, n_atoms))
    params = model.init(jax.random.PRNGKey(0), ae_vectors, r_ae)

    tags = get_tags(model.apply, params, ae_vectors, r_ae)
    for env_key in env_keys:
        assert "tag_variant=scale_and_shift" in tags["params"][env_key]["pi"]
        assert "tag_variant=scale_and_shift" in tags["params"][env_key]["sigma"]


@pytest.mark.parametrize(
    "envelope_type",
    [EnvelopeType.isotropic, EnvelopeType.abs_isotropic],
)
@pytest.mark.parametrize("ndim_feat", [3, 6])
def test_envelope_layer_registration_variable_dim(envelope_type, ndim_feat):
    """Test K-FAC tag registration for Envelope with variable feature dimensions.

    This tests that Envelope works with different input dimensions (e.g., 3 for
    nu_distance, 6 for tri_distance in PBC).

    Note: DIAGONAL envelope uses einsum which falls back to generic K-FAC,
    so we only test isotropic variants here.
    """
    n_elec = 3
    n_atoms = 2
    ndets = 4

    model = Envelope(
        envelope_type=envelope_type,
        ndets=ndets,
        nspins=(1, 2),
        orbitals_spin_split=False,
    )
    ae_vectors = jnp.zeros((n_elec, n_atoms, ndim_feat))
    r_ae = jnp.zeros((n_elec, n_atoms))
    params = model.init(jax.random.PRNGKey(0), ae_vectors, r_ae)

    tags = get_tags(model.apply, params, ae_vectors, r_ae)
    assert "tag_variant=scale_and_shift" in tags["params"]["_env"]["pi"]
    assert "tag_variant=scale_and_shift" in tags["params"]["_env"]["sigma"]
