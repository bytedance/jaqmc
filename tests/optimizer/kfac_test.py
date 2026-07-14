# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import kfac_jax
import pytest

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.wavefunction.ferminet import FermiNetWavefunction
from jaqmc.data import BatchedData
from jaqmc.optimizer.kfac.complex_support import patch_block_diagonal_curvature
from jaqmc.optimizer.kfac.curvature_blocks import make_tag_to_block_ctor
from jaqmc.optimizer.kfac.tag_registration import make_graph_patterns
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType


def get_tag_tree(func, params, *args) -> Any:
    """Return the parameter pytree with tag labels at each leaf.

    Couples to kfac_jax ``auto_register_tags`` internals (``_param_labels``,
    ``_func_graph``) because no public tag-introspection API exists.
    """

    def value_fn(p, *a):
        out = func(p, *a)
        kfac_jax.register_normal_predictive_distribution(jnp.reshape(out, (-1, 1)))
        return out

    tagged_func = kfac_jax.tag_graph_matcher.auto_register_tags(
        value_fn, (params, *args), graph_patterns=make_graph_patterns()
    )
    labels = [
        "|".join(tagged_func._param_labels.get(p, ["Orphan"]))
        for p in tagged_func._func_graph.params_vars
    ]
    tag_tree = jax.tree_util.tree_unflatten(tagged_func._func_graph.params_tree, labels)
    return tag_tree


def assert_tag(tag: str, *, variant: str, match_type: str) -> None:
    assert tag != "Orphan"
    assert "Auto[generic(" not in tag
    assert f"tag_variant={variant}" in tag
    assert f"match_type={match_type}" in tag


def assert_params_f32(params: Any) -> None:
    assert all(
        isinstance(param, jax.Array) and param.dtype == jnp.float32
        for param in jax.tree_util.tree_leaves(params)
    )


def activation_dtype(x64_mode: bool | None) -> jnp.dtype:
    return jnp.float64 if x64_mode else jnp.float32


def test_single_dense_layer_registration():
    model = nn.Dense(5)
    x = jnp.ones((1, 3))
    params = model.init(jax.random.PRNGKey(0), x)
    tag_tree = get_tag_tree(model.apply, params, x)

    assert_tag(
        tag_tree["params"]["kernel"],
        variant="dense",
        match_type="flax_dense_with_bias",
    )
    assert "tag_variant=dense(0)" in tag_tree["params"]["bias"]


@pytest.mark.x64_modes
def test_vmapped_repeated_dense_layer_registration_mixed_x64_activations(x64_mode):
    model = nn.Dense(5)
    act_dtype = activation_dtype(x64_mode)
    x = jnp.ones((2, 3), dtype=act_dtype)
    params = model.init(jax.random.PRNGKey(0), x)
    batch = jnp.ones((4, *x.shape), dtype=act_dtype)
    assert_params_f32(params)

    def apply_batch(p, xb):
        return jax.vmap(model.apply, in_axes=(None, 0))(p, xb)

    tag_tree = get_tag_tree(apply_batch, params, batch)

    assert params["params"]["kernel"].dtype == jnp.float32
    assert batch.dtype == act_dtype
    assert_tag(
        tag_tree["params"]["kernel"],
        variant="repeated_dense",
        match_type="repeated[1]_flax_dense_with_bias",
    )
    assert_tag(
        tag_tree["params"]["bias"],
        variant="repeated_dense",
        match_type="repeated[1]_flax_dense_with_bias",
    )


@pytest.mark.x64_modes
def test_scalar_scale_registration_mixed_x64_activation(x64_mode):
    params = {"alpha": jnp.asarray(2.0, dtype=jnp.float32)}

    def apply_scale(p, x):
        return p["alpha"] * x

    x = jnp.asarray(3.0, dtype=activation_dtype(x64_mode))
    tag_tree = get_tag_tree(apply_scale, params, x)

    assert x.dtype == activation_dtype(x64_mode)
    assert_tag(
        tag_tree["alpha"],
        variant="scale_and_shift",
        match_type="scale_only_broadcast_0",
    )


@pytest.mark.x64_modes
def test_cast_scale_only_broadcast_registration(x64_mode):
    params = {"scale": jnp.asarray(jnp.arange(13.0), dtype=jnp.float32)}

    def apply_scale(p, x):
        return x * p["scale"].astype(x.dtype)

    x = jnp.ones((2, 13), dtype=activation_dtype(x64_mode))
    tag_tree = get_tag_tree(apply_scale, params, x)

    assert x.dtype == activation_dtype(x64_mode)
    assert_tag(
        tag_tree["scale"],
        variant="scale_and_shift",
        match_type="scale_only_broadcast_2",
    )


class KfacPatternComposite(nn.Module):
    """Compact graph exercising JaQMC flax dense and scalar-scale patterns."""

    hidden: int = 5
    out_features: tuple[int, int] = (4, 5)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        flat = x.reshape(x.shape[0], -1)
        h = nn.Dense(self.hidden)(flat)
        h = nn.Dense(self.hidden, use_bias=False)(h)
        h2 = nn.Dense(self.hidden)(x)
        h3 = nn.DenseGeneral(self.out_features, axis=(-2, -1), use_bias=True)(x)
        alpha = self.param("alpha", nn.initializers.ones, (), jnp.float32)
        return alpha * (h.sum() + h2.sum() + h3.sum())


@pytest.mark.x64_modes
def test_kfac_pattern_composite_vmapped(x64_mode):
    model = KfacPatternComposite()
    x = jnp.ones((7, 2, 3))
    params = model.init(jax.random.PRNGKey(0), x)
    batch_x = jnp.ones((4, *x.shape), dtype=activation_dtype(x64_mode))
    assert_params_f32(params)

    def apply_batch(p, xb):
        return jax.vmap(model.apply, in_axes=(None, 0))(p, xb)

    tag_tree = get_tag_tree(apply_batch, params, batch_x)

    assert_tag(
        tag_tree["params"]["Dense_0"]["kernel"],
        variant="repeated_dense",
        match_type="repeated[1]_flax_dense_with_bias",
    )
    assert_tag(
        tag_tree["params"]["Dense_0"]["bias"],
        variant="repeated_dense",
        match_type="repeated[1]_flax_dense_with_bias",
    )
    assert_tag(
        tag_tree["params"]["Dense_1"]["kernel"],
        variant="repeated_dense",
        match_type="repeated[1]_flax_dense_no_bias",
    )
    assert_tag(
        tag_tree["params"]["Dense_2"]["kernel"],
        variant="repeated_dense",
        match_type="repeated[2]_flax_dense_with_bias",
    )
    assert_tag(
        tag_tree["params"]["Dense_2"]["bias"],
        variant="repeated_dense",
        match_type="repeated[2]_flax_dense_with_bias",
    )
    assert_tag(
        tag_tree["params"]["DenseGeneral_0"]["kernel"],
        variant="repeated_dense",
        match_type="repeated[1]_flax_dense_with_bias",
    )
    assert_tag(
        tag_tree["params"]["DenseGeneral_0"]["bias"],
        variant="repeated_dense",
        match_type="repeated[1]_flax_dense_with_bias",
    )
    assert_tag(
        tag_tree["params"]["alpha"],
        variant="scale_and_shift",
        match_type="scale_only_broadcast_0",
    )


@pytest.mark.x64_modes
@pytest.mark.parametrize(
    "envelope_type",
    [EnvelopeType.abs_isotropic, EnvelopeType.diagonal],
)
def test_envelope_mixed_dtype_registration(x64_mode, envelope_type):
    n_elec = 3
    n_atoms = 2
    model = Envelope(
        envelope_type=envelope_type,
        ndets=4,
        nspins=(1, 2),
        orbitals_spin_split=True,
    )
    ae_vectors = jnp.zeros((n_elec, n_atoms, 3))
    r_ae = jnp.zeros((n_elec, n_atoms))
    params = model.init(jax.random.PRNGKey(0), ae_vectors, r_ae)
    act_dtype = activation_dtype(x64_mode)
    assert_params_f32(params)
    tag_tree = get_tag_tree(
        model.apply,
        params,
        ae_vectors.astype(act_dtype),
        r_ae.astype(act_dtype),
    )
    for env_tags in tag_tree["params"].values():
        assert_tag(
            env_tags["pi"],
            variant="scale_and_shift",
            match_type="scale_only_broadcast_2",
        )
        assert_tag(
            env_tags["sigma"],
            variant="scale_and_shift",
            match_type="scale_only_broadcast_2",
        )


@pytest.mark.x64_modes
def test_ferminet_registration_smoke(x64_mode):
    wf = FermiNetWavefunction(
        nspins=(1, 1),
        ndets=1,
        hidden_dims_single=[4, 4],
        hidden_dims_double=[2, 2],
    )
    act_dtype = activation_dtype(x64_mode)
    walker = MoleculeData(
        electrons=jnp.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=act_dtype),
        atoms=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([2.0]),
    )
    batch = BatchedData(
        MoleculeData(
            jnp.stack([walker.electrons, walker.electrons + 0.1]),
            walker.atoms,
            walker.charges,
        ),
        ["electrons"],
    )
    params = wf.init_params(
        MoleculeData(
            walker.electrons.astype(jnp.float32),
            walker.atoms,
            walker.charges,
        ),
        jax.random.PRNGKey(0),
    )
    assert_params_f32(params)

    def apply_logpsi(p, batched_data):
        return jax.vmap(wf.logpsi, in_axes=(None, batched_data.vmap_axis))(
            p, batched_data.data
        )

    tag_tree = get_tag_tree(apply_logpsi, params, batch)
    for env_tags in tag_tree["params"]["envelope_layer"].values():
        for key in ("pi", "sigma"):
            assert_tag(
                env_tags[key],
                variant="scale_and_shift",
                match_type="scale_only_broadcast_2",
            )
    for layer_tags in tag_tree["params"]["backbone_layer"].values():
        for key in ("kernel", "bias"):
            tag = layer_tags[key]
            assert "tag_variant=repeated_dense" in tag
            assert "match_type=repeated[" in tag
            assert "_flax_dense" in tag
            assert "Auto[generic(" not in tag
    for split_channel_tags in tag_tree["params"]["orbital_layer"].values():
        for dense_general_tags in split_channel_tags.values():
            tag = dense_general_tags["kernel"]
            assert "tag_variant=repeated_dense" in tag
            assert "match_type=repeated[" in tag
            assert "_flax_dense" in tag
            assert "Auto[generic(" not in tag


class CurvatureProbe(nn.Module):
    """Synthetic graph covering dense, repeated_dense, scale, and generic blocks."""

    @nn.compact
    def __call__(self, x_flat: jax.Array, x_rep: jax.Array) -> jax.Array:
        scale = self.param("scale", lambda *_: jnp.asarray(1.5, jnp.float32), ())
        w_generic = self.param(
            "w_generic", lambda *_: jnp.asarray(0.7, jnp.float32), ()
        )
        y0 = nn.Dense(1, use_bias=False)(x_flat)[..., 0]
        y1 = nn.Dense(1, use_bias=False)(x_rep).mean(axis=1)[..., 0]
        y2 = scale * jnp.ones(x_flat.shape[0])
        y3 = jnp.broadcast_to(jnp.sin(w_generic), x_flat.shape[:1])
        return y0 + y1 + y2 + y3


@dataclass(frozen=True)
class FactorSnapshot:
    block_index: int
    variant: str
    factor_index: int
    shape: tuple[int, ...]
    dtype: jnp.dtype
    weight: float
    value: jax.Array


def _weighted_factors(
    block_state: Any,
) -> list[kfac_jax.utils.WeightedMovingAverage]:
    if hasattr(block_state, "factors"):
        return list(block_state.factors)
    if hasattr(block_state, "diagonal_factors"):
        return list(block_state.diagonal_factors)
    raise AssertionError(f"unsupported KFAC block state layout: {type(block_state)!r}")


def _snapshot_preconditioner_factors(
    precon: kfac_jax.OptaxPreconditioner,
    state: kfac_jax.OptaxPreconditionState,
) -> dict[tuple[int, str, int], FactorSnapshot]:
    estimator = precon.estimator
    snapshots: dict[tuple[int, str, int], FactorSnapshot] = {}
    for block_index, (block_state, tag_eqn) in enumerate(
        zip(
            state.estimator_state.blocks_states,
            estimator.jaxpr.layer_tags,
            strict=True,
        )
    ):
        variant = tag_eqn.params["meta"].variant
        for factor_index, factor in enumerate(_weighted_factors(block_state)):
            assert factor.value is not None
            value = cast(jax.Array, factor.value)
            key = (block_index, variant, factor_index)
            snapshots[key] = FactorSnapshot(
                block_index=block_index,
                variant=variant,
                factor_index=factor_index,
                shape=value.shape,
                dtype=value.dtype,
                weight=float(factor.weight),
                value=value,
            )
    return snapshots


@pytest.mark.x64_modes
def test_curvature_updates_all_block_variants(x64_mode):
    probe = CurvatureProbe()
    act_dtype = activation_dtype(x64_mode)
    x_flat = jnp.linspace(0.1, 1.0, 4 * 3, dtype=act_dtype).reshape(4, 3)
    x_rep = jnp.linspace(0.2, 1.0, 4 * 2 * 3, dtype=act_dtype).reshape(4, 2, 3)
    params = probe.init(jax.random.PRNGKey(0), x_flat[0], x_rep[0])
    assert_params_f32(params)

    tag_tree = get_tag_tree(probe.apply, params, x_flat, x_rep)
    assert "Auto[generic(" in tag_tree["params"]["w_generic"]
    for tag, variant in (
        (tag_tree["params"]["Dense_0"]["kernel"], "dense"),
        (tag_tree["params"]["Dense_1"]["kernel"], "repeated_dense"),
        (tag_tree["params"]["scale"], "scale_and_shift"),
    ):
        assert f"tag_variant={variant}" in tag
        assert "Auto[generic(" not in tag

    def value_func(p, xb_flat, xb_rep):
        out = cast(jax.Array, probe.apply(p, xb_flat, xb_rep))
        kfac_jax.register_normal_predictive_distribution(out[:, None])
        return out

    def batch_size_extractor(xb_flat: jax.Array) -> int:
        return xb_flat.shape[0]

    precon = kfac_jax.OptaxPreconditioner(
        value_func,
        damping=1e-3,
        l2_reg=0.0,
        estimation_mode="fisher_exact",
        layer_tag_to_block_ctor=make_tag_to_block_ctor(),
        auto_register_kwargs=dict(graph_patterns=make_graph_patterns()),
        batch_size_extractor=batch_size_extractor,
        distributed_inverses=False,
        distributed_precon_apply=False,
    )
    patch_block_diagonal_curvature(precon.estimator)

    rng = jax.random.PRNGKey(0)
    state = precon.init((params, x_flat, x_rep), rng)
    found_variants = {
        eqn.params["meta"].variant for eqn in precon.estimator.jaxpr.layer_tags
    }
    assert {"dense", "repeated_dense", "scale_and_shift", "generic"}.issubset(
        found_variants
    )

    before = _snapshot_preconditioner_factors(precon, state)
    for snapshot in before.values():
        assert jnp.all(jnp.isfinite(snapshot.value))
        assert snapshot.weight == 0

    state = precon.maybe_update_estimator_curvature(
        state, (params, x_flat, x_rep), rng, sync=False
    )
    after = _snapshot_preconditioner_factors(precon, state)

    assert set(after) == set(before)
    for key, before_snapshot in before.items():
        after_snapshot = after[key]
        assert before_snapshot.shape == after_snapshot.shape
        assert before_snapshot.dtype == after_snapshot.dtype
        assert after_snapshot.dtype == jnp.float32
        assert after_snapshot.weight > 0
        assert jnp.all(jnp.isfinite(after_snapshot.value))
        assert jnp.any(after_snapshot.value != 0)


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
    ae_vectors = jnp.zeros((n_elec, n_atoms, 3))
    r_ae = jnp.zeros((n_elec, n_atoms))
    params = model.init(jax.random.PRNGKey(0), ae_vectors, r_ae)

    tag_tree = get_tag_tree(model.apply, params, ae_vectors, r_ae)
    for env_key in env_keys:
        assert "tag_variant=scale_and_shift" in tag_tree["params"][env_key]["pi"]
        assert "tag_variant=scale_and_shift" in tag_tree["params"][env_key]["sigma"]


@pytest.mark.parametrize(
    "envelope_type",
    [EnvelopeType.isotropic, EnvelopeType.abs_isotropic],
)
@pytest.mark.parametrize("ndim_feat", [3, 6])
def test_envelope_layer_registration_variable_dim(envelope_type, ndim_feat):
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

    tag_tree = get_tag_tree(model.apply, params, ae_vectors, r_ae)
    assert "tag_variant=scale_and_shift" in tag_tree["params"]["_env"]["pi"]
    assert "tag_variant=scale_and_shift" in tag_tree["params"]["_env"]["sigma"]


@pytest.mark.x64_modes
def test_x64_mode_fixture_scopes_off_under_process_x64(x64_mode):
    """Lightweight check that scoped x64-off works when the process starts x64-on."""
    if not x64_mode:
        assert not jax.config.read("jax_enable_x64")
