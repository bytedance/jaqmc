# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import chex
import jax
import numpy as np
import optax
import pytest
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from jaqmc.optimizer.sr.jasr import (
    FisherInverseState,
    _enable_x64,
    active_leaf_indices,
    calc_gram_matrix,
    calc_gram_matrix_chunked,
    chunked_vmap,
    estimate_required_damping,
    get_preset_params,
    get_structural_active_mask,
    lift_null_space,
    robust_sr,
    scale_by_constrained_norm,
    scale_by_fisher_inverse,
    take_single_sample,
)

if jax.__version_info__ < (0, 8, 0):
    pytest.skip(
        "jasr_test.py requires jax >= 0.8.0 for enable_x64",
        allow_module_level=True,
    )


def _simple_log_psi(params, sample):
    """Linear log-psi: log_psi = dot(params, sample).

    This yields the centered score matrix `samples - mean(samples, axis=0)`.

    Returns:
        Scalar log-amplitude for the given sample.
    """
    w = params["w"]
    return jnp.vdot(w, sample)


def _setup_linear_case(n_batch=4, n_params=3, complex=False):
    # Deterministic parameters and samples
    key = jax.random.key(42)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    params = {"w": jax.random.uniform(key1, shape=(n_params,))}
    samples = jax.random.uniform(key2, shape=(n_batch, n_params))
    if complex:
        samples += 1j * jax.random.uniform(key4, shape=(n_batch, n_params))
    grads = {"w": jax.random.uniform(key3, shape=(n_params,))}

    return params, samples, grads


def _random_score_and_scale(n_batch=4, n_params=3):
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key, 2)

    score = jax.random.normal(key1, (n_batch, n_params), dtype=jnp.float32)
    score = score - jnp.mean(score, axis=0, keepdims=True)
    score = score / jnp.sqrt(n_batch)
    scale = jnp.abs(jax.random.normal(key2, (n_params,), dtype=jnp.float32))

    return score, scale


def _structural_mask_and_idx(log_psi_fn, params, samples, score_in_axes=0):
    sample = take_single_sample(samples, score_in_axes)
    mask = get_structural_active_mask(log_psi_fn, params, sample)
    idx = active_leaf_indices(params, mask)
    return mask, idx


def test_chunked_vmap():
    def f(inputs):
        return {
            "a": (inputs["x"][0] + 1, inputs["x"][1] - 1),
            "b": jnp.sum(inputs["z"]),
            "c": inputs["x"][0] ** 2 + inputs["y"],
        }

    inputs = {
        "x": (jnp.arange(21), jnp.arange(21)),
        "y": jnp.ones((3, 21)),
        "z": jnp.arange(5),
    }
    in_axes = ({"x": 0, "y": 1, "z": None},)
    out_axes = {"a": 0, "b": None, "c": 1}
    if not hasattr(jax.tree, "broadcast"):
        # no broadcast available, need to fully specify in_axes and out_axes
        in_axes[0]["x"] = (0, 0)
        out_axes["a"] = (0, 0)

    y_ref = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)(inputs)
    y = chunked_vmap(f, in_axes=in_axes, out_axes=out_axes, chunk_size=6)(inputs)

    chex.assert_trees_all_close(y, y_ref)


@pytest.mark.parametrize("complex", [False, True])
@pytest.mark.parametrize("with_jit", [False, True])
@pytest.mark.parametrize("chunks", [None, 2])
def test_sr_matches_kernel_formulation(with_jit, complex, chunks):
    """Check SR implementation matches the efficient kernel formula."""
    params, samples, grads = _setup_linear_case(n_batch=5, n_params=7, complex=complex)

    lam = 1e-2
    opt = scale_by_fisher_inverse(
        _simple_log_psi,
        damping=lam,
        spring_mu=None,
        score_chunk_size=chunks,
        gram_num_chunks=chunks,
    )
    state = opt.init(params)

    update_fn = opt.update if not with_jit else jax.jit(opt.update)
    updates, new_state = update_fn(grads, state, params, samples)
    delta_num, _ = ravel_pytree(updates)

    # Explicit kernel computation for comparison.
    def grad_log_psi(p, s):
        # For linear log psi, grad wrt w is just sample.
        return {"w": s}

    score_fn = jax.vmap(lambda s: ravel_pytree(grad_log_psi(params, s))[0])
    score_matrix = score_fn(samples)
    score_matrix = score_matrix - jnp.mean(score_matrix, axis=0, keepdims=True)
    score_matrix /= score_matrix.shape[0] ** 0.5
    if complex:
        score_matrix = jnp.concatenate([score_matrix.real, score_matrix.imag], axis=0)

    g_flat, _ = ravel_pytree(grads)
    u_g = score_matrix @ g_flat

    F = score_matrix @ score_matrix.T + lam * jnp.eye(
        score_matrix.shape[0], dtype=score_matrix.dtype
    )
    w = jnp.linalg.solve(F, u_g)
    z = jnp.linalg.solve(F, u_g - w)
    delta_ref = g_flat - score_matrix.T @ z

    chex.assert_trees_all_close(delta_num, delta_ref, atol=1e-5)
    assert new_state.counter == 1


@pytest.mark.parametrize("complex", [False, True])
def test_sr_reduces_to_identity_when_large_damping(complex):
    """If damping is large, the update approaches the raw gradient."""
    params, samples, grads = _setup_linear_case(n_batch=4, n_params=3, complex=complex)

    opt = scale_by_fisher_inverse(_simple_log_psi, damping=1e8)
    state = opt.init(params)

    updates, new_state = opt.update(grads, state, params, samples)

    flat_g, _ = ravel_pytree(grads)
    flat_u, _ = ravel_pytree(updates)

    # With huge damping and centered score ~ 0, delta ~ g.
    chex.assert_trees_all_close(flat_u, flat_g, atol=1e-5)
    assert isinstance(new_state, FisherInverseState)


@pytest.mark.parametrize("complex", [False, True])
def test_sr_reduces_to_standard_when_small_damping(complex):
    """For small damping, SR reduces to standard Fisher inverse."""
    n_batch = 5
    n_params = 4 if not complex else 8
    params, samples, grads = _setup_linear_case(
        n_batch=n_batch, n_params=n_params, complex=complex
    )

    lam = 1e-10
    opt = scale_by_fisher_inverse(_simple_log_psi, damping=lam, spring_mu=None)
    state = opt.init(params)

    updates, _ = opt.update(grads, state, params, samples)
    delta_num, _ = ravel_pytree(updates)

    # Build score as in the implementation.
    def grad_log_psi(p, s):
        return {"w": s}

    score_fn = jax.vmap(lambda s: ravel_pytree(grad_log_psi(params, s))[0])
    score_matrix = score_fn(samples)
    score_matrix = score_matrix - jnp.mean(score_matrix, axis=0, keepdims=True)
    score_matrix = score_matrix / n_batch**0.5

    g_flat, _ = ravel_pytree(grads)
    F = (score_matrix.conj().T @ score_matrix).real + lam * jnp.eye(
        score_matrix.shape[-1]
    )
    delta_direct = jnp.linalg.solve(F, g_flat)

    chex.assert_trees_all_close(delta_num, delta_direct, atol=5e-4, rtol=5e-4)


def test_spring_adds_momentum():
    """SPRING term should move in the direction of previous delta when mu>0."""
    params, samples, grads = _setup_linear_case(n_batch=5, n_params=4)

    lam = 1e-2
    mu = 0.5
    opt = scale_by_fisher_inverse(_simple_log_psi, damping=lam, spring_mu=mu)
    state = opt.init(params)

    # First step: prev_grads=0, so same as pure SR.
    updates1, state1 = opt.update(grads, state, params, samples)

    # Second step with same grads and samples: momentum should now contribute.
    updates2, _state2 = opt.update(grads, state1, params, samples)

    u1, _ = ravel_pytree(updates1)
    u2, _ = ravel_pytree(updates2)

    # Second update should differ from the first due to momentum.
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(u1, u2)
    # Momentum pushes in the direction of previous delta.
    dot = jnp.vdot(u2 - u1, u1)
    assert dot.real > 0


def test_march_reduces_to_spring_on_first_step():
    """On the first step, MARCH uses initial acc_var, close to identity metric."""
    params, samples, grads = _setup_linear_case(n_batch=5, n_params=4)

    lam = 1e-2
    mu = 0.5
    beta = 0.9

    opt_spring = scale_by_fisher_inverse(
        _simple_log_psi, damping=lam, spring_mu=mu, march_beta=None
    )
    opt_march = scale_by_fisher_inverse(
        _simple_log_psi, damping=lam, spring_mu=mu, march_beta=beta
    )

    state_s = opt_spring.init(params)
    state_m = opt_march.init(params)

    updates_s, _ = opt_spring.update(grads, state_s, params, samples)
    updates_m, state_m1 = opt_march.update(grads, state_m, params, samples)

    u_s, _ = ravel_pytree(updates_s)
    u_m, _ = ravel_pytree(updates_m)

    # Initial acc_var is all ones, so M^{-1} ~ 1; expect similar behavior.
    chex.assert_trees_all_close(u_s, u_m, atol=1e-4)

    # After first step, acc_var should be updated.
    assert isinstance(state_m1, FisherInverseState)
    assert state_m1.counter == 1
    assert not jnp.allclose(state_m1.acc_var, jnp.ones_like(state_m1.acc_var))


def test_march_var_mode_updates_acc_var_correctly():
    """Test that march 'var' mode updates acc_var with score variance."""
    params, samples, grads = _setup_linear_case(n_batch=5, n_params=4)

    lam = 1e-2
    mu = 0.5
    beta = 0.9

    opt = scale_by_fisher_inverse(
        _simple_log_psi, damping=lam, spring_mu=mu, march_beta=beta, march_mode="var"
    )

    state = opt.init(params)

    # First step
    _updates, state1 = opt.update(grads, state, params, samples)

    # Calculate score variance manually for comparison
    def grad_log_psi(p, s):
        return {"w": s}

    score_fn = jax.vmap(lambda s: ravel_pytree(grad_log_psi(params, s))[0])
    score_matrix = score_fn(samples)
    score_matrix = score_matrix - jnp.mean(score_matrix, axis=0, keepdims=True)
    score_matrix = score_matrix / score_matrix.shape[0] ** 0.5
    score_var = jnp.var(score_matrix, axis=0)
    score_var /= score_var.mean()

    # The acc_var should be updated with score variance
    expected_acc_var = beta * jnp.ones_like(score_var) + (1 - beta) * score_var
    chex.assert_trees_all_close(state1.acc_var, expected_acc_var, atol=1e-5)


def test_sr_matches_parameter_space_formula_in_1d():
    """In 1D, robust SR update matches analytic G(h) operator."""
    # 1D parameter, simple samples
    params = {"w": jnp.array([1.0], dtype=jnp.float32)}
    samples = jnp.array([[0.0], [1.0], [2.0], [3.0]], dtype=jnp.float32)
    grads = {"w": jnp.array([2.0], dtype=jnp.float32)}  # tilde{delta}

    lam = 0.5
    opt = scale_by_fisher_inverse(_simple_log_psi, damping=lam, spring_mu=None)
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params, samples)
    delta, _ = ravel_pytree(updates)

    # Manually construct the centered score and curvature h in this 1D case.
    x = samples[:, 0]
    centered_score = x - jnp.mean(x)
    h = jnp.dot(centered_score, centered_score) / centered_score.shape[0]

    # Analytic robust SR operator in 1D:
    # G(h) = 1 - h/(h+lambda) + h/(h+lambda)^2
    g = grads["w"][0]
    G_h = 1.0 - h / (h + lam) + h / (h + lam) ** 2
    delta_expected = G_h * g

    chex.assert_trees_all_close(delta[0], delta_expected, atol=1e-5)


def test_spring_matches_analytic_formula_in_1d():
    """In 1D, SPRING update matches analytic (g,p,h,lambda) expression."""
    params = {"w": jnp.array([0.5], dtype=jnp.float32)}
    samples = jnp.array([[0.0], [1.0], [2.0], [3.0]], dtype=jnp.float32)
    grads = {"w": jnp.array([1.2], dtype=jnp.float32)}

    lam = 0.3
    mu = 0.8
    opt = scale_by_fisher_inverse(_simple_log_psi, damping=lam, spring_mu=mu)

    # Custom state with known previous delta.
    s0 = opt.init(params)
    prev_delta = jnp.array([0.4], dtype=jnp.float32)
    state = FisherInverseState(counter=0, prev_delta=prev_delta, acc_var=s0.acc_var)

    updates, _ = opt.update(grads, state, params, samples)
    delta_num, _ = ravel_pytree(updates)

    # 1D score statistics
    x = samples[:, 0]
    centered_score = x - jnp.mean(x)
    h = jnp.dot(centered_score, centered_score) / centered_score.shape[0]

    g = grads["w"][0]
    p = mu * prev_delta[0]

    # Analytic SPRING formula in 1D derived from the kernel algorithm:
    # delta = (g + p) * [1 - h/(h+lambda)] + g * h/(h+lambda)^2
    G1 = 1.0 - h / (h + lam)
    G2 = h / (h + lam) ** 2
    delta_expected = (g + p) * G1 + g * G2

    chex.assert_trees_all_close(delta_num[0], delta_expected, atol=1e-5)


@pytest.mark.parametrize("march_mode", ["var", "diff"])
def test_march_matches_analytic_formula_in_1d(march_mode):
    """In 1D, MARCH update matches analytic expression for fixed M^{-1}."""
    params = {"w": jnp.array([0.7], dtype=jnp.float32)}
    samples = jnp.array([[0.0], [1.0], [2.0], [3.0]], dtype=jnp.float32)
    grads = {"w": jnp.array([0.9], dtype=jnp.float32)}

    lam = 0.2
    mu = 0.6
    m_inv = 1.5  # M^{-1}
    beta = 0.9

    opt = scale_by_fisher_inverse(
        _simple_log_psi,
        damping=lam,
        spring_mu=mu,
        march_beta=beta,
        march_mode=march_mode,
        eps=0.0,
    )

    # Custom state with known previous delta and metric M^{-1}.
    s0 = opt.init(params)
    prev_delta = jnp.array([0.3], dtype=jnp.float32)
    # We want M_inv = 1 / sqrt(v_prev) = m_inv => v_prev = 1 / m_inv^2.
    v_prev = jnp.full_like(s0.acc_var, 1.0 / (m_inv**2))
    state = FisherInverseState(counter=0, prev_delta=prev_delta, acc_var=v_prev)

    updates, _ = opt.update(grads, state, params, samples)
    delta_num, _ = ravel_pytree(updates)

    x = samples[:, 0]
    centered_score = x - jnp.mean(x)
    h = jnp.dot(centered_score, centered_score) / centered_score.shape[0]

    g = grads["w"][0]
    p = mu * prev_delta[0]
    m = (
        m_inv
        if march_mode == "diff"
        # mode diff: acc_var is not updated
        else 1 / jnp.sqrt(beta * v_prev + (1 - beta) * 1.0)
    )  # mode var: rel var is 1 in 1D

    # Analytic MARCH formula in 1D for fixed M^{-1} = m:
    # u_g = score (m g), u_p = score p, F_M = lambda I + m score score^T.
    # The kernel algorithm simplifies to:
    #   delta = m g + p - m h/(lambda + m h) * (m g + p - m g/(lambda + m h))
    A = m * g + p
    B = m * g / (lam + m * h)
    delta_expected = A - m * h / (lam + m * h) * (A - B)

    chex.assert_trees_all_close(delta_num[0], delta_expected, atol=1e-5)


def test_sr_score_in_axes_none():
    """Score computed from pytree samples where one leaf is unmapped (None)."""
    n_batch = 6
    n1 = 4  # dims for batched leaf 'x'
    n2 = 3  # dims for constant leaf 'y'

    key = jax.random.key(123)
    kx, ky, kw1, kw2, kg1, kg2 = jax.random.split(key, 6)

    # params and grads split to match two leaves
    params = {
        "w1": jax.random.normal(kw1, (n1,)),
        "w2": jax.random.normal(kw2, (n2,)),
    }
    grads = {
        "w1": jax.random.normal(kg1, (n1,)),
        "w2": jax.random.normal(kg2, (n2,)),
    }

    # samples as a pytree: 'x' batched, 'y' constant (unmapped)
    samples = {
        "x": jax.random.normal(kx, (n_batch, n1)),
        "y": jax.random.normal(ky, (n2,)),
    }

    # log-psi uses both leaves
    def log_psi_fn(p, s):
        return jnp.vdot(p["w1"], s["x"]) + jnp.vdot(p["w2"], s["y"])

    score_axes = ({"x": 0, "y": None},)
    lam = 1e-2

    opt = scale_by_fisher_inverse(
        log_psi_fn, damping=lam, spring_mu=None, score_in_axes=score_axes
    )
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params, samples)
    delta_num, _ = ravel_pytree(updates)

    # Explicit kernel computation
    def grad_log_psi(p, s):
        return {"w1": s["x"], "w2": s["y"]}

    score_fn = jax.vmap(
        lambda s: ravel_pytree(grad_log_psi(params, s))[0],
        in_axes=score_axes,
        out_axes=0,
    )
    score_matrix = score_fn(samples)
    score_matrix = score_matrix - jnp.mean(score_matrix, axis=0, keepdims=True)
    score_matrix = score_matrix / jnp.sqrt(n_batch)

    g_flat, _ = ravel_pytree(grads)
    u_g = score_matrix @ g_flat
    F = score_matrix @ score_matrix.T + lam * jnp.eye(
        score_matrix.shape[0], dtype=score_matrix.dtype
    )
    w = jnp.linalg.solve(F, u_g)
    z = jnp.linalg.solve(F, u_g - w)
    delta_ref = g_flat - score_matrix.T @ z

    chex.assert_trees_all_close(delta_num, delta_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("complex", [False, True])
def test_sr_score_norm_clip_matches_reference(complex):
    """`score_norm_clip` clips each row (mean abs) before centering/scaling."""
    n_params = 3
    params = {"w": jnp.ones((n_params,), dtype=jnp.float32)}
    grads = {"w": jnp.array([0.2, -0.3, 0.4], dtype=jnp.float32)}

    samples = jnp.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    if complex:
        samples = samples + 1j * (0.1 * samples)

    lam = 1e-2
    clip = 1.0

    opt = scale_by_fisher_inverse(
        _simple_log_psi,
        damping=lam,
        spring_mu=None,
        march_beta=None,
        score_norm_clip=clip,
    )
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params, samples)
    delta_num, _ = ravel_pytree(updates)

    # Manual reference matching the implementation.
    score_matrix = samples
    row_abs_mean = jnp.mean(jnp.abs(score_matrix), axis=1, keepdims=True)
    row_scale = jnp.clip(clip / row_abs_mean, min=0.0, max=1.0)
    score_matrix = score_matrix * row_scale
    score_matrix = (score_matrix - score_matrix.mean(0)) / jnp.sqrt(
        score_matrix.shape[0]
    )
    if complex:
        score_matrix = jnp.concatenate([score_matrix.real, score_matrix.imag], axis=0)

    g_flat, _ = ravel_pytree(grads)
    u_g = score_matrix @ g_flat
    F = score_matrix @ score_matrix.T + lam * jnp.eye(
        score_matrix.shape[0], dtype=score_matrix.dtype
    )
    w = jnp.linalg.solve(F, u_g)
    z = jnp.linalg.solve(F, u_g - w)
    delta_ref = g_flat - score_matrix.T @ z

    # Sanity: ensure at least one row was clipped and at least one was not.
    assert (row_scale < 1.0).any()
    assert jnp.isclose(row_scale, 1.0).any()
    chex.assert_trees_all_close(delta_num, delta_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("complex", [False, True])
def test_sr_score_norm_clip_large_is_noop(complex):
    """Very large `score_norm_clip` should be a no-op vs `None`."""
    params, samples, grads = _setup_linear_case(n_batch=6, n_params=5, complex=complex)
    lam = 1e-2

    opt_none = scale_by_fisher_inverse(
        _simple_log_psi,
        damping=lam,
        spring_mu=None,
        march_beta=None,
        score_norm_clip=None,
    )
    opt_large = scale_by_fisher_inverse(
        _simple_log_psi,
        damping=lam,
        spring_mu=None,
        march_beta=None,
        score_norm_clip=1e9,
    )

    st0 = opt_none.init(params)
    u_none, _ = opt_none.update(grads, st0, params, samples)
    u_large, _ = opt_large.update(grads, st0, params, samples)

    chex.assert_trees_all_close(u_none, u_large, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("with_jit", [False, True])
def test_prune_inactive_unused_leaf_matches_baseline(with_jit):
    """Compile-time pruning should preserve updates when an extra leaf is unused."""
    params, samples, grads = _setup_linear_case(n_batch=5, n_params=4)
    params = {**params, "unused": jnp.array([1.0, -2.0], dtype=jnp.float32)}
    grads = {**grads, "unused": jnp.zeros((2,), dtype=jnp.float32)}
    lam = 1e-2

    def log_psi_fn(p, s):
        return jnp.vdot(p["w"], s)

    active_mask, active_idx = _structural_mask_and_idx(log_psi_fn, params, samples)
    assert active_mask == {"unused": False, "w": True}
    np.testing.assert_array_equal(active_idx, np.arange(2, 6, dtype=np.int32))

    opt_ref = scale_by_fisher_inverse(log_psi_fn, damping=lam, spring_mu=None)
    opt_pruned = scale_by_fisher_inverse(
        log_psi_fn, damping=lam, spring_mu=None, prune_inactive=True
    )

    state_ref = opt_ref.init(params)
    state_pruned = opt_pruned.init(params)
    update_ref = opt_ref.update if not with_jit else jax.jit(opt_ref.update)
    update_pruned = opt_pruned.update if not with_jit else jax.jit(opt_pruned.update)

    updates_ref, new_state_ref = update_ref(grads, state_ref, params, samples)
    updates_pruned, new_state_pruned = update_pruned(
        grads, state_pruned, params, samples
    )

    chex.assert_trees_all_close(updates_pruned, updates_ref, atol=1e-6, rtol=1e-6)
    chex.assert_trees_all_close(
        new_state_pruned.prev_delta, new_state_ref.prev_delta, atol=1e-5, rtol=1e-5
    )
    chex.assert_trees_all_close(
        new_state_pruned.acc_var, new_state_ref.acc_var, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("with_jit", [False, True])
def test_prune_inactive_scan_and_stop_gradient_matches_baseline(with_jit):
    """Pruning should recurse into scan and ignore forward-only stop-gradient leaves."""
    key = jax.random.key(7)
    ka, kb, ku, kg, ks = jax.random.split(key, 5)
    n_batch, n_steps, n_features = 4, 3, 5

    params = {
        "active": jax.random.normal(ka, (n_features,)),
        "stopped": jax.random.normal(kb, (n_features,)),
        "unused": jax.random.normal(ku, (2,)),
    }
    grads = {
        "active": jax.random.normal(kg, (n_features,)),
        "stopped": jnp.zeros((n_features,), dtype=jnp.float32),
        "unused": jnp.zeros((2,), dtype=jnp.float32),
    }
    samples = jax.random.normal(ks, (n_batch, n_steps, n_features))
    lam = 5e-3

    def log_psi_fn(p, sample):
        stopped = jax.lax.stop_gradient(p["stopped"])

        def body(carry, x):
            y = jnp.vdot(p["active"], x) + jnp.vdot(stopped, x)
            return carry + y, y

        carry, ys = jax.lax.scan(body, 0.0, sample)
        return carry + jnp.sum(ys)

    active_mask, active_idx = _structural_mask_and_idx(log_psi_fn, params, samples)
    assert active_mask == {"active": True, "stopped": False, "unused": False}
    np.testing.assert_array_equal(active_idx, np.arange(n_features, dtype=np.int32))

    opt_ref = scale_by_fisher_inverse(log_psi_fn, damping=lam, spring_mu=None)
    opt_pruned = scale_by_fisher_inverse(
        log_psi_fn, damping=lam, spring_mu=None, prune_inactive=True
    )

    state_ref = opt_ref.init(params)
    state_pruned = opt_pruned.init(params)
    update_ref = opt_ref.update if not with_jit else jax.jit(opt_ref.update)
    update_pruned = opt_pruned.update if not with_jit else jax.jit(opt_pruned.update)

    updates_ref, new_state_ref = update_ref(grads, state_ref, params, samples)
    updates_pruned, new_state_pruned = update_pruned(
        grads, state_pruned, params, samples
    )

    chex.assert_trees_all_close(updates_pruned, updates_ref, atol=1e-5, rtol=1e-5)
    chex.assert_trees_all_close(
        new_state_pruned.prev_delta, new_state_ref.prev_delta, atol=1e-5, rtol=1e-5
    )
    chex.assert_trees_all_close(
        new_state_pruned.acc_var, new_state_ref.acc_var, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("with_jit", [False, True])
def test_dense_var_mode_matches_pruned_for_structurally_inactive_leaves(with_jit):
    """Dense var-mode MARCH should ignore structurally inactive leaves like pruning."""
    params, samples, grads = _setup_linear_case(n_batch=5, n_params=4)
    params = {**params, "unused": jnp.array([1.0, -2.0], dtype=jnp.float32)}
    grads = {**grads, "unused": jnp.array([0.5, -0.25], dtype=jnp.float32)}

    def log_psi_fn(p, s):
        return jnp.vdot(p["w"], s)

    opt_dense = scale_by_fisher_inverse(
        log_psi_fn,
        damping=1e-2,
        spring_mu=None,
        march_beta=0.5,
        march_mode="var",
        prune_inactive=False,
    )
    opt_pruned = scale_by_fisher_inverse(
        log_psi_fn,
        damping=1e-2,
        spring_mu=None,
        march_beta=0.5,
        march_mode="var",
        prune_inactive=True,
    )

    state_dense = opt_dense.init(params)
    state_pruned = opt_pruned.init(params)
    update_dense = opt_dense.update if not with_jit else jax.jit(opt_dense.update)
    update_pruned = opt_pruned.update if not with_jit else jax.jit(opt_pruned.update)

    updates_dense, new_state_dense = update_dense(grads, state_dense, params, samples)
    updates_pruned, new_state_pruned = update_pruned(
        grads, state_pruned, params, samples
    )

    chex.assert_trees_all_close(updates_dense, updates_pruned, atol=1e-6, rtol=1e-6)
    chex.assert_trees_all_close(
        new_state_dense.prev_delta, new_state_pruned.prev_delta, atol=1e-6, rtol=1e-6
    )
    chex.assert_trees_all_close(
        new_state_dense.acc_var, new_state_pruned.acc_var, atol=1e-6, rtol=1e-6
    )


def test_prune_inactive_raises_when_all_params_inactive():
    params = {"w": jnp.array([1.0, 2.0], dtype=jnp.float32)}
    grads = {"w": jnp.zeros((2,), dtype=jnp.float32)}
    samples = jnp.ones((4, 2), dtype=jnp.float32)

    def log_psi_fn(p, s):
        del p, s
        return jnp.array(0.0, dtype=jnp.float32)

    opt = scale_by_fisher_inverse(
        log_psi_fn,
        damping=1e-2,
        spring_mu=None,
        prune_inactive=True,
    )
    state = opt.init(params)

    with pytest.raises(ValueError, match="pruned all parameter leaves"):
        opt.update(grads, state, params, samples)


@pytest.mark.parametrize("mode", ["pmap", "smap"])
@pytest.mark.parametrize("complex", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 10])
def test_multi_device_parity(complex, mode, chunk_size):
    n_devices = jax.local_device_count()
    if n_devices < 2:
        pytest.skip("need >=2 devices for pmap parity test")

    n_params = 300
    n_local = 25
    n_batch = n_devices * n_local
    params, samples, grads = _setup_linear_case(
        n_batch=n_batch, n_params=n_params, complex=complex
    )

    lam = 1e-2
    opt_single = scale_by_fisher_inverse(
        _simple_log_psi,
        damping=lam,
        spring_mu=None,
        axis_name=None,
        score_chunk_size=chunk_size,
        mixed_precision=True,
    )
    opt_multi = scale_by_fisher_inverse(
        _simple_log_psi,
        damping=lam,
        spring_mu=None,
        axis_name="i",
        score_chunk_size=chunk_size,
        mixed_precision=True,
    )

    state_s = opt_single.init(params)
    updates_s, _ = opt_single.update(grads, state_s, params, samples)

    map_args = dict(axis_name="i", in_axes=(None, None, None, 0), out_axes=None)

    if mode == "pmap":
        samples = samples.reshape(n_devices, n_local, n_params)
        update_mapped = jax.pmap(opt_multi.update, **map_args)
        updates_m, _ = update_mapped(grads, state_s, params, samples)
    elif mode == "smap":
        if not hasattr(jax, "smap") or not hasattr(jax, "set_mesh"):
            pytest.skip("smap not available")
        update_mapped = jax.smap(opt_multi.update, **map_args)
        mesh = jax.sharding.Mesh(np.array(jax.devices()), ("i",))
        with jax.set_mesh(mesh):
            updates_m, _ = update_mapped(grads, state_s, params, samples)
    else:
        pytest.skip(f"mode {mode} not supported")

    chex.assert_trees_all_close(updates_m, updates_s, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_x64", [False, True])
def test_calc_gram_chunked_parity(use_x64):
    n_batch = 8
    n_params = 11
    num_chunks = 4

    score_rows, s = _random_score_and_scale(n_batch, n_params)
    gram_ref = (score_rows * s) @ score_rows.T

    with _enable_x64(use_x64):
        gram_chunked = calc_gram_matrix_chunked(
            score_rows,
            s,
            num_chunks=num_chunks,
            use_x64=use_x64,
        )

    assert gram_chunked.dtype == jnp.float64 if use_x64 else jnp.float32
    chex.assert_trees_all_close(gram_chunked, gram_ref, atol=1e-6, rtol=1e-6)


def test_calc_gram_chunked_no_all_gather():
    if not hasattr(jax, "smap") or not hasattr(jax, "set_mesh"):
        pytest.skip("smap not available")
    n_devices = jax.local_device_count()
    if n_devices < 2:
        pytest.skip("need >=2 devices")

    n_local = 2
    n_batch = n_devices * n_local
    n_params = 8
    num_chunks = 3

    score_rows, s = _random_score_and_scale(n_batch, n_params)

    fn = partial(
        calc_gram_matrix_chunked,
        num_chunks=num_chunks,
        use_x64=False,
        axis_name="i",
    )
    jit_fn = jax.jit(jax.smap(fn, in_axes=(0, None), out_axes=0, axis_name="i"))

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("i",))
    with jax.set_mesh(mesh):
        lowered = jit_fn.lower(score_rows, s)

    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    assert hlo.count("all-gather") == 1  # only once for row factor


@pytest.mark.parametrize("mode", ["pmap", "smap"])
@pytest.mark.parametrize("num_chunks", [None, 2, 4])
def test_calc_gram_multi_device_parity(mode, num_chunks):
    n_devices = jax.local_device_count()
    if n_devices < 2:
        pytest.skip("need >=2 devices for pmap gram test")

    n_params = 6
    n_local = 3
    n_batch = n_devices * n_local

    score_rows, s = _random_score_and_scale(n_batch, n_params)
    gram_ref = (score_rows * s) @ score_rows.T

    calc_fn = partial(
        calc_gram_matrix, num_chunks=num_chunks, use_x64=False, axis_name="i"
    )
    map_args = dict(axis_name="i", in_axes=(0, None), out_axes=None)

    if mode == "pmap":
        score_rows = score_rows.reshape(n_devices, n_local, n_params)
        gram_m = jax.pmap(calc_fn, **map_args)(score_rows, s)
    elif mode == "smap":
        if not hasattr(jax, "smap") or not hasattr(jax, "set_mesh"):
            pytest.skip("smap not available")
        mesh = jax.sharding.Mesh(np.array(jax.devices()), ("i",))
        with jax.set_mesh(mesh):
            gram_m = jax.smap(calc_fn, **map_args)(score_rows, s)
    else:
        pytest.skip(f"mode {mode} not supported")

    chex.assert_trees_all_close(gram_m, gram_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("complex", [False, True])
def test_lift_null_space_properties(complex):
    """`lift_null_space` adds a low-rank shift that removes the mean-null mode.

    Checks:
    - shift rank is 1 (real) / 2 (complex-concatenated)
    - shift annihilates centered score: `shift @ score == 0`
    - lifted gram is full rank even when damping is zero, for n_params > 2*n_batch
    """
    n_batch = 5
    n_params = 20  # ensures n_params > 2*n_batch

    n_rows = 2 * n_batch if complex else n_batch

    # Recover the shift matrix by lifting a zero gram.
    shift = lift_null_space(jnp.zeros((n_rows, n_rows)), n_batch)

    # Rank checks via SVD thresholding.
    s_shift = jnp.linalg.svd(shift, compute_uv=False)
    rank_shift = int(jnp.sum(s_shift > 1e-7))
    assert rank_shift == (2 if complex else 1)

    # Build a centered score matrix (matching the implementation's centering).
    key = jax.random.key(0)
    if complex:
        k1, k2 = jax.random.split(key)
        score_c = jax.random.normal(k1, (n_batch, n_params))
        score_c = score_c + 1j * jax.random.normal(k2, (n_batch, n_params))
        score_c = score_c - jnp.mean(score_c, axis=0, keepdims=True)
        score_c = score_c / jnp.sqrt(n_batch)
        score = jnp.concatenate([score_c.real, score_c.imag], axis=0)
    else:
        score = jax.random.normal(key, (n_batch, n_params))
        score = score - jnp.mean(score, axis=0, keepdims=True)
        score = score / jnp.sqrt(n_batch)

    # The shift averages within each real/imag block, so it must kill
    # the centered score.
    residual = shift @ score
    assert jnp.max(jnp.abs(residual)) < 1e-7

    # Without damping, the lifted gram should be full rank.
    gram = score @ score.T
    gram_lifted = lift_null_space(gram, n_batch)
    s_lifted = jnp.linalg.svd(gram_lifted, compute_uv=False)
    rank_lifted = int(jnp.sum(s_lifted > 1e-7))
    assert rank_lifted == n_rows


def test_estimate_required_damping():
    """Test estimate_required_damping function correctness.

    This test verifies that estimate_required_damping correctly calculates
    the damping needed to achieve a target condition number when the smallest
    eigenvalue is 0.
    """
    # Test case 1: Simple matrix with known eigenvalues [10, 0]
    A = jnp.zeros((2, 2)).at[0, 0].set(10)

    # Test with target condition number = 10
    target_cond = 10.0
    required_lambda = estimate_required_damping(jnp.array(A), target_cond)
    required_lambda = jax.device_get(required_lambda)

    # Expected value: (10 - 10*0)/(10 - 1) = 10/9 ≈ 1.111
    expected = 10 / 9
    assert abs(required_lambda - expected) < 1e-6, (
        f"Expected {expected:.6f}, got {required_lambda:.6f}"
    )

    # Verify the condition number of (A + λI) is ≤ target_cond
    damped_A = A + required_lambda * jnp.eye(2)
    cond_num = jnp.linalg.cond(damped_A)
    assert cond_num <= target_cond + 1e-6, (
        f"Condition number {cond_num:.2f} exceeds target {target_cond}"
    )

    # Test case 2: Larger matrix with eigenvalues [100, 0, 0, 0]
    A = jnp.zeros((4, 4)).at[0, 0].set(100)

    target_cond = 100.0
    required_lambda = estimate_required_damping(jnp.array(A), target_cond)
    required_lambda = jax.device_get(required_lambda)

    # Expected value: (100 - 100*0)/(100 - 1) = 100/99 ≈ 1.0101
    expected = 100 / 99
    assert abs(required_lambda - expected) < 1e-6, (
        f"Expected {expected:.6f}, got {required_lambda:.6f}"
    )

    damped_A = A + required_lambda * jnp.eye(4)
    cond_num = jnp.linalg.cond(damped_A)
    assert cond_num <= target_cond + 1e-4, (
        f"Condition number {cond_num:.2f} exceeds target {target_cond}"
    )

    # Test case 3: Small target condition number (<= 1)
    with pytest.raises(ValueError):
        estimate_required_damping(jnp.array(A), 1.0)

    with pytest.raises(ValueError):
        estimate_required_damping(jnp.array(A), 0.5)

    # Test case 4: Small target condition number (2.0)
    target_cond = 2.0
    required_lambda = estimate_required_damping(jnp.array(A), target_cond)
    required_lambda = jax.device_get(required_lambda)

    expected = 100 / 1  # (100 - 2*0)/(2 - 1) = 100
    assert abs(required_lambda - expected) < 1e-6, (
        f"Expected {expected:.6f}, got {required_lambda:.6f}"
    )

    damped_A = A + required_lambda * jnp.eye(4)
    cond_num = jnp.linalg.cond(damped_A)
    assert cond_num <= target_cond + 1e-6, (
        f"Condition number {cond_num:.2f} exceeds target {target_cond}"
    )


@pytest.mark.parametrize("max_norm", [0.01, 1, None])
def test_scale_by_constrained_norm_correctness(max_norm):
    """Checks update = grad * min(lr, C/||grad||) for both modes.

    - Unconstrained: `C` is None → scale by `lr` only.
    - Constrained: `C` set and chosen to bind (lr large) → scale by `C/||g||`.
    """
    # Simple gradient tree
    grads = {"w": jnp.array([1.0, 2.0], dtype=jnp.float32)}

    # Choose parameters so constraint may bind when enabled
    lr = 0.1
    opt = scale_by_constrained_norm(learning_rate=lr, max_norm=max_norm, eps=1e-8)
    state = opt.init(params=grads)  # params unused

    updates, _new_state = opt.update(grads, state)
    u_flat, _ = ravel_pytree(updates)

    g_flat, _ = ravel_pytree(grads)
    g_norm = jnp.linalg.norm(g_flat)
    if max_norm is None:
        expected_scale = lr
    else:
        constraint_scale = max_norm / (g_norm + 1e-8)
        expected_scale = jnp.minimum(lr, constraint_scale)
    expected = g_flat * expected_scale

    chex.assert_trees_all_close(u_flat, expected, atol=1e-8, rtol=1e-8)


def test_scale_by_constrained_norm_stability():
    """Stability when grads are tiny or zero: no NaNs/Infs, correct zeros.

    - Zero grads → updates remain exactly zero regardless of `lr`/`C`.
    - Tiny grads → finite updates; scale selection remains well-defined.
    """
    # Zero gradients
    zero_grads = {"w": jnp.zeros((4,), dtype=jnp.float32)}
    opt_z = scale_by_constrained_norm(learning_rate=1.0, max_norm=0.5, eps=1e-8)
    st_z = opt_z.init(params=zero_grads)
    upd_z, _ = opt_z.update(zero_grads, st_z)
    z_flat, _ = ravel_pytree(upd_z)
    assert jnp.all(jnp.isclose(z_flat, 0.0))
    assert jnp.isfinite(z_flat).all()

    # Very small gradients
    tiny = 1e-12
    tiny_grads = {"w": jnp.full((4,), tiny, dtype=jnp.float32)}
    opt_t = scale_by_constrained_norm(learning_rate=0.1, max_norm=1.0, eps=1e-8)
    st_t = opt_t.init(params=tiny_grads)
    upd_t, _ = opt_t.update(tiny_grads, st_t)
    t_flat, _ = ravel_pytree(upd_t)
    assert jnp.isfinite(t_flat).all()

    # Expected to scale by min(lr, C/(||g||+eps)); ensure magnitude consistent
    g_flat, _ = ravel_pytree(tiny_grads)
    g_norm = jnp.linalg.norm(g_flat)
    expected_scale = jnp.minimum(0.1, 1.0 / (g_norm + 1e-8))
    chex.assert_trees_all_close(t_flat, g_flat * expected_scale, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "with_jit, cfg",
    [
        (
            False,
            dict(max_norm=None, spring_mu=None, march_beta=None, score_norm_clip=None),
        ),
        (
            True,
            dict(max_norm=0.1, spring_mu=0.95, march_beta=0.995, score_norm_clip=0.1),
        ),
        (True, "preset"),
        (True, "preset_f32"),
    ],
)
def test_robust_sr_runs_without_error(with_jit, cfg):
    """Smoke test: `robust_sr` init/update executes and produces finite updates.

    Covers both unconstrained and constrained-norm modes, with optional SPRING/MARCH.
    """
    params, samples, grads = _setup_linear_case(n_batch=5, n_params=4, complex=False)
    if cfg == "preset":
        cfg = get_preset_params(f32_dot=False)
    elif cfg == "preset_f32":
        cfg = get_preset_params(f32_dot=True)

    opt = robust_sr(
        _simple_log_psi,
        learning_rate=0.05,
        **cfg,
    )
    state = opt.init(params)

    update_fn = opt.update if not with_jit else jax.jit(opt.update)
    updates, _new_state = update_fn(grads, state, params, data=samples)

    # Finite outputs and correct pytree structure
    u_flat, _ = ravel_pytree(updates)
    g_flat, _ = ravel_pytree(grads)
    assert u_flat.shape == g_flat.shape
    assert jnp.isfinite(u_flat).all()

    # Can be applied to params (structure check)
    new_params = optax.apply_updates(params, updates)
    chex.assert_trees_all_equal_shapes(params, new_params)
