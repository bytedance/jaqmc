# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections.abc import Callable, Sequence
from enum import Enum
from functools import partial, reduce
from typing import Any, Literal, NamedTuple

import jax
import numpy as np
import optax
from jax import Array, lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax.flatten_util import ravel_pytree
from optax import tree_utils as otu

try:
    from jax import enable_x64 as _enable_x64
except ImportError:
    from jax.experimental import enable_x64 as _enable_x64  # type: ignore[no-redef]

try:
    from jax.extend import core as jcore
except ImportError:
    from jax._src import core as jcore  # type: ignore[no-redef]


def get_preset_params(f32_dot: bool = False) -> dict:
    """Get preset parameters for the optimizer.

    Args:
        f32_dot: Whether to use float32 for dot product in calculating the gram matrix.
        This is useful for GPUs with very limited F64 compute, like H20.

    Returns:
        A dictionary of preset parameters.
    """
    params = dict(
        damping=1e-3,
        max_cond_num=1e7,
        spring_mu=0.9,
        march_beta=0.5,
        march_mode="var",
        eps=1e-8,
        mixed_precision=True,  # double precision is important in inversion
        score_norm_clip=None,  # do not use score norm clipping
        score_chunk_size=128,  # to save memory; set to None for small model
        gram_num_chunks=4,  # to save memory; set to None for small model
        gram_dot_prec="F64",  # double precision to minimize numerical error
        prune_inactive=True,  # prune inactive parameters to speed up calculation
    )
    if f32_dot:  #
        params.update(
            gram_num_chunks=16,  # This is necessary to have smaller numerical error
            gram_dot_prec="F32_HIGH",  # use single precision to speed up dot product
            march_beta=0.8,  # larger beta might smooth out some numerical instability
        )
    return params


def robust_sr(
    log_psi_fn: Callable,
    *,
    learning_rate: optax.ScalarOrSchedule = 1e-2,
    max_norm: optax.ScalarOrSchedule | None = None,
    damping: optax.ScalarOrSchedule = 1e-3,
    max_cond_num: float | None = None,
    spring_mu: optax.ScalarOrSchedule | None = None,
    march_beta: optax.ScalarOrSchedule | None = None,
    march_mode: Literal["diff", "var"] = "diff",
    eps: float = 1e-8,
    mixed_precision: bool = True,
    score_in_axes: int | Sequence[Any] = 0,
    score_chunk_size: int | None = None,
    score_norm_clip: float | None = None,
    gram_num_chunks: int | None = None,
    gram_dot_prec: "PrecisionLike" = None,
    axis_name: str | None = None,
    prune_inactive: bool = False,
) -> optax.GradientTransformationExtraArgs:
    """Robust SR optimizer using constrained-norm scaling via ``optax.chain``.

    Constructs an Optax transformation by chaining:
      ``scale_by_fisher_inverse`` → ``scale_by_constrained_norm`` → ``scale(-1)``.

    Args:
        log_psi_fn: Callable ``log_psi_fn(params, sample) -> scalar`` used to build
        the centered score matrix.
        learning_rate: Step size (scalar or schedule).
        max_norm: Constrained update norm ``C`` (scalar or schedule). If ``None``,
        only the learning-rate scaling is applied.
        damping: Damping ``lambda`` (scalar or schedule).
        max_cond_num: Maximum condition number for the gram matrix by tuning damping.
        If ``None``, no adaptive damping is applied.
        spring_mu: Decay factor for the SPRING momentum accumulator (scalar or
        schedule). If ``None``, no SPRING momentum is used.
        march_beta: Decay factor for the MARCH variance accumulator (scalar or
        schedule). If ``None``, no MARCH metric is used.
        march_mode: Mode for calculating the MARCH variance accumulator. Can be
        'diff' (default, using update differences) or 'var' (using score variance
        along sample axis).
        eps: Small numerical constant for numerical stability.
        mixed_precision: Whether to use mixed precision for the gram factorization.
        score_in_axes: Axes for score computation that will be passed to vmap when
        computing the score matrix.
        score_chunk_size: Chunk size for score computation. If ``None``, full-batch
        score computation is used.
        score_norm_clip: Optional clip value for the mean of absolute values of
        each row of the score matrix. If ``None``, no clipping is applied.
        gram_num_chunks: Number of chunks for gram computation. If ``None``,
        full-batch gram computation is used.
        gram_dot_prec: Precision for the gram matrix computation. If ``None``,
        use the default (highest in dtype) precision.
        axis_name: Axis name for multi-device mapping.
        prune_inactive: Whether to structurally prune parameter leaves that do not
        contribute to the linearized ``log_psi_fn`` for the current sample
        shape. This reduces the width of the score matrix and the SR solve, but
        keeps the optimizer state layout unchanged.

    Returns:
        An ``optax.GradientTransformationExtraArgs`` that supports
        ``update(grads, state, params, data=...)``.

    Example:
        >>> from jax import numpy as jnp
        >>> import optax
        >>> def log_psi_fn(params, sample):
        ...     return params["w"] * sample
        >>> params = {"w": jnp.array(1.0)}
        >>> grads = {"w": jnp.array(0.25)}
        >>> samples = jnp.array([0.5, 1.5, 2.0, 3.0])
        >>> opt = robust_sr(
        ...     log_psi_fn,
        ...     learning_rate=optax.cosine_decay_schedule(1e-2, 1000),
        ...     max_norm=0.1,
        ...     damping=1e-3,
        ...     spring_mu=0.95,
        ...     march_beta=0.995,
        ...     march_mode="var",
        ... )
        >>> state = opt.init(params)
        >>> updates, state = opt.update(grads, state, params, data=samples)
        >>> new_params = optax.apply_updates(params, updates)
    """
    # Core SR preconditioner (requires extra args `data`).
    sr = scale_by_fisher_inverse(
        log_psi_fn,
        damping=damping,
        max_cond_num=max_cond_num,
        spring_mu=spring_mu,
        march_beta=march_beta,
        march_mode=march_mode,
        eps=eps,
        mixed_precision=mixed_precision,
        score_in_axes=score_in_axes,
        score_chunk_size=score_chunk_size,
        score_norm_clip=score_norm_clip,
        gram_num_chunks=gram_num_chunks,
        gram_dot_prec=gram_dot_prec,
        axis_name=axis_name,
        prune_inactive=prune_inactive,
    )

    # Constrained-norm scaling stage.
    constrain = scale_by_constrained_norm(
        learning_rate=learning_rate,
        max_norm=max_norm,
        eps=eps,
    )

    return optax.chain(
        sr,
        constrain,
        optax.scale(-1.0),
    )


class FisherInverseState(NamedTuple):
    """State for robust stochastic reconfiguration.

    Attributes:
        counter: Number of update steps taken.
        prev_grad: Flattened previous preconditioned update (delta_{k-1}).
        acc_var: Accumulated variance used by the MARCH metric.
    """

    counter: int
    prev_delta: Array
    acc_var: Array


def scale_by_fisher_inverse(
    log_psi_fn: Callable,
    *,
    damping: optax.ScalarOrSchedule = 1e-3,
    max_cond_num: float | None = None,
    spring_mu: optax.ScalarOrSchedule | None = None,
    march_beta: optax.ScalarOrSchedule | None = None,
    march_mode: Literal["diff", "var"] = "diff",
    eps: float = 1e-8,
    mixed_precision: bool = True,
    score_in_axes: int | Sequence[Any] = 0,
    score_chunk_size: int | None = None,
    score_norm_clip: float | None = None,
    gram_num_chunks: int | None = None,
    gram_dot_prec: "PrecisionLike" = None,
    axis_name: str | None = None,
    prune_inactive: bool = False,
) -> optax.GradientTransformationExtraArgs:
    """Robust stochastic reconfiguration preconditioner.

    This implements the robust SR, SPRING, and MARCH variants described in
    ``note.md`` using the efficient kernel formulations.

    The update operates on the flattened gradient ``g`` (``tilde{delta}``) and
    uses the centered score matrix ``O`` built from ``log_psi_fn``:

      * Robust SR (no SPRING, no MARCH):

        delta_k = (g + p) - O^T z

        with p = 0 and

          F = O O^T + lambda I
          u_g = O g
          w   = F^{-1} u_g
          z   = F^{-1} (u_g - w)

      * Robust SPRING: same as above but with momentum

          p = mu * delta_{k-1}

      * Robust MARCH: additionally introduces a diagonal metric M based on an
        exponential moving average of squared update differences (mode='diff')
        or score variance along sample axis (mode='var'), using the
        previous step statistics to build the metric for the current update.

    Args:
        log_psi_fn: Callable ``log_psi_fn(params, sample) -> scalar``.
        damping: Damping lambda (scalar or schedule).
        max_cond_num: Maximum condition number for the gram matrix by tuning damping.
            If ``None``, no adaptive damping is applied.
        spring_mu: Decay factor for the SPRING momentum accumulator (scalar or
            schedule). If ``None``, no SPRING momentum is used.
        march_beta: Decay factor for the MARCH variance accumulator (scalar or
            schedule). If ``None``, no MARCH metric is used.Goo
        march_mode: Mode for calculating the MARCH variance accumulator. Can be
            'diff' (default, using update differences) or 'var' (using score variance
            along sample axis). For 'var' mode, the acc_var is updated with current
            score before it is used to calculate the scale.
        eps: Small numerical constant for numerical stability.
        mixed_precision: Whether to use mixed precision for the Cholesky
            factorization.
        score_in_axes: Axes for score computation that will be passed to vmap when
            computing the score matrix.
        score_chunk_size: Chunk size for the score matrix computation.
            If ``None``, use vmap, otherwise use lax.map to reduce memory footprint.
        score_norm_clip: Optional clip value for the mean of absolute values of
            each row of the score matrix. If ``None``, no clipping is applied.
        gram_num_chunks: Number of chunks for the gram matrix computation.
            If not ``None``, use lax.scan to reduce memory footprint.
        gram_dot_prec: Precision for the gram matrix computation. If ``None``,
            use the default (highest in dtype) precision.
        axis_name: Axis name for multi-device mapping.
        prune_inactive: Whether to structurally prune parameter leaves that are
            inactive in the linearized ``log_psi_fn``. When enabled, the score
            matrix and SR solve operate only on active leaves, while the optimizer
            state remains full-sized for API compatibility.

    Returns:
        An ``optax.GradientTransformationExtraArgs`` taking ``(params, data)``.
    """
    grad_log_psi = adaptive_grad(log_psi_fn, argnums=0)

    get_damping = ensure_schedule(damping)
    get_mu = ensure_schedule(spring_mu)
    get_beta = ensure_schedule(march_beta)

    paxis = PAxis(axis_name)

    def init_fn(params):
        flat, _ = ravel_pytree(params)
        zeros = jnp.zeros_like(flat)
        ones = jnp.ones_like(flat)
        return FisherInverseState(counter=0, prev_delta=zeros, acc_var=ones)

    def update_fn(grads, state: FisherInverseState, params, data):
        """Apply robust Fisher preconditioning.

        Args:
          grads: Pytree of raw gradients ``tilde{delta}``.
          state: ``FisherInverseState``.
          params: Parameter pytree.
          data: Monte Carlo samples; a pytree whose leaves have leading
            dimension ``n_batch``.

        Returns:
          Tuple of preconditioned updates and the new optimizer state.

        Raises:
          ValueError: If structural pruning removes every parameter leaf.
        """
        count, prev_delta, acc_var = state

        # Schedules for this step.
        lam = get_damping(count) if get_damping is not None else 0.0
        mu = get_mu(count) if get_mu is not None else 0.0
        beta = get_beta(count) if get_beta is not None else 0.0

        # Flatten gradient.
        grad_flat, unravel_fn = ravel_pytree(grads)

        active_leaf_mask = None
        active_idx = None
        if prune_inactive:
            with jax.ensure_compile_time_eval():
                sample = take_single_sample(data, score_in_axes)
                active_leaf_mask = get_structural_active_mask(
                    log_psi_fn, params, sample
                )
                active_idx = active_leaf_indices(params, active_leaf_mask)
                if active_idx.size == 0:
                    raise ValueError(
                        "prune_inactive=True pruned all parameter leaves; "
                        "log_psi_fn does not depend on params for the given "
                        "input signature."
                    )

        # Build raw score matrix (shape: [n_batch, n_params])
        # Use lax.map to reduce memory footprint when chunk_size is given.
        score_tree = chunked_vmap(
            lambda sample: grad_log_psi(paxis.pvary(params), sample),
            in_axes=score_in_axes,
            out_axes=0,
            chunk_size=score_chunk_size,
        )(data)
        score = jax.vmap(lambda x: ravel_selected_tree(x, active_leaf_mask)[0], 0, 0)(
            score_tree
        )

        n_local, _n_params = score.shape
        n_device = paxis.size()
        n_batch = n_local * n_device

        # Preprocess score matrix.
        if score_norm_clip is not None:
            # Clip score matrix to have mean absolute value of each row <= clip.
            row_abs_mean = jnp.mean(jnp.abs(score), axis=1)
            rescale = jnp.clip(score_norm_clip / row_abs_mean, min=0.0, max=1.0)
            score = score * rescale[:, None]
            mean_tree = jax.tree.map(
                lambda x: jnp.einsum("i,i...->...", rescale, x) / n_local,
                score_tree,
            )
            mean_score = paxis.pmean(
                ravel_selected_tree(mean_tree, active_leaf_mask)[0]
            )
        else:
            mean_tree = jax.tree.map(
                lambda x: jnp.mean(x, axis=0, keepdims=True),
                score_tree,
            )
            mean_score = paxis.pmean(
                ravel_selected_tree(mean_tree, active_leaf_mask)[0]
            )
        score -= mean_score
        score /= n_batch**0.5

        # Momentum term p = mu * delta_{k-1}.
        momentum = mu * prev_delta if spring_mu is not None else 0.0
        momentum_active = get_idx(momentum, active_idx)

        # Update acc_var first for 'var' mode before calculating scale
        if march_beta is not None and march_mode == "var":
            score_var = paxis.psum(jnp.sum((score * score.conj()).real, axis=0))
            var_mask = score_var > eps  # only update those that are not very small
            score_var = normalize_masked(score_var, var_mask, mode="mean")
            acc_var_active = get_idx(acc_var, active_idx)
            acc_var_updated = beta * acc_var_active + (1.0 - beta) * score_var
            acc_var_active = jnp.where(var_mask, acc_var_updated, acc_var_active)
            acc_var = set_idx(acc_var, active_idx, acc_var_active)

        # Spring/SR => scale = 1, MARCH => learned scale.
        scale = jax.lax.rsqrt(acc_var + eps) if march_beta is not None else 1.0
        scale_active = get_idx(scale, active_idx)

        # Complex SR: concat real/imag scores along batch dim
        if jnp.iscomplexobj(score):
            score = jnp.concatenate([score.real, score.imag], axis=0)

        # Follow equations to perform robust march update. See notes for details
        scaled_grad = scale * grad_flat
        scaled_grad_active = get_idx(scaled_grad, active_idx)
        u_g = paxis.all_gather(
            score @ scaled_grad_active, tiled=True
        )  # shape: [n_batch]
        u_p = (
            paxis.all_gather(score @ momentum_active, tiled=True)
            if spring_mu is not None
            else 0.0
        )

        # Use mixed precision for calculation and inversion of gram matrix
        with _enable_x64(mixed_precision):
            # calculate and regularize gram matrix (score @ score.T)
            gram = calc_gram_matrix(
                score,
                scale_active,
                num_chunks=gram_num_chunks,
                use_x64=mixed_precision,
                dot_prec=gram_dot_prec,
                axis_name=axis_name,
            )
            gram = lift_null_space(gram, n_batch, n_device)
            if max_cond_num is not None:
                lam = jnp.maximum(lam, estimate_required_damping(gram, max_cond_num))
            gram = gram + lam * jnp.eye(gram.shape[0])

            # inversion is done with cho_solve
            chol = jsp.linalg.cho_factor(gram)
            w = jsp.linalg.cho_solve(chol, u_g)
            z = jsp.linalg.cho_solve(chol, u_g + u_p - w).astype(grad_flat.dtype)

        # Final update
        otz = paxis.psum(score.T @ paxis.pscatter(z, axis=0, tiled=True))
        delta_active = scaled_grad_active + momentum_active - scale_active * otz
        delta = set_idx(scaled_grad + momentum, active_idx, delta_active)

        # Update variance accumulator for 'diff' mode
        if march_beta is not None and march_mode == "diff":
            diff = delta - prev_delta
            acc_var = beta * acc_var + (1.0 - beta) * (diff**2)

        new_state = FisherInverseState(
            counter=count + 1,
            prev_delta=delta,
            acc_var=acc_var,
        )

        return unravel_fn(delta), new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def calc_gram_matrix(
    score: Array,
    scale: Array,
    *,
    num_chunks: int | None = None,
    use_x64: bool = True,
    dot_prec: "PrecisionLike" = None,
    axis_name: str | None = None,
) -> Array:
    """Calculate the kernel matrix ``score diag(scale) score^T``.

    Returns:
        The dense Gram matrix, optionally accumulated in chunks.
    """
    if num_chunks is not None:
        return calc_gram_matrix_chunked(
            score,
            scale,
            num_chunks=num_chunks,
            use_x64=use_x64,
            dot_prec=dot_prec,
            axis_name=axis_name,
        )
    paxis = PAxis(axis_name)
    score_w = score * jnp.sqrt(scale)  # assume positive scale

    # normalize score matrix on the row dimension
    local_row_factor = jnp.max(jnp.abs(score_w), axis=1, keepdims=True)
    local_row_factor = jnp.clip(local_row_factor, min=1e-4)
    score_norm = score_w / local_row_factor

    # communication and resharding
    score_resharded = paxis.all_to_all(
        pad_to_multiple(score_norm, paxis.size(), axis=1),
        split_axis=1,
        concat_axis=0,
        tiled=True,
    )  # [n_batch (global), n_params (local)]

    # accumulate gram matrix
    prec: Precision = get_precision(dot_prec, use_x64)
    gram_local = dot_with_precision(score_resharded, score_resharded.T, prec)
    gram_local = gram_local.astype(jnp.float64 if use_x64 else score.dtype)
    gram = paxis.psum(gram_local)

    # put back the row scales
    row_factor = paxis.all_gather(local_row_factor, axis=0, tiled=True)
    row_factor = paxis.pmax(row_factor)  # hack to make it invariant
    row_factor = row_factor.astype(gram.dtype)
    gram = gram * (row_factor @ row_factor.T)

    # extra symmetrization for better numerics
    gram = (gram + gram.T) / 2
    return gram


def calc_gram_matrix_chunked(
    score: Array,
    scale: Array,
    *,
    num_chunks: int,
    use_x64: bool = True,
    dot_prec: "PrecisionLike" = None,
    axis_name: str | None = None,
) -> Array:
    # handle shapes
    paxis = PAxis(axis_name)
    n_devices = paxis.size()
    batch_local, n_params = score.shape
    batch_global = batch_local * n_devices
    chunk_size = (n_params + num_chunks - 1) // num_chunks
    max_start = n_params - chunk_size
    prec: Precision = get_precision(dot_prec, use_x64)

    # get the row normalization factor
    local_row_factor = jnp.max(jnp.abs(score * jnp.sqrt(scale)), axis=1, keepdims=True)
    local_row_factor = jnp.clip(local_row_factor, min=1e-4)

    # loop body to be used in scan
    def body_fn(gram, chunk_id):
        # select chunk of score matrix, make sure it's within bounds
        raw_start = chunk_id * chunk_size
        start = jnp.minimum(raw_start, max_start)
        score_chunk = lax.dynamic_slice_in_dim(score, start, chunk_size, axis=1)
        scale_chunk = (
            lax.dynamic_slice_in_dim(scale, start, chunk_size, axis=0)
            if isinstance(scale, Array)
            else scale
        )

        # masking out-of-bounds elements; normalize row dimension
        param_idx = start + jnp.arange(chunk_size)
        valid = (param_idx >= raw_start) & (param_idx < n_params)
        score_chunk_w = score_chunk * jnp.sqrt(scale_chunk) * valid
        score_chunk_norm = score_chunk_w / local_row_factor

        # resharding score chunk
        score_resharded = paxis.all_to_all(
            pad_to_multiple(score_chunk_norm, n_devices, axis=1),
            split_axis=1,
            concat_axis=0,
            tiled=True,
        )  # [batch_global, chunk_size / n_devices]

        # gram accumulation
        gram_local = dot_with_precision(score_resharded, score_resharded.T, prec)
        gram_local = gram_local.astype(jnp.float64 if use_x64 else gram.dtype)
        gram = gram + paxis.psum(gram_local)
        return gram, None

    # scan over chunks
    gram0 = jnp.zeros(
        (batch_global, batch_global), dtype=jnp.float64 if use_x64 else score.dtype
    )
    gram, _ = lax.scan(body_fn, gram0, jnp.arange(num_chunks))

    # put back the row scales
    row_factor = paxis.all_gather(local_row_factor, axis=0, tiled=True)
    row_factor = paxis.pmax(row_factor)  # hack to make it invariant
    row_factor = row_factor.astype(gram.dtype)
    gram = gram * (row_factor @ row_factor.T)

    # extra symmetrization for better numerics
    gram = (gram + gram.T) / 2
    return gram


def lift_null_space(gram: Array, n_batch: int, n_device: int = 1):
    """Lift the null space of the Gram matrix so that it is full-rank.

    Returns:
        A shifted Gram matrix with the null-space directions lifted.
    """
    # for complex case, lift the null space of the real and imaginary parts separately
    assert gram.shape[0] in (n_batch, 2 * n_batch), (
        f"{gram.shape=} is not compatible with {n_batch=}"
    )
    # prepare shift matrix and vector used to calculate mean
    with jax.ensure_compile_time_eval():
        n_local = n_batch // n_device
        null_rank = gram.shape[0] // n_batch  # real: 1, complex: 2
        base = 1 - np.indices((null_rank, n_device * null_rank)).sum(0) % null_rank
        null_basis = np.kron(base, np.ones((1, n_local)))
        mean_vec = jnp.array(null_basis / np.sqrt(n_batch), dtype=gram.dtype)
        shift = jnp.array(null_basis.T @ null_basis / n_batch, dtype=gram.dtype)
    # double centering by subtracting mean of row and col and adding mean of all
    mean_oneside = (gram @ mean_vec.T) @ mean_vec
    mean_all = mean_vec.T @ (mean_vec @ mean_oneside)
    gram = gram - mean_oneside - mean_oneside.T + mean_all
    # add shift back to make sure the gram matrix is positive semi-definite
    return gram + shift


def power_iteration(A: Array, num_iters: int = 10) -> Array:
    """Estimate the largest eigenvalue with power iteration.

    Returns:
        Approximation to the largest eigenvalue of ``A``.
    """
    with jax.ensure_compile_time_eval():
        n = A.shape[0]
        key = jax.random.key(42)
        v0 = jax.random.uniform(key, (n,), dtype=A.dtype)
        v0 = v0 / jnp.linalg.norm(v0)

    def body_fun(i, v):
        v_next = jnp.dot(A, v)
        return v_next / jnp.linalg.norm(v_next)

    v_final = jax.lax.fori_loop(0, num_iters, body_fun, v0)
    max_eigenvalue = jnp.dot(v_final.conj(), jnp.dot(A, v_final))
    return max_eigenvalue


def estimate_required_damping(gram: Array, target_cond_num: float) -> Array:
    """Calculate damping required to reach a target condition number.

    Returns:
        The minimum non-negative damping that meets ``target_cond_num``.

    Raises:
        ValueError: If ``target_cond_num`` is not greater than 1.
    """
    if target_cond_num <= 1.0:
        raise ValueError(
            f"target_cond_num must be greater than 1, got {target_cond_num}"
        )
    eig_max = power_iteration(gram, num_iters=10)
    eig_min = 0  # assume the smallest eigval is 0
    required_lambda = (eig_max - target_cond_num * eig_min) / (target_cond_num - 1)
    return jnp.maximum(required_lambda, 0.0)


def scale_by_constrained_norm(
    learning_rate: optax.ScalarOrSchedule = 1e-2,
    max_norm: optax.ScalarOrSchedule | None = None,
    eps: float | None = 1e-8,
) -> optax.GradientTransformationExtraArgs:
    """Scale updates by a constrained norm with optional schedules.

    Implements the rule:
      ``update = grad * min(eta_t, C_t / ||grad||)``

    Args:
        learning_rate: Step size ``eta`` (scalar or schedule).
        max_norm: Constraint ``C`` (scalar or schedule). If ``None``, scales only
        by the learning rate schedule (unconstrained).
        eps: Small constant for numerical stability.

    Returns:
        An Optax transform that rescales updates by the constrained norm rule.
    """
    get_lr = ensure_schedule(learning_rate)
    get_norm = ensure_schedule(max_norm)

    def init_fn(params):
        return optax.ScaleByScheduleState(count=0)

    def update_fn(updates, state: optax.ScaleByScheduleState, params=None):
        # Current schedules
        lr = get_lr(state.count) if get_lr is not None else 1.0
        if get_norm is None:
            new_updates = otu.tree_scale(lr, updates)
        else:
            max_norm = get_norm(state.count)
            grad_norm = otu.tree_norm(updates)
            constraint_scale = max_norm / (grad_norm + eps)
            effective_scale = jnp.minimum(lr, constraint_scale)
            new_updates = otu.tree_scale(effective_scale, updates)

        new_state = optax.ScaleByScheduleState(count=state.count + 1)
        return new_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# Below are helper functions ##########


class Precision(Enum):
    F32_LOW = lax.Precision.DEFAULT.value
    F32_MID = lax.Precision.HIGH.value
    F32_HIGH = lax.Precision.HIGHEST.value
    F64 = 100


PrecisionLike = int | str | None | Precision


def get_precision(label: PrecisionLike, under_x64: bool) -> Precision:
    if label is None:
        precision = Precision.F64 if under_x64 else Precision.F32_HIGH
    if isinstance(label, int):
        precision = Precision(label)
    if isinstance(label, str):
        precision = Precision[label.upper()]
    if not under_x64 and precision == Precision.F64:
        raise ValueError(f"Precision {precision} is not supported without x64.")
    return precision


def dot_with_precision(a, b, precision: Precision):
    if precision == Precision.F64:
        a, b = a.astype(jnp.float64), b.astype(jnp.float64)
        return jnp.dot(a, b)
    else:
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        return jnp.dot(a, b, precision=lax.Precision(precision.value))


def ensure_schedule(value: float | optax.ScalarOrSchedule | None):
    """Convert a scalar or schedule into a callable schedule.

    None is passed through and should be handled by the caller.

    Returns:
      ``None`` or a callable ``schedule(step)``.
    """
    if value is None:
        return None
    if callable(value):
        return value
    return lambda step: value


def normalize_masked(
    x: Array, mask: Array, mode: Literal["sum", "mean"] = "mean"
) -> Array:
    """Normalize ``x`` over ``mask`` and preserve masked-off entries.

    Returns:
        The normalized array with masked-off entries left unchanged.

    Raises:
        ValueError: If ``mode`` is not ``"sum"`` or ``"mean"``.
    """
    mask = mask.astype(bool)
    masked_sum = jnp.sum(jnp.where(mask, x, jnp.zeros_like(x)))

    if mode == "sum":
        norm = masked_sum
    elif mode == "mean":
        masked_count = jnp.sum(mask.astype(x.dtype))
        norm = jnp.where(
            masked_count > 0, masked_sum / masked_count, jnp.array(1.0, x.dtype)
        )
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    norm = jnp.where(norm > 0, norm, jnp.array(1.0, x.dtype))
    return jnp.where(mask, x / norm, x)


def get_structural_active_mask(log_psi_fn, params, sample):
    """Detect parameter leaves that structurally affect ``log_psi_fn``.

    Returns:
        A pytree of booleans matching the parameter leaf structure.
    """
    tangent_params = jax.tree.map(jnp.zeros_like, params)
    flat_tangents, treedef = jax.tree_util.tree_flatten(params)
    _, lin_fn = jax.linearize(lambda p: log_psi_fn(p, sample), params)
    closed_jaxpr = jax.make_jaxpr(lin_fn)(tangent_params)
    active_invars = get_active_input_positions(closed_jaxpr.jaxpr)
    flat_mask = [idx in active_invars for idx in range(len(flat_tangents))]
    return jax.tree_util.tree_unflatten(treedef, flat_mask)


def get_active_input_positions(jaxpr, active_out_positions=None):
    """Return input positions that can reach selected outputs in a jaxpr."""
    if active_out_positions is None:
        active = {outvar for outvar in jaxpr.outvars if is_jaxpr_var(outvar)}
    else:
        active = {
            jaxpr.outvars[pos]
            for pos in active_out_positions
            if is_jaxpr_var(jaxpr.outvars[pos])
        }

    for eqn in reversed(jaxpr.eqns):
        eqn_out_positions = [
            pos
            for pos, outvar in enumerate(eqn.outvars)
            if is_jaxpr_var(outvar) and outvar in active
        ]
        if not eqn_out_positions:
            continue
        input_positions = get_eqn_active_input_positions(eqn, eqn_out_positions)
        if input_positions is None:
            for invar in eqn.invars:
                if is_jaxpr_var(invar):
                    active.add(invar)
        else:
            for pos in input_positions:
                if pos < len(eqn.invars) and is_jaxpr_var(eqn.invars[pos]):
                    active.add(eqn.invars[pos])

    return {
        idx
        for idx, invar in enumerate(jaxpr.invars)
        if is_jaxpr_var(invar) and invar in active
    }


def get_eqn_active_input_positions(eqn, active_out_positions):
    """Propagate activity through a primitive using nested jaxprs when present.

    Returns:
        Input positions that may influence ``active_out_positions``, or ``None``
        when every input should be treated as active.
    """
    if "branches" in eqn.params:
        active_inputs = set()
        for branch in eqn.params["branches"]:
            branch_jaxpr = as_jaxpr(branch)
            branch_active = get_active_input_positions(
                branch_jaxpr, active_out_positions
            )
            active_inputs.update(pos + 1 for pos in branch_active)
        return active_inputs

    if eqn.primitive.name == "scan" and "jaxpr" in eqn.params:
        scan_jaxpr = as_jaxpr(eqn.params["jaxpr"])
        return get_active_input_positions(scan_jaxpr, active_out_positions)

    for key in ("call_jaxpr", "jaxpr"):
        if key in eqn.params and eqn.primitive.name != "scan":
            child_jaxpr = as_jaxpr(eqn.params[key])
            return get_active_input_positions(child_jaxpr, active_out_positions)

    if "cond_jaxpr" in eqn.params or "body_jaxpr" in eqn.params:
        active_inputs = set()
        for key in ("cond_jaxpr", "body_jaxpr"):
            if key not in eqn.params:
                continue
            child_jaxpr = as_jaxpr(eqn.params[key])
            active_inputs.update(get_active_input_positions(child_jaxpr))
        return active_inputs

    return None


def is_jaxpr_var(value):
    return isinstance(value, jcore.Var)


def as_jaxpr(value):
    if isinstance(value, jcore.ClosedJaxpr):
        return value.jaxpr
    return value


def active_leaf_indices(params, active_leaf_mask):
    """Expand a leaf-level active mask to flattened parameter indices.

    Returns:
        Flattened parameter indices for active leaves only.
    """
    flat_params, _ = jax.tree_util.tree_flatten(params)
    flat_mask, _ = jax.tree_util.tree_flatten(active_leaf_mask)
    active_idx = []
    start = 0
    for leaf, is_active in zip(flat_params, flat_mask):
        leaf_size = int(np.prod(leaf.shape, dtype=np.int64))
        if is_active:
            active_idx.extend(range(start, start + leaf_size))
        start += leaf_size
    return np.asarray(active_idx, dtype=np.int32)


def select_tree_leaves(tree, leaf_mask):
    if leaf_mask is None:
        return tree
    flat_tree, _ = jax.tree_util.tree_flatten(tree)
    flat_mask, _ = jax.tree_util.tree_flatten(leaf_mask)
    return tuple(leaf for leaf, is_active in zip(flat_tree, flat_mask) if is_active)


def ravel_selected_tree(tree, leaf_mask):
    return ravel_pytree(select_tree_leaves(tree, leaf_mask))


def get_idx(x, active_idx):
    if active_idx is None or not jnp.shape(x):
        return x
    return x[active_idx]


def set_idx(x, active_idx, active_value):
    if active_idx is None:
        return active_value
    return x.at[active_idx].set(active_value)


def pad_to_multiple(x, k, axis=0):
    return jnp.pad(
        x,
        [(0, 0)] * axis + [(0, (-x.shape[axis]) % k)] + [(0, 0)] * (x.ndim - axis - 1),
    )


def compose(*funcs):
    def c2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))

    return reduce(c2, funcs)


def r2c_grad(f, argnums=0, has_aux=False):
    """Gradient helper that supports real and complex-valued outputs.

    Returns:
        A gradient function that preserves complex information.
    """
    if has_aux:
        return r2c_grad_with_aux(f, argnums=argnums)
    f_splited = compose(lambda x: jnp.stack([x.real, x.imag]), f)

    def grad_f(*args, **kwargs):
        jac = jax.jacrev(f_splited, argnums=argnums)(*args, **kwargs)
        return jax.tree.map(lambda x: x[0] + 1j * x[1], jac)

    return grad_f


def r2c_grad_with_aux(f, argnums=0):
    f_splited = compose(lambda x: (jnp.array([x[0].real, x[0].imag]), x[1]), f)

    def grad_f(*args, **kwargs):
        jac, aux = jax.jacrev(f_splited, argnums=argnums, has_aux=True)(*args, **kwargs)
        return jax.tree.map(lambda x: x[0] + 1j * x[1], jac), aux

    return grad_f


def adaptive_grad(f, argnums=0, has_aux=False):
    """Use real grad when possible, fall back to complex-safe grad.

    Returns:
        A gradient function that chooses the cheaper real path when possible.
    """
    rgrad_f = jax.grad(f, argnums=argnums, has_aux=has_aux)
    cgrad_f = r2c_grad(f, argnums=argnums, has_aux=has_aux)

    def agrad_f(*args, **kwargs):
        try:
            return rgrad_f(*args, **kwargs)
        except TypeError:
            return cgrad_f(*args, **kwargs)

    return agrad_f


def wrap_if_mapped(p_func):

    def p_func_if_mapped(obj, axis_name, **kwargs):
        try:
            _ = lax.axis_index(axis_name)
            return p_func(obj, axis_name, **kwargs)
        except NameError:
            return obj

    return p_func_if_mapped


def pscatter(x, axis_name, axis=0, tiled=False):
    size = lax.psum(1, axis_name)
    if tiled:
        if x.shape[axis] % size != 0:
            raise ValueError(
                f"The size of split axis ({x.shape[axis]}) "
                f"has to be divisible by the size of "
                f"the named axis {axis_name} ({size})"
            )
        x = x.reshape(*x.shape[:axis], size, -1, *x.shape[axis + 1 :])
    else:
        if x.shape[axis] != size:
            raise ValueError(
                f"The size of split axis ({x.shape[axis]}) "
                f"has to be equal to the size of "
                f"the named axis {axis_name} ({size})"
            )
    return x.take(lax.axis_index(axis_name), axis=axis)


def safe_pvary(x, axis_name):
    if hasattr(jax.lax, "pcast"):
        try:
            return jax.lax.pcast(x, axis_name, to="varying")
        except (AssertionError, ValueError):
            return x
    if hasattr(jax.lax, "pvary"):
        try:
            return jax.lax.pvary(x, axis_name)
        except (AssertionError, ValueError):
            return x
    return x


@dataclasses.dataclass(frozen=True)
class PAxis:
    name: str | None = None
    pmax: Callable = dataclasses.field(init=False)
    pmin: Callable = dataclasses.field(init=False)
    psum: Callable = dataclasses.field(init=False)
    pmean: Callable = dataclasses.field(init=False)
    all_gather: Callable = dataclasses.field(init=False)
    all_to_all: Callable = dataclasses.field(init=False)
    pscatter: Callable = dataclasses.field(init=False)
    pvary: Callable = dataclasses.field(init=False)

    def __post_init__(self):
        for nm, fn in (
            ("pmax", lax.pmax),
            ("pmin", lax.pmin),
            ("psum", lax.psum),
            ("pmean", lax.pmean),
            ("all_gather", lax.all_gather),
            ("all_to_all", lax.all_to_all),
            ("pscatter", pscatter),
            ("pvary", safe_pvary),
        ):
            object.__setattr__(
                self, nm, partial(wrap_if_mapped(fn), axis_name=self.name)
            )

    def index(self):
        try:
            return lax.axis_index(self.name)
        except NameError:
            return 0

    def size(self):
        try:
            return lax.psum(1, self.name)
        except NameError:
            return 1


def chunked_vmap(
    fun,
    in_axes: int | Sequence[Any] = 0,
    out_axes: Any = 0,
    chunk_size: int | None = None,
):
    """Chunked vmap implemented via ``lax.scan``.

    Returns:
        A wrapped function with the same signature as ``fun``.
    """

    def wrapped(*args):
        return _chunked_vmap_wrapped(fun, in_axes, out_axes, chunk_size, *args)

    return wrapped


def tree_broadcast(prefix_tree, full_tree, **kw):
    # use jax.tree.broadcast if available
    if hasattr(jax.tree, "broadcast"):
        return jax.tree.broadcast(prefix_tree, full_tree, **kw)
    # broadcast scalars/None to full tree
    if isinstance(prefix_tree, int) or prefix_tree is None:
        return jax.tree.map(lambda _: prefix_tree, full_tree)
    # otherwise, return prefix_tree as is
    return prefix_tree


def _none_is_leaf(x):
    return x is None


def _mapped_axis_sizes(args, full_in_axes):
    return jax.tree.leaves(
        jax.tree.map(
            lambda x, ax: x.shape[ax] if ax is not None else None,
            args,
            full_in_axes,
            is_leaf=_none_is_leaf,
        )
    )


def _validate_batch_sizes(axes_sizes):
    if not axes_sizes:
        raise ValueError("chunked_vmap requires at least one mapped axis")
    total_size = axes_sizes[0]
    for size in axes_sizes[1:]:
        if size != total_size:
            raise ValueError("Inconsistent batch sizes")
    return total_size


def _take_chunk_from_args(args, full_in_axes, start, size):
    return jax.tree.map(
        lambda x, ax: (
            lax.dynamic_slice_in_dim(x, start, size, ax) if ax is not None else x
        ),
        args,
        full_in_axes,
        is_leaf=_none_is_leaf,
    )


def _reorg_chunked_output(scan, rem, ax, *, num_chunks, chunk_size):
    if ax is None:
        return scan[0]
    scan = scan.reshape(num_chunks * chunk_size, *scan.shape[2:])
    full = jnp.concatenate([scan, rem], axis=0)
    return jnp.moveaxis(full, 0, ax)


def _chunked_vmap_wrapped(fun, in_axes, out_axes, chunk_size, *args):
    full_in_axes = tree_broadcast(in_axes, args, is_leaf=_none_is_leaf)
    zero_out_axes = jax.tree.map(lambda _: 0, out_axes)
    vmapped_f = jax.vmap(fun, in_axes=full_in_axes, out_axes=zero_out_axes)

    axes_sizes = _mapped_axis_sizes(args, full_in_axes)
    total_size = _validate_batch_sizes(axes_sizes)
    if chunk_size is None or chunk_size >= total_size:
        return jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)(*args)

    num_chunks = total_size // chunk_size
    remainder = total_size % chunk_size

    def scan_body(_, chunk_idx):
        start = chunk_idx * chunk_size
        chunk_args = _take_chunk_from_args(args, full_in_axes, start, chunk_size)
        return None, vmapped_f(*chunk_args)

    _, scan_ys = lax.scan(scan_body, None, jnp.arange(num_chunks))

    start = num_chunks * chunk_size
    rem_args = _take_chunk_from_args(args, full_in_axes, start, remainder)
    rem_ys = vmapped_f(*rem_args)
    full_out_axes = tree_broadcast(out_axes, rem_ys, is_leaf=_none_is_leaf)
    return jax.tree.map(
        lambda scan, rem, ax: _reorg_chunked_output(
            scan,
            rem,
            ax,
            num_chunks=num_chunks,
            chunk_size=chunk_size,
        ),
        scan_ys,
        rem_ys,
        full_out_axes,
        is_leaf=_none_is_leaf,
    )


def take_single_sample(data, in_axes, index: int = 0):
    """Extract a representative single sample matching ``score_in_axes``.

    Returns:
        A pytree with one sample selected from each mapped leaf.
    """

    def _take_single_sample_leaf(x, axis):
        if axis is None:
            return x
        return lax.index_in_dim(x, index, axis=axis, keepdims=False)

    if isinstance(in_axes, (tuple, list)) and len(in_axes) == 1:
        in_axes = in_axes[0]
    in_axes_full = tree_broadcast(in_axes, data, is_leaf=lambda x: x is None)
    return jax.tree.map(_take_single_sample_leaf, data, in_axes_full)
