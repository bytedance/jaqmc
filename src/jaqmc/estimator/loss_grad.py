# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections.abc import Mapping
from functools import partial
from typing import Any, Literal

import jax
import optax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import Estimator, PerWalkerEstimator, mean_reduce
from jaqmc.utils import parallel_jax
from jaqmc.utils.array import match_first_axis_of
from jaqmc.utils.clip import clip_observable
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.func_transform import transform_maybe_complex
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate


@configurable_dataclass
class LossAndGrad(PerWalkerEstimator):
    r"""Estimator that computes the VMC loss and parameter gradients.

    The gradient of the variational energy with respect to wavefunction
    parameters :math:`\theta` is:

    .. math::

        \nabla_\theta \langle E_L \rangle
          = 2 \left\langle
              (E_L - \langle E_L \rangle) \,
              \nabla_\theta \log|\psi_\theta|
          \right\rangle

    The computation proceeds in three stages across the estimator
    lifecycle:

    1. ``evaluate_single_walker`` — evaluates :math:`\log\psi` and its
       parameter gradient for each walker, and reads the loss value.
    2. ``reduce`` — optionally clips the local energies for outlier
       robustness, then forms the per-walker product
       :math:`\nabla\log\psi \cdot E_L^{\text{clipped}}`.
    3. ``finalize`` — averages over walkers and subtracts the baseline
       to assemble the final gradient.

    Args:
        loss_key: Key in prev_walker_stats to use as the loss.
        clip_method: Outlier clipping rule used for gradient weighting.
            ``"iqr"`` clips to :math:`[Q_1 - s \cdot \text{IQR},\;
            Q_3 + s \cdot \text{IQR}]`. ``"mad"`` clips to
            :math:`\mathrm{median}(E_L) \pm s \cdot
            \mathrm{median}(|E_L - \mathrm{median}(E_L)|)`.
            ``"none"`` disables clipping.
        clip_scale: Width multiplier for the selected clipping window.
            Smaller values clip more aggressively, which stabilises
            gradients but biases the estimator. Ignored when
            ``clip_method="none"``.
        f_log_psi: Log-psi function to differentiate (runtime dep).
    """

    loss_key: str = "total_energy"
    clip_method: Literal["iqr", "mad", "none"] = "mad"
    clip_scale: float = 5.0
    validity_key: str | None = None
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del rngs
        value_and_grad_f = transform_maybe_complex(self.f_log_psi, jax.value_and_grad)
        # Wrap params with pvary so grads inherit the correct (varying) sharding.
        # Without this, JAX assumes grads share params' non-varying sharding,
        # which confuses the compiler (known bug in JAX 0.8.1).
        primals, grads = value_and_grad_f(parallel_jax.pvary(params), data)
        if not jnp.isscalar(prev_walker_stats[self.loss_key]):
            raise ValueError(
                "Expected scalar loss value (i.e. ndim=0), got shape "
                f"{prev_walker_stats[self.loss_key].shape}."
            )
        # We copy over loss stats because we need to do customized reduce
        result = {
            "logpsi": primals,
            "grad_logpsi": grads,
            "loss": prev_walker_stats[self.loss_key],
        }
        if self.validity_key is not None:
            result["loss_valid"] = prev_walker_stats[self.validity_key]
        return result, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        loss = walker_stats["loss"]
        grad_logpsi = walker_stats["grad_logpsi"]
        if self.validity_key is not None:
            valid = walker_stats["loss_valid"]
            loss = jnp.where(valid, loss, jnp.nan)
            grad_logpsi = jax.tree.map(
                lambda x: jnp.where(match_first_axis_of(valid, x), x, jnp.nan),
                grad_logpsi,
            )
        clipped_loss = clip_observable(loss, self.clip_method, scale=self.clip_scale)
        grad_logpsi_and_loss = jax.tree.map(
            lambda x: jnp.real(jnp.conj(x) * match_first_axis_of(clipped_loss, x)),
            grad_logpsi,
        )
        return mean_reduce(
            {
                "grad_logpsi_and_loss": grad_logpsi_and_loss,
                "loss": loss,
                "clipped_loss": clipped_loss,
                "grad_logpsi": grad_logpsi,
            },
            include_variance=False,
        )

    def finalize_stats(
        self, batch_stats: Mapping[str, Any], state: None
    ) -> dict[str, Any]:
        del state
        batch_mean = partial(jnp.mean, axis=0)
        grad_logpsi_and_loss = jax.tree.map(
            batch_mean, batch_stats["grad_logpsi_and_loss"]
        )
        grad_logpsi = jax.tree.map(batch_mean, batch_stats["grad_logpsi"])
        clipped_loss = batch_mean(batch_stats["clipped_loss"])
        loss = batch_mean(batch_stats["loss"])
        grads = optax.tree.add(
            grad_logpsi_and_loss,
            optax.tree.real(
                optax.tree.scale(
                    -clipped_loss, jax.tree.map(jnp.conj, grad_logpsi)
                )
            ),
        )
        grads = jax.tree.map(
            lambda grad, log_grad: grad.astype(jnp.real(log_grad).dtype),
            grads,
            grad_logpsi,
        )
        return {"loss": loss, "grads": optax.tree.scale(2, grads)}


@configurable_dataclass
class StreamingLossAndGrad(Estimator):
    """Compute VMC gradients with chunk-local sufficient-statistic reduction.

    Unlike :class:`LossAndGrad`, this estimator never materializes a
    ``[walkers, ...parameters]`` gradient tree. Each chunk's per-walker
    gradients are reduced immediately inside ``lax.scan``.
    """

    loss_key: str = "total_energy"
    vmap_chunk_size: int | None = None
    clip_method: Literal["iqr", "mad", "none"] = "mad"
    clip_scale: float = 5.0
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()

    def evaluate_batch_walkers(
        self,
        params: Params,
        batched_data: BatchedData,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del rngs
        loss = prev_walker_stats[self.loss_key]
        if loss.ndim != 1:
            raise ValueError(
                f"Expected one scalar loss per walker, got shape {loss.shape}."
            )
        batch_size = batched_data.batch_size
        if batch_size < 1:
            raise ValueError("StreamingLossAndGrad requires a non-empty batch.")
        chunk_size = self.vmap_chunk_size or batch_size
        if chunk_size < 1:
            raise ValueError("vmap_chunk_size must be positive or None.")
        chunk_size = min(chunk_size, batch_size)
        n_chunks = (batch_size + chunk_size - 1) // chunk_size
        padded_size = n_chunks * chunk_size
        pad_size = padded_size - batch_size

        clipped_loss = clip_observable(
            loss, self.clip_method, scale=self.clip_scale
        )

        def pad_leaf(x):
            if pad_size == 0:
                return x
            pad_width = ((0, pad_size),) + ((0, 0),) * (x.ndim - 1)
            return jnp.pad(x, pad_width, mode="edge")

        padded_data = dataclasses.replace(
            batched_data.data,
            **{
                name: jax.tree.map(pad_leaf, batched_data.data[name])
                for name in batched_data.fields_with_batch
            },
        )
        padded_loss = pad_leaf(clipped_loss)
        padded_mask = jnp.arange(padded_size) < batch_size

        value_and_grad_f = transform_maybe_complex(
            self.f_log_psi, jax.value_and_grad
        )
        grad_shape = jax.eval_shape(
            lambda p, d: value_and_grad_f(p, d)[1],
            params,
            batched_data.unbatched_example(),
        )
        sum_grad = jax.tree.map(
            lambda x: jnp.zeros(x.shape, x.dtype), grad_shape
        )
        sum_grad_loss = jax.tree.map(
            lambda x: jnp.zeros(x.shape, jnp.real(jnp.zeros((), x.dtype)).dtype),
            grad_shape,
        )
        # Chunk sums vary across data shards even though parameters are shared.
        # Mark the scan carry accordingly so shard_map VMA stays stable.
        sum_grad = parallel_jax.pvary(sum_grad)
        sum_grad_loss = parallel_jax.pvary(sum_grad_loss)
        vmap_axis = batched_data.vmap_axis
        vmapped_grad = jax.vmap(value_and_grad_f, in_axes=(None, vmap_axis))

        def scan_body(carry, chunk_index):
            start = chunk_index * chunk_size
            chunk_data = dataclasses.replace(
                padded_data,
                **{
                    name: jax.tree.map(
                        lambda x: jax.lax.dynamic_slice_in_dim(
                            x, start, chunk_size, axis=0
                        ),
                        padded_data[name],
                    )
                    for name in batched_data.fields_with_batch
                },
            )
            loss_chunk = jax.lax.dynamic_slice_in_dim(
                padded_loss, start, chunk_size, axis=0
            )
            mask_chunk = jax.lax.dynamic_slice_in_dim(
                padded_mask, start, chunk_size, axis=0
            )
            _, grads = vmapped_grad(parallel_jax.pvary(params), chunk_data)
            grads = jax.tree.map(
                lambda x: jnp.where(match_first_axis_of(mask_chunk, x), x, 0),
                grads,
            )
            grad_loss = jax.tree.map(
                lambda x: jnp.real(
                    jnp.conj(x) * match_first_axis_of(loss_chunk, x)
                ).astype(jnp.real(x).dtype),
                grads,
            )
            chunk_sum_grad = jax.tree.map(lambda x: jnp.sum(x, axis=0), grads)
            chunk_sum_grad_loss = jax.tree.map(
                lambda x: jnp.sum(x, axis=0), grad_loss
            )
            return (
                jax.tree.map(jnp.add, carry[0], chunk_sum_grad),
                jax.tree.map(jnp.add, carry[1], chunk_sum_grad_loss),
            ), None

        (sum_grad, sum_grad_loss), _ = jax.lax.scan(
            scan_body, (sum_grad, sum_grad_loss), jnp.arange(n_chunks)
        )
        return {
            "sum_grad_logpsi": sum_grad,
            "sum_grad_logpsi_and_loss": sum_grad_loss,
            "sum_loss": jnp.sum(loss),
            "sum_clipped_loss": jnp.sum(clipped_loss),
            "count": jnp.asarray(batch_size),
        }, state

    def reduce(self, stats: Mapping[str, Any]) -> dict[str, Any]:
        count = parallel_jax.psum(stats["count"])
        return {
            "grad_logpsi": jax.tree.map(
                lambda x: parallel_jax.psum(x) / count,
                stats["sum_grad_logpsi"],
            ),
            "grad_logpsi_and_loss": jax.tree.map(
                lambda x: parallel_jax.psum(x) / count,
                stats["sum_grad_logpsi_and_loss"],
            ),
            "loss": parallel_jax.psum(stats["sum_loss"]) / count,
            "clipped_loss": (
                parallel_jax.psum(stats["sum_clipped_loss"]) / count
            ),
        }

    def finalize_stats(
        self, batch_stats: Mapping[str, Any], state: None
    ) -> dict[str, Any]:
        del state
        batch_mean = partial(jnp.mean, axis=0)
        grad_logpsi_and_loss = jax.tree.map(
            batch_mean, batch_stats["grad_logpsi_and_loss"]
        )
        grad_logpsi = jax.tree.map(batch_mean, batch_stats["grad_logpsi"])
        clipped_loss = batch_mean(batch_stats["clipped_loss"])
        loss = batch_mean(batch_stats["loss"])
        grads = optax.tree.add(
            grad_logpsi_and_loss,
            optax.tree.real(
                optax.tree.scale(
                    -clipped_loss, jax.tree.map(jnp.conj, grad_logpsi)
                )
            ),
        )
        grads = jax.tree.map(
            lambda grad, log_grad: grad.astype(jnp.real(log_grad).dtype),
            grads,
            grad_logpsi,
        )
        return {"loss": loss, "grads": optax.tree.scale(2, grads)}
