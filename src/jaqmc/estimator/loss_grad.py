# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from functools import partial
from typing import Any, Literal

import jax
import optax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
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
        return {
            "logpsi": primals,
            "grad_logpsi": grads,
            "loss": prev_walker_stats[self.loss_key],
        }, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        clipped_loss = clip_observable(
            walker_stats["loss"], self.clip_method, scale=self.clip_scale
        )
        grad_logpsi_and_loss = jax.tree.map(
            lambda x: jnp.real(jnp.conj(x) * match_first_axis_of(clipped_loss, x)),
            walker_stats["grad_logpsi"],
        )
        return mean_reduce(
            {
                "grad_logpsi_and_loss": grad_logpsi_and_loss,
                "loss": walker_stats["loss"],
                "clipped_loss": clipped_loss,
                "grad_logpsi": walker_stats["grad_logpsi"],
            }
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
            optax.tree.real(optax.tree.scale(-clipped_loss, grad_logpsi)),
        )
        return {"loss": loss, "grads": optax.tree.scale(2, grads)}
