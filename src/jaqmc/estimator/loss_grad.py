# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Shared VMC loss and covariance-gradient helpers.

Provides :func:`loss_term` and :class:`LossGrad` used by estimators and
:class:`~jaqmc.workflow.stage.vmc.VMCWorkStage` to compute VMC parameter
gradients:

.. math::

    \nabla_\theta \langle \mathcal{O}_L \rangle
      = 2 \operatorname{Re} \left\langle
          (\nabla_\theta \log\psi_\theta)^*
          (\mathcal{O}_L - \langle \mathcal{O}_L \rangle)
      \right\rangle
      + \langle \nabla_\theta \mathcal{O}_L \rangle

Note:
    :class:`LossGrad` sums all per-walker :math:`\partial \mathcal{L} /
    \partial \log\psi` terms into one effective local Hamiltonian, clips
    and centers it once per step, and contracts the centered weights in a
    single batched :func:`jax.vjp` backward pass. As a consequence, averaging
    ``reduce`` outputs across multiple steps is not mathematically valid for
    covariance centering or nonlinear ``loss_fn`` penalties; multi-step
    gradient accumulation requires separate handling.
"""

from collections.abc import Callable, Mapping
from functools import partial, reduce
from typing import Any, Literal

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import Estimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.array import match_first_axis_of
from jaqmc.utils.clip import clip_observable
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

__all__ = ["LossGrad", "loss_term"]


def loss_term(
    tag: str,
    local_val: jnp.ndarray,
    *,
    weight: float | None = 1.0,
    direct_grad: Params | None = None,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> dict[str, Any]:
    r"""Build per-walker auxiliary loss and gradient statistics for a loss term.

    Args:
        tag: Identifier for this loss term (reported under ``loss:{tag}``).
        local_val: Per-walker local observable values.
        weight: Multiplier applied to this term's loss and parameter gradient.
            ``0.0`` or ``None`` disables gradient computation for this term.
        direct_grad: Optional per-walker parameter gradient
            :math:`\nabla_\theta \mathcal{O}_L` for observables that depend
            directly on wavefunction parameters.
        loss_fn: Optional scalar function :math:`f(\langle \operatorname{Re}
            \mathcal{O}_L \rangle)` wrapping the expectation value.

    Returns:
        Dictionary with ``aux:loss:{tag}``, ``aux:dloss_dlogpsi:{tag}``, and
        (when ``direct_grad`` is provided) ``aux:direct_grad:{tag}``, or an
        empty dict when ``weight`` is zero or ``None``.
    """
    if not weight:
        return {}
    real_local = jnp.real(local_val)
    if loss_fn is None:
        loss = weight * real_local
        grad_scale: float | jnp.ndarray = weight
    else:
        mean_local = parallel_jax.pnanmean(real_local)
        loss = jnp.where(jnp.isnan(real_local), jnp.nan, weight * loss_fn(mean_local))
        grad_scale = weight * jax.grad(loss_fn)(mean_local)

    out: dict[str, Any] = {
        f"aux:loss:{tag}": loss,
        f"aux:dloss_dlogpsi:{tag}": grad_scale * local_val,
    }
    if direct_grad is not None:
        out[f"aux:direct_grad:{tag}"] = jax.tree.map(
            lambda x: grad_scale * x, direct_grad
        )
    return out


@configurable_dataclass
class LossGrad[DataT: Data](Estimator[DataT]):
    r"""Assemble ``aux:loss:*``, ``aux:dloss_dlogpsi:*``, and ``aux:direct_grad:*``.

    Reads per-walker ``aux:loss:*``, ``aux:dloss_dlogpsi:*``, and
    ``aux:direct_grad:*`` from ``prev_walker_stats``, sums all
    ``aux:dloss_dlogpsi:*`` terms into one effective local Hamiltonian, clips
    and centers it over valid walkers, contracts the centered sensitivity
    vector with :math:`\nabla_\theta\log\psi` in a single batched
    :func:`jax.vjp` backward pass in :meth:`evaluate_batch_walkers`, and
    reduces losses and gradients across walkers and devices in :meth:`reduce`.

    Args:
        wavefunction_precision: Wavefunction-defined precision mode passed to
            wavefunction evaluation. ``None`` inherits the wavefunction default.
        clip_method: Outlier clipping rule applied to the combined effective
            local Hamiltonian before gradient centering. ``"iqr"`` clips to
            :math:`[Q_1 - s \cdot \text{IQR},\; Q_3 + s \cdot \text{IQR}]`.
            ``"mad"`` clips to :math:`\mathrm{median}(x) \pm s \cdot
            \mathrm{median}(|x - \mathrm{median}(x)|)`. ``"none"`` disables
            clipping.
        clip_scale: Width multiplier for the selected clipping window.
            Smaller values clip more aggressively, which stabilises
            gradients but biases the estimator. Ignored when
            ``clip_method="none"``.
        mode: Gradient evaluation mode (runtime dep). ``"standard"`` contracts
            centered ``dloss_dlogpsi`` weights with :math:`\nabla_\theta\log\psi`
            via :func:`jax.vjp` and returns ``grads``. ``"sr"`` skips the VJP
            and returns centered per-walker ``dloss_dlogpsi`` weights for
            :class:`~jaqmc.optimizer.sr.SROptimizer` to contract with its
            score matrix.
        f_log_psi: Log-psi function to differentiate (runtime dep).
    """

    clip_method: Literal["iqr", "mad", "none"] = "mad"
    clip_scale: float = 5.0
    wavefunction_precision: str | None = None
    mode: Literal["standard", "sr"] = runtime_dep(default="standard")
    f_log_psi: NumericWavefunctionEvaluate[DataT] = runtime_dep()

    def _vjp_log_psi(
        self,
        params: Params,
        batched_data: BatchedData[DataT],
        dloss_dlogpsi: jnp.ndarray,
    ) -> Params:
        batch_log_psi = jax.vmap(
            partial(self.f_log_psi, precision=self.wavefunction_precision),
            in_axes=(None, batched_data.vmap_axis),
        )
        # Wrap params with pvary so grads inherit the correct (varying) sharding.
        # Without this, JAX assumes grads share params' non-varying sharding,
        # which confuses the compiler (known bug in JAX 0.8.1).
        log_psi, vjp_fn = jax.vjp(
            lambda p: batch_log_psi(p, batched_data.data),
            parallel_jax.pvary(params),
        )
        # JAX's complex VJP computes Re(cotangent * d(log_psi)), so conjugate
        # the weights to obtain Re(conj(grad log_psi) * weights).
        cotangent = jnp.conj(dloss_dlogpsi)
        if jnp.isrealobj(log_psi):
            cotangent = cotangent.real
        (local_grads,) = vjp_fn(cotangent.astype(log_psi.dtype))
        return local_grads

    def evaluate_batch_walkers(
        self,
        params: Params,
        batched_data: BatchedData[DataT],
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del rngs
        losses = {
            k.removeprefix("aux:"): v
            for k, v in prev_walker_stats.items()
            if k.startswith("aux:loss:")
        }
        if not losses:
            return {}, state

        dloss_terms = [
            v
            for k, v in prev_walker_stats.items()
            if k.startswith("aux:dloss_dlogpsi:")
        ]
        if dloss_terms:
            total_dloss = reduce(jnp.add, dloss_terms)
            clipped = clip_observable(
                total_dloss, self.clip_method, scale=self.clip_scale
            )
            valid = ~jnp.isnan(clipped)
            # In IEEE-754, NaN == NaN is False, so keep is False for both NaN
            # walkers and clipped outlier walkers.
            keep = clipped == total_dloss
            n_valid = parallel_jax.psum(jnp.sum(valid))
            mean_clipped = parallel_jax.pnanmean(clipped)
            dloss_dlogpsi: jnp.ndarray | None = jnp.where(
                valid, 2.0 * (clipped - mean_clipped) / n_valid, 0.0
            )
        else:
            valid = keep = ~jnp.isnan(reduce(jnp.add, losses.values()))
            n_valid = parallel_jax.psum(jnp.sum(valid))
            dloss_dlogpsi = None

        direct_terms = [
            v for k, v in prev_walker_stats.items() if k.startswith("aux:direct_grad:")
        ]
        if direct_terms:
            total_direct = reduce(
                lambda a, b: jax.tree.map(jnp.add, a, b), direct_terms
            )
            # Only mask walkers by observable validity/clipping (keep); do not
            # mask NaNs in direct_grad itself so wavefunction gradient NaNs
            # propagate and abort training.
            grads = jax.tree.map(
                lambda x: (
                    jnp.sum(jnp.where(match_first_axis_of(keep, x), x, 0.0), axis=0)
                    / n_valid
                ),
                total_direct,
            )
        else:
            grads = jax.tree.map(jnp.zeros_like, params)

        if self.mode == "sr":
            return {**losses, "grads": grads, "dloss_dlogpsi": dloss_dlogpsi}, state

        if dloss_dlogpsi is None:
            return {**losses, "grads": grads}, state

        vjp_grads = self._vjp_log_psi(params, batched_data, dloss_dlogpsi)
        grads = jax.tree.map(jnp.add, grads, vjp_grads)
        return {**losses, "grads": grads}, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        if not walker_stats:
            return {}
        losses = {
            k: parallel_jax.pnanmean(v)
            for k, v in walker_stats.items()
            if k.startswith("loss:")
        }
        out: dict[str, Any] = {
            **losses,
            "loss": reduce(jnp.add, losses.values()),
            "grads": parallel_jax.psum(walker_stats["grads"]),
        }
        if "dloss_dlogpsi" in walker_stats:
            out["dloss_dlogpsi"] = walker_stats["dloss_dlogpsi"]
        return out

    def finalize_stats(
        self, batched_stats: Mapping[str, Any], state: None
    ) -> dict[str, Any]:
        del state
        return jax.tree.map(lambda x: jnp.mean(x, axis=0), dict(batched_stats))
