# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared orbital-reference pretraining utilities."""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params
from jaqmc.data import Data
from jaqmc.estimator.base import Estimator, PerWalkerEstimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep, wire
from jaqmc.wavefunction import NumericWavefunctionEvaluate, WavefunctionEvaluate


@runtime_checkable
class OrbitalReference(Protocol):
    """Protocol for references that evaluate separate alpha and beta orbitals.

    References usually come from a prepared orbital file, but may also be
    analytic, such as the free-electron plane waves used for electron-gas
    pretraining.
    """

    def eval_orbitals(
        self, pos: jnp.ndarray, nspins: tuple[int, int]
    ) -> tuple[jnp.ndarray, jnp.ndarray]: ...


@configurable_dataclass
class PretrainLossEstimator(PerWalkerEstimator):
    """Estimator matching neural-network orbitals to a reference."""

    orbitals_fn: NumericWavefunctionEvaluate = runtime_dep()
    orbital_ref: OrbitalReference = runtime_dep()
    nspins: tuple[int, int] = runtime_dep()
    full_det: bool = False

    def evaluate_single_walker(self, params, data, prev_walker_stats, state, rngs):
        del prev_walker_stats, rngs
        target = self.orbital_ref.eval_orbitals(data["electrons"], self.nspins)
        if self.full_det:
            na = target[0].shape[-2]
            nb = target[1].shape[-2]
            concat_target = jnp.block(
                [
                    [target[0], jnp.zeros((na, nb))],
                    [jnp.zeros((nb, na)), target[1]],
                ]
            )

        def loss_fn(current_params):
            orbitals = self.orbitals_fn(current_params, data)
            if self.full_det:
                return jnp.mean(jnp.abs(concat_target - orbitals) ** 2)
            losses = [
                jnp.mean(jnp.abs(target_spin - orbital_spin) ** 2)
                for count, target_spin, orbital_spin in zip(
                    self.nspins, target, orbitals, strict=True
                )
                if count
            ]
            return jnp.sum(jnp.stack(losses)) if losses else jnp.array(0.0)

        # Wrap params with pvary so grads inherit the correct (varying) sharding.
        # grads varies across the batch axis but params does not, and without pvary
        # the JAX compiler can miscompile this (observed in JAX 0.8.1).
        loss, grads = jax.value_and_grad(loss_fn)(parallel_jax.pvary(params))
        return {
            "aux:loss:pretrain": loss,
            "aux:direct_grad:pretrain": grads,
        }, state


def make_pretrain_log_amplitude[DataT: Data](
    log_psi_fn: WavefunctionEvaluate[DataT, jnp.ndarray],
    ref_log_amplitude_fn: Callable[[DataT], jnp.ndarray],
    ref_fraction: float = 0.0,
) -> WavefunctionEvaluate[DataT, jnp.ndarray]:
    """Create a log amplitude function for pretraining sampling.

    The reference normally comes from a prepared orbital file, but may also be
    an analytic model. The returned function evaluates the reference ansatz,
    the neural ansatz, or a weighted mixture of the two.

    Args:
        log_psi_fn: Neural network log amplitude function.
        ref_log_amplitude_fn: Function that takes data and returns the reference
            log amplitude.
        ref_fraction: Mixing fraction for the reference
            (0.0 = pure neural ansatz, 1.0 = pure reference).

    Returns:
        A log amplitude function for sampling.

    Type Parameters:
        DataT: Concrete ``Data`` subtype consumed by both input callables.

    Raises:
        ValueError: If ref_fraction is not between 0 and 1.
    """
    if ref_fraction > 1 or ref_fraction < 0:
        raise ValueError("ref_fraction must be in between 0 and 1, inclusive.")

    if ref_fraction <= 0.0:
        return log_psi_fn

    def ref_network(
        params: Params, data: DataT, *, precision: str | None = None
    ) -> jnp.ndarray:
        del params, precision
        return ref_log_amplitude_fn(data)

    if ref_fraction >= 1.0:
        return ref_network

    def log_amplitude(
        params: Params, data: DataT, *, precision: str | None = None
    ) -> jnp.ndarray:
        log_psi = log_psi_fn(params, data, precision=precision)
        log_ref = ref_network(params, data)
        return (1 - ref_fraction) * log_psi + ref_fraction * log_ref

    return log_amplitude


def make_pretrain_loss(
    orbitals_fn: NumericWavefunctionEvaluate,
    orbital_ref: OrbitalReference,
    nspins: tuple[int, int],
    full_det: bool = False,
) -> Estimator:
    """Return a loss estimator matching neural and reference orbitals.

    The reference may come from a prepared orbital file or an analytic model
    such as free-electron plane waves.

    Args:
        orbitals_fn: Neural orbital evaluator.
        orbital_ref: Spin-separated orbital reference.
        nspins: Electron spin counts as (n_alpha, n_beta).
        full_det: Whether ``orbitals_fn`` returns one full determinant matrix
            instead of separate matrices for the two spin channels.
    """
    estimator = PretrainLossEstimator(full_det=full_det)
    return wire(
        estimator,
        orbitals_fn=orbitals_fn,
        orbital_ref=orbital_ref,
        nspins=nspins,
    )
