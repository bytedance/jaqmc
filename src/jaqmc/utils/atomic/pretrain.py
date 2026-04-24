# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared pretrain loss estimator for atomic systems (molecule, solid)."""

from collections.abc import Callable, Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator import FunctionEstimator
from jaqmc.estimator.base import Estimator
from jaqmc.utils import parallel_jax
from jaqmc.wavefunction import NumericWavefunctionEvaluate
from jaqmc.wavefunction.base import WavefunctionEvaluate

from .scf import MolecularSCF, PeriodicSCF


def make_pretrain_log_amplitude[DataT: Data](
    log_psi_fn: WavefunctionEvaluate[DataT, jnp.ndarray],
    scf_log_amplitude_fn: Callable[[DataT], jnp.ndarray],
    scf_fraction: float = 0.0,
) -> WavefunctionEvaluate[DataT, jnp.ndarray]:
    """Creates a log amplitude function for pretraining sampling.

    Creates a function that returns either the SCF ansatz, the neural network
    ansatz, or a weighted mixture of the two. This allows sampling from an
    SCF-biased distribution during pretraining.

    Args:
        log_psi_fn: Neural network log amplitude function.
        scf_log_amplitude_fn: Function that takes data and returns the SCF
            log amplitude.
        scf_fraction: Mixing fraction for SCF (0.0 = pure NN, 1.0 = pure SCF).

    Returns:
        A log amplitude function for sampling.

    Type Parameters:
        DataT: Concrete ``Data`` subtype consumed by both input callables.

    Raises:
        ValueError: If scf_fraction is not between 0 and 1.
    """
    if scf_fraction > 1 or scf_fraction < 0:
        raise ValueError("scf_fraction must be in between 0 and 1, inclusive.")

    if scf_fraction <= 0.0:
        return log_psi_fn

    def scf_network(params, data):
        del params
        return scf_log_amplitude_fn(data)

    if scf_fraction >= 1.0:
        return scf_network

    def log_amplitude(params, data):
        log_psi = log_psi_fn(params, data)
        log_scf = scf_network(None, data)
        return (1 - scf_fraction) * log_psi + scf_fraction * log_scf

    return log_amplitude


def make_pretrain_loss(
    orbitals_fn: NumericWavefunctionEvaluate,
    scf: MolecularSCF | PeriodicSCF,
    nspins: tuple[int, int],
    full_det: bool = False,
) -> Estimator:
    """Returns a pretrain loss estimator matching NN orbitals to HF orbitals.

    Used by both molecule and solid workflows to pretrain the wavefunction
    against Hartree-Fock orbitals from an SCF calculation.

    Args:
        orbitals_fn: Function to evaluate NN orbitals.
        scf: SCF instance (MolecularSCF or PeriodicSCF).
        nspins: Electron spin counts as (n_alpha, n_beta).
        full_det: Whether to use full determinant.
    """

    def loss_fn(params: Params, data: Data) -> jnp.ndarray:
        target = scf.eval_orbitals(data["electrons"], nspins)
        orbitals = orbitals_fn(params, data)
        if full_det:
            na = target[0].shape[-2]
            nb = target[1].shape[-2]
            concat_target = jnp.block(
                [
                    [target[0], jnp.zeros((na, nb))],
                    [jnp.zeros((nb, na)), target[1]],
                ]
            )
            return jnp.mean(jnp.abs(concat_target - orbitals) ** 2)
        return jnp.array(
            [jnp.mean(jnp.abs(t - o) ** 2) for t, o in zip(target, orbitals)]
        ).sum()

    loss_and_grad_fn = jax.value_and_grad(loss_fn, argnums=0)

    def evaluate(
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats, rngs
        # By default the sharding of grads will follow params in JAX. However, grads is
        # varying but params is not varying, and this can confuse the JAX compiler.
        # This can cause bug in JAX 0.8.1. To fix this, simply add pvary to params.
        loss, grads = loss_and_grad_fn(parallel_jax.pvary(params), data)
        return {"loss": loss, "grads": grads}, state

    return FunctionEstimator(evaluate)
