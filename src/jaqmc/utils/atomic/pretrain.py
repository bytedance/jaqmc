# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared orbital-reference pretraining utilities."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import jax
import serde
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator import FunctionEstimator
from jaqmc.estimator.base import Estimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.wavefunction import NumericWavefunctionEvaluate
from jaqmc.wavefunction.base import WavefunctionEvaluate


class OrbitalReference(Protocol):
    """Reference wavefunction that can evaluate spin-separated orbitals.

    References usually come from an SCF calculation, but may also be analytic,
    such as the free-electron plane waves used for electron-gas pretraining.
    """

    def eval_orbitals(
        self, pos: jnp.ndarray, nspins: tuple[int, int]
    ) -> tuple[jnp.ndarray, jnp.ndarray]: ...


@configurable_dataclass
class PretrainReferenceConfig:
    """Configuration for the Hartree-Fock reference used during pretraining.

    Args:
        basis: The basis set for Hartree-Fock pretrain. Can be a string
            (e.g., "sto-3g", "ccecpccpvdz") or a dict mapping element
            symbols to basis names (e.g., {"Fe": "ccecpccpvdz", "O": "cc-pvdz"}).
        sample_fraction: Mixing fraction for SCF during pretrain sampling.
            (0.0 = pure NN, 1.0 = pure SCF.)
        extra: Extra options for the PySCF mean-field object.
            When specifying in CLI, all unknown/extra fields are captured.
    """

    basis: str | Mapping[str, str] | None = "cc-pVDZ"
    sample_fraction: float = 1.0
    verbose: int = 4
    extra: dict[str, Any] = serde.field(flatten=True, default_factory=dict)


def make_pretrain_log_amplitude[DataT: Data](
    log_psi_fn: WavefunctionEvaluate[DataT, jnp.ndarray],
    ref_log_amplitude_fn: Callable[[DataT], jnp.ndarray],
    ref_fraction: float = 0.0,
) -> WavefunctionEvaluate[DataT, jnp.ndarray]:
    """Create a log amplitude function for pretraining sampling.

    The reference normally comes from an SCF calculation, but may also be an
    analytic reference. The returned function evaluates the reference ansatz,
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

    def ref_network(params, data):
        del params
        return ref_log_amplitude_fn(data)

    if ref_fraction >= 1.0:
        return ref_network

    def log_amplitude(params, data):
        log_psi = log_psi_fn(params, data)
        log_ref = ref_network(None, data)
        return (1 - ref_fraction) * log_psi + ref_fraction * log_ref

    return log_amplitude


def make_pretrain_loss(
    orbitals_fn: NumericWavefunctionEvaluate,
    orbital_ref: OrbitalReference,
    nspins: tuple[int, int],
    full_det: bool = False,
) -> Estimator:
    """Return a loss estimator matching neural and reference orbitals.

    The reference may come from an SCF calculation or an analytic model such as
    free-electron plane waves.

    Args:
        orbitals_fn: Function to evaluate NN orbitals.
        orbital_ref: Spin-separated orbital reference.
        nspins: Electron spin counts as (n_alpha, n_beta).
        full_det: Whether to use full determinant.
    """

    def loss_fn(params: Params, data: Data) -> jnp.ndarray:
        target = orbital_ref.eval_orbitals(data["electrons"], nspins)
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
