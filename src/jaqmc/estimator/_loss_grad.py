# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared mathematical helpers for VMC loss-gradient estimators."""

import jax
import optax
from jax import numpy as jnp

from jaqmc.array_types import Params


def vmc_energy_gradient(
    mean_grad_logpsi: Params,
    mean_grad_logpsi_loss: Params,
    mean_clipped_loss: jax.Array,
) -> Params:
    r"""Return the real VMC energy gradient from sufficient statistics.

    This helper is shared by the materialized and streaming estimators so
    complex conjugation, real projection, scaling, and dtype preservation
    cannot drift between the two implementations.
    """
    grads = optax.tree.add(
        mean_grad_logpsi_loss,
        optax.tree.real(
            optax.tree.scale(
                -mean_clipped_loss,
                jax.tree.map(jnp.conj, mean_grad_logpsi),
            )
        ),
    )
    grads = jax.tree.map(
        lambda grad, log_grad: grad.astype(jnp.real(log_grad).dtype),
        grads,
        mean_grad_logpsi,
    )
    return optax.tree.scale(2, grads)
