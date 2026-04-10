# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict

from flax import linen as nn
from jax import numpy as jnp


class ComplexLogDetOutput(TypedDict):
    """Output of :class:`LogDet` for complex-valued orbital matrices.

    Attributes:
        logpsi: Log wavefunction value (complex scalar).
        sign_logdets: Signs (phases) of individual determinants.
        abs_logdets: Log absolute values of individual determinants.
    """

    logpsi: jnp.ndarray
    sign_logdets: jnp.ndarray
    abs_logdets: jnp.ndarray


class RealLogDetOutput(TypedDict):
    """Output of :class:`LogDet` for real-valued orbital matrices.

    Attributes:
        logpsi: Log absolute wavefunction value (real scalar).
        sign_logpsi: Sign of the wavefunction (+1 or -1).
        sign_logdets: Signs of individual determinants.
        abs_logdets: Log absolute values of individual determinants.
    """

    logpsi: jnp.ndarray
    sign_logpsi: jnp.ndarray
    sign_logdets: jnp.ndarray
    abs_logdets: jnp.ndarray


class LogDet(nn.Module):
    r"""Compute log-determinant sum from orbital matrices.

    Given an orbital tensor of shape ``(ndets, n, n)``, computes

    .. math::

        \log\psi = \log\!\sum_k \operatorname{sign}(\det M_k)\,
        \exp(\log|\det M_k|)

    using the logsumexp trick for numerical stability.
    """

    def __call__(self, xs: jnp.ndarray) -> RealLogDetOutput | ComplexLogDetOutput:
        """Compute the log-sum of determinants for orbital matrices.

        Args:
            xs: Orbital matrices of shape ``(ndets, n, n)``. Complex-valued
                inputs produce complex output; real-valued inputs produce a
                log-amplitude and sign.

        Returns:
            Structured log-determinant output containing the total ``logpsi``
            and per-determinant signs and log absolute values.
        """
        signs, logdets = jnp.linalg.slogdet(xs)
        logmax = jnp.max(logdets)  # logsumexp trick
        if jnp.iscomplexobj(xs):
            return ComplexLogDetOutput(
                logpsi=jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax,
                sign_logdets=signs,
                abs_logdets=logdets,
            )
        signed_sum = jnp.sum(signs * jnp.exp(logdets - logmax))
        return RealLogDetOutput(
            sign_logpsi=jnp.sign(signed_sum),
            logpsi=jnp.log(jnp.abs(signed_sum)) + logmax,
            sign_logdets=signs,
            abs_logdets=logdets,
        )
