# Copyright 2023 Microsoft Corporation
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2026.
#
# Original file was released under MIT, with the full license text
# available at licenses/folx_MIT.txt
#
# This file is distributed under the Apache License 2.0,
# with portions originally licensed under the MIT License.

"""Forward Laplacian rules for linear algebra primitives."""

from typing import Any

import jax.numpy as jnp
from jax.extend.core import Primitive

from ..guards import is_sparse_laptuple
from ..types import ArrayOrLapTuple, LaplacianHandler, LapTuple
from .core import log_dense_fallback


def handle_slogdet(
    args: tuple[ArrayOrLapTuple, ...],
    kwargs: dict[str, Any],
) -> (
    tuple[jnp.ndarray, jnp.ndarray | LapTuple[jnp.ndarray]]
    | list[LapTuple[jnp.ndarray]]
):
    del kwargs
    x = args[0]
    if is_sparse_laptuple(x):
        log_dense_fallback(
            site="slogdet",
            kind="unrepresentable",
            reason="determinant output has global input dependence",
        )
        x = x.to_dense()

    if not isinstance(x, LapTuple):
        sign, logdet = jnp.linalg.slogdet(x)
        return sign, logdet

    A = x.x
    J = x.jacobian
    L = x.laplacian
    sign_val, logdet_val = jnp.linalg.slogdet(A)
    A_inv = jnp.linalg.inv(A)
    # Matrix identities:
    #   d log det(A) = tr(A^{-1} dA)
    #   d^2 log det(A)[H, H] contributes -tr(A^{-1} H A^{-1} H).
    Ainv_J = jnp.einsum("...ab,k...bc->k...ac", A_inv, J)
    full_jac = jnp.trace(Ainv_J, axis1=-1, axis2=-2)
    full_lapl_jvp = jnp.trace(jnp.matmul(A_inv, L), axis1=-1, axis2=-2)
    hess_trace = -jnp.sum(Ainv_J * jnp.swapaxes(Ainv_J, -1, -2), axis=(0, -1, -2))
    logdet_jac = full_jac.real
    logdet_lapl = full_lapl_jvp.real + hess_trace.real
    logdet = LapTuple(logdet_val, logdet_jac, logdet_lapl)

    if not jnp.iscomplexobj(A):
        return sign_val, logdet

    # For complex matrices, sign = exp(i theta) with theta = Im(log det(A)),
    # so the phase path is differentiated through that scalar angle.
    theta_grad = full_jac.imag
    theta_lapl = full_lapl_jvp.imag + hess_trace.imag
    sign_jac = 1j * sign_val * theta_grad
    sign_lapl = sign_val * (1j * theta_lapl - jnp.sum(theta_grad * theta_grad, axis=0))
    sign = LapTuple(sign_val, sign_jac, sign_lapl)
    return [sign, logdet]


# JAX internally JITs `slogdet` so a string match can work
SLOGDET_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    "slogdet": handle_slogdet,
}
