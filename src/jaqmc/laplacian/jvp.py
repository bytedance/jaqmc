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

"""Dense JVP strategies for Forward Laplacian propagation."""

from typing import Literal

import jax
import jax.numpy as jnp

from .ad import vjp as complex_vjp
from .types import Axes, ForwardFn, LapArgs

# ---------------------------------------------------------------------------
# JVP strategies
# ---------------------------------------------------------------------------

type DenseJvpStrategy = Literal["split", "elementwise"]


def dense_elementwise_jvp(
    fwd: ForwardFn, laplace_args: LapArgs[jnp.ndarray]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Elementwise JVP via vjp trick: vjp(fwd, x)(ones_like(y)).

    Only valid for unary elementwise ops where output shape == input shape.

    WARNING: The vjp trick only recovers the *diagonal* of the Jacobian.
    This is correct for elementwise ops (diagonal Jacobian) but silently
    wrong for non-elementwise ops (e.g. rev, transpose, slice).  Those
    MUST use dense_split_jvp instead.  The ``strategy`` parameter in
    ``dense_jvp`` gates this: only ``wrap_elementwise`` and
    ``wrap_multiplication`` pass
    ``strategy="elementwise"``; all others use ``"split"``.

    Returns:
        Tuple of (value, Jacobian, Laplacian).

    Raises:
        ValueError: If output shape differs from input shape, indicating the
            handler was registered for a non-elementwise op.
    """
    y = fwd(laplace_args.x[0])
    if y.shape != laplace_args.x[0].shape:
        raise ValueError(
            f"Elementwise JVP requires output shape == input shape, "
            f"got {y.shape} != {laplace_args.x[0].shape}. "
            f"Use wrap_general instead of wrap_elementwise for this op."
        )

    jac = complex_vjp(fwd, laplace_args.x[0])(jnp.ones_like(y))[0]
    grad_y = jac * laplace_args.jacobian[0]
    lapl_y = jac * laplace_args.laplacian[0]
    return y, grad_y, lapl_y


def dense_split_jvp(
    fwd: ForwardFn, laplace_args: LapArgs[jnp.ndarray]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (value, Jacobian, Laplacian) via jax.linearize + vmap."""
    y, jvp_fn = jax.linearize(fwd, *laplace_args.x)
    jacobians = laplace_args.jacobian
    grad_y = jax.vmap(jvp_fn)(*jacobians)
    lapl_y = jvp_fn(*laplace_args.laplacian)
    return y, grad_y, lapl_y


def dense_jvp(
    fwd: ForwardFn,
    laplace_args: LapArgs[jnp.ndarray],
    in_axes: Axes,
    strategy: DenseJvpStrategy = "split",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Dispatch to the appropriate dense JVP strategy.

    Args:
        fwd: The forward function to differentiate.
        laplace_args: Tracked arguments carrying (value, Jacobian, Laplacian).
        in_axes: Axes specification for vmap scheduling.
        strategy: One of ``"split"`` or ``"elementwise"``.
            - ``"elementwise"``: vjp trick for unary elementwise ops. When the
              fast-path preconditions are not met, ``dense_jvp`` uses the
              general split strategy instead. If the fast path does run and the
              op is not shape-preserving, ``dense_elementwise_jvp`` raises.
            - "split": jax.linearize + vmap (default, general-purpose).

    Returns:
        Tuple of (value, Jacobian, Laplacian).

    Raises:
        ValueError: If ``strategy`` is unknown, or if the elementwise fast path
            is selected for a non-shape-preserving operation.
    """
    if strategy == "elementwise" and in_axes == () and len(laplace_args) == 1:
        return dense_elementwise_jvp(fwd, laplace_args)
    if strategy == "elementwise":
        return dense_split_jvp(fwd, laplace_args)
    if strategy == "split":
        return dense_split_jvp(fwd, laplace_args)
    raise ValueError(f"Unknown dense_jvp strategy: {strategy}")
