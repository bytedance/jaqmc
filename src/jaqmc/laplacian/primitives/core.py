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


"""Handler setup, dense wrapper families, and dense fallback mechanics."""

import functools
import logging
from collections.abc import Callable
from typing import Any, Literal, overload

import jax.numpy as jnp
import jax.tree_util as jtu
from optax.tree_utils import tree_add

from jaqmc.array_types import PyTree

from ..guards import (
    dense_jacobian_needs_materialization,
    is_dense_laptuple,
    is_laptuple,
    is_sparse_laptuple,
    lap_args_are_dense,
)
from ..hessian import (
    elementwise_jac_hessian_jac,
    general_jac_hessian_jac,
)
from ..jvp import dense_jvp
from ..types import (
    ArrayOrLapTuple,
    Arrays,
    ExtraArgs,
    ForwardFn,
    LapArgs,
    LaplacianHandler,
    LapTuple,
    LapTuples,
    MergeFn,
)

logger = logging.getLogger(__name__)

type DenseFallbackKind = Literal[
    "unrepresentable",
    "not_implemented",
]


def log_dense_fallback(*, site: str, kind: str, reason: str) -> None:
    """Log why a sparse-preserving path fell back to dense handling.

    ``not_implemented`` means the sparse model could represent the operation,
    but JaQMC has no sparse-preserving rule for this case yet. It is warning
    level because users may want to know about missing sparse support.

    ``unrepresentable`` means the operation's dependency pattern no longer fits
    the current Local1/Local2 sparse model. That is expected for some functions,
    so it stays at debug level.
    """
    level = logging.WARNING if kind == "not_implemented" else logging.DEBUG
    logger.log(level, "dense-fallback[%s] %s: %s", site, kind, reason)


def densify_tree(tree: PyTree) -> PyTree:
    """Densify every sparse ``LapTuple`` leaf inside a pytree.

    Returns:
        A pytree with the same structure where each ``LapTuple`` leaf has been
        converted to its dense representation.
    """
    return jtu.tree_map(
        lambda x: x.to_dense() if isinstance(x, LapTuple) else x,
        tree,
        is_leaf=lambda value: isinstance(value, LapTuple),
    )


def broadcast_dense_jacobian(
    jacobian: jnp.ndarray,
    output_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Broadcast a dense Jacobian to a primitive output shape.

    The dense layout keeps the tracked-input basis on axis ``0``. Scalar
    primal operands therefore carry a Jacobian shaped ``(basis,)`` and must
    gain singleton value axes before the Jacobian can broadcast over an array
    primitive result.

    Returns:
        The Jacobian in the full ``(basis, *output_shape)`` layout.
    """
    if jacobian.shape[1:] == output_shape:
        return jacobian
    if jacobian.ndim == 1:
        jacobian = jnp.reshape(
            jacobian,
            (jacobian.shape[0], *(1,) * len(output_shape)),
        )
    return jnp.broadcast_to(jacobian, (jacobian.shape[0], *output_shape))


def _partition_tracked_args(
    args: tuple[ArrayOrLapTuple, ...],
) -> tuple[LapTuples, ExtraArgs, MergeFn]:
    """Split primitive args into tracked ``LapTuple`` values and everything else.

    This stays local to handler setup because it is JAX tree plumbing, not a
    Forward Laplacian domain concept.

    Returns:
        A tuple of ``(tracked, extra, merge_fn)``.
    """
    leaves, tree_def = jtu.tree_flatten(
        args,
        is_leaf=lambda x: (
            isinstance(x, LapTuple) or not isinstance(x, (dict, list, tuple))
        ),
    )
    mask = [isinstance(leaf, LapTuple) for leaf in leaves]
    tracked = tuple(leaf for leaf, keep in zip(leaves, mask) if keep)
    extra = tuple(leaf for leaf, keep in zip(leaves, mask) if not keep)

    def merge(tracked_args: Arrays, extra_args: ExtraArgs) -> Arrays:
        tracked_iter = iter(tracked_args)
        extra_iter = iter(extra_args)
        return jtu.tree_unflatten(
            tree_def,
            [(next(tracked_iter) if keep else next(extra_iter)) for keep in mask],
        )

    return tracked, extra, merge  # type: ignore[return-value]


def setup_handler(
    fn: ForwardFn,
    args: tuple[ArrayOrLapTuple, ...],
    kwargs: dict[str, Any],
) -> tuple[ForwardFn, LapArgs[Any] | None]:
    """Prepare a registry handler invocation for derivative-aware execution.

    Registry handlers receive primitive calls as packed ``(args, kwargs)``
    rather than the primitive's original Python signature. ``setup_handler`` is
    the usual first step inside those handlers: it separates tracked
    ``LapTuple`` inputs from ordinary values and returns a forward function that
    replays ``fn`` on primal values only.

    When ``lapl_args`` is ``None``, no argument carries derivative state and
    the handler should immediately return ``merged_fwd()``.

    Args:
        fn: Primitive bind function or other primal-only callable to replay.
        args: Positional call arguments supplied to the registry handler.
        kwargs: Keyword call arguments supplied to the registry handler.

    Returns:
        A tuple ``(merged_fwd, lapl_args)``. ``merged_fwd(*tracked_x)``
        reconstructs the original call by splicing the provided primal tracked
        values back together with the untracked arguments and bound keyword
        arguments. ``lapl_args`` is ``None`` when nothing is tracked.

    Raises:
        ValueError: If dense tracked ``LapTuple`` inputs disagree on their
            tracked-input basis dimension. Dense wrappers need a shared
            tracked-input basis before Jacobian cross-terms can be combined.
    """
    tracked_args, extra, merge = _partition_tracked_args(args)

    partial_fn = functools.partial(fn, **kwargs)
    partial_fn.__name__ = getattr(fn, "__name__", "fn")  # type: ignore[attr-defined]

    def merged_fwd(*tracked_x):
        return partial_fn(*merge(tracked_x, extra))

    merged_fwd.__name__ = getattr(fn, "__name__", "fn")

    if len(tracked_args) == 0:
        return merged_fwd, None

    dense_basis_dims = [
        arg.jacobian.shape[0] for arg in tracked_args if is_dense_laptuple(arg)
    ]
    if len(dense_basis_dims) > 1 and len(set(dense_basis_dims)) != 1:
        raise ValueError(
            "Dense LapTuple Jacobians must share the same tracked-input basis: "
            f"got leading dimensions {tuple(dense_basis_dims)}."
        )
    return merged_fwd, LapArgs(tracked_args)


def fallback_dense(
    handler: LaplacianHandler,
    args: tuple[ArrayOrLapTuple, ...],
    kwargs: dict[str, Any],
    *,
    kind: DenseFallbackKind,
    reason: str = "generic dense fallback",
) -> PyTree:
    op = getattr(handler, "__name__", "handler")
    log_dense_fallback(site=op, kind=kind, reason=reason)
    return handler(densify_tree(args), kwargs)


def pack_laptuple(y: PyTree, grad_y: PyTree, lapl_y: PyTree) -> PyTree:
    """Returns a LapTuple pytree wrapping (y, grad_y, lapl_y)."""
    return jtu.tree_map(
        lambda x, jac, lapl: LapTuple(x, jac, lapl).astype(x.dtype),
        y,
        grad_y,
        lapl_y,
    )


@overload
def laplacian_handler(
    body_fn: Callable[
        [ForwardFn, LapArgs[jnp.ndarray], Any], tuple[PyTree, PyTree, PyTree]
    ],
    *,
    in_axes: Any = (),
) -> Callable[[ForwardFn], LaplacianHandler]: ...


@overload
def laplacian_handler(
    body_fn: None = None,
    *,
    in_axes: Any = (),
) -> Callable[
    [Callable[[ForwardFn, LapArgs[jnp.ndarray], Any], tuple[PyTree, PyTree, PyTree]]],
    Callable[[ForwardFn], LaplacianHandler],
]: ...


def laplacian_handler(
    body_fn: Callable[
        [ForwardFn, LapArgs[jnp.ndarray], Any], tuple[PyTree, PyTree, PyTree]
    ]
    | None = None,
    *,
    in_axes: Any = (),
) -> (
    Callable[[ForwardFn], LaplacianHandler]
    | Callable[
        [
            Callable[
                [ForwardFn, LapArgs[jnp.ndarray], Any], tuple[PyTree, PyTree, PyTree]
            ]
        ],
        Callable[[ForwardFn], LaplacianHandler],
    ]
):
    """Returns a decorator that handles setup/teardown while body does the math.

    Transforms a body function with signature
        (merged_fwd, lapl_args, in_axes) -> (y, grad_y, lapl_y)
    into a handler factory with signature
        (fn) -> handler(args, kwargs)

    Can be used with or without arguments::

        @laplacian_handler              # in_axes=() default
        @laplacian_handler(in_axes=(-1,))  # custom in_axes
    """

    def decorator(
        body_fn: Callable[
            [ForwardFn, LapArgs[jnp.ndarray], Any], tuple[PyTree, PyTree, PyTree]
        ],
    ) -> Callable[[ForwardFn], LaplacianHandler]:
        def factory(fn: ForwardFn, *, name: str = "") -> LaplacianHandler:
            def handler(
                args: tuple[ArrayOrLapTuple, ...],
                kwargs: dict[str, Any],
            ) -> PyTree:
                # The reusable wrap_* helpers implement dense Jacobian algebra.
                # Primitive-specific sparse handlers should bypass this path and
                # preserve sparse structure explicitly when they have an exact rule.
                if any(is_sparse_laptuple(arg) for arg in args):
                    return fallback_dense(
                        handler,
                        args,
                        kwargs,
                        kind="not_implemented",
                        reason=f"{body_fn.__name__} has no sparse rule",
                    )
                if any(dense_jacobian_needs_materialization(arg) for arg in args):
                    return fallback_dense(
                        handler,
                        args,
                        kwargs,
                        kind="unrepresentable",
                        reason=(
                            f"{body_fn.__name__} materializes compact dense Jacobians"
                        ),
                    )
                merged_fwd, lapl_args = setup_handler(fn, args, kwargs)
                if lapl_args is None:
                    return merged_fwd()
                y, grad_y, lapl_y = body_fn(merged_fwd, lapl_args, in_axes)
                return pack_laptuple(y, grad_y, lapl_y)

            if name:
                handler.__name__ = name
            # fn is usually a Primitive's bound `.bind` method, whose own
            # __name__ is always "bind"; the primitive itself carries the
            # real name (e.g. "sub", "gather").
            elif getattr(fn, "__name__", None) == "bind":
                primitive_name = getattr(getattr(fn, "__self__", None), "name", None)
                handler.__name__ = (
                    primitive_name if isinstance(primitive_name, str) else "handler"
                )
            return handler

        return factory

    if body_fn is not None:
        return decorator(body_fn)
    return decorator


# ---------------------------------------------------------------------------
# Named handler helpers
# ---------------------------------------------------------------------------


@laplacian_handler
def wrap_linear(
    merged_fwd: ForwardFn,
    lapl_args: LapArgs[jnp.ndarray],
    in_axes: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a dense handler for linear primitives.

    Use this for operations whose second derivative is identically zero, such
    as reshapes, transposes, slicing, and many other structure-changing array
    transforms. The wrapper computes the primal and Jacobian with the usual
    split JVP path and reuses the incoming Laplacian unchanged.

    Sparse ``LapTuple`` inputs are densified by the outer
    ``@laplacian_handler`` wrapper before this function runs.

    Returns:
        The dense Forward Laplacian triple ``(y, jacobian, laplacian)`` for
        the wrapped primitive.

    Raises:
        RuntimeError: If handler setup failed to densify tracked Jacobians.
    """
    if not lap_args_are_dense(lapl_args):
        msg = "internal error: expected LapArgs with dense Jacobian arrays"
        raise RuntimeError(msg)
    dense_args = lapl_args
    return dense_jvp(merged_fwd, dense_args, in_axes=in_axes, strategy="split")


@laplacian_handler
def wrap_elementwise(
    merged_fwd: ForwardFn,
    lapl_args: LapArgs[jnp.ndarray],
    in_axes: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a dense handler for unary shape-preserving elementwise functions.

    This wrapper is for nonlinear array functions such as ``exp``, ``log``,
    ``tanh``, or ``sin`` where each output element depends only on the
    matching input element. It uses the elementwise JVP path and then adds the
    analytic diagonal Hessian correction for that unary function.

    The wrapped primitive must accept exactly one tracked array argument and
    return an array with the same shape. Sparse ``LapTuple`` inputs are
    densified by the outer ``@laplacian_handler`` wrapper before this function
    runs.

    Returns:
        The dense Forward Laplacian triple ``(y, jacobian, laplacian)`` for
        the wrapped primitive, including the elementwise Hessian correction.

    Raises:
        RuntimeError: If handler setup failed to densify tracked Jacobians.
        RuntimeError: If the wrapped function is not unary or does not preserve
            array shape.
    """
    if not lap_args_are_dense(lapl_args):
        msg = "internal error: expected LapArgs with dense Jacobian arrays"
        raise RuntimeError(msg)
    dense_args = lapl_args
    y, grad_y, lapl_y = dense_jvp(
        merged_fwd, dense_args, in_axes=in_axes, strategy="elementwise"
    )
    if not (
        len(dense_args) == 1
        and in_axes == ()
        and isinstance(y, jnp.ndarray)
        and y.shape == dense_args.x[0].shape
    ):
        raise RuntimeError(
            "wrap_elementwise requires a unary shape-preserving array function; "
            f"got len(dense_args)={len(dense_args)}, in_axes={in_axes}, "
            f"type(y)={type(y).__name__}, y.shape={getattr(y, 'shape', None)}, "
            f"x.shape={dense_args.x[0].shape if len(dense_args) == 1 else None}."
        )
    hessian = elementwise_jac_hessian_jac(merged_fwd, dense_args)
    return y, grad_y, lapl_y + hessian


@laplacian_handler
def wrap_general(
    merged_fwd: ForwardFn,
    lapl_args: LapArgs[jnp.ndarray],
    in_axes: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the dense correctness-first fallback handler.

    Use this when no more structured wrapper applies. The helper computes the
    primal and Jacobian with the split JVP path, then adds the full dense
    Hessian contraction. It is the most general wrapper in this module and is
    appropriate for "correct first, optimize later" handler registration.

    Sparse ``LapTuple`` inputs are densified by the outer
    ``@laplacian_handler`` wrapper before this function runs.

    Returns:
        The dense Forward Laplacian triple ``(y, jacobian, laplacian)`` for
        the wrapped primitive, including the full dense Hessian contraction.

    Raises:
        RuntimeError: If handler setup failed to densify tracked Jacobians.
    """
    if not lap_args_are_dense(lapl_args):
        msg = "internal error: expected LapArgs with dense Jacobian arrays"
        raise RuntimeError(msg)
    dense_args = lapl_args
    y, grad_y, lapl_y = dense_jvp(
        merged_fwd, dense_args, in_axes=in_axes, strategy="split"
    )
    hessian = general_jac_hessian_jac(merged_fwd, dense_args)
    return y, grad_y, tree_add(lapl_y, hessian)


@laplacian_handler
def wrap_multiplication(
    merged_fwd: ForwardFn,
    lapl_args: LapArgs[jnp.ndarray],
    in_axes: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a dense handler for multiplication-style primitives.

    This wrapper covers unary scaling and binary product rules. With one
    tracked argument, it treats the primitive as multiplication by an
    untracked factor and scales the incoming Jacobian and Laplacian. With two
    tracked arguments, it applies the product rule and adds the mixed
    ``2 tr(J_lhs J_rhs^T)`` correction to the Laplacian.

    Sparse ``LapTuple`` inputs are densified by the outer
    ``@laplacian_handler`` wrapper before this function runs.

    Returns:
        The dense Forward Laplacian triple ``(y, jacobian, laplacian)`` for
        the wrapped primitive after applying the scaling or product rule.

    Raises:
        RuntimeError: If handler setup failed to densify tracked Jacobians.
    """
    del in_axes
    if not lap_args_are_dense(lapl_args):
        msg = "internal error: expected LapArgs with dense Jacobian arrays"
        raise RuntimeError(msg)
    dense_args = lapl_args
    y = merged_fwd(*dense_args.x)
    if len(dense_args) == 1:
        jac_x = dense_args.jacobian[0]
        lapl_x = dense_args.laplacian[0]
        multiplier = merged_fwd(jnp.ones_like(dense_args.x[0]))
        jac_x = broadcast_dense_jacobian(jac_x, y.shape)
        grad_y = jac_x * multiplier
        lapl_y = lapl_x * multiplier
        return y, grad_y, lapl_y
    lhs_x, rhs_x = dense_args.x
    # Cross terms need both Jacobians in the same tracked-input basis before we can
    # form tr(J_lhs J_rhs^T).
    lhs_jac, rhs_jac = dense_args.jacobian
    lhs_lapl, rhs_lapl = dense_args.laplacian
    lhs_jac = broadcast_dense_jacobian(lhs_jac, y.shape)
    rhs_jac = broadcast_dense_jacobian(rhs_jac, y.shape)
    grad_y = lhs_jac * rhs_x + rhs_jac * lhs_x
    # Product rule:
    #   Delta(x y) = y Delta(x) + x Delta(y) + 2 tr(J_x J_y^T).
    lapl_y = (
        lhs_lapl * rhs_x + rhs_lapl * lhs_x + 2 * jnp.sum(lhs_jac * rhs_jac, axis=0)
    )
    return y, grad_y, lapl_y


def wrap_componentwise(fn: ForwardFn) -> LaplacianHandler:
    """Returns a handler for ops that distribute over LapTuple components.

    Applies fn directly to value, jacobian, and laplacian without any JVP
    or Hessian computation.  Valid for linear ops where fn commutes with
    addition and scalar multiplication (e.g., conj, real, imag).
    """

    def handler(
        args: tuple[ArrayOrLapTuple, ...],
        kwargs: dict[str, Any],
    ) -> PyTree:
        x = args[0]
        if not isinstance(x, LapTuple):
            return fn(x, **kwargs)
        # Structured sparse payloads stay sparse under componentwise transforms
        # because the operation applies independently to each support block.
        if is_sparse_laptuple(x):
            return LapTuple(
                fn(x.x, **kwargs),
                x.jacobian.with_blocks(fn(x.jacobian.blocks, **kwargs)),
                fn(x.laplacian, **kwargs),
            )
        return LapTuple(
            fn(x.x, **kwargs),
            fn(x.jacobian, **kwargs),
            fn(x.laplacian, **kwargs),
        )

    return handler


def wrap_without_fwd_laplacian(fn: ForwardFn) -> LaplacianHandler:
    """Build a handler that drops derivative tracking entirely.

    The returned handler unwraps any ``LapTuple`` inputs to their primal values
    before calling ``fn`` and always returns the plain primal result. Use this
    for primitives that should act as a barrier to Forward Laplacian
    propagation rather than preserving or transforming derivative state.

    Returns:
        A registry handler that strips ``LapTuple`` inputs to their primal
        values before calling ``fn``.
    """

    def handler(
        args: tuple[ArrayOrLapTuple, ...],
        kwargs: dict[str, Any],
    ) -> PyTree:
        args, kwargs = jtu.tree_map(
            lambda a: a.x if isinstance(a, LapTuple) else a,
            (args, kwargs),
            is_leaf=is_laptuple,
        )
        return fn(*args, **kwargs)

    return handler
