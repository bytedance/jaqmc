# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Custom Forward Laplacian primitive plumbing and registry state."""

import logging
from collections.abc import Callable
from itertools import count
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.api_util import shaped_abstractify
from jax.extend.core import Primitive, jaxpr_as_fun
from jax.interpreters import ad as ad_interp
from jax.interpreters import batching, mlir

from ..guards import is_laptuple
from ..types import LaplacianHandler, LapTuple

logger = logging.getLogger(__name__)


_CUSTOM_LAPLACIAN_REGISTRY: dict[int, dict[str, Any]] = {}
_CUSTOM_LAPLACIAN_COUNTER = count(1)


custom_laplacian_call_p = Primitive("custom_laplacian_call")
custom_laplacian_call_p.multiple_results = True
custom_laplacian_call_p.call_primitive = True


class AutoLaplacianFallback(Exception):
    """Signal that a custom rule should delegate to the dense auto path."""

    def __init__(self, reason: str = "custom rule requested auto fallback"):
        super().__init__(reason)
        self.reason = reason


def _replay_custom_rule_vmaps(rule, tree_args, in_tree, vmap_history):
    """Replay outer ``vmap`` transforms around a custom Laplacian rule.

    JAX batches the staged primal primitive, but the user rule is recovered later
    from JaQMC's registry and still expects the original structured inputs. The
    recorded ``vmap_history`` lets this helper wrap the rule in equivalent
    ``jax.vmap`` calls before executing it on ``LapTuple`` arguments.

    Returns:
        A rule wrapped in the same ``vmap`` history as the staged primitive.
    """
    batched_rule = rule
    for raw_dims in vmap_history:
        dims = tuple(None if dim is batching.not_mapped else dim for dim in raw_dims)
        tree_dims = jtu.tree_unflatten(in_tree, dims)

        # A custom rule sees a LapTuple as one logical argument whose primal,
        # Jacobian, and Laplacian components are batched together. Plain array
        # in_axes would make JAX treat the three fields as unrelated leaves.
        in_axes = jtu.tree_map(
            lambda arg, axis: (
                LapTuple.pytree_spec(None, None, None)
                if isinstance(arg, LapTuple) and axis is None
                else LapTuple.pytree_spec(axis, axis + 1, axis)
                if isinstance(arg, LapTuple)
                else axis
            ),
            tree_args,
            tree_dims,
            is_leaf=is_laptuple,
        )
        prev_rule = batched_rule

        def vmapped_rule(*args, _prev_rule=prev_rule, _in_axes=in_axes):
            result = jax.vmap(_prev_rule, in_axes=_in_axes)(*args)
            return jtu.tree_map(
                lambda leaf: (
                    LapTuple(
                        leaf.x,
                        jnp.moveaxis(leaf.jacobian, 0, 1),
                        leaf.laplacian,
                    )
                    if isinstance(leaf, LapTuple)
                    else leaf
                ),
                result,
                is_leaf=is_laptuple,
            )

        batched_rule = vmapped_rule
    return batched_rule


def _custom_laplacian_impl(*flat_args, custom_id, primal_jaxpr, in_tree, vmap_history):
    del custom_id, in_tree, vmap_history
    return jaxpr_as_fun(primal_jaxpr)(*flat_args)


custom_laplacian_call_p.def_impl(_custom_laplacian_impl)


@custom_laplacian_call_p.def_abstract_eval
def _(
    *flat_args,
    custom_id,
    primal_jaxpr,
    in_tree,
    vmap_history,
):
    del flat_args, custom_id, in_tree, vmap_history
    return list(primal_jaxpr.out_avals)


def _custom_laplacian_jvp(
    primals,
    tangents,
    *,
    custom_id,
    primal_jaxpr,
    in_tree,
    vmap_history,
):
    Zero = ad_interp.Zero

    del custom_id, in_tree, vmap_history

    def materialize_tangent(primal, tangent):
        if not isinstance(tangent, Zero):
            return tangent
        if jax.dtypes.issubdtype(primal.dtype, jnp.floating) or jax.dtypes.issubdtype(
            primal.dtype, jnp.complexfloating
        ):
            return jnp.zeros_like(primal)
        return jnp.zeros_like(primal, dtype=jax.dtypes.float0)

    tangents = tuple(map(materialize_tangent, primals, tangents))

    def flat_fn(*args):
        return tuple(jaxpr_as_fun(primal_jaxpr)(*args))

    primals_out, tangents_out = jax.jvp(flat_fn, primals, tangents)
    return list(primals_out), list(tangents_out)


ad_interp.primitive_jvps[custom_laplacian_call_p] = _custom_laplacian_jvp


def _custom_laplacian_batch(
    vals,
    dims,
    *,
    custom_id,
    primal_jaxpr,
    in_tree,
    vmap_history,
):
    dims = tuple(None if dim is batching.not_mapped else dim for dim in dims)
    # Batch the primal jaxpr now so the primitive remains valid under JAX's
    # batching interpreter. The handwritten LapTuple rule is replayed later in
    # handle_custom_laplacian with the same vmap history.
    batched_fun = jax.vmap(jaxpr_as_fun(primal_jaxpr), in_axes=dims)
    dummy_vals = tuple(shaped_abstractify(v) for v in vals)
    batched_jaxpr = jax.make_jaxpr(batched_fun)(*dummy_vals)
    out_vals = custom_laplacian_call_p.bind(
        *vals,
        custom_id=custom_id,
        primal_jaxpr=batched_jaxpr,
        in_tree=in_tree,
        vmap_history=(*vmap_history, dims),
    )
    return out_vals, [0] * len(batched_jaxpr.out_avals)


batching.primitive_batchers[custom_laplacian_call_p] = _custom_laplacian_batch


def _custom_laplacian_lowering(
    ctx,
    *flat_args,
    custom_id,
    primal_jaxpr,
    in_tree,
    vmap_history,
):
    del custom_id, in_tree, vmap_history

    def flat_fn(*args):
        return tuple(jaxpr_as_fun(primal_jaxpr)(*args))

    return mlir.lower_fun(flat_fn, multiple_results=True)(ctx, *flat_args)


mlir.register_lowering(custom_laplacian_call_p, _custom_laplacian_lowering)


def handle_custom_laplacian(args, kwargs):
    custom_id = kwargs["custom_id"]
    primal_jaxpr = kwargs["primal_jaxpr"]
    in_tree = kwargs["in_tree"]
    vmap_history = kwargs["vmap_history"]

    if all(not isinstance(x, LapTuple) for x in args):
        return jaxpr_as_fun(primal_jaxpr)(*args)

    rule = _CUSTOM_LAPLACIAN_REGISTRY[custom_id]["rule"]
    if rule is None:
        raise ValueError(
            "custom_laplacian function was used inside forward_laplacian "
            "without a registered def_laplacian_rule."
        )

    tree_args = jtu.tree_unflatten(in_tree, args)
    batched_rule = _replay_custom_rule_vmaps(rule, tree_args, in_tree, vmap_history)
    try:
        result = batched_rule(*tree_args)
    except AutoLaplacianFallback as exc:
        logger.info(
            "custom_laplacian %s auto fallback for %s", rule.__name__, exc.reason
        )
        # Custom rules should keep only genuinely specialized logic by hand.
        # Unsupported cases are delegated back to the ordinary interpreter so
        # the dense formulas stay centralized in one implementation path.
        from ..interpreter import eval_jaxpr_with_forward_laplacian

        return eval_jaxpr_with_forward_laplacian(
            primal_jaxpr.jaxpr,
            primal_jaxpr.literals,
            *args,
        )
    return list(jtu.tree_leaves(result, is_leaf=is_laptuple))


CUSTOM_LAPLACIAN_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    custom_laplacian_call_p: handle_custom_laplacian,
}


def create_custom_laplacian_entry(fn: Callable) -> int:
    """Allocate a registry entry for a ``custom_laplacian``-decorated function.

    Returns:
        The integer registry id used to bind later staged calls back to this
        function and its optional handwritten rule.
    """
    custom_id = next(_CUSTOM_LAPLACIAN_COUNTER)
    _CUSTOM_LAPLACIAN_REGISTRY[custom_id] = {"fn": fn, "rule": None}
    return custom_id


def set_custom_laplacian_rule(custom_id: int, rule: Callable) -> None:
    """Install or replace the handwritten Forward Laplacian rule."""
    _CUSTOM_LAPLACIAN_REGISTRY[custom_id]["rule"] = rule


def bind_custom_laplacian(fn: Callable, custom_id: int, args: tuple):
    """Stage the custom primitive call for the traced function invocation.

    Returns:
        The traced result pytree reconstructed with the original output
        structure from the primitive's flattened outputs.
    """
    flat_args, in_tree = jtu.tree_flatten(args)
    dummy_args = jtu.tree_map(shaped_abstractify, args)

    def flat_fn(*flat_fn_args):
        tree_args = jtu.tree_unflatten(in_tree, flat_fn_args)
        result = fn(*tree_args)
        return tuple(jtu.tree_leaves(result))

    dummy_flat_args = tuple(jtu.tree_leaves(dummy_args))
    primal_jaxpr = jax.make_jaxpr(flat_fn)(*dummy_flat_args)
    out_shape = jax.eval_shape(fn, *dummy_args)
    _, out_tree = jtu.tree_flatten(out_shape)

    flat_out = custom_laplacian_call_p.bind(
        *flat_args,
        custom_id=custom_id,
        primal_jaxpr=primal_jaxpr,
        in_tree=in_tree,
        vmap_history=(),
    )
    return jtu.tree_unflatten(out_tree, flat_out)
