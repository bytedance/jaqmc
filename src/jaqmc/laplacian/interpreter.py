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

"""Jaxpr interpreter with Forward Laplacian propagation."""

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import overload

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var
from jax.extend.source_info_util import summarize

from jaqmc.array_types import PyTree

from .guards import is_laptuple, is_sparse_laptuple
from .primitives.core import densify_tree, log_dense_fallback, wrap_general
from .primitives.registry import get_laplacian
from .tracing import forward_laplacian_tracing
from .types import ArrayOrLapTuple, LapTuple

logger = logging.getLogger(__name__)


class JaxExprEnvironment:
    """Variable environment with reference counting.

    Frees intermediate variables as soon as they are no longer needed,
    reducing peak memory for large computation graphs.

    Reference counting scheme:
    - Invars/constvars start at 1 (write) + N (equation reads). The write
      ref is never consumed, so these persist in env for the lifetime of
      the interpreter. This keeps input and constant bindings available
      throughout the walk.
    - Equation outvars start at N (downstream reads only). They are freed
      as soon as their last downstream equation reads them.
    - Output vars are pinned to int32.max and never freed.
    """

    env: dict[Var, ArrayOrLapTuple]
    reference_counter: dict[Var, int]

    def __init__(
        self, jaxpr: Jaxpr, consts: Sequence[jnp.ndarray], *args: ArrayOrLapTuple
    ):
        """Build environment from Jaxpr constants and input arguments."""
        self.env = {}
        self.reference_counter = defaultdict(int)
        # +1 for invars/constvars ensures write() stores them (never consumed,
        # so these vars persist in env even after all equation reads).
        for v in jaxpr.invars + jaxpr.constvars:
            if isinstance(v, Literal):
                continue
            self.reference_counter[v] += 1
        # Count all reads in equations
        for eqn in jaxpr.eqns:
            for eqn_in in eqn.invars:
                if isinstance(eqn_in, Literal):
                    continue
                self.reference_counter[eqn_in] += 1
        # Output vars should never be freed
        for out in jaxpr.outvars:
            if isinstance(out, Literal):
                continue
            self.reference_counter[out] = np.iinfo(np.int32).max
        self.write_many(jaxpr.constvars, consts)
        self.write_many(jaxpr.invars, args)

    def read(self, var: Var | Literal) -> ArrayOrLapTuple:
        """Returns the variable's value, freeing it when no references remain."""
        if isinstance(var, Literal):
            return var.val
        self.reference_counter[var] -= 1
        result = self.env[var]
        if self.reference_counter[var] == 0:
            del self.env[var]
            del self.reference_counter[var]
        return result

    def write(self, var: Var, val: ArrayOrLapTuple) -> None:
        """Store a value if the variable has remaining references."""
        if self.reference_counter[var] > 0:
            self.env[var] = val

    def read_many(self, vars: Sequence[Var | Literal]) -> list[ArrayOrLapTuple]:
        """Returns values for multiple variables at once."""
        return list(map(self.read, vars))

    def write_many(
        self,
        vars: Sequence[Var],
        vals: Sequence[ArrayOrLapTuple],
    ) -> None:
        """Write multiple variable-value pairs at once."""
        list(map(self.write, vars, vals))


def eval_jaxpr_with_forward_laplacian(
    jaxpr: Jaxpr,
    consts: Sequence[jnp.ndarray],
    *args: ArrayOrLapTuple,
) -> list[ArrayOrLapTuple]:
    """Returns output values by walking equations and dispatching to handlers."""
    env = JaxExprEnvironment(jaxpr, consts, *args)

    for eqn in jaxpr.eqns:
        invals = env.read_many(eqn.invars)

        try:
            outvals = _eval_eqn(eqn, invals)
        except Exception:
            logger.error(
                "[lapjax](%s) Error in operation %s.",
                summarize(eqn.source_info),
                eqn.primitive.name,
            )
            raise

        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        env.write_many(eqn.outvars, outvals)

    return env.read_many(jaxpr.outvars)


def _eval_eqn(
    eqn: JaxprEqn,
    invals: Sequence[ArrayOrLapTuple],
) -> PyTree:
    # No LapTuple inputs -> use standard primitive (no overhead).
    if all(not isinstance(x, LapTuple) for x in invals):
        old_style, subfuns, params = _resolve_bind_params(eqn)
        if old_style:
            return eqn.primitive.bind(*subfuns, *invals, **params)
        if subfuns:
            params = dict(params, subfuns=subfuns)
        return eqn.primitive.bind(*invals, **params)
    if eqn.primitive.name == "while":
        return _eval_while(eqn, invals)
    if eqn.primitive.name == "scan":
        return _eval_scan(eqn, invals)
    if eqn.primitive.name in ("jit", "pjit"):
        return _eval_pjit(eqn, invals)
    if eqn.primitive.name == "custom_jvp_call":
        return _eval_custom_jvp(eqn, invals)
    old_style, subfuns, params = _resolve_bind_params(eqn)
    if not old_style and subfuns:
        params = dict(params, subfuns=subfuns)
    fn = get_laplacian(eqn.primitive, wrap_if_missing=True)
    assert fn is not None
    return fn(tuple(invals), params)


def _eval_scan(
    eqn: JaxprEqn,
    invals: Sequence[ArrayOrLapTuple],
) -> tuple[ArrayOrLapTuple, ...]:
    if any(is_sparse_laptuple(value) for value in invals):
        log_dense_fallback(
            site="scan",
            kind="not_implemented",
            reason="scan interpreter currently operates on dense LapTuple state",
        )
        invals = densify_tree(invals)

    n_carry = eqn.params["num_carry"]
    n_const = eqn.params["num_consts"]
    in_const = invals[:n_const]
    in_carry = invals[n_const : n_const + n_carry]
    in_inp = invals[n_const + n_carry :]

    jacobian_size = next(
        (
            value.jacobian.shape[0]
            for value in (*in_const, *in_carry, *in_inp)
            if isinstance(value, LapTuple) and isinstance(value.jacobian, jnp.ndarray)
        ),
        None,
    )

    def as_laptuple(value):
        if isinstance(value, LapTuple):
            return value
        assert jacobian_size is not None
        value = jnp.asarray(value)
        return LapTuple(
            value,
            jnp.zeros((jacobian_size, *value.shape), dtype=value.dtype),
            jnp.zeros_like(value),
        )

    def is_differentiable_plain_value(value) -> bool:
        if isinstance(value, LapTuple):
            return False
        dtype = jnp.result_type(value)
        return jax.dtypes.issubdtype(dtype, jnp.floating) or jax.dtypes.issubdtype(
            dtype, jnp.complexfloating
        )

    if jacobian_size is not None:
        in_carry = [
            as_laptuple(value) if is_differentiable_plain_value(value) else value
            for value in in_carry
        ]

    in_inp = [
        LapTuple(
            x.x,
            jnp.moveaxis(x.jacobian, 0, 1),
            x.laplacian,
        )
        if isinstance(x, LapTuple) and isinstance(x.jacobian, jnp.ndarray)
        else x
        for x in in_inp
    ]

    def wrapped(
        carry: Sequence[ArrayOrLapTuple],
        x: Sequence[ArrayOrLapTuple],
    ) -> tuple[list[ArrayOrLapTuple], list[ArrayOrLapTuple]]:
        result = eval_jaxpr_with_forward_laplacian(
            eqn.params["jaxpr"].jaxpr,
            (),
            *in_const,
            *carry,
            *x,
        )
        carry_out = result[:n_carry]
        return carry_out, result[n_carry:]

    carry, y = jax.lax.scan(
        wrapped,
        in_carry,
        in_inp,
        length=eqn.params["length"],
        reverse=eqn.params["reverse"],
        unroll=eqn.params["unroll"],
    )

    y = [
        LapTuple(
            v.x,
            jnp.moveaxis(v.jacobian, 1, 0),
            v.laplacian,
        )
        if isinstance(v, LapTuple) and isinstance(v.jacobian, jnp.ndarray)
        else v
        for v in y
    ]

    return (*carry, *y)


def _eval_while(
    eqn: JaxprEqn,
    invals: Sequence[ArrayOrLapTuple],
) -> Sequence[ArrayOrLapTuple]:
    """Evaluate ``lax.while_loop`` with derivative state in differentiable leaves.

    Returns:
        A tuple of loop-carry leaves where differentiable array leaves remain
        ``LapTuple`` values and non-differentiable leaves remain plain values.
    """
    if any(is_sparse_laptuple(value) for value in invals):
        log_dense_fallback(
            site="while",
            kind="not_implemented",
            reason="while interpreter currently operates on dense LapTuple state",
        )
        invals = densify_tree(invals)

    n_cond_consts = eqn.params["cond_nconsts"]
    n_body_consts = eqn.params["body_nconsts"]
    cond_consts = invals[:n_cond_consts]
    body_consts = invals[n_cond_consts : n_cond_consts + n_body_consts]
    carry = invals[n_cond_consts + n_body_consts :]

    def cond_fun(loop_carry: Sequence[ArrayOrLapTuple]) -> jnp.ndarray:
        cond_inputs = [
            value.x if isinstance(value, LapTuple) else value
            for value in (*cond_consts, *loop_carry)
        ]
        (pred,) = eval_jaxpr_with_forward_laplacian(
            eqn.params["cond_jaxpr"].jaxpr,
            (),
            *cond_inputs,
        )
        return pred.x if isinstance(pred, LapTuple) else pred

    def body_fun(loop_carry: Sequence[ArrayOrLapTuple]) -> list[ArrayOrLapTuple]:
        return eval_jaxpr_with_forward_laplacian(
            eqn.params["body_jaxpr"].jaxpr,
            (),
            *body_consts,
            *loop_carry,
        )

    return jax.lax.while_loop(cond_fun, body_fun, carry)


def _eval_pjit(
    eqn: JaxprEqn,
    invals: Sequence[ArrayOrLapTuple],
) -> list[ArrayOrLapTuple]:
    name = eqn.params["name"]
    if fn := get_laplacian(name):
        outvals = fn(tuple(invals), {})
        if isinstance(outvals, (LapTuple, jnp.ndarray)):
            outvals = [outvals]
        return outvals
    sub_expr: ClosedJaxpr = eqn.params["jaxpr"]
    return eval_jaxpr_with_forward_laplacian(
        sub_expr.jaxpr,
        sub_expr.literals,
        *invals,
    )


def _eval_custom_jvp(
    eqn: JaxprEqn,
    invals: Sequence[ArrayOrLapTuple],
) -> PyTree:
    # Wrap the *bound primitive* so the user's custom JVP rule is invoked.
    # Do NOT evaluate the inner call_jaxpr directly — that would bypass the
    # custom rule and fall back to the default implementation.
    old_style, subfuns, params = _resolve_bind_params(eqn)
    if old_style:
        return wrap_general(
            lambda *args: eqn.primitive.bind(*subfuns, *args, **params)
        )(tuple(invals), {})
    if subfuns:
        params = dict(params, subfuns=subfuns)
    return wrap_general(lambda *args: eqn.primitive.bind(*args, **params))(
        tuple(invals), {}
    )


@overload
def init_forward_laplacian_state(*x: jnp.ndarray) -> list[LapTuple[jnp.ndarray]]: ...
@overload
def init_forward_laplacian_state(*x: ArrayOrLapTuple) -> list[ArrayOrLapTuple]: ...
def init_forward_laplacian_state(
    *x: ArrayOrLapTuple,
) -> list[LapTuple[jnp.ndarray]] | list[ArrayOrLapTuple]:
    """Build ``LapTuple`` inputs for Forward Laplacian evaluation.

    Args:
        *x: Positional arguments. Each may be a JAX array or a nested pytree
            whose leaves are arrays. The return-value rule below also applies
            when some leaves are already ``LapTuple`` instances.

    Returns:
        When every leaf under ``*x`` is a plain array, the return value is a
        ``list`` of length ``len(x)`` that lines up with positional arguments in
        order. At each index, the list element mirrors that argument's pytree
        structure, but each leaf becomes a ``LapTuple`` with identity Jacobian
        and zero Laplacian as described above.

        When any leaf under ``*x`` is a ``LapTuple``, the function returns
        ``list(x)`` and does not wrap anything. This is deliberately
        all-or-nothing: if even one leaf is already a ``LapTuple``, bare arrays
        elsewhere are left as bare arrays. Callers who need pre-built
        derivatives on some inputs and default initialization on others must
        arrange that outside this helper (for example by constructing the
        remaining ``LapTuple`` leaves themselves).
    """
    from .seed import make_laplacian_input

    # If already LapTuple, pass through
    if any(is_laptuple(x_) for x_ in jtu.tree_leaves(x, is_leaf=is_laptuple)):
        return list(x)

    return list(make_laplacian_input(x))


def forward_laplacian(fn: Callable) -> Callable:
    """Trace fn to Jaxpr, init state, evaluate with Forward Laplacian.

    Args:
        fn: Function to trace (positional-only usage in typical call sites).

    Returns:
        A function that accepts plain arrays or pre-constructed LapTuple
        inputs and returns a pytree whose array leaves usually carry
        ``LapTuple`` derivatives. Non-floating outputs may still drop back to
        plain arrays, matching :meth:`LapTuple.astype`.

        When every input leaf is a plain array, each leaf is tracked with the
        default identity Jacobian and zero Laplacian. Once any input leaf is
        already a ``LapTuple``, those tracked leaves are used as-is and every
        remaining plain-array leaf is treated as an untracked constant.

        Returned ``LapTuple.jacobian`` payloads may stay structured sparse when
        the propagated state remains representable in JaQMC's sparse model.
        Callers that need a uniform dense Jacobian array should read
        ``LapTuple.dense_jacobian`` instead of assuming ``jacobian`` is already
        dense.
    """

    def wrapped(*args: ArrayOrLapTuple) -> PyTree:
        # Extract plain arrays for tracing
        args_arr = jtu.tree_map(
            lambda x: x.x if is_laptuple(x) else x,
            args,
            is_leaf=is_laptuple,
        )
        with forward_laplacian_tracing():
            closed_jaxpr, out_shape = jax.make_jaxpr(fn, return_shape=True)(*args_arr)

        flat_args = jtu.tree_leaves(args, is_leaf=is_laptuple)
        lapl_args = init_forward_laplacian_state(*flat_args)

        out = eval_jaxpr_with_forward_laplacian(
            closed_jaxpr.jaxpr,
            closed_jaxpr.literals,
            *lapl_args,
        )

        out_structure = jtu.tree_structure(out_shape)
        return jtu.tree_unflatten(out_structure, out)

    return wrapped


def _resolve_bind_params(
    eqn: JaxprEqn,
) -> tuple[bool, tuple[Callable, ...], dict[str, object]]:
    """Normalize bind params across old and new JAX contracts.

    Returns:
        A tuple ``(old_style, subfuns, params)`` where ``old_style`` indicates
        whether JAX returned ``(subfuns, params)`` directly.
    """
    bind_params = eqn.primitive.get_bind_params(eqn.params)
    if isinstance(bind_params, tuple) and len(bind_params) == 2:
        subfuns, params = bind_params
        return True, tuple(subfuns), dict(params)
    params = dict(bind_params)
    subfuns = tuple(params.pop("subfuns", ()))
    return False, subfuns, params
