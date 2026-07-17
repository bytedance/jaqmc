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

"""Complex-aware autodiff primitives.

Provides vjp, jacrev, jacfwd, and hessian that correctly handle all four
combinations of real/complex inputs and outputs (R→R, R→C, C→R, C→C).

JAX's built-in vjp has issues with the R→C case; this module works around
that via an explicit decomposition into real and imaginary VJPs.

Ported from folx/ad.py (Microsoft, MIT license).
"""

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu


def is_tree_complex(tree) -> bool:
    """Returns True if any leaf of a pytree is complex."""
    leaves = jtu.tree_leaves(tree)
    return any(jnp.iscomplexobj(leaf) for leaf in leaves)


# ---------------------------------------------------------------------------
# Complex-aware VJP
# ---------------------------------------------------------------------------


def _vjp_rc(fun, *primals: jax.Array):
    """Custom VJP for R→C functions.

    Decomposes into real and imaginary parts to avoid JAX's incorrect
    conjugation behavior on R→C VJPs.

    Returns:
        A VJP function that correctly handles R→C differentiation.
    """

    def real_fun(*primals):
        return jnp.real(fun(*primals))

    def imag_fun(*primals):
        return jnp.imag(fun(*primals))

    _, vjp_r = jax.vjp(real_fun, *primals)
    _, vjp_i = jax.vjp(imag_fun, *primals)

    def vjp_fn(*tangents: jax.Array):
        real_tangents = jtu.tree_map(jnp.real, tangents)
        imag_tangents = jtu.tree_map(jnp.imag, tangents)

        # letters: v=vector, j=jacobian, r=real, i=imag
        vr_jr = vjp_r(*real_tangents)
        vi_jr = vjp_r(*imag_tangents)
        vr_ji = vjp_i(*real_tangents)
        vi_ji = vjp_i(*imag_tangents)

        return jtu.tree_map(
            lambda a, b, c, d: a - d + 1j * (c + b),
            vr_jr,
            vi_jr,
            vr_ji,
            vi_ji,
        )

    return vjp_fn


def vjp(fun, *primals: jax.Array):
    """Returns a complex-aware VJP function for all dtype combinations.

    Unlike ``jax.vjp``, this returns only the VJP function (not the primal
    output).
    """
    out, vjp_fn = jax.vjp(fun, *primals)
    if is_tree_complex(primals) or not is_tree_complex(out):
        # C→C, C→R, R→R: standard VJP is correct
        return vjp_fn
    # R→C: use custom decomposition
    return _vjp_rc(fun, *primals)


# ---------------------------------------------------------------------------
# Complex-aware Jacobians and Hessian
# ---------------------------------------------------------------------------


def jacrev(f):
    """Returns a reverse-mode Jacobian function for complex inputs and outputs.

    Always flattens the output to a 1-D array (unlike ``jax.jacrev``).
    """

    def jacfun(*primals):
        flat_primals, unravel = jfu.ravel_pytree(primals)

        def flat_f(x):
            return jfu.ravel_pytree(f(*unravel(x)))[0]

        out = flat_f(flat_primals)

        eye = jnp.eye(out.size, dtype=out.dtype)
        if hasattr(jax.lax, "pcast"):
            eye = jax.lax.pcast(eye, tuple(jax.typeof(out).vma), to="varying")
        elif hasattr(jax.lax, "pvary"):
            eye = jax.lax.pvary(eye, tuple(jax.typeof(out).vma))
        result = jax.vmap(vjp(flat_f, flat_primals))(eye)[0]
        result = jax.vmap(unravel, out_axes=0)(result)
        if len(primals) == 1:
            return result[0]
        return result

    return jacfun


def jacfwd(f):
    """Returns a forward-mode Jacobian function for complex inputs and outputs.

    Always flattens the input to a 1-D array (unlike ``jax.jacfwd``).
    """

    def jacfun(*primals):
        flat_primals, unravel = jfu.ravel_pytree(primals)

        def jvp_fun(s):
            return jax.jvp(f, primals, unravel(s))[1]

        eye = jnp.eye(flat_primals.size, dtype=flat_primals.dtype)
        if hasattr(jax.lax, "pcast"):
            eye = jax.lax.pcast(eye, tuple(jax.typeof(flat_primals).vma), to="varying")
        elif hasattr(jax.lax, "pvary"):
            eye = jax.lax.pvary(eye, tuple(jax.typeof(flat_primals).vma))
        return jax.vmap(jvp_fun, out_axes=-1)(eye)

    return jacfun


def hessian(f):
    """Returns a complex-aware Hessian function: ``jacfwd(jacrev(f))``."""
    return jacfwd(jacrev(f))
