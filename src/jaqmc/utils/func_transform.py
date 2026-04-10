# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Concatenate

import jax
from jax import numpy as jnp

from jaqmc.array_types import ArrayTree, Params
from jaqmc.data import Data


def with_real[**P, ReturnT](f: Callable[P, ReturnT]) -> Callable[P, ReturnT]:
    """Wrap ``f`` so that only real parts of its outputs are returned.

    Args:
        f: Callable to wrap.

    Returns:
        A wrapped function that applies ``jnp.real`` to all outputs.

    Type Parameters:
        P: Parameter specification of ``f`` (arguments are preserved).
        ReturnT: Return type of ``f`` before applying ``jnp.real`` tree-wise.

    Examples:
        >>> import jax.numpy as jnp
        >>> f = lambda x: x + 1j * x**2
        >>> float(with_real(f)(2.0))
        2.0
    """

    def wrapped(*args: P.args, **kwargs: P.kwargs):
        return jax.tree.map(jnp.real, f(*args, **kwargs))

    return wrapped


def with_imag[**P, ReturnT](f: Callable[P, ReturnT]) -> Callable[P, ReturnT]:
    """Wrap ``f`` so that only imaginary parts of its outputs are returned.

    Args:
        f: Callable to wrap.

    Returns:
        A wrapped function that applies ``jnp.imag`` to all outputs.

    Type Parameters:
        P: Parameter specification of ``f`` (arguments are preserved).
        ReturnT: Return type of ``f`` before applying ``jnp.imag`` tree-wise.

    Examples:
        >>> import jax.numpy as jnp
        >>> f = lambda x: x + 1j * x**2
        >>> float(with_imag(f)(2.0))
        4.0
    """

    def wrapped(*args: P.args, **kwargs: P.kwargs):
        return jax.tree.map(jnp.imag, f(*args, **kwargs))

    return wrapped


def with_output[**P](f: Callable[P, Mapping[str, Any]], key: str) -> Callable[P, Any]:
    """Wrap ``f`` to return only ``f(...)[key]``.

    Args:
        f: Callable returning a mapping.
        key: Mapping key to extract.

    Returns:
        A wrapped function that extracts ``key`` from the output mapping.

    Type Parameters:
        P: Parameter specification of ``f`` (arguments are preserved).

    Examples:
        >>> g = lambda x: {"energy": x**2, "force": -2*x}
        >>> with_output(g, "energy")(3.0)
        9.0
    """

    def wrapped(*args: P.args, **kwargs: P.kwargs):
        return f(*args, **kwargs)[key]

    return wrapped


def transform_maybe_complex[**P](
    f: Callable[P, Any], jaxfun, argnums: int | Sequence[int] = 0
) -> Callable[P, Any]:
    """Apply a JAX transform to functions with real or complex outputs.

    If ``f`` has real outputs, this delegates directly to ``jaxfun``. If ``f``
    has complex outputs, the real and imaginary parts are transformed
    separately and recombined into a complex result.

    Args:
        f: Function to transform.
        jaxfun: JAX transformation such as ``jax.grad``, ``jax.hessian``, or
            ``jax.value_and_grad``.
        argnums: Positional argument index or indices to transform with
            respect to.

    Returns:
        Wrapped function with the same call signature as ``f``.

    Type Parameters:
        P: Parameter specification of ``f``.
    """

    def transformed(*args: P.args, **kwargs: P.kwargs):
        if jnp.isrealobj(jax.eval_shape(f, *args, **kwargs)):
            return jaxfun(f, argnums=argnums)(*args, **kwargs)
        real_fun = jaxfun(with_real(f), argnums=argnums)
        imag_fun = jaxfun(with_imag(f), argnums=argnums)
        return jax.tree.map(
            lambda r, i: r + 1j * i,
            real_fun(*args, **kwargs),
            imag_fun(*args, **kwargs),
        )

    return transformed


def linearize_maybe_complex(f: Callable, *args) -> tuple[Any, Callable]:
    """Wraps ``jax.linearize`` to handle complex inputs/outputs.

    Splits complex values into real and imaginary parts if needed, or passes
    through if real.

    Args:
        f: The function to linearize.
        *args: Arguments to ``f``.

    Returns:
        A tuple ``(primal, jvp_fn)`` where ``primal`` is the value of ``f(*args)``
        and ``jvp_fn`` is the function that computes the Jacobian-vector product.
    """
    if jnp.isrealobj(jax.eval_shape(f, *args)):
        return jax.linearize(f, *args)
    real_primal, real_fun = jax.linearize(with_real(f), *args)
    imag_primal, imag_fun = jax.linearize(with_imag(f), *args)

    def transformed(*args):
        return jax.tree.map(lambda r, i: r + 1j * i, real_fun(*args), imag_fun(*args))

    return real_primal + 1j * imag_primal, transformed


def grad_maybe_complex[**P](
    f: Callable[P, Any], argnums: int | Sequence[int] = 0
) -> Callable[P, Any]:
    """Return ``jax.grad`` wrapped to support complex-valued outputs.

    Args:
        f: Function to differentiate.
        argnums: Positional argument index or indices to differentiate with
            respect to.

    Returns:
        Gradient function with the same call signature as ``f``.

    Type Parameters:
        P: Parameter specification of ``f``.
    """
    return transform_maybe_complex(f, jax.grad, argnums)


def hessian_maybe_complex[**P](
    f: Callable[P, Any], argnums: int | Sequence[int] = 0
) -> Callable[P, Any]:
    """Return ``jax.hessian`` wrapped to support complex-valued outputs.

    Args:
        f: Function to differentiate twice.
        argnums: Positional argument index or indices to differentiate with
            respect to.

    Returns:
        Hessian function with the same call signature as ``f``.

    Type Parameters:
        P: Parameter specification of ``f``.
    """
    return transform_maybe_complex(f, jax.hessian, argnums)


type CompatibleFunc[DataT: Data, **P] = Callable[Concatenate[Params, DataT, P], Any]
"""Callable that receives ``params`` first, ``data`` second, then extra args."""


def transform_with_data[DataT: Data, **P](
    f: CompatibleFunc[DataT, P], key: str, jaxfun
) -> CompatibleFunc[DataT, P]:
    r"""Make grad of functions like ``f(params, data, *args)``.

    Args:
        f: Function to grad.
        key: With respect to which part in data to take grad.
        jaxfun: the type of gradient to take, e.g. ``jax.grad``.

    Returns:
        The grad function.

    Type Parameters:
        DataT: Concrete ``Data`` subtype passed through ``f`` and the wrapper.
        P: Extra parameter specification after ``params`` and ``data``.
    """

    def wrap_f(params: Params, x: ArrayTree, data: DataT, *args, **kwargs):
        return f(params, data.merge({key: x}), *args, **kwargs)

    transformed_func = jaxfun(wrap_f, argnums=1)

    def wrap_transformed_func(params: Params, data: DataT, *args, **kwargs):
        return transformed_func(params, data[key], data, *args, **kwargs)

    return wrap_transformed_func
