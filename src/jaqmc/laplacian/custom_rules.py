# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public helper-level API for custom Forward Laplacian rules."""

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, Protocol, cast

from jaqmc.array_types import PyTree

from .primitives.custom_laplacian import (
    bind_custom_laplacian,
    create_custom_laplacian_entry,
    set_custom_laplacian_rule,
)
from .tracing import is_forward_laplacian_tracing


class CustomLaplacianCallable[**P, R: PyTree](Protocol):
    """Callable returned by ``custom_laplacian`` with rule registration."""

    __name__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    def def_laplacian_rule[F: Callable[..., Any]](self, rule: F) -> F: ...


def _is_effectively_immutable(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_is_effectively_immutable(v) for v in value)
    return False


def _is_flax_bound_method_wrapper(rule: Callable) -> bool:
    """Return whether ``rule`` has Linen's generated bound-method shape."""
    try:
        from flax import linen as nn
    except ImportError:
        return False

    if not isinstance(getattr(rule, "__self__", None), nn.Module):
        return False

    closure = rule.__closure__ or ()
    if len(closure) != 1:
        return False

    wrapped_function = closure[0].cell_contents
    return inspect.isfunction(wrapped_function) and wrapped_function.__closure__ is None


def _warn_on_suspicious_rule_closure(rule: Callable):
    if _is_flax_bound_method_wrapper(rule):
        return

    closure = rule.__closure__ or ()
    captured = [
        cell.cell_contents
        for cell in closure
        if cell is not None and not _is_effectively_immutable(cell.cell_contents)
    ]
    if captured:
        warnings.warn(
            "custom_laplacian rule closes over nontrivial Python state. "
            "Because rules are recovered lazily from a registry, later mutations "
            "of captured state may not be visible to JAX staging. Prefer passing "
            "changing values as explicit function arguments.",
            stacklevel=3,
        )


def custom_laplacian[**P, R: PyTree](
    fn: Callable[P, R],
) -> CustomLaplacianCallable[P, R]:
    """Attach a custom Forward Laplacian rule to a function.

    Analogous to ``jax.custom_jvp`` but for the Forward Laplacian transform.
    The decorated function works normally outside ``forward_laplacian`` and
    only stages the custom primitive while ``forward_laplacian`` traces the
    surrounding function. When used inside ``forward_laplacian``, the
    registered rule receives ``LapTuple`` inputs and must return ``LapTuple``
    outputs with correctly propagated Jacobian and Laplacian.

    A rule must be registered via ``def_laplacian_rule`` before the function
    can be used inside ``forward_laplacian``. A registered rule may raise
    ``AutoLaplacianFallback`` to delegate unsupported cases back to the
    interpreter's dense auto-rule path.

    Supports pytree inputs and outputs (dicts, lists, tuples, nested
    structures).

    Notes:
        - Positional arguments only (no kwargs through the primitive).
        - The rule may receive a mix of ``LapTuple`` and plain ``jnp.ndarray``
          arguments (the latter for non-tracked inputs).
        - ``LapTuple.jacobian`` may be sparse. Use ``dense_jacobian`` unless
          the rule intentionally specializes a sparse payload layout.

    Returns:
        A wrapped callable with a ``def_laplacian_rule`` attribute.
    """
    custom_id = create_custom_laplacian_entry(fn)

    @functools.wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        if not is_forward_laplacian_tracing():
            return fn(*args, **kwargs)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                "custom_laplacian functions used inside forward_laplacian "
                f"accept positional arguments only; got keyword arguments: {names}."
            )
        return bind_custom_laplacian(fn, custom_id, args)

    def def_laplacian_rule[F: Callable[..., Any]](rule: F) -> F:
        _warn_on_suspicious_rule_closure(rule)
        set_custom_laplacian_rule(custom_id, rule)
        return rule

    wrapped.def_laplacian_rule = def_laplacian_rule  # type: ignore[attr-defined]
    return cast(CustomLaplacianCallable[P, R], wrapped)
