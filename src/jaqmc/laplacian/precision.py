# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Jacobian contraction precision modes for the Forward Laplacian transform."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import Any

import jax
from jax import numpy as jnp
from jax.extend.core import JaxprEqn

__all__ = [
    "ContractionMode",
    "current_contraction_mode",
    "eqn_contraction_scope",
    "jacobian_contraction_scope",
    "scaled_dot_general",
    "scaled_einsum",
]

_CONTRACTION_TAG = "jaqmc_jacobian_contraction="


class ContractionMode(StrEnum):
    """How Forward Laplacian Jacobian contractions are evaluated.

    Attributes:
        STANDARD: Contract in the operands' own dtype and precision.
        SCALED_FP16: Normalize operands by a power of two along their contracted
            axes, contract in FP16 with FP32 accumulation, and restore the
            magnitude of each result.
    """

    STANDARD = "standard"
    SCALED_FP16 = "scaled_fp16"


_RULE_MODE: ContextVar[ContractionMode] = ContextVar(
    "jaqmc_jacobian_rule_mode", default=ContractionMode.STANDARD
)


def current_contraction_mode() -> ContractionMode:
    """Return the contraction mode active during Forward Laplacian interpretation.

    Returns:
        The resolved equation mode, or ``STANDARD`` outside interpretation.
    """
    return _RULE_MODE.get()


@contextmanager
def jacobian_contraction_scope(
    mode: ContractionMode | str = ContractionMode.SCALED_FP16,
) -> Generator[None]:
    """Record a fixed ``mode`` on enclosed equations for Forward Laplacian.

    Tags are recorded during ordinary tracing too, so cached primal Jaxprs retain
    their derivative precision. An inner ``STANDARD`` scope overrides an outer
    scaled scope. Place scopes inside the traced function, not around calls to
    ``forward_laplacian``. Scopes do not bind ambient Python precision or
    partition JIT caches by mode.

    Args:
        mode: Jacobian contraction mode to apply.
    """
    resolved = ContractionMode(mode)
    with jax.named_scope(f"{_CONTRACTION_TAG}{resolved.value}"):
        yield


@contextmanager
def eqn_contraction_scope(eqn: JaxprEqn) -> Generator[None]:
    """Bind an equation's mode for kernels and recursive Jaxpr interpretation.

    Untagged equations inherit the enclosing call equation's mode. This carries
    fixed scopes across cached JIT bodies without retracing them per mode.
    """
    mode = _RULE_MODE.get()
    name_stack = str(eqn.source_info.name_stack)
    if _CONTRACTION_TAG in name_stack:
        tag = name_stack.rsplit(_CONTRACTION_TAG, 1)[1].split("/", 1)[0].rstrip(")")
        mode = ContractionMode(tag)
    token = _RULE_MODE.set(mode)
    try:
        yield
    finally:
        _RULE_MODE.reset(token)


def _scale_to_fp16(
    x: jnp.ndarray, contract_axes: tuple[int, ...]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    axes = tuple(a % x.ndim for a in contract_axes)
    x_fp32 = x.astype(jnp.float32)
    amax = jax.lax.stop_gradient(
        jnp.max(jnp.abs(x_fp32), axis=axes, keepdims=True, initial=jnp.float32(0.0))
    )
    _, exponent = jnp.frexp(amax)
    exponent = jnp.clip(exponent, -126, 126)
    scale = jnp.ldexp(jnp.float32(1.0), exponent)
    inv_scale = jnp.ldexp(jnp.float32(1.0), -exponent)
    return (x_fp32 * inv_scale).astype(jnp.float16), scale


def scaled_dot_general(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
    out_sharding: Any = None,
    mode: ContractionMode | None = None,
) -> jnp.ndarray:
    """Evaluate ``jax.lax.dot_general`` under the active contraction mode.

    In ``SCALED_FP16`` mode, each real operand is normalized by a power of two
    along its contracted axes, contracted in FP16 with FP32 accumulation, and
    rescaled to ``result_dtype``.

    Args:
        lhs: Left-hand side operand.
        rhs: Right-hand side operand.
        dimension_numbers: Contraction and batch axes
            ``((lhs_c, rhs_c), (lhs_b, rhs_b))``.
        precision: Optional JAX matmul precision for ``STANDARD`` mode.
        preferred_element_type: Optional accumulation/output dtype.
        out_sharding: Optional output sharding passed to ``dot_general``.
        mode: Explicit mode override. Defaults to :func:`current_contraction_mode`.

    Returns:
        The contracted array in the promoted output dtype.
    """
    sharding_kw = {} if out_sharding is None else {"out_sharding": out_sharding}
    resolved = current_contraction_mode() if mode is None else mode
    if (
        resolved is ContractionMode.STANDARD
        or jnp.iscomplexobj(lhs)
        or jnp.iscomplexobj(rhs)
    ):
        return jax.lax.dot_general(
            lhs,
            rhs,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            **sharding_kw,
        )
    result_dtype = jnp.result_type(
        lhs.dtype, rhs.dtype, preferred_element_type or lhs.dtype
    )
    (lhs_c, rhs_c), _ = dimension_numbers
    lhs_fp16, s_lhs = _scale_to_fp16(lhs, tuple(lhs_c))
    rhs_fp16, s_rhs = _scale_to_fp16(rhs, tuple(rhs_c))
    out = jax.lax.dot_general(
        lhs_fp16,
        rhs_fp16,
        dimension_numbers=dimension_numbers,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
        **sharding_kw,
    )
    scale = jax.lax.dot_general(
        s_lhs,
        s_rhs,
        dimension_numbers=dimension_numbers,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    return (out * scale).astype(result_dtype)


def scaled_einsum(
    subscripts: str,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    *,
    precision: Any = None,
    preferred_element_type: Any = None,
    mode: ContractionMode | None = None,
) -> jnp.ndarray:
    """Evaluate a two-operand contraction ``"lhs,rhs->out"`` under the active mode.

    Lowers directly to :func:`scaled_dot_general` and an optional transpose.
    Each subscript term must use unique axis labels without ellipsis (``...``),
    and every axis omitted from ``out`` must appear in both ``lhs`` and ``rhs``
    (single-operand reductions and traces are not supported).

    Args:
        subscripts: Explicit two-operand contraction string ``"lhs,rhs->out"``.
        lhs: Left-hand side operand.
        rhs: Right-hand side operand.
        precision: Optional JAX matmul precision for ``STANDARD`` mode.
        preferred_element_type: Optional accumulation/output dtype.
        mode: Explicit mode override. Defaults to :func:`current_contraction_mode`.

    Returns:
        The contracted array in the promoted output dtype.

    Raises:
        ValueError: If ``subscripts`` is not a supported two-operand contraction.
    """
    in_subs, out_sub = subscripts.replace(" ", "").split("->")
    lhs_sub, rhs_sub = in_subs.split(",")
    contract_chars = [c for c in lhs_sub if c in rhs_sub and c not in out_sub]
    batch_chars = [c for c in out_sub if c in lhs_sub and c in rhs_sub]
    dot_sub = "".join(
        [
            *batch_chars,
            *(c for c in lhs_sub if c not in contract_chars and c not in batch_chars),
            *(c for c in rhs_sub if c not in contract_chars and c not in batch_chars),
        ]
    )
    if not (
        "." not in subscripts
        and len(set(lhs_sub)) == len(lhs_sub) == lhs.ndim
        and len(set(rhs_sub)) == len(rhs_sub) == rhs.ndim
        and len(set(out_sub)) == len(out_sub) == len(dot_sub)
        and set(out_sub) == set(dot_sub)
    ):
        raise ValueError(f"Unsupported scaled_einsum subscripts: {subscripts!r}")
    dn = (
        (
            tuple(lhs_sub.index(c) for c in contract_chars),
            tuple(rhs_sub.index(c) for c in contract_chars),
        ),
        (
            tuple(lhs_sub.index(c) for c in batch_chars),
            tuple(rhs_sub.index(c) for c in batch_chars),
        ),
    )
    out = scaled_dot_general(
        lhs,
        rhs,
        dn,
        precision=precision,
        preferred_element_type=preferred_element_type,
        mode=mode,
    )
    if dot_sub != out_sub:
        out = jnp.transpose(out, tuple(dot_sub.index(c) for c in out_sub))
    return out
