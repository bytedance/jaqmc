# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from jax import numpy as jnp

from jaqmc.utils import parallel_jax


def iqr_clip(x: jnp.ndarray, scale: float = 100.0) -> jnp.ndarray:
    """Clip observables using an interquartile-range window.

    For complex numbers, the clip is applied on real and imag parts separately.

    Returns:
        Values clipped to the IQR-based window.
    """
    if jnp.iscomplexobj(x):
        return iqr_clip(x.real, scale) + 1j * iqr_clip(x.imag, scale)
    all_x = parallel_jax.all_gather(x)
    q1 = jnp.nanquantile(all_x, 0.25)
    q3 = jnp.nanquantile(all_x, 0.75)
    iqr = q3 - q1
    return jnp.clip(x, q1 - scale * iqr, q3 + scale * iqr)


def mad_clip(x: jnp.ndarray, scale: float = 100.0) -> jnp.ndarray:
    """Clip observables using a median absulute deviation window.

    For complex numbers, the clip is applied on real and imag parts separately.

    Returns:
        Values clipped to the median-centered mean-abs-deviation window.
    """
    if jnp.iscomplexobj(x):
        return mad_clip(x.real, scale) + 1j * mad_clip(x.imag, scale)
    all_x = parallel_jax.all_gather(x)
    median = jnp.nanmedian(all_x)
    absdev = jnp.nanmedian(jnp.abs(all_x - median))
    return jnp.clip(x, median - scale * absdev, median + scale * absdev)


def clip_observable(
    x: jnp.ndarray,
    method: Literal["iqr", "mad", "none"],
    scale: float = 100.0,
) -> jnp.ndarray:
    """Clip observables with a named method.

    Args:
        x: Observable values to clip.
        method: One of ``"iqr"``, ``"mad"``, or ``"none"``.
        scale: Width multiplier for methods that use a clipping window.

    Returns:
        Clipped values, or the original values when ``method`` is ``"none"``.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method == "iqr":
        return iqr_clip(x, scale)
    if method == "mad":
        return mad_clip(x, scale)
    if method == "none":
        return x
    raise ValueError(f"Unknown clip method {method!r}.")
