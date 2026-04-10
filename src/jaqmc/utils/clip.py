# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import numpy as jnp

from jaqmc.utils.parallel_jax import BATCH_AXIS_NAME


def iqr_clip_real(x: jnp.ndarray, scale=100.0) -> jnp.ndarray:
    """Returns the clipped the observables based on interquartile range (IQR)."""
    all_x = jax.lax.all_gather(x, axis_name=BATCH_AXIS_NAME, tiled=True)
    q1 = jnp.nanquantile(all_x, 0.25)
    q3 = jnp.nanquantile(all_x, 0.75)
    iqr = q3 - q1
    return jnp.clip(x, q1 - scale * iqr, q3 + scale * iqr)


def iqr_clip(x: jnp.ndarray, scale=100.0) -> jnp.ndarray:
    """Returns the clipped complex observables by applying IQR clip.

    The clip is applied on real and imag parts separately.
    """
    if jnp.isrealobj(x):
        return iqr_clip_real(x, scale)
    return iqr_clip_real(x.real, scale) + 1j * iqr_clip_real(x.imag, scale)
