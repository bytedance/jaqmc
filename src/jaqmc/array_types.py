# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp
from jax.typing import ArrayLike

type PyTree = Any

type ArrayTree = jnp.ndarray | Sequence[ArrayTree] | Mapping[str, ArrayTree]
"""Native PyTree whose leaves are JAX arrays."""

type ArrayLikeTree = ArrayLike | Sequence[ArrayLikeTree] | Mapping[str, ArrayLikeTree]
"""Native PyTree whose leaves can be converted to JAX arrays."""

type ArrayDict = dict[str, jnp.ndarray]
type NumpyArrayDict = dict[str, np.ndarray]

type PRNGKey = jax.Array
"""Random key array created by :func:`jax.random.key`."""

type Params = ArrayTree
"""Parameter PyTree for wavefunction."""


def is_jax_key(x: Any) -> bool:
    """Return whether ``x`` is a typed JAX random key array."""
    return isinstance(x, jax.Array) and jax.dtypes.issubdtype(
        x.dtype, jax.dtypes.prng_key
    )
