# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Forward Laplacian tracing state."""

import contextlib
import contextvars
from collections.abc import Generator

from jax.experimental.xla_metadata import set_xla_metadata

_FORWARD_LAPLACIAN_TRACING = contextvars.ContextVar(
    "forward_laplacian_tracing", default=False
)


@contextlib.contextmanager
def forward_laplacian_tracing() -> Generator[None]:
    """Mark derivative tracing and isolate it from cached ordinary JIT traces."""
    token = _FORWARD_LAPLACIAN_TRACING.set(True)
    try:
        # Custom Laplacian calls stage differently in this trace. JAX includes
        # XLA metadata in both its tracing and JIT cache keys.
        with set_xla_metadata(jaqmc_forward_laplacian=True):
            yield
    finally:
        _FORWARD_LAPLACIAN_TRACING.reset(token)


def is_forward_laplacian_tracing() -> bool:
    """Return whether execution is currently tracing a Forward Laplacian."""
    return _FORWARD_LAPLACIAN_TRACING.get()
