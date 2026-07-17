# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Forward Laplacian tracing state."""

import contextlib
import contextvars

_FORWARD_LAPLACIAN_TRACING = contextvars.ContextVar(
    "forward_laplacian_tracing", default=False
)


@contextlib.contextmanager
def forward_laplacian_tracing():
    """Mark the current trace as being under ``forward_laplacian``."""
    token = _FORWARD_LAPLACIAN_TRACING.set(True)
    try:
        yield
    finally:
        _FORWARD_LAPLACIAN_TRACING.reset(token)


def is_forward_laplacian_tracing() -> bool:
    """Return whether execution is currently tracing a Forward Laplacian."""
    return _FORWARD_LAPLACIAN_TRACING.get()
