# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Primitive registry and named handler helpers for Forward Laplacian."""

from .core import (
    setup_handler,
    wrap_elementwise,
    wrap_general,
    wrap_linear,
    wrap_multiplication,
    wrap_without_fwd_laplacian,
)
from .custom_laplacian import AutoLaplacianFallback
from .registry import deregister_function, get_laplacian, register_function

__all__ = [
    "AutoLaplacianFallback",
    "deregister_function",
    "get_laplacian",
    "register_function",
    "setup_handler",
    "wrap_elementwise",
    "wrap_general",
    "wrap_linear",
    "wrap_multiplication",
    "wrap_without_fwd_laplacian",
]
